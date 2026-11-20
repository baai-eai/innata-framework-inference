# Copyright (c) 2026, BAAI. All rights reserved.
# Adapted from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/data/generator/processors/qwen3vl_processor.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import logging
import os
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers.models.auto.processing_auto import AutoProcessor

from cosmos_framework.data.generator.processors.cosmos3_edge_processing import (
    build_cosmos3_edge_processor,
    is_cosmos3_edge_native_snapshot,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.reasoner.pretrained_models_downloader import maybe_download_hf_model_from_s3

_ENABLE_THINKING_PROCESSOR_WARNING = (
    "Keyword argument `enable_thinking` is not a valid argument for this processor and will be ignored."
)


class _EnableThinkingWarningFilter(logging.Filter):
    """Drop a Transformers false-positive while preserving chat-template kwargs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != _ENABLE_THINKING_PROCESSOR_WARNING


@contextmanager
def _suppress_enable_thinking_processor_warning(enabled: bool):
    """Suppress the warning caused when HF leaks a template kwarg into ``Processor.__call__``."""
    if not enabled:
        yield
        return
    logger = logging.getLogger("transformers.processing_utils")
    warning_filter = _EnableThinkingWarningFilter()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)


def convert_string_content_to_list_content(messages: List[Dict]) -> List[Dict]:
    """Normalize chat messages to Qwen's typed content-list form."""
    for index, message in enumerate(messages):
        if isinstance(message["content"], str):
            messages[index]["content"] = [{"type": "text", "text": message["content"]}]
    return messages


def maybe_parse_video_content(
    messages: List[Dict],
) -> tuple[int, Optional[list[float]], Optional[list[int]], Optional[list[list[int]]]]:
    """Collect metadata for videos that were already decoded into frame lists."""
    video_fps: list[float] = []
    video_total_num_frames: list[int] = []
    video_frames_indices: list[list[int]] = []
    for message in messages:
        if not isinstance(message["content"], list):
            continue
        for content in message["content"]:
            if content.get("type") != "video" or not isinstance(content.get("video"), list):
                continue
            fps = content.get("fps")
            if fps is None:
                log.critical(f"fps is None for video {content}. Better to set the fps explicitly", rank0_only=False)
            video_fps.append(fps)
            video_total_num_frames.append(len(content["video"]))
            video_frames_indices.append(list(range(video_total_num_frames[-1])))
    return len(video_fps), video_fps, video_total_num_frames, video_frames_indices


def maybe_get_max_pixels_from_images_kwargs(messages: List[Dict]) -> tuple[Optional[int], Optional[int]]:
    """Read an optional image pixel budget from the first matching content item."""
    for message in messages:
        if not isinstance(message["content"], list):
            continue
        for content in message["content"]:
            if content.get("type") == "image" and content.get("max_pixels") is not None:
                return content["max_pixels"], content.get("min_pixels")
    return None, None


class Qwen3VLProcessor:
    """The single public training processor for Cosmos3 Nano and Edge."""

    # Qwen3-VL does not expose a single vision-end token in its vocabulary, so
    # leave ``vision_end_id`` unset (None) rather than the legacy ``</img>``,
    # which silently resolved to the UNK token id.
    VISION_END_TOKEN: Optional[str] = None

    def __init__(
        self,
        name: str = "Qwen/Qwen3-VL-2B-Init",
        credentials: str = "",
        bucket: str = "",
        cache_dir: Optional[str] = None,
    ):
        self.name = name
        if os.path.isdir(name):
            local_path = name
        else:
            local_path = maybe_download_hf_model_from_s3(
                name,
                credentials,
                bucket,
                include_model_weights=False,
                cache_dir=cache_dir,
            )
        # The processor and visual tower belong to the same family-native VLM
        # snapshot. Keep its resolved local path so the unified model can build
        # the matching visual module without a second checkpoint-path option.
        self.checkpoint_path = local_path
        if is_cosmos3_edge_native_snapshot(local_path):
            self.processor = build_cosmos3_edge_processor(local_path)
        else:
            self.processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
        log.info("Successfully loaded processor from local cache")

        self.image_token_id = (
            self.tokenizer.convert_tokens_to_ids(self.processor.image_token)
            if hasattr(self.processor, "image_token")
            else None
        )
        self.video_token_id = (
            self.tokenizer.convert_tokens_to_ids(self.processor.video_token)
            if hasattr(self.processor, "video_token")
            else None
        )
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_id
        self.vision_end_id = None
        # Helper attributes consumed by the dataloader video decoding path.
        shortest_edge = self.processor.image_processor.size["shortest_edge"]
        self.min_height_width = int(np.sqrt(shortest_edge))
        self.patch_size = self.processor.video_processor.patch_size
        self.temporal_patch_size = self.processor.video_processor.temporal_patch_size
        self.merge_size = self.processor.video_processor.merge_size
        self.use_smart_resize = True

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def tokenize_text(
        self,
        caption: str,
        is_video: bool = False,
        use_system_prompt: bool = False,
        system_prompt: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        strip_empty_thinking: bool = False,
    ) -> list[int]:
        return tokenize_caption(
            caption,
            self.tokenizer,
            is_video=is_video,
            use_system_prompt=use_system_prompt,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            strip_empty_thinking=strip_empty_thinking,
        )

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=False,
        return_tensors="pt",
        tokenize=True,
        **kwargs,
    ):
        """
        Return:
            inputs: dict
                input_ids: torch.Tensor, shape: (N_token)
                attention_mask: torch.Tensor, shape: (N_token)
                texts: str, the raw text
                image_sizes: torch.Tensor, shape (N_img, 2)
                pixel_values: torch.Tensor, shape (N_img_patch, 3, 224, 224)
        """
        assert tokenize, "tokenize must be True"
        assert return_tensors == "pt", "return_tensors must be pt"
        # Note: this tokenizer does not support "content": str, it always expect "content" entry to be a list of dicts
        messages = convert_string_content_to_list_content(messages)
        # Preserve chat-template controls supplied by callers.  Generation
        # catalogs pass ``enable_thinking=False`` so the modality-control token
        # can immediately follow the native assistant prefix.
        processor_kwargs = dict(kwargs)
        max_pixels, min_pixels = maybe_get_max_pixels_from_images_kwargs(messages)

        if max_pixels is not None:
            min_pixels = min_pixels if min_pixels is not None else self.processor.image_processor.size["shortest_edge"]
            processor_kwargs["images_kwargs"] = {"max_pixels": max_pixels, "min_pixels": min_pixels}

        num_video, video_fps, video_total_num_frames, video_frames_indices = maybe_parse_video_content(messages)
        if num_video > 0:
            # Here we add the args to avoid the error:
            # File "/invalid_dir", line 321, in _decode_and_sample_videos
            #     raise ValueError(
            # ValueError: Sampling frames from a list of images is not supported! Set `do_sample_frames=False`.
            video_metadata = [
                dict(fps=fps, total_num_frames=total_num_frames, frames_indices=frames_indices)
                for fps, total_num_frames, frames_indices in zip(
                    video_fps,
                    video_total_num_frames,
                    video_frames_indices,
                )
            ]
            processor_kwargs["videos_kwargs"] = {
                "do_sample_frames": False,
                "video_metadata": video_metadata[0] if num_video == 1 else video_metadata,
            }
        # ``enable_thinking`` is a valid Jinja chat-template control. In the
        # Transformers version used by the training image, ProcessorMixin also
        # leaks custom template kwargs into ``Processor.__call__`` and emits an
        # invalid-processor-argument warning for every dataloader worker. Keep
        # forwarding the control so tokenization is unchanged, and suppress
        # only that known false-positive warning around the upstream call.
        with _suppress_enable_thinking_processor_warning("enable_thinking" in processor_kwargs):
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
                return_tensors=return_tensors,
                **processor_kwargs,
            )

        # Convert batch features into single features
        # By default, the processor returns a batch of features, but we use processor in dataloader, so we need to convert it to single features
        inputs["input_ids"] = inputs["input_ids"][0]  # [N_token]
        inputs["attention_mask"] = inputs["attention_mask"][0]  # [N_token]
        return inputs

    def add_assistant_tokens_mask(self, tokens):
        """
        Add a mask to the assistant tokens.
        This is used to mask out tokens that are not generated by the assistant (e.g.,  system prompts, user prompts, chat templates), such that in the loss computation, only the tokens generated by the assistant are used.
        If there are multiple turns in the conversation, the mask will mask all the assistant tokens in each turn.

        Args:
            tokens (Union[List[int], torch.Tensor]): The tokens to add the mask to.
        Returns:
            Union[List[bool], torch.Tensor]: The mask. True for tokens generated by the assistant (i.e. should apply loss on), False for tokens not generated by the assistant.
        """
        if isinstance(tokens, torch.Tensor) and tokens.ndim == 2:
            mask = torch.stack(
                [self.add_assistant_tokens_mask(tokens[i]) for i in range(tokens.shape[0])]
            )  # [B,N_token]
            assert mask.shape == tokens.shape
            return mask
        np_tokens = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else np.array(tokens)
        assert np_tokens.ndim == 1

        # Constants defining chat boundaries.
        BOS_TOKEN = "<|im_start|>"
        EOS_TOKEN = "<|im_end|>"

        # Match the complete native role header.  Qwen tokenizers do not all
        # encode ``assistant`` as one token (Cosmos3-Edge splits it), so a
        # single convert_tokens_to_ids("assistant") check silently masks every
        # target in those checkpoints.
        bos_token_id = self.processor.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        eos_token_id = self.processor.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        role_ids = self.processor.tokenizer.encode("assistant\n", add_special_tokens=False)

        # Locate all positions where the start and end markers appear.
        start_indices = np.where(np_tokens == bos_token_id)[0]
        end_indices = np.where(np_tokens == eos_token_id)[0]

        # Initialize the mask with False values.
        masks = np.zeros_like(np_tokens, dtype=bool)
        assert len(start_indices) == len(end_indices)
        # For each pair of bos/eos, check if the role is 'assistant'
        # and apply the mask accordingly.
        for start, end in zip(start_indices, end_indices):
            header_start = start + 1
            content_start = header_start + len(role_ids)
            if content_start <= end and np.array_equal(np_tokens[header_start:content_start], np.asarray(role_ids)):
                # The assistant turn terminator is itself an autoregressive
                # target: the model must learn when text reasoning is done.
                masks[content_start : end + 1] = True

        # Role/template/media placeholders are conditioning only. im_end is the
        # one special-token exception because it terminates assistant generation.
        for special_id in self.processor.tokenizer.all_special_ids:
            if special_id == eos_token_id:
                continue
            masks[np_tokens == special_id] = False
        vocabulary = self.processor.tokenizer.get_vocab()
        for token in ("<think>", "</think>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"):
            if token in vocabulary:
                masks[np_tokens == self.processor.tokenizer.convert_tokens_to_ids(token)] = False

        assert masks.shape == np_tokens.shape
        if isinstance(tokens, torch.Tensor):
            return torch.from_numpy(masks)
        else:
            return masks.tolist()


if __name__ == "__main__":
    """
    PYTHONPATH=. python3 cosmos_framework/data/generator/processors/qwen3vl_processor.py

    For image, expected output:
        input_ids: type: <class 'torch.Tensor'>      shape: torch.Size([2772])
        attention_mask: type: <class 'torch.Tensor'> shape: torch.Size([2772])
        pixel_values: type: <class 'torch.Tensor'>   shape: torch.Size([11008, 1536])
        image_grid_thw: tensor([[  1,  86, 128]])
        num_image_token_id_tokens: 2752
        decoded_assistant_tokens: Paris

    For video, expected output:
        input_ids: type: <class 'torch.Tensor'>      shape: torch.Size([5538])
        pixel_values_videos: type: <class 'torch.Tensor'> shape: torch.Size([22016, 1536])
        video_grid_thw: tensor([[  2,  86, 128]])
        num_video_token_id_tokens: 5504
        decoded_assistant_tokens: Paris
    """
    processor = Qwen3VLProcessor("Qwen/Qwen3-VL-2B-Init")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": ["https://invalid_url"] * 4,
                    "fps": 12,
                },
                {"type": "text", "text": "What is the capital of France?"},
            ],
        },
        {"role": "assistant", "content": "Paris"},
    ]
    inputs = processor.apply_chat_template(messages)
    input_ids = inputs["input_ids"]
    decoded_text = processor.decode(input_ids, skip_special_tokens=True)
    print(decoded_text)
    print(inputs.keys())
    for k, v in inputs.items():
        print(f"{k}: type: {type(v)}")
        if isinstance(v, torch.Tensor):
            print(f"shape: {v.shape}")
        if "grid" in k:
            print(f"{k}: {v}")
    num_image_token_id_tokens = inputs["input_ids"] == processor.image_token_id
    print(f"num_image_token_id_tokens: {num_image_token_id_tokens.sum()}")
    num_video_token_id_tokens = inputs["input_ids"] == processor.video_token_id
    print(f"num_video_token_id_tokens: {num_video_token_id_tokens.sum()}")

    assistant_tokens_mask = processor.add_assistant_tokens_mask(inputs["input_ids"])
    print(f"assistant_tokens_mask: {assistant_tokens_mask.sum()}")
    assistant_tokens = inputs["input_ids"][assistant_tokens_mask]
    print(f"assistant_tokens: {assistant_tokens}")
    decoded_assistant_tokens = processor.decode(assistant_tokens, skip_special_tokens=False)
    print(f"decoded_assistant_tokens: {decoded_assistant_tokens}")

    print("done")
