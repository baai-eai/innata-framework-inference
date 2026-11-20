# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Window-local DROID latent datasets; dynamic text transforms remain online."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.offline_vae import VaeLatentRef


class OfflineActionDataset(Dataset):
    """Read immutable raw action metadata and window-local VAE latents.

    The manifest keeps episode blocks and original source indices. No pixels are
    decoded, and CFG/dropout is evaluated anew by the normal action transform.
    """

    def __init__(self, manifest_path, tokenizer_config=None):
        self.root = Path(manifest_path).resolve().parent
        self.manifest = json.loads(Path(manifest_path).read_text())
        if self.manifest.get("format") != "cosmos_action_latents_v1" or not self.manifest.get("complete"):
            raise ValueError("Action latent manifest must be complete and use cosmos_action_latents_v1")
        self.records = self.manifest["records"]
        self.blocks = [tuple(x) for x in self.manifest["blocks"]]
        if sum(n for _, n in self.blocks) != len(self.records):
            raise ValueError("Action latent block coverage does not match record count")
        self.transform = ActionTransformPipeline(tokenizer_config=tokenizer_config, **self.manifest["transform"])
        self.resolution = self.manifest["resolution"]

    def __len__(self):
        return len(self.records)

    def get_shuffle_blocks(self):
        return self.blocks

    def __getitem__(self, index):
        record = self.records[index]
        path = str(self.root / record["shard"])
        prefix = record["key"]
        sample = dict(record["metadata"])
        choices = sample.pop("_offline_caption_choices", None)
        if choices is not None:
            sample["ai_caption"] = random.choice(choices)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in record["tensor_keys"]:
                sample[key] = f.get_tensor(prefix + "." + key)
        # Caption augmentors inspect only the shape. No pixel buffer is allocated.
        sample["video"] = torch.zeros((), dtype=torch.uint8).expand(record["video_shape"])
        sample["_offline_action_media"] = True
        sample = self.transform(sample, self.resolution)
        sample.pop("video")
        sample["video_latent_ref"] = VaeLatentRef(
            path=path,
            tensor_key=prefix + ".latent",
            latent_shape=tuple(record["latent_shape"]),
            target_fps=float(sample["conditioning_fps"]),
            catalog_id=str(record["source_index"]),
            dataset="droid_offline",
        )
        sample["source_sample_index"] = torch.tensor(record["source_index"], dtype=torch.int64)
        return sample


class SelectedActionDataset(Dataset):
    """Online control with exactly the offline manifest's indices and blocks."""

    def __init__(self, dataset, manifest_path):
        self.dataset = dataset
        manifest = json.loads(Path(manifest_path).read_text())
        self.indices = [r["source_index"] for r in manifest["records"]]
        self.blocks = [tuple(x) for x in manifest["blocks"]]

    def __len__(self):
        return len(self.indices)

    def get_shuffle_blocks(self):
        return self.blocks

    def __getitem__(self, index):
        sample = self.dataset[self.indices[index]]
        sample["source_sample_index"] = torch.tensor(self.indices[index], dtype=torch.int64)
        return sample
