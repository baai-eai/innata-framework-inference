# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Load visual-generation inference samples from unified Parquet catalogs."""

import glob
import json
import math
import re
from pathlib import Path
from typing import Any

import pydantic

from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.inference.args import OmniSampleOverrides

_CATALOG_COLUMNS = {"id", "type", "dataset", "data", "provenance"}


class InferenceCatalogError(ValueError):
    """A catalog row cannot be represented by the offline inference CLI."""


def _decode_object(value: Any, *, field: str, row_id: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise InferenceCatalogError(f"catalog row {row_id!r} has invalid JSON in {field}") from error
    if not isinstance(value, dict):
        raise InferenceCatalogError(f"catalog row {row_id!r} requires an object in {field}")
    return value


def _sample_name(row_id: str, provenance: dict[str, Any]) -> str:
    extra = provenance.get("extra")
    if isinstance(extra, dict):
        index = extra.get("eval_index")
        mode = extra.get("eval_mode") or extra.get("mode")
        if isinstance(index, int) and not isinstance(index, bool) and isinstance(mode, str) and mode:
            safe_mode = re.sub(r"[^A-Za-z0-9._-]+", "_", mode).strip("_")
            return f"{index + 1:03d}_{safe_mode}"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", row_id).strip("_")
    if not name:
        raise InferenceCatalogError(f"catalog row {row_id!r} cannot produce a non-empty sample name")
    return name


def _prompt_text(value: Any) -> str:
    """Unwrap the catalog's legacy single-field caption envelope."""

    text = str(value or "")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(decoded, dict) and set(decoded) == {"caption"} and isinstance(decoded["caption"], str):
        return decoded["caption"]
    return text


def _nearest_video_bucket(width: int, height: int) -> tuple[str, str, tuple[int, int]]:
    candidates = [
        (resolution, aspect_ratio, size)
        for resolution, aspect_ratios in VIDEO_RES_SIZE_INFO.items()
        for aspect_ratio, size in aspect_ratios.items()
        if resolution in {"256", "480", "720", "768"}
    ]
    resolution, aspect_ratio, size = min(
        candidates,
        key=lambda candidate: (
            abs(math.log((width / height) / (int(candidate[1].split(",")[0]) / int(candidate[1].split(",")[1])))),
            abs(math.log(width / candidate[2][0])) + abs(math.log(height / candidate[2][1])),
        ),
    )
    return resolution, aspect_ratio, size


def _catalog_row_to_overrides(
    row: dict[str, Any],
    *,
    source: Path,
    global_overrides: pydantic.BaseModel | None,
) -> OmniSampleOverrides:
    row_id = str(row.get("id", ""))
    if row.get("type") != "visual_generation":
        raise InferenceCatalogError(
            f"catalog row {row_id!r} has type={row.get('type')!r}; only visual_generation video rows are supported"
        )
    data = _decode_object(row.get("data"), field="data", row_id=row_id)
    provenance = _decode_object(row.get("provenance"), field="provenance", row_id=row_id)
    messages = data.get("messages")
    if not isinstance(messages, list):
        raise InferenceCatalogError(f"catalog row {row_id!r} requires data.messages")

    prompt_parts: list[str] = []
    user_media: list[str] = []
    generation_parts: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise InferenceCatalogError(f"catalog row {row_id!r} contains a non-object message")
        role = message.get("role")
        content = message.get("content", [])
        if not isinstance(content, list):
            content = [{"type": "text", "text": content}]
        for part in content:
            if not isinstance(part, dict):
                raise InferenceCatalogError(f"catalog row {row_id!r} contains non-object message content")
            kind = part.get("type")
            if role == "user" and kind == "text":
                prompt_parts.append(_prompt_text(part.get("text")))
            elif role == "user" and kind in {"image", "video"}:
                user_media.append(str(kind))
            elif role == "assistant" and kind in {"gen_image", "gen_video"}:
                generation_parts.append(part)

    if user_media:
        raise InferenceCatalogError(
            f"catalog row {row_id!r} is media-conditioned ({user_media}); this input path currently supports T2V only"
        )
    if len(generation_parts) != 1 or generation_parts[0].get("type") != "gen_video":
        raise InferenceCatalogError(f"catalog row {row_id!r} requires exactly one assistant gen_video declaration")
    prompt = "\n".join(prompt_parts).strip()
    if not prompt:
        raise InferenceCatalogError(f"catalog row {row_id!r} has an empty user prompt")

    generation = generation_parts[0]
    num_frames = generation.get("valid_frames", generation.get("num_frames"))
    fps = generation.get("fps")
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        raise InferenceCatalogError(f"catalog row {row_id!r} requires a positive valid_frames/num_frames")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0 or not float(fps).is_integer():
        raise InferenceCatalogError(f"catalog row {row_id!r} requires a positive integer-valued fps")
    conditioning_latent_frames = generation.get("conditioning_latent_frames", 0)
    if (
        isinstance(conditioning_latent_frames, bool)
        or not isinstance(conditioning_latent_frames, int)
        or conditioning_latent_frames not in {0, 1, 2}
    ):
        raise InferenceCatalogError(f"catalog row {row_id!r} requires conditioning_latent_frames to be 0, 1, or 2")

    sample_data: dict[str, Any] = {
        "model_mode": "text2video",
        "name": _sample_name(row_id, provenance),
        "prompt": prompt,
        "fps": int(fps),
        "num_frames": num_frames,
    }
    if conditioning_latent_frames:
        target_path = generation.get("gen_video")
        if not isinstance(target_path, str) or not target_path:
            raise InferenceCatalogError(
                f"catalog row {row_id!r} requires a gen_video path for clean-prefix conditioning"
            )
        if "://" not in target_path:
            path = Path(target_path).expanduser()
            target_path = str(path if path.is_absolute() else source.parent / path)
        sample_data.update(
            {
                # One clean latent is TI2V (one pixel frame for the causal Wan VAE);
                # two clean latents are the short-prefix V2V training mode.
                "model_mode": "image2video" if conditioning_latent_frames == 1 else "video2video",
                "vision_path": target_path,
                "condition_frame_indexes_vision": list(range(conditioning_latent_frames)),
                "condition_video_keep": "first",
            }
        )
    provenance_extra = provenance.get("extra")
    if isinstance(provenance_extra, dict):
        seed = provenance_extra.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            sample_data["seed"] = seed

    width, height = generation.get("width"), generation.get("height")
    catalog_metadata: dict[str, Any] = {
        "id": row_id,
        "dataset": row.get("dataset"),
        "source": str(source),
        "conditioning_latent_frames": conditioning_latent_frames,
    }
    if all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (width, height)):
        resolution, aspect_ratio, output_size = _nearest_video_bucket(width, height)
        sample_data.update({"resolution": resolution, "aspect_ratio": aspect_ratio})
        catalog_metadata.update(
            {
                "source_width": width,
                "source_height": height,
                "output_width": output_size[0],
                "output_height": output_size[1],
            }
        )
    sample_data["extra"] = {"catalog": catalog_metadata}

    if global_overrides is not None:
        override_data = global_overrides.model_dump(exclude_none=True)
        for key, value in override_data.items():
            if key == "extra" and isinstance(value, dict):
                sample_data["extra"] = {**sample_data["extra"], **value}
            else:
                sample_data[key] = value
    return OmniSampleOverrides.model_validate(sample_data)


def _catalog_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(item for item in path.rglob("*.parquet") if item.is_file())
        if not files:
            raise FileNotFoundError(f"No Parquet files found under inference input directory: {path}")
        return files
    if path.suffix.lower() == ".parquet":
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]
    raise InferenceCatalogError(f"Not a Parquet catalog input: {path}")


def _load_catalog(path: Path, *, global_overrides: pydantic.BaseModel | None = None) -> list[OmniSampleOverrides]:
    import pyarrow.parquet as pq

    samples: list[OmniSampleOverrides] = []
    for parquet_path in _catalog_files(path):
        parquet = pq.ParquetFile(parquet_path)
        missing = _CATALOG_COLUMNS - set(parquet.schema_arrow.names)
        if missing:
            raise InferenceCatalogError(f"Parquet catalog {parquet_path} is missing columns: {sorted(missing)}")
        for batch in parquet.iter_batches(columns=sorted(_CATALOG_COLUMNS), batch_size=1024):
            for row in batch.to_pylist():
                samples.append(_catalog_row_to_overrides(row, source=parquet_path, global_overrides=global_overrides))
    return samples


def load_inference_inputs(
    paths: list[Path], *, global_overrides: pydantic.BaseModel | None = None
) -> list[OmniSampleOverrides]:
    """Load ordinary JSON/JSONL/YAML inputs and visual-generation Parquet catalogs."""

    if not paths:
        raise ValueError("No inference parameter files")
    expanded: list[Path] = []
    for path in paths:
        pattern = str(path)
        if glob.has_magic(pattern):
            expanded.extend(Path(match) for match in glob.glob(pattern, recursive=True))
        else:
            expanded.append(path)
    expanded = sorted(set(expanded))
    if not expanded:
        raise FileNotFoundError(f"Inference input patterns matched no paths: {[str(path) for path in paths]}")

    catalog_paths = [path for path in expanded if path.is_dir() or path.suffix.lower() == ".parquet"]
    regular_paths = [path for path in expanded if path not in catalog_paths]
    samples = OmniSampleOverrides.from_files(regular_paths, overrides=global_overrides) if regular_paths else []
    for path in catalog_paths:
        samples.extend(_load_catalog(path, global_overrides=global_overrides))

    names: set[str] = set()
    for sample in samples:
        if not sample.name:
            raise InferenceCatalogError("Every inference sample requires a name")
        if sample.name in names:
            raise InferenceCatalogError(f"Duplicate inference sample name: {sample.name!r}")
        names.add(sample.name)
    return samples
