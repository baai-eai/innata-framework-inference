# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-local exact UND and compact metadata reuse for small Action towers.

UND is causally self-contained: no timestep, VIDEO or ACTION feeds its layers.
Keep separate instances for cond/uncond and discard them after each request.
"""

from dataclasses import dataclass

import torch

from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryState, MemoryValue


@dataclass
class SmallActionMemoryValue(MemoryValue):
    und_output: torch.Tensor | None = None
    und_key: torch.Tensor | None = None
    und_value: torch.Tensor | None = None


class SmallActionMemoryState(MemoryState):
    def __init__(self, num_layers: int, *, reuse_und: bool = True):
        self.reuse_und = reuse_und
        self.layers = [SmallActionMemoryValue() for _ in range(num_layers)]
        self.compact_metadata: dict | None = None

    def init(self, hidden_states: dict, device: torch.device) -> None:
        pass

    def read_for_layer(self, layer_idx: int) -> SmallActionMemoryValue | None:
        return self.layers[layer_idx] if self.reuse_und else None

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        output, _, key, value = kv_to_store
        # Own these tensors: compiled/graph outputs may be overwritten later.
        self.layers[layer_idx] = SmallActionMemoryValue(*(t.detach().clone() for t in (output, key, value)))

    def is_gen_only(self) -> bool:
        # Unlike legacy GEN-only memory, preserve UND layout for three-tower routing.
        return False

    def requires_natten_metadata(self) -> bool:
        return False
