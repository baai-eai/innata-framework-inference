# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-block attention for the three physically separate small towers.

The reference SDPA mask is dense over the entire pack. A BlockMask lets Flex
skip cross-sample tiles while preserving the same within-sample visibility.
Metadata is constructed once outside the checkpointed/compiled decoder layers.
The optional VIDEO Flash3 path groups KV by sample; UND/ACTION keep Flex.
Unpadded inference can use FA3 for all three streams, with causal UND attention.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_compiled_attention = torch.compile(flex_attention, dynamic=False)


class VideoVarlenMask(NamedTuple):
    kv_order: torch.Tensor
    cumulative_q: torch.Tensor
    cumulative_kv: torch.Tensor
    max_q: int
    max_kv: int
    causal: bool = False


def video_varlen_mask(query_ids, key_ids, num_samples, *, fixed_capacity=False, causal=False):
    """Group KV by sample; VIDEO queries already have sample order, then padding.

    Padding is a separate final sequence. Its outputs never enter real-token
    losses or later real-token attention. Keep physical tensor lengths static.
    Fixed-capacity packs use physical lengths as safe sequence upper bounds and
    symbolic cumulative-length metadata. This keeps changing sample counts and
    lengths from specializing the static decoder; token-compute shapes stay fixed.
    """
    q_counts = torch.bincount(query_ids + 1, minlength=num_samples + 1)
    k_counts = torch.bincount(key_ids + 1, minlength=num_samples + 1)

    def cumulative(counts):
        counts = torch.cat((counts[1:], counts[:1])).to(torch.int32)
        return torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))

    max_q = len(query_ids) if fixed_capacity else min(len(query_ids), ((int(q_counts.max().item()) + 127) // 128) * 128)
    # Every VIDEO query is also a KV token. A sequence can contain at most
    # max_q VIDEO tokens plus all the remaining physical UND/ACTION tokens.
    # Unlike exact caption maxima, this bound stays constant across pack92.
    max_kv = len(key_ids) - len(query_ids) + max_q
    order = torch.argsort(torch.where(key_ids >= 0, key_ids, num_samples), stable=True)
    cu_q, cu_kv = cumulative(q_counts), cumulative(k_counts)
    if fixed_capacity:
        for cu in (cu_q, cu_kv):
            torch._dynamo.mark_dynamic(cu, 0)
    return VideoVarlenMask(order, cu_q, cu_kv, max_q, max_kv, causal)


def video_varlen_attend(q, k, v, mask):
    from cosmos_framework.model.attention import attention
    from cosmos_framework.model.attention.masks import CausalType

    out = attention(
        q[None],
        k[mask.kv_order][None],
        v[mask.kv_order][None],
        backend="flash3",
        cumulative_seqlen_Q=mask.cumulative_q,
        cumulative_seqlen_KV=mask.cumulative_kv,
        max_seqlen_Q=mask.max_q,
        max_seqlen_KV=mask.max_kv,
        is_causal=mask.causal,
        causal_type=CausalType.BottomRight if mask.causal else None,
    )
    return out[0].flatten(1)


def sample_block_mask(query_ids, key_ids, *, causal=False):
    nq, nk = len(query_ids), len(key_ids)
    if not nq or not nk:
        raise ValueError("Block-sparse attention requires nonempty physical streams")
    # The builder evaluates full tiles, including indices beyond physical length.
    qids = F.pad(query_ids, (0, (-nq) % 128), value=-1)
    kids = F.pad(key_ids, (0, (-nk) % 128), value=-1)

    def allowed(b, h, q, k):
        real = (qids[q] >= 0) & (qids[q] == kids[k])
        if causal:
            real = real & (q >= k)
        # Like the reference: padding queries see one finite key; real queries
        # cannot see padding keys. Padding outputs never enter the loss.
        return real | ((qids[q] < 0) & (k == 0))

    return create_block_mask(allowed, None, None, nq, nk, device=str(query_ids.device), _compile=query_ids.is_cuda)


def sparse_attend(q, k, v, mask):
    dim = q.shape[-1]
    # Flex requires a power-of-two head dimension; retain the original scale
    # when padding the width_head_dim_1b preset's Dh=48 to Dh=64.
    padded = 1 << (dim - 1).bit_length()
    q, k, v = [F.pad(x, (0, padded - dim)).transpose(0, 1)[None] for x in (q, k, v)]
    fn = _compiled_attention if q.is_cuda else flex_attention
    out = fn(q, k, v, block_mask=mask, enable_gqa=True, scale=dim**-0.5)
    return out[0, :, :, :dim].transpose(0, 1).flatten(1)
