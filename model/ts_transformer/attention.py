import math
from collections.abc import Callable
from functools import partial
from typing import Optional, Union, Type

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from model.ts_transformer.position import AttentionBias, QueryKeyProjection


def native_scaled_dot_product_attention(
    query: torch.FloatTensor,  # (batch_size, num_heads, 1, q_len, head_dim)
    key: torch.FloatTensor,  # (batch_size, num_heads, 1, kv_len, head_dim)
    value: torch.FloatTensor,  # (batch_size, num_heads, 1, kv_len, head_dim)
    attn_mask: Optional[Union[torch.BoolTensor, torch.FloatTensor]] = None,  # (batch_size, num_heads, 1, q_len, kv_len)
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
):
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor  # (batch_size, num_heads, 1, q_len, kv_len)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros_like(attn_weight)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask
        attn_weight = attn_weight + attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def packed_attention_mask(
    sample_id: torch.IntTensor  # (batch_size, seq_len)
) -> torch.BoolTensor:  # (batch_size, seq_len, seq_len)
    sample_id = sample_id.unsqueeze(-1)
    attention_mask = sample_id.eq(sample_id.mT)
    return attention_mask


def packed_causal_attention_mask(
    sample_id: torch.IntTensor,  # (batch_size, seq_len)
    time_id: torch.IntTensor,  # (batch_size, seq_len)
) -> torch.BoolTensor:  # (batch_size, seq_len, seq_len)
    attention_mask = packed_attention_mask(sample_id)  # (batch_size, seq_len, seq_len)
    expanded_id1 = time_id.unsqueeze(-2)
    expanded_id2 = time_id.unsqueeze(-1)
    compare_res = expanded_id1 <= expanded_id2
    attention_mask = attention_mask * compare_res  # (batch_size, seq_len, seq_len)
    return attention_mask


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        bias: bool = True,
        norm_layer: Optional[Union[Type[nn.Module], partial[nn.Module]]] = nn.LayerNorm,
        softmax_scale: Optional[float] = None,
        attn_dropout_p: float = 0.0,
        var_attn_bias: Optional[Callable[[], AttentionBias]] = None,
        time_attn_bias: Optional[Callable[[], AttentionBias]] = None,
        var_qk_proj: Optional[Callable[[], QueryKeyProjection]] = None,
        time_qk_proj: Optional[Callable[[], QueryKeyProjection]] = None,
    ):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = dim // num_heads
        self.heads_per_group = num_heads // num_groups
        self.var_attn_bias = var_attn_bias() if var_attn_bias is not None else None
        self.time_attn_bias = time_attn_bias() if time_attn_bias is not None else None
        self.var_qk_proj = var_qk_proj() if var_qk_proj is not None else None
        self.time_qk_proj = time_qk_proj() if time_qk_proj is not None else None

        self.softmax_scale = softmax_scale or 1 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.v_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.q_norm = (
            norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        )
        self.attn_dropout_p = attn_dropout_p
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _get_var_id(
        self,
        query: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        key: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        query_var_id: Optional[torch.IntTensor],  # (batch_size, seq_len)
        kv_var_id: Optional[torch.IntTensor],  # (batch_size, seq_len)
    ) -> tuple[
        Optional[torch.IntTensor],
        Optional[torch.IntTensor],
    ]:  # (batch_size, 1, 1, seq_len)
        if self.var_attn_bias is not None or self.var_qk_proj is not None:
            if query_var_id is None:
                query_var_id = repeat(
                    torch.zeros((), device=query.device, dtype=torch.long),
                    f" -> {' '.join(map(str, query.shape[:-4]))} 1 1 {query.shape[-2]}",
                )
            else:
                query_var_id = rearrange(query_var_id, "... q_len -> ... 1 1 q_len")

            if kv_var_id is None:
                kv_var_id = repeat(
                    torch.zeros((), device=key.device, dtype=torch.long),
                    f" -> {' '.join(map(str, key.shape[:-4]))} 1 1 {key.shape[-2]}",
                )
            else:
                kv_var_id = rearrange(kv_var_id, "... kv_len -> ... 1 1 kv_len")

        return query_var_id, kv_var_id

    def _get_time_id(
        self,
        query: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        key: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        query_time_id: Optional[torch.IntTensor],  # (batch_size, seq_len)
        kv_time_id: Optional[torch.IntTensor],  # (batch_size, seq_len)
    ) -> tuple[
        Optional[torch.IntTensor],
        Optional[torch.IntTensor],
    ]:  # (batch_size, 1, 1, seq_len)
        if self.time_attn_bias is not None or self.time_qk_proj is not None:
            if query_time_id is None:
                query_time_id = repeat(
                    torch.arange(
                        query.shape[-2], device=query.device, dtype=torch.long
                    ),
                    f"q_len -> {' '.join(map(str, query.shape[:-4]))} 1 1 q_len",
                )
            else:
                query_time_id = rearrange(query_time_id, "... q_len -> ... 1 1 q_len")

            if kv_time_id is None:
                kv_time_id = repeat(
                    torch.arange(key.shape[-2], device=key.device, dtype=torch.long),
                    f"kv_len -> {' '.join(map(str, key.shape[:-4]))} 1 1 kv_len",
                )
            else:
                kv_time_id = rearrange(kv_time_id, "... kv_len-> ... 1 1 kv_len")

        return query_time_id, kv_time_id

    def _update_attn_mask(
        self,
        attn_mask: Optional[torch.BoolTensor],  # (batch_size, seq_len, seq_len)
        query: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        key: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        query_var_id: Optional[torch.IntTensor] = None,  # (batch_size, 1, 1, seq_len)
        kv_var_id: Optional[torch.IntTensor] = None,  # (batch_size, 1, 1, seq_len)
        query_time_id: Optional[torch.IntTensor] = None,  # (batch_size, 1, 1, seq_len)
        kv_time_id: Optional[torch.IntTensor] = None,  # (batch_size, 1, 1, seq_len)
    ) -> Optional[Union[torch.BoolTensor, torch.FloatTensor]]:  # (batch_size, num_heads, 1, seq_len, seq_len)
        if attn_mask is not None:
            attn_mask = rearrange(
                attn_mask,
                "... q_len kv_len -> ... 1 1 q_len kv_len",
            )  # (batch_size, 1, 1, seq_len, seq_len)

        attn_bias = 0
        if self.var_attn_bias is not None:
            attn_bias = attn_bias + self.var_attn_bias(
                query,
                key,
                query_id=query_var_id,
                kv_id=kv_var_id,
            )  # (batch_size, num_heads, 1, seq_len, seq_len)

        if self.time_attn_bias is not None:  # skip this branch
            attn_bias = attn_bias + self.time_attn_bias(
                query,
                key,
                query_id=query_time_id,
                kv_id=kv_time_id,
            )

        attn_mask = (
            attn_mask
            if isinstance(attn_bias, int)
            else (
                attn_bias
                if attn_mask is None
                else attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
            )
        )
        return attn_mask

    def _qk_proj(
        self,
        query: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        key: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        query_var_id: Optional[torch.IntTensor],  # (batch_size, 1, 1, seq_len)
        kv_var_id: Optional[torch.IntTensor],  # (batch_size, 1, 1, seq_len)
        query_time_id: Optional[torch.IntTensor],  # (batch_size, 1, 1, seq_len)
        kv_time_id: Optional[torch.IntTensor],  # (batch_size, 1, 1, seq_len)
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:  # (batch_size, num_heads, 1, seq_len, head_dim)
        if self.var_qk_proj is not None:  # skip
            query, key = self.var_qk_proj(
                query, key, query_id=query_var_id, kv_id=kv_var_id
            )

        if self.time_qk_proj is not None:
            query, key = self.time_qk_proj(
                query, key, query_id=query_time_id, kv_id=kv_time_id
            )

        return query, key

    def forward(
        self,
        query: torch.FloatTensor,  # (batch_size, seq_len, dim)
        key: torch.FloatTensor,  # (batch_size, seq_len, dim)
        value: torch.FloatTensor,  # (batch_size, seq_len, dim)
        attn_mask: Optional[torch.BoolTensor] = None,  # (batch_size, seq_len, seq_len)
        query_var_id: Optional[torch.IntTensor] = None,  # (batch_size, seq_len)
        kv_var_id: Optional[torch.IntTensor] = None,  # (batch_size, seq_len)
        query_time_id: Optional[torch.IntTensor] = None,  # (batch_size, seq_len)
        kv_time_id: Optional[torch.IntTensor] = None,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, dim)
        query = self.q_proj(query)  # (batch_size, seq_len, dim)
        key = self.k_proj(key)  # (batch_size, seq_len, dim)
        value = self.v_proj(value)  # (batch_size, seq_len, dim)

        query = self.q_norm(
            rearrange(
                query,
                "... q_len (group hpg dim) -> ... group hpg q_len dim",
                group=self.num_groups,
                hpg=self.heads_per_group,
            )
        )  # (batch_size, num_heads, 1, seq_len, head_dim)
        key = self.k_norm(
            repeat(
                key,
                "... kv_len (group dim) -> ... group hpg kv_len dim",
                group=self.num_groups,
                hpg=self.heads_per_group,
            )
        )  # (batch_size, num_heads, 1, seq_len, head_dim)
        value = repeat(
            value,
            "... kv_len (group dim) -> ... group hpg kv_len dim",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )  # (batch_size, num_heads, 1, seq_len, head_dim)

        query_var_id, kv_var_id = self._get_var_id(
            query,
            key,
            query_var_id,
            kv_var_id
        )  # (batch_size, 1, 1, seq_len)
        query_time_id, kv_time_id = self._get_time_id(
            query,
            key,
            query_time_id,
            kv_time_id,
        )  # (batch_size, 1, 1, seq_len)

        attn_mask = self._update_attn_mask(
            attn_mask,
            query,
            key,
            query_var_id=query_var_id,
            kv_var_id=kv_var_id,
            query_time_id=query_time_id,
            kv_time_id=kv_time_id,
        )  # (batch_size, num_heads, 1, seq_len, seq_len)

        query, key = self._qk_proj(  # Rotatory Position Embedding
            query,
            key,
            query_var_id=query_var_id,
            kv_var_id=kv_var_id,
            query_time_id=query_time_id,
            kv_time_id=kv_time_id,
        )  # (batch_size, num_heads, 1, seq_len, head_dim)

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p,
            scale=self.softmax_scale,
        )  # (batch_size, num_heads, 1, seq_len, head_dim)
        out = rearrange(out, "... group hpg q_len dim -> ... q_len (group hpg dim)")  # (batch_size, seq_len, dim)
        return self.out_proj(out)  # (batch_size, seq_len, dim)



