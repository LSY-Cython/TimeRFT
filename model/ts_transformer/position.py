import abc
import math
from functools import cached_property
from typing import Any, Optional, Type

import torch
from einops import einsum, rearrange, repeat
from torch import nn


class AttentionBias(nn.Module, abc.ABC):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
    ):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.head_dim = dim // num_heads

    @abc.abstractmethod
    def forward(
        self,
        query: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        key: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        query_id: torch.IntTensor,  # (batch_size, 1, 1, seq_len)
        kv_id: torch.IntTensor,  # (batch_size, 1, 1, seq_len)
    ) -> torch.FloatTensor: ...  # (batch_size, num_heads, 1, seq_len, seq_len)


class BinaryAttentionBias(AttentionBias):
    def __init__(self, dim: int, num_heads: int, num_groups: int):
        super().__init__(dim, num_heads, num_groups)
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)

    def forward(
        self,
        query: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        key: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        query_id: torch.IntTensor,  # (batch_size, 1, 1, seq_len)
        kv_id: torch.IntTensor,  # (batch_size, 1, 1, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, num_heads, 1, seq_len, seq_len)
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))  # (batch_size, 1, 1, seq_len, seq_len), only diagonal True
        weight = rearrange(self.emb.weight, "two num_heads -> two num_heads 1 1")
        bias = rearrange(  # try to avoid advanced indexing
            ~ind * weight[:1] + ind * weight[1:],  # (batch_size, 1, num_heads, seq_len, seq_len)
            "... 1 (group hpg) q_len kv_len -> ... group hpg q_len kv_len",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )
        return bias


class Projection(nn.Module, abc.ABC):
    def __init__(self, proj_width: int, num_heads: int, num_groups: int, **kwargs: Any):
        super().__init__()
        self.proj_width = proj_width
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups

    @abc.abstractmethod
    def forward(
        self,
        x: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, dim)
        seq_id: Optional[torch.IntTensor],  # (batch_size, 1, 1, seq_len)
    ) -> torch.FloatTensor: ...  # (batch_size, num_heads, 1, seq_len, dim)


class RotaryProjection(Projection):
    def __init__(
        self,
        *,
        proj_width: int,
        num_heads: int,
        num_groups: int,
        max_len: int = 512,
        base: int = 10000,
    ):
        super().__init__(proj_width, num_heads, num_groups)
        assert (
            self.proj_width % 2 == 0
        ), f"proj_width must be even, got {self.proj_width}"
        self.register_buffer(
            "theta",
            1.0
            / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float)
                / self.proj_width,
            ),
            persistent=False,
        )  # (proj_width, )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len: int):
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(
                max_len, device=self.theta.device, dtype=self.theta.dtype
            )
            m_theta = einsum(position, self.theta, "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x: torch.FloatTensor) -> torch.FloatTensor:  # (batch_size, num_heads, 1, seq_len, dim)
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)  # noqa

    def forward(
        self,
        x: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, dim)
        seq_id: Optional[torch.IntTensor],  # (batch_size, 1, 1, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, num_heads, 1, seq_len, dim)
        self._init_freq(max_len=seq_id.max() + 1)
        rot_cos = self.cos[seq_id]
        rot_sin = self.sin[seq_id]
        return rot_cos * x + rot_sin * self._rotate(x)


class QueryKeyProjection(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        proj_layer: Type[Projection],
        kwargs: Optional[dict[str, Any]] = None,
        key_proj_layer: Optional[Type[Projection]] = None,
        key_kwargs: Optional[dict[str, Any]] = None,
        partial_factor: Optional[tuple[float, float]] = None,
    ):
        super().__init__()
        if partial_factor is not None:
            assert (
                0.0 <= partial_factor[0] < partial_factor[1] <= 1.0
            ), f"got {partial_factor[0]}, {partial_factor[1]}"
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.head_dim = dim // num_heads
        self.partial_factor = partial_factor
        self.query_proj = proj_layer(
            proj_width=self.proj_width,
            num_heads=num_heads,
            num_groups=num_groups,
            **(kwargs or {}),
        )
        if key_proj_layer is None:
            self.key_proj = self.query_proj
        else:
            self.key_proj = key_proj_layer(
                proj_width=self.proj_width,
                num_heads=num_heads,
                num_groups=num_groups,
                **(key_kwargs or {}),
            )

    @cached_property
    def proj_width(self) -> int:
        if self.partial_factor is None:
            return self.head_dim
        return int(self.head_dim * (self.partial_factor[1] - self.partial_factor[0]))

    @cached_property
    def split_sizes(self) -> tuple[int, int, int]:
        if self.partial_factor is None:
            return 0, self.head_dim, 0
        return (
            int(self.partial_factor[0] * self.head_dim),
            self.proj_width,
            int((1.0 - self.partial_factor[1]) * self.head_dim),
        )

    def forward(
        self,
        query: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        key: torch.FloatTensor,  # (batch_size, num_heads, 1, seq_len, head_dim)
        query_id: Optional[torch.IntTensor],  # (batch_size, 1, 1, seq_len)
        kv_id: Optional[torch.IntTensor],  # (batch_size, 1, 1, seq_len)
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:  # (batch_size, num_heads, 1, seq_len, head_dim)
        if self.partial_factor is not None:
            queries = list(query.split(self.split_sizes, dim=-1))  # in half
            keys = list(key.split(self.split_sizes, dim=-1))  # in half
            queries[1] = self.query_proj(queries[1], seq_id=query_id)
            keys[1] = self.key_proj(keys[1], seq_id=kv_id)
            query = torch.cat(queries, dim=-1)
            key = torch.cat(keys, dim=-1)
        else:
            query = self.query_proj(query, seq_id=query_id)
            key = self.key_proj(key, seq_id=kv_id)
        return query, key
