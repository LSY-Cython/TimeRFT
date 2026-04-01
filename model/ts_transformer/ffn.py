from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * in_dim
        out_dim = out_dim or in_dim

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bias = bias
        self.ffn_dropout_p = ffn_dropout_p

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout1 = nn.Dropout(ffn_dropout_p)
        self.dropout2 = nn.Dropout(ffn_dropout_p)
        self.activation = activation

    def forward(
        self,
        x: torch.FloatTensor,  # (*batch_size, seq_len, in_dim)
        centroid: Optional[torch.FloatTensor] = None,  # (num_experts, in_dim)
    ) -> torch.FloatTensor:  # (*batch_size, seq_len, out_dim)
        x = self._in_proj(x)  # (*batch_size, seq_len, hidden_dim)
        return self.dropout2(self.fc2(self.dropout1(x)))

    def _in_proj(
        self, x: torch.FloatTensor  # (*batch_size, seq_len, in_dim)
    ) -> torch.FloatTensor:  # (*batch_size, seq_len, out_dim)
        return self.activation(self.fc1(x))


class GatedLinearUnitFeedForward(FeedForward):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__(
            in_dim,
            hidden_dim=hidden_dim or self.adjust_hidden_dim(4 * in_dim),
            out_dim=out_dim,
            activation=activation,
            bias=bias,
            ffn_dropout_p=ffn_dropout_p,
        )
        self.fc_gate = nn.Linear(self.in_dim, self.hidden_dim, bias=self.bias)

    @staticmethod
    def adjust_hidden_dim(dim):
        return (int(dim * 2 / 3) + 7) // 8 * 8

    def _in_proj(  # SwiGLU non-linearity
        self, x: torch.FloatTensor  # (*batch_size, seq_len, in_dim)
    ) -> torch.FloatTensor:  # (*batch_size, seq_len, out_dim)
        return self.activation(self.fc_gate(x)) * self.fc1(x)  # (*batch_size, seq_len, hidden_dim)


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_token: int,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        self.experts = nn.ModuleList(
            [
                GatedLinearUnitFeedForward(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    activation=activation,
                    bias=bias,
                    ffn_dropout_p=ffn_dropout_p,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        x: torch.FloatTensor,  # (batch_size, seq_len, in_dim)
        centroid: Optional[torch.FloatTensor] = None,  # (num_experts, in_dim)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, in_dim)
        x_squashed = x.view(-1, x.shape[-1])  # (batch_size*seq_len, in_dim)

        centroid = centroid.to(x.device).type_as(x)
        if len(x.shape) > 3:
            x_temp = x.view(-1, x.shape[-2], x.shape[-1])
        else:
            x_temp = x
        centroid = centroid.unsqueeze(0).repeat(x_temp.shape[0], 1, 1)  # (batch_size, num_experts, in_dim)
        cdist = torch.cdist(x_temp, centroid)  # (batch_size, seq_len, num_experts)
        gate_logits = cdist.view(-1, cdist.shape[-1])  # (batch_size*seq_len, num_experts)

        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_token)  # (batch_size*seq_len, num_experts_per_token)
        weights = nn.functional.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(x)  # (batch_size*seq_len, num_experts_per_token)

        results = torch.zeros_like(x_squashed)  # (batch_size*seq_len, in_dim)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                x_squashed[batch_idx]
            )

        results = results.view_as(x)  # (batch_size, seq_len, in_dim)
        return results
