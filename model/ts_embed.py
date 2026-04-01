import math
from typing import Optional

import torch
from einops import einsum, rearrange
from torch import nn


def size_to_mask(
    max_size: int,
    sizes: torch.IntTensor,  # (num_feats, )
) -> torch.BoolTensor:  # (num_feats, max_size)
    mask = torch.arange(max_size, device=sizes.device)
    return torch.lt(mask, sizes.unsqueeze(-1))  # all True


class MultiInSizeLinear(nn.Module):
    def __init__(
        self,
        in_features_ls: tuple[int, ...],
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features_ls = in_features_ls
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                (len(in_features_ls), out_features, max(in_features_ls)), dtype=dtype
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty((len(in_features_ls), out_features), dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self.register_buffer(
            "mask",
            rearrange(
                size_to_mask(max(in_features_ls), torch.as_tensor(in_features_ls)),
                "num_feats max_feat -> num_feats 1 max_feat",
            ),
            persistent=False,
        )

    def reset_parameters(self):
        for idx, feat_size in enumerate(self.in_features_ls):
            nn.init.kaiming_uniform_(self.weight[idx, :, :feat_size], a=math.sqrt(5))
            nn.init.zeros_(self.weight[idx, :, feat_size:])
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[idx, :, :feat_size]
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[idx], -bound, bound)

    def forward(
        self,
        x: torch.FloatTensor,  # (batch_size, seq_len, max_feat)
        in_feat_size: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, out_feat)
        out = 0
        for idx, feat_size in enumerate(self.in_features_ls):
            weight = self.weight[idx] * self.mask[idx]
            bias = self.bias[idx] if self.bias is not None else 0
            out = out + (
                torch.eq(in_feat_size, feat_size).unsqueeze(-1)  # all True
                * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)
            )
        return out  # (batch_size, seq_len, out_feat)

    def extra_repr(self) -> str:
        return (
            f"in_features_ls={self.in_features_ls}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"dtype={self.weight.dtype}"
        )


class FeatLinear(nn.Module):
    def __init__(
        self,
        in_features_ls: tuple[int, ...],
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features_ls = in_features_ls
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((len(in_features_ls), out_features, out_features), dtype=dtype)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty((len(in_features_ls), out_features), dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: torch.FloatTensor,  # (batch_size, seq_len, out_feat)
        in_feat_size: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, out_feat)
        out = 0
        for idx, feat_size in enumerate(self.in_features_ls):
            weight = self.weight[idx]
            bias = self.bias[idx] if self.bias is not None else 0
            out = out + (
                torch.eq(in_feat_size, feat_size).unsqueeze(-1)  # all True
                * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)
            )
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features_ls={self.in_features_ls}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"dtype={self.weight.dtype}"
        )


class MultiOutSizeLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features_ls: tuple[int, ...],
        dim: int = 1,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features_ls = out_features_ls
        self.dim = dim

        self.weight = nn.Parameter(
            torch.empty(
                (len(out_features_ls), max(out_features_ls), in_features), dtype=dtype
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty((len(out_features_ls), max(out_features_ls)), dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self.register_buffer(
            "mask",
            rearrange(
                size_to_mask(max(out_features_ls), torch.as_tensor(out_features_ls)),
                "num_feats max_feat -> num_feats max_feat 1",
            ),
            persistent=False,
        )

    def reset_parameters(self):
        for idx, feat_size in enumerate(self.out_features_ls):
            nn.init.kaiming_uniform_(self.weight[idx, :feat_size], a=math.sqrt(5))
            nn.init.zeros_(self.weight[idx, feat_size:])
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[idx, :feat_size]
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[idx, :feat_size], -bound, bound)
                nn.init.zeros_(self.bias[idx, feat_size:])

    def forward(
        self,
        x: torch.FloatTensor,  # (batch_size, seq_len, in_feat)
        out_feat_size: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, out_feat)
        out = 0
        for idx, feat_size in enumerate(self.out_features_ls):
            weight = self.weight[idx] * self.mask[idx]
            bias = self.bias[idx] if self.bias is not None else 0
            out = out + (
                torch.eq(out_feat_size, feat_size // self.dim).unsqueeze(-1)
                * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)
            )
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features_ls={self.out_features_ls}, "
            f"bias={self.bias is not None}, "
            f"dtype={self.weight.dtype}"
        )


# if __name__ == "__main__":
#     patch_sizes = (16, )
#     d_model = 384
#     in_proj = MultiInSizeLinear(
#             in_features_ls=patch_sizes,
#             out_features=d_model,
#         )
#     feat_proj = FeatLinear(
#         in_features_ls=patch_sizes,
#         out_features=d_model,
#     )
#     param_proj = MultiOutSizeLinear(
#         in_features=d_model,
#         out_features_ls=patch_sizes
#     )
#
#     batch_size = 1
#     patch_size = 16
#     max_seq_len = 512
#     num_token = max_seq_len//patch_size
#     scaled_target = torch.randn((batch_size, num_token, patch_size))
#     patch_size_per_token = torch.ones((batch_size, num_token)) * patch_size
#     in_reprs = in_proj(scaled_target, patch_size_per_token)
#     print("MultiInSizeLinear output shape:", in_reprs.shape)
#     in_reprs = feat_proj(in_reprs, patch_size_per_token)
#     print("FeatLinear output shape:", in_reprs.shape)
#     distr_param = param_proj(in_reprs, patch_size_per_token)
#     print("MultiOutSizeLinear output shape:", distr_param.shape)
