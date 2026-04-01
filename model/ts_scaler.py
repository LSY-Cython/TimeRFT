from typing import Optional

import torch
from einops import reduce
from torch import nn

from utils import safe_div


class PackedScaler(nn.Module):
    def forward(
        self,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        observed_mask: torch.BoolTensor = None,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor = None,  # (batch_size, seq_len)
        variate_id: Optional[torch.IntTensor] = None,  # (batch_size, seq_len)
    ):
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )
        if variate_id is None:
            variate_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )

        loc, scale = self._get_loc_scale(
            target.double(), observed_mask, sample_id, variate_id
        )
        return loc.float(), scale.float()

    def _get_loc_scale(
        self,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:  # (batch_size, seq_len, patch_size)
        raise NotImplementedError


class PackedNOPScaler(PackedScaler):
    def _get_loc_scale(
        self,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> tuple[
        torch.FloatTensor, torch.FloatTensor
    ]:  # (batch_size, 1, patch_size)
        loc = torch.zeros_like(target, dtype=target.dtype)
        scale = torch.ones_like(target, dtype=target.dtype)
        return loc, scale


class PackedStdScaler(PackedScaler):
    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale

    def _get_loc_scale(
        self,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> tuple[
        torch.FloatTensor, torch.FloatTensor
    ]:  # (batch_size, seq_len, 1)
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )  # (batch_size, seq_len, seq_len)
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )  # (batch_size, seq_len, 1)
        loc = reduce(
            id_mask * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = safe_div(loc, tobs)
        var = reduce(
            id_mask
            * reduce(
                ((target - loc) ** 2) * observed_mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        var = safe_div(var, (tobs - self.correction))
        scale = torch.sqrt(var + self.minimum_scale)
        loc[sample_id == 0] = 0
        scale[sample_id == 0] = 1
        return loc, scale
