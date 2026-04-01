from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution

from model.ts_transformer.attention import packed_causal_attention_mask
from model.ts_distribution.base import DistributionOutput
from model.ts_transformer.norm import RMSNorm
from model.ts_scaler import PackedNOPScaler, PackedStdScaler
from model.ts_transformer.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
)
from model.ts_transformer.transformer import TransformerEncoder
from model.ts_embed import FeatLinear, MultiInSizeLinear


class MoiraiMoEModule(nn.Module):
    """
    Contains components of Moirai, to ensure implementation is identical across models.
    """

    def __init__(
        self,
        distr_output: DistributionOutput,
        d_model: int,
        d_ff: int,
        num_layers: int,
        patch_sizes: tuple[int, ...],  # tuple[int, ...] | list[int]
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
    ):
        """
        :param distr_output: ts_distribution output object
        :param d_model: model hidden dimensions
        :param num_layers: number of transformer layers
        :param patch_sizes: sequence of patch sizes
        :param max_seq_len: maximum sequence length for inputs
        :param attn_dropout_p: dropout probability for attention layers
        :param dropout_p: dropout probability for all other layers
        :param scaling: whether to apply scaling (standardization)
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_sizes = patch_sizes
        self.max_seq_len = max_seq_len
        self.scaling = scaling

        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
        self.in_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        self.res_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        self.feat_proj = FeatLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=None,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_moe=True,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=d_ff,
        )
        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_param_proj(d_model, patch_sizes)

    def forward(
        self,
        target: torch.FloatTensor,  # (batch_size, seq_len, max_patch)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, max_patch)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        time_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
        prediction_mask: torch.BoolTensor,  # (batch_size, seq_len)
        patch_size: torch.IntTensor,  # (batch_size, seq_len)
    ) -> Distribution:
        """
        Defines the forward pass of MoiraiMoEModule.
        This method expects processed inputs.

        1. Apply scaling to observations
        2. Project from observations to representations
        3. Replace prediction window with learnable mask
        4. Apply transformer layers
        5. Project from representations to ts_distribution parameters
        6. Return ts_distribution object

        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :param prediction_mask: binary mask for prediction horizon, 1 if part of the horizon, 0 otherwise
        :param patch_size: patch size for each token
        :return: predictive ts_distribution
        """
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )  # (batch_size, seq_len, 1)
        scaled_target = (target - loc) / scale  # (batch_size, seq_len, patch_size)

        in_reprs = self.in_proj(scaled_target, patch_size)
        in_reprs = F.silu(in_reprs)
        in_reprs = self.feat_proj(in_reprs, patch_size)
        res_reprs = self.res_proj(scaled_target, patch_size)
        reprs = in_reprs + res_reprs  # (batch_size, seq_len, d_model)

        reprs = self.encoder(
            reprs,
            packed_causal_attention_mask(sample_id, time_id),
            time_id=time_id,
            var_id=variate_id,
        )  # (batch_size, seq_len, d_model)
        distr_param = self.param_proj(reprs, patch_size)
        distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)
        return distr
