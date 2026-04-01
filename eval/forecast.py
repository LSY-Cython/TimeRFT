import math
from typing import Any, Generator, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from gluonts.model import Input, InputSpec
from gluonts.torch import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
    CausalMeanValueImputation,
    ExpandDimArray,
    TestSplitSampler,
    Transformation,
)
from gluonts.transform.split import TFTInstanceSplitter

from model.ts_moe import MoiraiMoEModule


class MoiraiMoEForecast(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        target_dim: int,
        feat_dynamic_real_dim: int,
        past_feat_dynamic_real_dim: int,
        context_length: int,
        module: Optional[MoiraiMoEModule] = None,
        patch_size: int = 16,
        num_samples: int = 100,
    ):
        super().__init__()
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.context_length = context_length
        self.module = module
        self.patch_size = patch_size
        self.num_samples = num_samples

    def create_predictor(
        self,
        batch_size: int,
        device: str = "auto",
    ) -> PyTorchPredictor:
        ts_fields = []
        if self.feat_dynamic_real_dim > 0:
            ts_fields.append("feat_dynamic_real")
            ts_fields.append("observed_feat_dynamic_real")
        past_ts_fields = []
        if self.past_feat_dynamic_real_dim > 0:
            past_ts_fields.append("past_feat_dynamic_real")
            past_ts_fields.append("past_observed_feat_dynamic_real")
        instance_splitter = TFTInstanceSplitter(
            instance_sampler=TestSplitSampler(),
            past_length=self.past_length,
            future_length=self.prediction_length,
            observed_value_field="observed_target",
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )
        return PyTorchPredictor(
            input_names=self.prediction_input_names,
            prediction_net=self,  # run self.forward()
            batch_size=batch_size,
            prediction_length=self.prediction_length,
            input_transform=self.get_default_transform() + instance_splitter,
            device=device,
        )

    def describe_inputs(self, batch_size: int = 1) -> InputSpec:
        data = {
            "past_target": Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.target_dim,
                ),
                dtype=torch.float,
            ),
            "past_observed_target": Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.target_dim,
                ),
                dtype=torch.bool,
            ),
            "past_is_pad": Input(
                shape=(batch_size, self.past_length),
                dtype=torch.bool,
            ),
        }
        if self.feat_dynamic_real_dim > 0:
            data["feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length + self.prediction_length,
                    self.feat_dynamic_real_dim,
                ),
                dtype=torch.float,
            )
            data["observed_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length + self.prediction_length,
                    self.feat_dynamic_real_dim,
                ),
                dtype=torch.bool,
            )
        if self.past_feat_dynamic_real_dim > 0:
            data["past_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.past_feat_dynamic_real_dim,
                ),
                dtype=torch.float,
            )
            data["past_observed_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.past_feat_dynamic_real_dim,
                ),
                dtype=torch.bool,
            )
        return InputSpec(data=data, zeros_fn=torch.zeros)

    @property
    def prediction_input_names(self) -> list[str]:
        return list(self.describe_inputs())

    @property
    def training_input_names(self):
        return self.prediction_input_names + ["future_target", "future_observed_values"]

    @property
    def past_length(self) -> int:
        return (
            self.context_length + self.prediction_length
            if self.patch_size == "auto"
            else self.context_length
        )

    def context_token_length(self, patch_size: int) -> int:
        return math.ceil(self.context_length / patch_size)

    def prediction_token_length(self, patch_size) -> int:
        return math.ceil(self.prediction_length / patch_size)

    @property
    def max_patch_size(self) -> int:
        return max(self.module.patch_sizes)

    def forward(
        self,
        past_target: torch.FloatTensor,  # (batch_size, context_length, target_dim)
        past_observed_target: torch.BoolTensor,  # (batch_size, past_length, target_dim)
        past_is_pad: torch.BoolTensor,  # (batch_size, past_length)
        feat_dynamic_real: Optional[torch.FloatTensor] = None,  # (batch_size, past_length+future_length, feat_dynamic_real_dim)
        observed_feat_dynamic_real: Optional[torch.FloatTensor] = None,  # (batch_size, past_length+future_length, feat_dynamic_real_dim)
        past_feat_dynamic_real: Optional[torch.FloatTensor] = None,  # (batch_size, past_length, past_feat_dynamic_real_dim)
        past_observed_feat_dynamic_real: Optional[torch.FloatTensor] = None,  # (batch_size, past_length, past_feat_dynamic_real_dim)
        num_samples: Optional[int] = None,
    ) -> torch.FloatTensor:  # (batch_size, num_samples, prediction_length, target_dim)
        context_step = self.context_token_length(self.patch_size)
        context_token = self.target_dim * context_step
        predict_step = self.prediction_token_length(self.patch_size)
        predict_token = self.target_dim * predict_step

        """
        Note that the order of multivariate tokens in evaluation stage is different from that in finetuning stage.
        Below, the prediction tokens are gathered at the end of the whole sequence.
        """
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            self.patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )
        patch_size = (
            torch.ones_like(time_id, dtype=torch.long) * self.patch_size
        )

        pred_index = torch.arange(
            start=context_step - 1, end=context_token, step=context_step
        )
        assign_index = torch.arange(
            start=context_token,
            end=context_token + predict_token,
            step=predict_step,
        )

        if predict_step == 1:
            distr = self.module(
                target,
                observed_mask,
                sample_id,
                time_id,
                variate_id,
                prediction_mask,
                patch_size,

            )
            preds = distr.sample(torch.Size((num_samples or self.num_samples,)))  # (num_samples, batch_size, seq_len, patch_size)
            preds[..., assign_index, :] = preds[..., pred_index, :]
            return self._format_preds(self.patch_size, preds, self.target_dim)
        else:
            distr = self.module(
                target,
                observed_mask,
                sample_id,
                time_id,
                variate_id,
                prediction_mask,
                patch_size,
            )
            preds = distr.sample(torch.Size((self.num_samples,)))  # (num_samples, batch_size, seq_len, patch_size)
            # preds = distr.mean.unsqueeze(0).repeat(self.num_samples, 1, 1, 1)

            expand_target = target.unsqueeze(0).repeat(
                self.num_samples, 1, 1, 1
            )
            expand_prediction_mask = prediction_mask.unsqueeze(0).repeat(
                self.num_samples, 1, 1
            )
            expand_observed_mask = observed_mask.unsqueeze(0).expand(
                self.num_samples, -1, -1, -1
            )
            expand_sample_id = sample_id.unsqueeze(0).expand(
                self.num_samples, -1, -1
            )
            expand_time_id = time_id.unsqueeze(0).expand(
                self.num_samples, -1, -1
            )
            expand_variate_id = variate_id.unsqueeze(0).expand(
                self.num_samples, -1, -1
            )
            expand_patch_size = patch_size.unsqueeze(0).expand(
                self.num_samples, -1, -1
            )

            expand_target[..., assign_index, :] = preds[..., pred_index, :]
            expand_prediction_mask[..., assign_index] = False

            remain_step = predict_step - 1
            while remain_step > 0:
                distr = self.module(
                    expand_target,
                    expand_observed_mask,
                    expand_sample_id,
                    expand_time_id,
                    expand_variate_id,
                    expand_prediction_mask,
                    expand_patch_size,
                )
                preds = distr.sample(torch.Size((1,)))
                # preds = distr.mean.unsqueeze(0)
                _, _, bs, token, ps = preds.shape
                preds = preds.view(-1, bs, token, ps)

                pred_index = assign_index
                assign_index = assign_index + 1
                expand_target[..., assign_index, :] = preds[..., pred_index, :]
                expand_prediction_mask[..., assign_index] = False

                remain_step -= 1

            return self._format_preds(self.patch_size, expand_target, self.target_dim)

    @staticmethod
    def _patched_seq_pad(
        patch_size: int,
        x: torch.Tensor,
        dim: int,
        left: bool = True,
        value: Optional[float] = None,
    ) -> torch.Tensor:
        if dim >= 0:
            dim = -x.ndim + dim
        pad_length = -x.size(dim) % patch_size
        if left:
            pad = (pad_length, 0)
        else:
            pad = (0, pad_length)
        pad = (0, 0) * (abs(dim) - 1) + pad
        return torch.nn.functional.pad(x, pad, value=value)

    def _generate_time_id(
        self,
        patch_size: int,
        past_observed_target: torch.BoolTensor,  # (batch_size, past_length, target_dim)
    ) -> tuple[torch.IntTensor, torch.IntTensor]:  # (batch_size, past_token_len, future_token_len)
        past_seq_id = reduce(
            self._patched_seq_pad(patch_size, past_observed_target, -2, left=True),
            "... (seq patch) dim -> ... seq",
            "max",
            patch=patch_size,
        )
        past_seq_id = torch.clamp(
            past_seq_id.cummax(dim=-1).values.cumsum(dim=-1) - 1, min=0
        )
        batch_shape = " ".join(map(str, past_observed_target.shape[:-2]))
        future_seq_id = (
            repeat(
                torch.arange(
                    self.prediction_token_length(patch_size),
                    device=past_observed_target.device,
                ),
                f"prediction -> {batch_shape} prediction",
            )
            + past_seq_id.max(dim=-1, keepdim=True).values
            + 1
        )
        return past_seq_id, future_seq_id

    def _convert(
        self,
        patch_size: int,
        past_target: torch.FloatTensor,  # (batch_size, past_length, target_dim)
        past_observed_target: torch.BoolTensor,  # (batch_size, past_length, target_dim)
        past_is_pad: torch.BoolTensor,  # (batch_size, past_length)
        future_target: Optional[torch.FloatTensor] = None,  # (batch_size, future_length, target_dim)
        future_observed_target: Optional[torch.BoolTensor] = None,  # (batch_size, future_length, target_dim)
        future_is_pad: Optional[torch.BoolTensor] = None,  # (batch_size, future_length)
        feat_dynamic_real: Optional[torch.FloatTensor] = None,  # (batch_size, past_length+future_length, feat_dynamic_real_dim)
        observed_feat_dynamic_real: Optional[torch.FloatTensor] = None,  # (batch_size, past_length+future_length, feat_dynamic_real_dim)
        past_feat_dynamic_real: Optional[torch.FloatTensor] = None,  # (batch_size, past_length, past_feat_dynamic_real_dim)
        past_observed_feat_dynamic_real: Optional[torch.FloatTensor] = None,  # (batch_size, past_length, past_feat_dynamic_real_dim)
    ) -> tuple[
        torch.FloatTensor,  # target: (batch_size, combine_seq_len, patch_size)
        torch.BoolTensor,  # observed_mask: (batch_size, combine_seq_len, patch_size)
        torch.IntTensor,  # sample_id: (batch_size, combine_seq_len)
        torch.IntTensor,  # time_id: (batch_size, combine_seq_len)
        torch.IntTensor,  # variate_id: (batch_size, combine_seq_len)
        torch.BoolTensor,  # prediction_mask: (batch_size, combine_seq_len)
    ]:
        batch_shape = past_target.shape[:-2]
        device = past_target.device

        target = []
        observed_mask = []
        sample_id = []
        time_id = []
        variate_id = []
        prediction_mask = []
        dim_count = 0

        past_seq_id, future_seq_id = self._generate_time_id(
            patch_size, past_observed_target
        )

        if future_target is None:
            future_target = torch.zeros(
                batch_shape
                + (
                    self.prediction_length,
                    past_target.shape[-1],
                ),
                dtype=past_target.dtype,
                device=device,
            )
        target.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(patch_size, past_target, -2, left=True),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
            ]
        )
        if future_observed_target is None:
            future_observed_target = torch.ones(
                batch_shape
                + (
                    self.prediction_length,
                    past_observed_target.shape[-1],
                ),
                dtype=torch.bool,
                device=device,
            )
        observed_mask.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_target, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_observed_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
            ]
        )
        if future_is_pad is None:
            future_is_pad = torch.zeros(
                batch_shape + (self.prediction_length,),
                dtype=torch.long,
                device=device,
            )
        sample_id.extend(
            [
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, future_is_pad, -1, left=False, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
            ]
        )
        time_id.extend(
            [past_seq_id] * past_target.shape[-1]
            + [future_seq_id] * past_target.shape[-1]
        )
        variate_id.extend(
            [
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                ),
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                    future=self.prediction_token_length(patch_size),
                ),
            ]
        )
        dim_count += past_target.shape[-1]
        prediction_mask.extend(
            [
                torch.zeros(
                    batch_shape
                    + (self.context_token_length(patch_size) * past_target.shape[-1],),
                    dtype=torch.bool,
                    device=device,
                ),
                torch.ones(
                    batch_shape
                    + (
                        self.prediction_token_length(patch_size)
                        * past_target.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
            ]
        )

        if feat_dynamic_real is not None:
            if observed_feat_dynamic_real is None:
                raise ValueError(
                    "observed_feat_dynamic_real must be provided if feat_dynamic_real is provided"
                )

            target.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., : self.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., self.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                ]
            )
            observed_mask.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., : self.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., self.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                ]
            )
            sample_id.extend(
                [
                    repeat(
                        reduce(
                            (
                                self._patched_seq_pad(
                                    patch_size, past_is_pad, -1, left=True
                                )
                                == 0
                            ).int(),
                            "... (seq patch) -> ... seq",
                            "max",
                            patch=patch_size,
                        ),
                        "... seq -> ... (dim seq)",
                        dim=feat_dynamic_real.shape[-1],
                    ),
                    torch.ones(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                ]
            )
            time_id.extend(
                [past_seq_id] * feat_dynamic_real.shape[-1]
                + [future_seq_id] * feat_dynamic_real.shape[-1]
            )
            variate_id.extend(
                [
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                        past=self.context_token_length(patch_size),
                    ),
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                        future=self.prediction_token_length(patch_size),
                    ),
                ]
            )
            dim_count += feat_dynamic_real.shape[-1]
            prediction_mask.extend(
                [
                    torch.zeros(
                        batch_shape
                        + (
                            self.context_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.zeros(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                ]
            )

        if past_feat_dynamic_real is not None:
            if past_observed_feat_dynamic_real is None:
                raise ValueError(
                    "past_observed_feat_dynamic_real must be provided if past_feat_dynamic_real is provided"
                )
            target.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                )
            )
            observed_mask.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                )
            )
            sample_id.append(
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_feat_dynamic_real.shape[-1],
                )
            )
            time_id.extend([past_seq_id] * past_feat_dynamic_real.shape[-1])

            variate_id.append(
                repeat(
                    torch.arange(past_feat_dynamic_real.shape[-1], device=device)
                    + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                )
            )
            dim_count += past_feat_dynamic_real.shape[-1]
            prediction_mask.append(
                torch.zeros(
                    batch_shape
                    + (
                        self.context_token_length(patch_size)
                        * past_feat_dynamic_real.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                )
            )

        target = torch.cat(target, dim=-2)
        observed_mask = torch.cat(observed_mask, dim=-2)
        sample_id = torch.cat(sample_id, dim=-1)
        time_id = torch.cat(time_id, dim=-1)
        variate_id = torch.cat(variate_id, dim=-1)
        prediction_mask = torch.cat(prediction_mask, dim=-1)
        return (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        )

    def _format_preds(
        self,
        patch_size: int,
        preds: torch.FloatTensor,  # (num_samples, batch_size, combine_seq_len, patch_size)
        target_dim: int,
    ) -> torch.FloatTensor:  # (batch_size, num_samples, future_length, patch_size)
        start = target_dim * self.context_token_length(patch_size)
        end = start + target_dim * self.prediction_token_length(patch_size)
        preds = preds[..., start:end, :patch_size]
        preds = rearrange(
            preds,
            "sample ... (dim seq) patch -> ... sample (seq patch) dim",
            dim=target_dim,
        )[..., : self.prediction_length, :]
        return preds.squeeze(-1)

    def get_default_transform(self) -> Transformation:
        transform = AsNumpyArray(
            field="target",
            expected_ndim=1 if self.target_dim == 1 else 2,
            dtype=np.float32,
        )
        if self.target_dim == 1:
            transform += AddObservedValuesIndicator(
                target_field="target",
                output_field="observed_target",
                imputation_method=CausalMeanValueImputation(),
                dtype=bool,
            )
            transform += ExpandDimArray(field="target", axis=0)
            transform += ExpandDimArray(field="observed_target", axis=0)
        else:
            transform += AddObservedValuesIndicator(
                target_field="target",
                output_field="observed_target",
                dtype=bool,
            )

        if self.feat_dynamic_real_dim > 0:
            transform += AsNumpyArray(
                field="feat_dynamic_real",
                expected_ndim=2,
                dtype=np.float32,
            )
            transform += AddObservedValuesIndicator(
                target_field="feat_dynamic_real",
                output_field="observed_feat_dynamic_real",
                dtype=bool,
            )

        if self.past_feat_dynamic_real_dim > 0:
            transform += AsNumpyArray(
                field="past_feat_dynamic_real",
                expected_ndim=2,
                dtype=np.float32,
            )
            transform += AddObservedValuesIndicator(
                target_field="past_feat_dynamic_real",
                output_field="past_observed_feat_dynamic_real",
                dtype=bool,
            )
        return transform
