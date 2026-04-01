import abc
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Union
from jaxtyping import Num

import numpy as np
from numpy.typing import NDArray
from einops import pack, rearrange, repeat
# import pandas as pd
# from gluonts.time_feature import norm_freq_str

from data.sampler import Sampler, get_sampler
from data.ts_typing import UnivarTimeSeries

from data.ts_transform.base import Transformation, CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin, ApplyFuncMixin


@dataclass
class SampleDimension(  # randomly select a subset of channels in single/multi-channel time series and covariates
    CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin, Transformation
):
    max_dim: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    sampler: Sampler = get_sampler("uniform")

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # len(data_entry["target"]) + len(data_entry["past_feat_dynamic_real"])
        total_field_dim = sum(
            self.collect_func_list(
                self._get_dim,
                data_entry,
                self.fields,
                optional_fields=self.optional_fields,
            )
        )
        self.map_func(
            partial(self._process, total_field_dim=total_field_dim),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _get_dim(self, data_entry: dict[str, Any], field: str) -> int:
        self.check_ndim(field, data_entry[field], 2)
        return len(data_entry[field])

    def _process(
        self, data_entry: dict[str, Any], field: str, total_field_dim: int
    ) -> list[UnivarTimeSeries]:
        arr: list[UnivarTimeSeries] = data_entry[field]
        rand_idx = np.random.permutation(len(arr))
        field_max_dim = (self.max_dim * len(arr)) // total_field_dim  # avoid too many variates
        n = self.sampler(min(len(arr), field_max_dim))
        return [arr[idx] for idx in rand_idx[:n]]


# class PatchSizeConstraints(abc.ABC):
#     @abc.abstractmethod
#     def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]: ...
#
#     def __call__(self, freq: str) -> range:
#         offset = pd.tseries.frequencies.to_offset(freq)
#         start, stop = self._get_boundaries(offset.n, norm_freq_str(offset.name))
#         return range(start, stop + 1)
#

# @dataclass
# class FixedPatchSizeConstraints(PatchSizeConstraints):
#     start: int
#     stop: Optional[int] = None
#
#     def __post_init__(self):
#         if self.stop is None:
#             self.stop = self.start
#         assert self.start <= self.stop
#
#     def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]:
#         return self.start, self.stop


# class DefaultPatchSizeConstraints(PatchSizeConstraints):
#     # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
#     DEFAULT_RANGES = {
#         "S": (64, 128),  # 512s = 8.53min, 4096s = 68.26min
#         "T": (32, 128),  # 64min = 1.07h, 512min = 8.53h
#         "H": (32, 64),  # 128h = 5.33days
#         "D": (16, 32),
#         "B": (16, 32),
#         "W": (16, 32),
#         "M": (8, 32),
#         "Q": (1, 8),
#         "Y": (1, 8),
#         "A": (1, 8),
#     }
#
#     def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]:
#         start, stop = self.DEFAULT_RANGES[offset_name]
#         return start, stop


@dataclass
class GetPatchSize(Transformation):
    min_time_patches: int
    target_field: str = "target"
    patch_sizes: Union[tuple[int, ...], list[int], range] = (8, 16, 32, 64, 128)
    # patch_size_constraints: PatchSizeConstraints = DefaultPatchSizeConstraints()
    # offset: bool = True

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        freq = data_entry["freq"]
        # constraints = self.patch_size_constraints(freq)
        # largest patch size based on min_time_patches
        target: list[UnivarTimeSeries] = data_entry[self.target_field]
        length = target[0].shape[0]
        patch_size_ceil = length // self.min_time_patches

        if isinstance(self.patch_sizes, (tuple, list)):
            patch_size_candidates = [
                patch_size
                for patch_size in self.patch_sizes
                if patch_size <= patch_size_ceil
                # if (patch_size in constraints) and (patch_size <= patch_size_ceil)
            ]
        # elif isinstance(self.patch_sizes, range):
        #     patch_size_candidates = range(
        #         max(self.patch_sizes.start, constraints.start),
        #         min(self.patch_sizes.stop, constraints.stop, patch_size_ceil),
        #     )
        else:
            raise NotImplementedError

        if len(patch_size_candidates) <= 0:
            ts_shape = (len(target),) + target[0].shape
            raise AssertionError(
                "no valid patch size candidates for "
                f"time series shape: {ts_shape}, "
                f"freq: {freq}, "
                f"patch_sizes: {self.patch_sizes}, "
                # f"constraints: {constraints}, "
                f"min_time_patches: {self.min_time_patches}, "
                f"patch_size_ceil: {patch_size_ceil}"
            )

        data_entry["patch_size"] = np.random.choice(patch_size_candidates)
        return data_entry


@dataclass
class PatchCrop(MapFuncMixin, Transformation):  # randomly crop a time range, constrained by max_seq_len and patch_size
    """
    Crop fields in a data_entry in the temporal dimension based on a patch_size.
    :param rng: numpy random number generator
    :param min_time_patches: minimum number of patches for time dimension
    :param max_patches: maximum number of patches for time * dim dimension (if flatten)
    :param will_flatten: whether time series fields will be flattened subsequently
    :param offset: whether to offset the start of the crop
    :param fields: fields to crop
    """

    min_time_patches: int
    max_patches: int
    will_flatten: bool = False
    offset: bool = True
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = ("past_feat_dynamic_real",)

    def __post_init__(self):
        assert (
            self.min_time_patches <= self.max_patches
        ), "min_patches must be <= max_patches"
        assert len(self.fields) > 0, "fields must be non-empty"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        a, b = self._get_boundaries(data_entry)
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a:b] for ts in data_entry[field]]

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        patch_size = data_entry["patch_size"]
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]  # length of timestamps
        nvar = (
            sum(len(data_entry[f]) for f in self.fields)
            + sum(len(data_entry[f]) for f in self.optional_fields if f in data_entry)
            if self.will_flatten
            else 1
        )  # number of time series plus number of past covariates

        offset = (
            np.random.randint(
                time % patch_size + 1
            )  # offset by [0, patch_size) so that the start is not always a multiple of patch_size
            if self.offset
            else 0
        )
        total_patches = (
            time - offset
        ) // patch_size  # total number of patches in time series

        # 1. max_patches should be divided by nvar if the time series is subsequently flattened
        # 2. cannot have more patches than total available patches
        max_patches = min(self.max_patches // nvar, total_patches)
        if max_patches < self.min_time_patches:
            raise ValueError(
                f"max_patches={max_patches} < min_time_patches={self.min_time_patches}"
            )

        num_patches = np.random.randint(
            self.min_time_patches, max_patches + 1
        )  # number of patches to consider
        first = np.random.randint(
            total_patches - num_patches + 1
        )  # first patch to consider

        start = offset + first * patch_size
        stop = start + num_patches * patch_size
        return start, stop


@dataclass
class FinetunePatchCrop(MapFuncMixin, Transformation):
    """
    Similar to EvalCrop, crop training samples based on specific context_length and prediction_length
    """
    offset: int
    distance: int
    prediction_length: int
    context_length: int
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = ("past_feat_dynamic_real",)

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        a, b = self._get_boundaries(data_entry)
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )

        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a:b] for ts in data_entry[field]]

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]  # num of time steps of one series
        window = data_entry["window"]
        fcst_start = self.offset + self.context_length + window * self.distance
        a = fcst_start - self.context_length
        b = fcst_start + self.prediction_length

        assert time >= b > a >= 0

        return a, b


@dataclass
class EvalCrop(MapFuncMixin, Transformation):
    offset: int
    distance: int
    prediction_length: int
    context_length: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        a, b = self._get_boundaries(data_entry)
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a : b or None] for ts in data_entry[field]]

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]
        window = data_entry["window"]
        fcst_start = self.offset + window * self.distance
        a = fcst_start - self.context_length
        b = fcst_start + self.prediction_length

        if self.offset >= 0:
            assert time >= b > a >= 0
        else:
            assert 0 >= b > a >= -time

        return a, b


@dataclass
class PackFields(CollectFuncMixin, Transformation):
    output_field: str
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    feat: bool = False

    def __post_init__(self):
        self.pack_str: str = "* time feat" if self.feat else "* time"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        fields = self.collect_func_list(
            self.pop_field,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        if len(fields) > 0:
            output_field = pack(fields, self.pack_str)[0]  # (var_dim_rand, ts_len_rand)
            data_entry |= {self.output_field: output_field}
        return data_entry

    @staticmethod
    def pop_field(data_entry: dict[str, Any], field: str) -> Any:
        return np.asarray(data_entry.pop(field))


@dataclass
class AddObservedMask(CollectFuncMixin, Transformation):  # detect NaN values
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    observed_mask_field: str = "observed_mask"
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        observed_mask = self.collect_func(
            self._generate_observed_mask,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )  # {field: torch.BoolTensor}, (var_dim_rand, ts_len_rand)
        data_entry[self.observed_mask_field] = observed_mask
        return data_entry

    @staticmethod
    def _generate_observed_mask(data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr = data_entry[field]
        return ~np.isnan(arr)


class ImputationMethod:
    def __call__(
        self, x: Num[np.ndarray, "*dim length"]
    ) -> Num[np.ndarray, "*dim length"]: ...


@dataclass(frozen=True)
class DummyValueImputation(ImputationMethod):
    value: Union[int, float, complex] = 0.0

    def __call__(
        self, x: Num[np.ndarray, "*dim length"]
    ) -> Num[np.ndarray, "*dim length"]:
        x[np.isnan(x)] = self.value
        return x


@dataclass
class ImputeTimeSeries(ApplyFuncMixin, Transformation):  # impute NaN via zero
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    imputation_method: ImputationMethod = DummyValueImputation(value=0.0)

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.apply_func(
            self._impute,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _impute(self, data_entry: dict[str, Any], field: str):
        value = data_entry[field]
        nan_entries = np.isnan(value)
        if nan_entries.any():
            data_entry[field] = self.imputation_method(value)


@dataclass
class Patchify(MapFuncMixin, Transformation):
    max_patch_size: int
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = ("past_feat_dynamic_real",)
    pad_value: Union[int, float] = 0

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        patch_size = data_entry["patch_size"]
        self.map_func(
            partial(self._patchify, patch_size=patch_size),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _patchify(self, data_entry: dict[str, Any], field: str, patch_size: int):
        arr = data_entry[field]
        if isinstance(arr, list):
            return [self._patchify_arr(a, patch_size) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.fields or k in self.optional_fields:
                    arr[k] = self._patchify_arr(v, patch_size)
            return arr
        return self._patchify_arr(arr, patch_size)

    def _patchify_arr(
        self, arr: Num[np.ndarray, "var time*patch"], patch_size: int
    ) -> Num[np.ndarray, "var time max_patch"]:
        assert arr.shape[-1] % patch_size == 0
        arr = rearrange(arr, "... (time patch) -> ... time patch", patch=patch_size)
        pad_width = [(0, 0) for _ in range(arr.ndim)]
        pad_width[-1] = (0, self.max_patch_size - patch_size)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=self.pad_value)
        return arr  # (var_dim_rand, ts_len_rand//patch_size, patch_size)


@dataclass
class AddVariateIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add variate_id to data_entry
    """

    fields: tuple[str, ...]
    max_dim: int
    optional_fields: tuple[str, ...] = tuple()
    variate_id_field: str = "variate_id"
    expected_ndim: int = 2
    randomize: bool = False
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.counter = 0
        self.dimensions = (
            np.random.choice(self.max_dim, size=self.max_dim, replace=False)
            if self.randomize
            else list(range(self.max_dim))
        )
        data_entry[self.variate_id_field] = self.collect_func(
            self._generate_variate_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_variate_id(
        self, data_entry: dict[str, Any], field: str
    ) -> np.ndarray:
        arr = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        dim, time = arr.shape[:2]
        if self.counter + dim > self.max_dim:
            raise ValueError(
                f"Variate ({self.counter + dim}) exceeds maximum variate {self.max_dim}. "
            )
        field_dim_id = repeat(
            np.asarray(self.dimensions[self.counter:self.counter + dim], dtype=int),
            "var -> var time",
            time=time,
        )
        self.counter += dim
        return field_dim_id


@dataclass
class AddTimeIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add time_id to data_entry
    """

    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    time_id_field: str = "time_id"
    expected_ndim: int = 2
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """
        add sequence_id
        """
        data_entry[self.time_id_field] = self.collect_func(
            self._generate_time_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_time_id(self, data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        var, time = arr.shape[:2]
        field_seq_id = np.arange(time)
        field_seq_id = repeat(field_seq_id, "time -> var time", var=var)
        return field_seq_id


@dataclass
class AddSampleIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add sample_id when sequence packing is not used. Follow the implementation in MoiraiForecast.
    """

    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    sample_id_field: str = "sample_id"
    expected_ndim: int = 2
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:

        data_entry[self.sample_id_field] = self.collect_func(
            self._generate_sample_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_sample_id(self, data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        var, time = arr.shape[:2]
        # If not using sequence packing, then all patches in an entry are from the same sample.
        field_seq_id = np.ones(time, dtype=int)
        field_seq_id = repeat(field_seq_id, "time -> var time", var=var)
        return field_seq_id


@dataclass
class MaskedPrediction(MapFuncMixin, CheckArrNDimMixin, Transformation):  # fill True over prediction horizon
    min_mask_ratio: float
    max_mask_ratio: float
    target_field: str = "target"
    truncate_fields: tuple[str, ...] = tuple()
    optional_truncate_fields: tuple[str, ...] = tuple()
    prediction_mask_field: str = "prediction_mask"
    expected_ndim: int = 2

    def __post_init__(self):
        assert (
            self.min_mask_ratio <= self.max_mask_ratio
        ), "min_mask_ratio must be <= max_mask_ratio"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target = data_entry[self.target_field]
        prediction_mask = self._generate_prediction_mask(target)
        self.map_func(
            partial(self._truncate, mask=prediction_mask),  # noqa
            data_entry,
            self.truncate_fields,
            optional_fields=self.optional_truncate_fields,
        )
        data_entry[self.prediction_mask_field] = prediction_mask
        return data_entry

    def _generate_prediction_mask(
        self, target: NDArray[np.float64]  # (var_dim_rand, seq_len_rand, patch_size)
    ) -> NDArray[np.bool_]:  # (var_dim_rand, seq_len_rand)
        self.check_ndim("target", target, self.expected_ndim)
        var, time = target.shape[:2]
        prediction_mask = np.zeros((var, time), dtype=bool)
        mask_ratio = np.random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        mask_length = max(1, round(time * mask_ratio))
        prediction_mask[:, -mask_length:] = True
        return prediction_mask

    def _truncate(
        self,
        data_entry: dict[str, Any],
        field: str,
        mask: np.ndarray,
    ) -> Union[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
        arr: Union[np.ndarray, list[np.ndarray], dict[str, np.ndarray]] = data_entry[field]
        if isinstance(arr, list):
            return [self._truncate_arr(a, mask) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.truncate_fields or k in self.optional_truncate_fields:  # exclude "target" item in truncate_fields
                    arr[k] = self._truncate_arr(v, mask)
            return arr
        return self._truncate_arr(arr, mask)

    @staticmethod
    def _truncate_arr(
        arr: NDArray[np.float64], mask: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        return arr[:, ~mask[0]]


@dataclass
class EvalMaskedPrediction(MapFuncMixin, CheckArrNDimMixin, Transformation):
    mask_length: int
    target_field: str = "target"
    truncate_fields: tuple[str, ...] = tuple()
    optional_truncate_fields: tuple[str, ...] = tuple()
    prediction_mask_field: str = "prediction_mask"
    expected_ndim: int = 2

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target = data_entry[self.target_field]
        prediction_mask = self._generate_prediction_mask(target)
        self.map_func(
            partial(self._truncate, mask=prediction_mask),  # noqa
            data_entry,
            self.truncate_fields,
            optional_fields=self.optional_truncate_fields,
        )
        data_entry[self.prediction_mask_field] = prediction_mask
        return data_entry

    def _generate_prediction_mask(
        self, target: NDArray[np.float64]  # (var_dim_rand, seq_len_rand, patch_size)
    ) -> NDArray[np.bool_]:  # (var_dim_rand, seq_len_rand)
        self.check_ndim("target", target, self.expected_ndim)
        var, time = target.shape[:2]
        prediction_mask = np.zeros((var, time), dtype=bool)
        prediction_mask[:, -self.mask_length :] = True
        return prediction_mask

    def _truncate(
        self,
        data_entry: dict[str, Any],
        field: str,
        mask: np.ndarray,
    ) -> Union[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
        arr: Union[np.ndarray, list[np.ndarray], dict[str, np.ndarray]] = data_entry[field]
        if isinstance(arr, list):
            return [self._truncate_arr(a, mask) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.truncate_fields or k in self.optional_truncate_fields:
                    arr[k] = self._truncate_arr(v, mask)
            return arr
        return self._truncate_arr(arr, mask)

    @staticmethod
    def _truncate_arr(
        arr: NDArray[np.float64], mask: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        return arr[:, ~mask[0]]


@dataclass
class ExtendMask(CheckArrNDimMixin, CollectFuncMixin, Transformation):
    fields: tuple[str, ...]
    mask_field: str
    optional_fields: tuple[str, ...] = tuple()
    expected_ndim: int = 2

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target_mask: np.ndarray = data_entry[self.mask_field]
        aux_target_mask: list[np.ndarray] = self.collect_func_list(
            self._generate_target_mask,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )  # (var_dim_rand, seq_len_rand_truncated)
        data_entry[self.mask_field] = [target_mask] + aux_target_mask
        return data_entry

    def _generate_target_mask(
        self, data_entry: dict[str, Any], field: str
    ) -> np.ndarray:
        arr: np.ndarray = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        var, time = arr.shape[:2]
        field_target_mask = np.zeros((var, time), dtype=bool)
        return field_target_mask


@dataclass
class FlatPackCollection(Transformation):
    field: str
    feat: bool = False

    def __post_init__(self):
        self.pack_str: str = "* feat" if self.feat else "*"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        collection = data_entry[self.field]
        if isinstance(collection, dict):
            collection = list(collection.values())
        data_entry[self.field] = pack(collection, self.pack_str)[0]  # (seq_len_final, )
        return data_entry


@dataclass
class FlatPackFields(CollectFuncMixin, Transformation):  # merge target and covariate into a flattened sequence
    output_field: str
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    feat: bool = False

    def __post_init__(self):
        self.pack_str: str = "* feat" if self.feat else "*"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        fields = self.collect_func_list(
            self.pop_field,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        if len(fields) > 0:
            output_field = pack(fields, self.pack_str)[0]  # (seq_len_final, patch_size)
            data_entry |= {self.output_field: output_field}  # delete "past_feat_dynamic_real" field
        return data_entry

    @staticmethod
    def pop_field(data_entry: dict[str, Any], field: str) -> Any:
        return np.asarray(data_entry.pop(field))


@dataclass
class SequencifyField(Transformation):
    field: str
    axis: int = 0
    target_field: str = "target"
    target_axis: int = 0

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        data_entry[self.field] = data_entry[self.field].repeat(
            data_entry[self.target_field].shape[self.target_axis], axis=self.axis
        )  # (seq_len_final, )
        return data_entry


@dataclass
class SelectFields(Transformation):
    fields: list[str]
    allow_missing: bool = False

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        if self.allow_missing:
            return {f: data_entry[f] for f in self.fields if f in data_entry}
        return {f: data_entry[f] for f in self.fields}


@dataclass
class EvalPad(MapFuncMixin, Transformation):
    prediction_pad: int
    context_pad: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.map_func(
            self.map,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def map(self, data_entry: dict[str, Any], field: str) -> Any:
        arr = data_entry[field]
        pad_width = [(0, 0) for _ in range(arr.ndim)]
        # pad_width[-1] = (self.context_pad, self.prediction_pad)  # pad nan at both the head and end of sequence
        pad_width[-1] = (self.context_pad + self.prediction_pad, 0)  # pad nan at the head of sequence
        arr = np.pad(arr, pad_width, mode="constant", constant_values=np.nan)
        return arr
