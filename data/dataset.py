from enum import Enum
from typing import Any, Union

import numpy as np
from torch.utils.data import Dataset

from data.sampler import Sampler, get_sampler
from data.ts_typing import (
    BatchedData,
    BatchedDateTime,
    BatchedString,
    Data,
    FlattenedData,
    MultivarTimeSeries,
    UnivarTimeSeries,
)
from data.indexer import Indexer
from data.ts_transform.base import Transformation


class SampleTimeSeriesType(Enum):
    """
    How to sample from the dataset.
    - none: do not sample, return the current index.
    - uniform: each time series sampled with equal probability
    - proportional: each time series sampled with probability proportional to it's length
    """

    NONE = "none"
    UNIFORM = "uniform"
    PROPORTIONAL = "proportional"


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        indexer: Indexer,
        transform: Transformation,
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
    ):
        """
        :param indexer: Underlying Indexer object
        :param transform: Transformation to apply to time series
        :param sample_time_series: defines how a time series is obtained from the dataset
        :param dataset_weight: multiplicative factor to apply to dataset size
        """
        self.indexer = indexer
        self.transform = transform
        self.sample_time_series = sample_time_series
        self.dataset_weight = dataset_weight

        if sample_time_series == SampleTimeSeriesType.NONE:
            self.probabilities = None
        elif sample_time_series == SampleTimeSeriesType.UNIFORM:
            self.probabilities = indexer.get_uniform_probabilities()
        elif sample_time_series == SampleTimeSeriesType.PROPORTIONAL:
            self.probabilities = indexer.get_proportional_probabilities()
        else:
            raise ValueError(f"Unknown sample type {sample_time_series}")

    def __getitem__(self, idx: int) -> dict[str, FlattenedData]:
        """
        Obtain a time series from the dataset, flatten
        :param idx: index of time series to retrieve. if sample_time_series is specified, this will be ignored.
        :return: transformed time series data
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        if self.sample_time_series != SampleTimeSeriesType.NONE:
            idx = np.random.choice(len(self.probabilities), p=self.probabilities)

        return self.transform(self._flatten_data(self._get_data(idx)))

    @property
    def num_ts(self) -> int:
        """
        Get the number of time series in the dataset
        """
        return len(self.indexer)

    def __len__(self) -> int:
        """
        Length is the number of time series multiplied by dataset_weight
        """
        return int(np.ceil(self.num_ts * self.dataset_weight))  # self.dataset_weight = windows

    def _get_data(self, idx: int) -> dict[str, Union[Data, BatchedData]]:
        """
        Obtains time series from Indexer object
        """
        return self.indexer[idx % self.num_ts]

    @staticmethod
    def _flatten_data(data: dict[str, Data]) -> dict[str, FlattenedData]:
        """
        Convert time series type data into a list of univariate time series
        """
        return {
            k: (
                [v]
                if isinstance(v, UnivarTimeSeries)
                else list(v) if isinstance(v, MultivarTimeSeries) else v
            )
            for k, v in data.items()
        }


class MultiSampleTimeSeriesDataset(TimeSeriesDataset):
    """
    Samples multiple time series and stacks them into a single time series.
    Underlying dataset should have aligned time series, meaning same start and end dates.
    """

    def __init__(
        self,
        indexer: Indexer,
        transform: Transformation,
        max_ts: int,
        combine_fields: tuple[str, ...],
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
        sampler: Sampler = get_sampler("beta_binomial", a=2, b=5),
    ):
        """
        :param indexer: Underlying Indexer object
        :param transform: Transformation to apply to time series
        :param max_ts: maximum number of time series that can be stacked together
        :param combine_fields: fields which should be stacked
        :param sample_time_series: defines how a time series is obtained from the dataset
        :param dataset_weight: multiplicative factor to apply to dataset size
        :param sampler: how to sample the other time series
        """
        super().__init__(indexer, transform, sample_time_series, dataset_weight)
        self.max_ts = max_ts
        self.combine_fields = combine_fields
        self.sampler = sampler

    def _get_data(self, idx: int) -> dict[str, BatchedData]:
        n_series = self.sampler(min(self.num_ts, self.max_ts))
        choices = np.concatenate([np.arange(idx), np.arange(idx + 1, self.num_ts)])
        others = np.random.choice(choices, n_series - 1, replace=False)
        samples = self.indexer[np.concatenate([[idx], others])]
        return samples

    def _flatten_data(
        self, samples: dict[str, BatchedData]
    ) -> dict[str, FlattenedData]:
        for field in samples.keys():
            if field in self.combine_fields:
                item = samples[field]
                if isinstance(item, list) and isinstance(item[0], MultivarTimeSeries):
                    samples[field] = [
                        univar for sample in samples[field] for univar in sample
                    ]
            elif isinstance(samples[field], BatchedDateTime):
                samples[field] = np.asarray(samples[field][0])
            elif isinstance(samples[field], BatchedString):
                samples[field] = samples[field][0]
            else:
                raise AssertionError(
                    f"Field {field} not accounted for in {self.indexer} MultiSampleTimeSeriesDataset"
                )
        return samples


class EvalDataset(TimeSeriesDataset):
    """
    Dataset class for validation.
    Should be used in conjunction with Eval transformations.
    """

    def __init__(
        self,
        windows: int,
        indexer: Indexer,
        transform: Transformation,
    ):
        """
        :param windows: number of windows to perform evaluation on
        """
        super().__init__(
            indexer,
            transform,
            SampleTimeSeriesType.NONE,
            dataset_weight=windows,
        )

    def _get_data(self, idx: int) -> dict[str, Data]:
        window, idx = divmod(idx, self.num_ts)
        item = self.indexer[idx]
        item["window"] = window
        return item


class FinetuneDataset(TimeSeriesDataset):
    """
    This class is identical to EvalDataset. It is created solely to avoid confusion due to naming.
    """

    def __init__(
        self,
        windows: int,
        indexer: Indexer,
        transform: Transformation,
    ):
        """
        :param windows: number of windows to perform evaluation on
        """
        super().__init__(
            indexer,
            transform,
            SampleTimeSeriesType.NONE,
            dataset_weight=windows,
        )

    def _get_data(self, idx: int) -> dict[str, Data]:
        """
        Window serve as an index in FinetunePatchCrop or EvalCrop to segment an instance of (context_length+prediction_length),
        its maximum value is the number of training windows.
        """
        window, idx = divmod(idx, self.num_ts)
        item = self.indexer[idx]
        item["window"] = window
        return item  # item["target"].shape: (num_var, train_length)


# if __name__ == "__main__":
#     from datasets import load_from_disk
#     from indexer import HuggingFaceDatasetIndexer
#     from ts_transform.transforms import SampleDimension, GetPatchSize, PatchCrop, PackFields, AddObservedMask, \
#         ImputeTimeSeries, DummyValueImputation, Patchify, AddVariateIndex, AddTimeIndex, MaskedPrediction, ExtendMask, \
#         FlatPackCollection, FlatPackFields, SequencifyField, SelectFields
#
#     hf_dataset_indexer = HuggingFaceDatasetIndexer(load_from_disk("LOTSA_V1/covid19_energy"), uniform=False)
#     hf_dataset_indexer_multi = HuggingFaceDatasetIndexer(load_from_disk("LOTSA_V1/hog"), uniform=False)
#     patch_sizes = (16, )
#     ts_transforms = SampleDimension(max_dim=128, fields=("target",), optional_fields=("past_feat_dynamic_real",)) + \
#                     GetPatchSize(min_time_patches=2, target_field="target", patch_sizes=patch_sizes,) + \
#                     PatchCrop(min_time_patches=2, max_patches=128, will_flatten=True, offset=True,
#                               fields=("target",), optional_fields=("past_feat_dynamic_real",),) + \
#                     PackFields(output_field="target", fields=("target",), feat=False,) + \
#                     PackFields(output_field="past_feat_dynamic_real", fields=tuple(),
#                                optional_fields=("past_feat_dynamic_real",), feat=False,) + \
#                     AddObservedMask(fields=("target",), optional_fields=("past_feat_dynamic_real",),
#                                     observed_mask_field="observed_mask", collection_type=dict,) + \
#                     ImputeTimeSeries(fields=("target",), optional_fields=("past_feat_dynamic_real",),
#                                      imputation_method=DummyValueImputation(value=0.0),) + \
#                     Patchify(max_patch_size=max(patch_sizes), fields=("target", "observed_mask"),
#                              optional_fields=("past_feat_dynamic_real",),) + \
#                     AddVariateIndex(fields=("target",), optional_fields=("past_feat_dynamic_real",),
#                                     variate_id_field="variate_id", expected_ndim=3, max_dim=128,
#                                     randomize=True, collection_type=dict,) + \
#                     AddTimeIndex(fields=("target",), optional_fields=("past_feat_dynamic_real",),
#                                  time_id_field="time_id", expected_ndim=3, collection_type=dict,) + \
#                     MaskedPrediction(min_mask_ratio=0.15, max_mask_ratio=0.5, target_field="target",
#                                      truncate_fields=("variate_id", "time_id", "observed_mask"),
#                                      optional_truncate_fields=("past_feat_dynamic_real",),
#                                      prediction_mask_field="prediction_mask", expected_ndim=3,) + \
#                     ExtendMask(fields=tuple(), optional_fields=("past_feat_dynamic_real",),
#                                mask_field="prediction_mask", expected_ndim=3,) + \
#                     FlatPackCollection(field="variate_id", feat=False,) + \
#                     FlatPackCollection(field="time_id", feat=False, ) + \
#                     FlatPackCollection(field="prediction_mask", feat=False, ) + \
#                     FlatPackCollection(field="observed_mask", feat=True, ) + \
#                     FlatPackFields(output_field="target", fields=("target",),
#                                    optional_fields=("past_feat_dynamic_real",), feat=True,) + \
#                     SequencifyField(field="patch_size", target_field="target")
#
#     ts_dataset = TimeSeriesDataset(indexer=hf_dataset_indexer, transform=ts_transforms)
#     # ts_dataset = MultiSampleTimeSeriesDataset(indexer=hf_dataset_indexer_multi, transform=ts_transforms,
#     #                                           max_ts=128, combine_fields=("target", "past_feat_dynamic_real"))
#     # idx = 0
#     # ts_sampled = ts_dataset[idx]
#     # ts_target = ts_sampled["target"]
#     # try:
#     #     ts_covariate = ts_sampled["past_feat_dynamic_real"]
#     # except:
#     #     print("past_feat_dynamic_real has been merged into target")
#     # ts_observed_mask = ts_sampled["observed_mask"]
#     # ts_variate_id = ts_sampled["variate_id"]
#     # ts_time_id = ts_sampled["time_id"]
#     # ts_pred_mask = ts_sampled["prediction_mask"]
#     # ts_patch_size = ts_sampled["patch_size"]
#
#
#     from loader import PackCollate, pad_func_map
#     from ts_transform.mapping import seq_fields
#     from torch.utils.data import DataLoader
#
#     pack_collate_fn = PackCollate(max_length=512, seq_fields=seq_fields, pad_func_map=pad_func_map)
#
#     ts_dataloader = DataLoader(
#         dataset=ts_dataset,
#         batch_size=2,
#         shuffle=True,
#         collate_fn=pack_collate_fn,
#         pin_memory=True,
#         drop_last=False
#     )
#
#     for i, ts_batch in enumerate(ts_dataloader):
#         batch_target = ts_batch["target"]
#         batch_observed_mask = ts_batch["observed_mask"]
#         batch_variate_id = ts_batch["variate_id"]
#         batch_time_id = ts_batch["time_id"]
#         batch_pred_mask = ts_batch["prediction_mask"]
#         batch_patch_size = ts_batch["patch_size"]
#         batch_sample_id = ts_batch["sample_id"]
#
#     pass
