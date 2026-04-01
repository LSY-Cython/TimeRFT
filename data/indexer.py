import abc
from collections.abc import Iterable, Sequence
from typing import Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset
from datasets.features import Sequence
from datasets.formatting import query_table

from data.ts_typing import BatchedData, Data, MultivarTimeSeries, UnivarTimeSeries


class Indexer(abc.ABC, Sequence):
    """
    Base class for all Indexers.

    An Indexer is responsible for extracting data from an underlying file format.
    """

    def __init__(self, uniform: bool = False):
        """
        :param uniform: whether the underlying data has uniform length
        """
        self.uniform = uniform

    def check_index(self, idx: Union[int, slice, Iterable[int]]):
        """
        Check the validity of a given index.

        :param idx: index to check
        :return: None
        :raises IndexError: if idx is out of bounds
        :raises NotImplementedError: if idx is not a valid type
        """
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of bounds for length {len(self)}")
        elif isinstance(idx, slice):
            if idx.start is not None and idx.start < 0:
                raise IndexError(
                    f"Index {idx.start} out of bounds for length {len(self)}"
                )
            if idx.stop is not None and idx.stop >= len(self):
                raise IndexError(
                    f"Index {idx.stop} out of bounds for length {len(self)}"
                )
        elif isinstance(idx, Iterable):
            idx = np.fromiter(idx, np.int64)
            if np.logical_or(idx < 0, idx >= len(self)).any():
                raise IndexError(f"Index out of bounds for length {len(self)}")
        else:
            raise NotImplementedError(f"Unable to index on type: {type(idx)}")

    def __getitem__(
        self, idx: Union[int, slice, Iterable[int]]
    ) -> dict[str, Union[Data, BatchedData]]:
        """
        Retrive the data from the underlying storage in dictionary format.

        :param idx: index to retrieve
        :return: underlying data with given index
        """
        self.check_index(idx)

        if isinstance(idx, int):
            item = self._getitem_int(idx)
        elif isinstance(idx, slice):
            item = self._getitem_slice(idx)
        elif isinstance(idx, Iterable):
            item = self._getitem_iterable(idx)
        else:
            raise NotImplementedError(f"Unable to index on type: {type(idx)}")

        return {k: v for k, v in item.items()}

    def _getitem_slice(self, idx: slice) -> dict[str, BatchedData]:
        indices = list(range(len(self))[idx])
        return self._getitem_iterable(indices)

    @abc.abstractmethod
    def _getitem_int(self, idx: int) -> dict[str, Data]: ...

    @abc.abstractmethod
    def _getitem_iterable(self, idx: Iterable[int]) -> dict[str, BatchedData]: ...

    def get_uniform_probabilities(self) -> np.ndarray:
        """
        Obtains uniform probability distribution over all time series.

        :return: uniform probability distribution
        """
        return np.ones(len(self)) / len(self)

    def get_proportional_probabilities(self, field: str = "target") -> np.ndarray:
        """
        Obtain proportion of each time series based on number of time steps.

        :param field: field name to measure time series length
        :return: proportional probabilities
        """
        if self.uniform:
            return self.get_uniform_probabilities()

        lengths = np.asarray([sample[field].shape[-1] for sample in self])
        probs = lengths / lengths.sum()
        return probs


class HuggingFaceDatasetIndexer(Indexer):
    """
    Indexer for Hugging Face Datasets
    """

    def __init__(self, dataset: Dataset, uniform: bool = False):
        """
        :param dataset: underlying Hugging Face Dataset
        :param uniform: whether the underlying data has uniform length
        """
        super().__init__(uniform=uniform)
        self.dataset = dataset
        self.features = dict(self.dataset.features)
        self.non_seq_cols = [
            name
            for name, feat in self.features.items()
            if not isinstance(feat, Sequence)
        ]  # ['item_id', 'start', 'freq']
        self.seq_cols = [
            name for name, feat in self.features.items() if isinstance(feat, Sequence)
        ]  # ['target', 'past_feat_dynamic_real (i.e. past covariates)']
        self.dataset.set_format("numpy", columns=self.non_seq_cols)

    def __len__(self) -> int:
        return len(self.dataset)  # num_rows in Huggingface dataset

    def _getitem_int(self, idx: int) -> dict[str, Data]:
        non_seqs = self.dataset[idx]
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
        seqs = {
            col: self._pa_column_to_numpy(pa_subtable, col)[0] for col in self.seq_cols
        }
        return non_seqs | seqs

    def _getitem_iterable(self, idx: Iterable[int]) -> dict[str, BatchedData]:
        non_seqs = self.dataset[idx]
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
        seqs = {
            col: self._pa_column_to_numpy(pa_subtable, col) for col in self.seq_cols
        }
        return non_seqs | seqs

    def _getitem_slice(self, idx: slice) -> dict[str, BatchedData]:
        non_seqs = self.dataset[idx]
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
        seqs = {
            col: self._pa_column_to_numpy(pa_subtable, col) for col in self.seq_cols
        }
        return non_seqs | seqs

    def _pa_column_to_numpy(
        self, pa_table: pa.Table, column_name: str
    ) -> Union[list[UnivarTimeSeries], list[MultivarTimeSeries]]:
        pa_array: pa.Array = pa_table.column(column_name)
        feature = self.features[column_name]

        if isinstance(pa_array, pa.ChunkedArray):
            if isinstance(feature.feature, Sequence):
                array = [
                    flat_slice.flatten().to_numpy(False).reshape(feat_length, -1)
                    for chunk in pa_array.chunks
                    for i in range(len(chunk))
                    if (flat_slice := chunk.slice(i, 1).flatten())
                    and (
                        feat_length := (
                            feature.length if feature.length != -1 else len(flat_slice)
                        )
                    )
                ]
            else:
                array = [
                    chunk.slice(i, 1).flatten().to_numpy(False)
                    for chunk in pa_array.chunks
                    for i in range(len(chunk))
                ]
        elif isinstance(pa_array, pa.ListArray):
            if isinstance(feature.feature, Sequence):
                flat_slice = pa_array.flatten()
                feat_length = (
                    feature.length if feature.length != -1 else len(flat_slice)
                )
                array = [flat_slice.flatten().to_numpy(False).reshape(feat_length, -1)]
            else:
                array = [pa_array.flatten().to_numpy(False)]
        else:
            raise NotImplementedError

        return array

    def get_proportional_probabilities(self, field: str = "target") -> np.ndarray:
        """
        Obtain proportion of each time series based on number of time steps.
        Leverages pyarrow.compute for fast implementation.

        :param field: field name to measure time series length
        :return: proportional probabilities
        """

        if self.uniform:
            return self.get_uniform_probabilities()

        if self[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(pc.list_slice(self.dataset.data.column(field), 0, 1))
            )
        else:
            lengths = pc.list_value_length(self.dataset.data.column(field))
        lengths = lengths.to_numpy()
        probs = lengths / lengths.sum()
        return probs


# if __name__ == "__main__":
#     import os
#     os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#
#     from datasets import load_from_disk
#     import matplotlib.pyplot as plt
#
#     hf_dataset_indexer = HuggingFaceDatasetIndexer(load_from_disk("LOTSA_V1/hog"), uniform=False)
#     past_cov = hf_dataset_indexer.dataset["past_feat_dynamic_real"]  # [(cov_dim, ts_len), ...] * ts_dim
#     idx = 1
#     ts_idx = hf_dataset_indexer[idx]["target"]  # (ts_len, )
#     past_cov_idx = hf_dataset_indexer[idx]["past_feat_dynamic_real"]  # (cov_dim, ts_len)
#     eq_cov = (past_cov_idx == past_cov[idx])  # True
#     assert isinstance(ts_idx, UnivarTimeSeries)  # True
#     assert isinstance(past_cov_idx, MultivarTimeSeries)  # True
#
#     pass
