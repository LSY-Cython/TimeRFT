from collections.abc import Callable, Iterable
from typing import Any, Union

import numpy as np
import torch
from jaxtyping import AbstractDtype, Num


class DateTime64(AbstractDtype):
    dtypes = ["datetime64"]


class Character(AbstractDtype):
    dtypes = ["str_"]


# Data preparation
GenFunc = Callable[[], Iterable[dict[str, Any]]]
SliceableGenFunc = Callable[..., Iterable[dict[str, Any]]]


# Indexer
DateTime = DateTime64[np.ndarray, ""]
BatchedDateTime = DateTime64[np.ndarray, "batch"]
String = np.character
BatchedString = Character[np.ndarray, "batch"]
UnivarTimeSeries = Num[np.ndarray, "time"]
MultivarTimeSeries = Num[np.ndarray, "var time"]
Data = Union[DateTime, String, UnivarTimeSeries, MultivarTimeSeries]
BatchedData = Union[BatchedDateTime, BatchedString, list[UnivarTimeSeries], list[MultivarTimeSeries]]
FlattenedData = Union[DateTime, String, list[UnivarTimeSeries]]


# Loader
Sample = dict[str, Num[torch.Tensor, "*sample"]]
BatchedSample = dict[str, Num[torch.Tensor, "batch *sample"]]
