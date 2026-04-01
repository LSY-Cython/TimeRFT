"""
Data curation: applying sliding windows to create offline datasets.
"""

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator, Optional
from itertools import product
import os

import datasets
import pandas as pd
import numpy as np
import pickle as pkl
from datasets import Features, Sequence, Value
from torch.utils.data import Dataset

from data.ts_typing import GenFunc
from data.dataset import EvalDataset, SampleTimeSeriesType, TimeSeriesDataset, FinetuneDataset
from data.indexer import HuggingFaceDatasetIndexer
from data.ts_transform.base import Transformation

CUSTOM_DATA_PATH = None


def _from_wide_dataframe(
    df: pd.DataFrame,
    offset: Optional[int] = None,
    date_offset: Optional[pd.Timestamp] = None,
    freq: str = "H",
) -> tuple[GenFunc, Features]:
    if offset is not None:
        df = df.iloc[:offset]
    elif date_offset is not None:
        df = df[df.index <= date_offset]

    print(df)

    # Infer the freq and generate the prompt
    inferred_freq = pd.infer_freq(df.index)

    if inferred_freq is not None:
        print(
            f"Inferred frequency: {inferred_freq}. Using this value for the 'freq' parameter."
        )
    else:
        print(
            f"Inferred frequency is None. Using predefined {freq} for the 'freq' parameter."
        )

    def example_gen_func() -> Generator[dict[str, Any], None, None]:
        for i in range(len(df.columns)):
            yield {
                "target": df.iloc[:, i].to_numpy(),
                "start": df.index[0],
                "freq": (
                    pd.infer_freq(df.index)
                    if pd.infer_freq(df.index) is not None
                    else freq
                ),
                "item_id": f"item_{i}",
            }

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(Value("float32")),
        )
    )

    return example_gen_func, features


def _from_wide_dataframe_multivariate(
    df: pd.DataFrame,
    offset: Optional[int] = None,
    date_offset: Optional[pd.Timestamp] = None,
    freq: str = "H",
) -> tuple[GenFunc, Features]:
    if offset is not None:
        df = df.iloc[:offset]
    elif date_offset is not None:
        df = df[df.index <= date_offset]

    # Infer the freq and generate the prompt
    inferred_freq = pd.infer_freq(df.index)

    if inferred_freq is not None:
        print(
            f"Inferred frequency: {inferred_freq}. Using this value for the 'freq' parameter."
        )
    else:
        print(
            f"Inferred frequency is None. Using predefined {freq} for the 'freq' parameter."
        )

    def example_gen_func() -> Generator[dict[str, Any], None, None]:
        yield {
            "target": df.to_numpy().T,
            "start": df.index[0],
            "freq": (
                pd.infer_freq(df.index) if pd.infer_freq(df.index) is not None else freq
            ),
            "item_id": "item_0",
        }

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(Sequence(Value("float32")), length=len(df.columns)),
        )
    )

    return example_gen_func, features


class DatasetBuilder(abc.ABC):
    """
    Base class for DatasetBuilders.
    """

    @abc.abstractmethod
    def build_dataset(self, *args, **kwargs):
        """
        Builds the dataset into the required file format.
        """
        ...

    @abc.abstractmethod
    def load_dataset(
        self, transform_map: Callable[..., Transformation]
    ) -> Dataset:
        """
        Load the dataset.

        :param transform_map: a map which returns the required dataset transformations to be applied
        :return: the dataset ready for training
        """
        ...


@dataclass
class SimpleFinetuneDatasetBuilder(DatasetBuilder):
    dataset: str
    windows: Optional[int]
    distance: Optional[int]
    prediction_length: Optional[int]
    context_length: Optional[int]
    patch_size: Optional[int]
    mode: Optional[str] = "univariate"
    storage_path: Path = CUSTOM_DATA_PATH
    mean = None
    std = None

    """
    Databuilder class for TSF fine-tuning, which is modified from SimpleEvalDatasetBuilder. 
    'mean' and 'std' are accepted for data normalization.
    """

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(
        self,
        file: Path,
        dataset_type: str,
        start: int = 0,
        offset: Optional[int] = None,
        date_offset: Optional[pd.Timestamp] = None,
        freq: str = "H",
        normalize: Optional[bool] = False,
    ):

        assert offset is None or date_offset is None, (
            "One or neither offset and date_offset must be specified, but not both. "
            f"Got offset: {offset}, date_offset: {date_offset}"
        )

        df = pd.read_csv(file, index_col=0, parse_dates=True)

        if normalize:  # Normalize data in TSF setup
            end = (
                offset
                if offset is not None
                else (
                    len(df[df.index <= date_offset].index)
                    if date_offset is not None
                    else len(df.index)
                )
            )
            df = self.scale(df, start, end)

        if dataset_type == "univariate":
            _from_dataframe = _from_wide_dataframe
        elif dataset_type in ["multivariate", "covariate"]:
            _from_dataframe = _from_wide_dataframe_multivariate
        else:
            raise ValueError(
                f"Unrecognized dataset_type, {dataset_type}."
                " Valid options are 'univariate', 'multivariate', and 'covariate'."
            )

        example_gen_func, features = _from_dataframe(
            df, freq=freq, offset=offset, date_offset=date_offset
        )
        hf_dataset = datasets.Dataset.from_generator(
            example_gen_func, features=features
        )
        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(
            self.storage_path / f"{dataset_type}" / self.dataset
        )

    def load_dataset(
        self, transform_map: Callable[..., Transformation]
    ) -> Dataset:

        if self.mode == "univariate":
            dataset_type = "univariate"
        elif self.mode == "multivariate":
            dataset_type = "multivariate"
        elif self.mode == "covariate":
            dataset_type = "covariate"
        else:
            raise NotImplementedError

        return FinetuneDataset(
            self.windows,
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / f"{dataset_type}" / self.dataset),
                )
            ),
            transform=transform_map,
        )

    def scale(self, data, start, end):
        train = data[start:end]
        self.mean = train.mean(axis=0)
        self.std = train.std(axis=0)
        return (data - self.mean) / (self.std + 1e-8)


@dataclass
class SimpleEvalDatasetBuilder(DatasetBuilder):
    dataset: str
    offset: Optional[int]
    windows: Optional[int]
    distance: Optional[int]
    prediction_length: Optional[int]
    context_length: Optional[int]
    patch_size: Optional[int]
    mode: Optional[str] = "univariate"
    storage_path: Path = CUSTOM_DATA_PATH

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(
        self,
        file: Path,
        dataset_type: str,
        freq: str = "H",
        mean: pd.Series = None,
        std: pd.Series = None,
    ):
        df = pd.read_csv(file, index_col=0, parse_dates=True)

        if mean is not None and std is not None:  # Normalize data in TSF setup
            df = (df - mean) / (std + 1e-8)

        if dataset_type == "univariate":
            _from_dataframe = _from_wide_dataframe
        elif dataset_type in ["multivariate", "covariate"]:
            _from_dataframe = _from_wide_dataframe_multivariate
        else:
            raise ValueError(
                f"Unrecognized dataset_type, {dataset_type}."
                " Valid options are 'univariate', 'multivariate', and 'covariate'."
            )

        example_gen_func, features = _from_dataframe(df, freq=freq)
        hf_dataset = datasets.Dataset.from_generator(
            example_gen_func, features=features
        )
        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(
            self.storage_path / f"{dataset_type}" / self.dataset
        )

        # Save normalization
        normalize = {"mean": mean.to_numpy(), "std": std.to_numpy()}
        with open(f"{self.storage_path}/{dataset_type}/{self.dataset}_normalize.pkl", "wb") as f:
            pkl.dump(normalize, f)

    def load_dataset(
        self, transform_map: Callable[..., Transformation]
    ) -> Dataset:
        if self.mode == "univariate":
            dataset_type = "univariate"
        elif self.mode == "multivariate":
            dataset_type = "multivariate"
        elif self.mode == "covariate":
            dataset_type = "covariate"
        else:
            raise NotImplementedError

        return EvalDataset(
            self.windows,
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / f"{dataset_type}" / self.dataset),
                )
            ),
            transform=transform_map,
        )


def generate_finetune_builder(
    dataset: str,
    offset: int,
    train_length: int,
    prediction_length: int,
    context_length: int,
    patch_size: int,
    mode: str,
    storage_path: Path = CUSTOM_DATA_PATH,
    distance=1,
) -> SimpleFinetuneDatasetBuilder:
    """
    By default, 'distance' is set to 1 for sliding window. A larger value can be used to reduce computational cost.
    """

    windows = (train_length - offset - context_length - prediction_length) // distance + 1
    return SimpleFinetuneDatasetBuilder(
        dataset=dataset,
        windows=windows,
        distance=distance,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        mode=mode,
        storage_path=storage_path,
    )


def generate_eval_builder(
    dataset: str,
    offset: int,
    eval_length: int,
    prediction_length: int,
    context_length: int,
    patch_size: int,
    mode: str,
    storage_path: Path = CUSTOM_DATA_PATH,
    distance=None,
) -> SimpleEvalDatasetBuilder:
    """
    By default, 'distance' is set to prediction length for rolling evaluation.
    Offer specific 'distance' to decrease the number of validation samples and to reduce computational cost.
    """

    if distance is not None:
        windows = (eval_length - prediction_length) // distance + 1
    else:
        distance = prediction_length
        windows = eval_length // prediction_length

    return SimpleEvalDatasetBuilder(
        dataset=dataset,
        offset=offset,
        windows=windows,
        distance=distance,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        mode=mode,
        storage_path=storage_path,
    )


# Convert GiftEval Huggingface datasets to csv files
def arrow_to_csv(
    storage_path: str,
    save_path: str,
    date_format="%Y/%m/%d %H:%M:%S",
):
    hf_dataset = datasets.load_from_disk(storage_path)
    start = hf_dataset[0]["start"]
    freq = hf_dataset[0]["freq"]
    target = np.array(hf_dataset[0]["target"]).T  # (L, C)
    col_target = [f"Var {i}" for i in range(int(target.shape[1]))]
    df_target = pd.DataFrame(target, columns=col_target)
    date_series = pd.date_range(start=start, periods=target.shape[0], freq=freq).strftime(date_format)
    df_target.insert(0, "date", date_series)
    df_target.to_csv(save_path, index=False)
    print(f"Saved to {save_path}: {df_target}")


# Convert fev-bench Huggingface datasets to csv files
def parquet_to_csv(
    storage_path: str,
    save_path: str,
    mode: str,
    freq: str,
    variates: list[str] = None,
    date_format="%Y/%m/%d %H:%M:%S",
    channel: int = 0,
):
    fev_dataset = datasets.load_dataset(storage_path, data_files="train-00000-of-00001.parquet")["train"]
    timestamp = fev_dataset[channel]["timestamp"]

    if "solar_with_weather" in save_path:
        timestamp = timestamp[125568:]

    if mode == "univariate":
        target = fev_dataset[channel]["target"]
        target = np.array(target, dtype=np.float64)[:, None]  # (total_length, 1)
    elif mode == "multivariate" or mode == "covariate":
        target = []
        for var in variates:
            sequence = fev_dataset[channel][var]
            target.append(sequence)
        target = np.array(target, dtype=np.float64).T  # (total_length, target_dim)

        if "solar_with_weather" in save_path:
            target = target[125568:]

    else:
        raise NotImplementedError

    col_target = [f"Var {i}" for i in range(int(target.shape[1]))]
    df_target = pd.DataFrame(target, columns=col_target)
    date_series = pd.date_range(start=timestamp[0], periods=len(timestamp), freq=freq).strftime(date_format)
    df_target.insert(0, "date", date_series)
    df_target.to_csv(save_path, index=False)
    print(f"Saved to {save_path}: {df_target}")


if __name__ == "__main__":  # must run in AutoDL environment
    import yaml

    # Load data curation configurations
    cfg_path = "configs/entsoe_30T_H96_100%_moirai_moe_1.0_R_small.yaml"
    with open(cfg_path, "r") as f:
        data_cfg = yaml.load(f, Loader=yaml.FullLoader)["train_dataset"]
    CUSTOM_DATA_PATH = data_cfg["storage_path"]
    dataset_name = data_cfg["dataset"]
    fevbench_path = data_cfg["fevbench_path"]
    fevbench_file = data_cfg["fevbench_file"]
    csv_path = data_cfg["csv_path"]
    mode = data_cfg["mode"]
    train_length = data_cfg["train_length"]
    train_offset = data_cfg["offset"]
    freq = data_cfg["freq"]
    date_offset = None  # default: None, type=str
    normalize = True  # TSF setup requires normalizing the data using training statistics

    # Transform fev-bench datasets to csv files
    if not os.path.exists(csv_path):
        # # ERCOT/Loop Seattle
        # parquet_to_csv(storage_path=fevbench_path, save_path=csv_path, mode=mode, freq=freq, channel=0)
        # parquet_to_csv(storage_path=fevbench_path, save_path=csv_path, mode=mode, freq=freq, channel=1)

        # # ETT
        # parquet_to_csv(storage_path=fevbench_path, save_path=csv_path, mode=mode, freq=freq,
        #                variates=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'], channel=0)
        # parquet_to_csv(storage_path=fevbench_path, save_path=csv_path, mode=mode, freq=freq,
        #                variates=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'], channel=1)

        # # Jena Weather
        # parquet_to_csv(storage_path=fevbench_path, save_path=csv_path, mode=mode, freq=freq,
        #                variates=[f"target_{i}" for i in range(21)])

        # # BOOMLET 963
        # parquet_to_csv(storage_path=fevbench_path, save_path=csv_path, mode=mode, freq=freq,
        #                variates=[f"target_{i}" for i in range(28)])

        # # UCI Air Quality
        # parquet_to_csv(storage_path=fevbench_path, save_path=csv_path, mode=mode, freq=freq,
        #                variates=['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)',
        #                          'T', 'RH', 'AH'])

        # # Solar with Weather
        # parquet_to_csv(storage_path=fevbench_path, save_path=csv_path, mode=mode, freq=freq,
        #                variates=['target', 'global_horizontal_irradiance', 'temp', 'pressure', 'humidity',
        #                          'wind_speed', 'rain_1h', 'snow_1h', 'clouds_all', 'day_length'])

        # ENTSO-e Load
        parquet_to_csv(storage_path=fevbench_path, save_path=csv_path, mode=mode, freq=freq,
                       variates=['target', 'solar_generation_actual', 'wind_onshore_generation_actual', 'temperature'], channel=0)
        # parquet_to_csv(storage_path=fevbench_path, save_path=csv_path, mode=mode, freq=freq,
        #                variates=['target', 'solar_generation_actual', 'wind_onshore_generation_actual', 'temperature'], channel=1)

    """
    Create training dataset.
    If offset/date_offset is not provided, the whole data will be used for training.
    Otherwise, only the part before offset is used for training.
    """
    train_dataset_builder = SimpleFinetuneDatasetBuilder(
        dataset=dataset_name,
        windows=None,
        distance=None,
        prediction_length=None,
        context_length=None,
        patch_size=None,
        storage_path=CUSTOM_DATA_PATH
    )
    train_dataset_builder.build_dataset(
        file=Path(csv_path),
        dataset_type=mode,
        start=train_offset,
        offset=train_length,
        date_offset=pd.Timestamp(date_offset) if date_offset else None,
        freq=freq,
        normalize=normalize,
    )

    """
    Create a validation dataset if offset/date_offset is provided.
    Eval dataset encompasses the whole train/val/test dataset.
    """
    if train_length is not None or date_offset is not None:
        SimpleEvalDatasetBuilder(
            f"{dataset_name}_eval",
            offset=None,
            windows=None,
            distance=None,
            prediction_length=None,
            context_length=None,
            patch_size=None,
            storage_path=CUSTOM_DATA_PATH
        ).build_dataset(
            file=Path(csv_path),
            dataset_type=mode,
            freq=freq,
            mean=train_dataset_builder.mean,
            std=train_dataset_builder.std,
        )
