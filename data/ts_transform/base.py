import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union

import numpy as np


class Transformation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]: ...

    def chain(self, other: "Transformation") -> "Chain":
        return Chain([self, other])

    def __add__(self, other: "Transformation") -> "Chain":
        return self.chain(other)

    def __radd__(self, other):
        if other == 0:
            return self
        return other + self


@dataclass
class Chain(Transformation):
    """
    Chain multiple transformations together.
    """

    transformations: list[Transformation]

    def __post_init__(self) -> None:
        transformations = []

        for transformation in self.transformations:
            if isinstance(transformation, Identity):
                continue
            elif isinstance(transformation, Chain):
                transformations.extend(transformation.transformations)
            else:
                assert isinstance(transformation, Transformation)
                transformations.append(transformation)

        self.transformations = transformations
        self.__init_passed_kwargs__ = {"transformations": transformations}

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        for t in self.transformations:
            data_entry = t(data_entry)
        return data_entry


class Identity(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        return data_entry


class MapFuncMixin:
    @staticmethod
    def map_func(
        func: Callable[[dict[str, Any], str], Any],
        data_entry: dict[str, Any],
        fields: tuple[str, ...],
        optional_fields: tuple[str, ...] = (),
    ):
        for field in fields:
            data_entry[field] = func(data_entry, field)
        for field in optional_fields:
            if field in data_entry:
                data_entry[field] = func(data_entry, field)


class ApplyFuncMixin:
    @staticmethod
    def apply_func(
        func: Callable[[dict[str, Any], str], None],
        data_entry: dict[str, Any],
        fields: tuple[str, ...],
        optional_fields: tuple[str, ...] = (),
    ):
        for field in fields:
            func(data_entry, field)
        for field in optional_fields:
            if field in data_entry:
                func(data_entry, field)


class CollectFuncMixin:
    @staticmethod
    def collect_func_list(
        func: Callable[[dict[str, Any], str], Any],
        data_entry: dict[str, Any],
        fields: tuple[str, ...],
        optional_fields: tuple[str, ...] = (),
    ) -> list[Any]:
        collect = []
        for field in fields:
            collect.append(func(data_entry, field))
        for field in optional_fields:
            if field in data_entry:
                collect.append(func(data_entry, field))
        return collect

    @staticmethod
    def collect_func_dict(
        func: Callable[[dict[str, Any], str], Any],
        data_entry: dict[str, Any],
        fields: tuple[str, ...],
        optional_fields: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        collect = {}
        for field in fields:
            collect[field] = func(data_entry, field)
        for field in optional_fields:
            if field in data_entry:
                collect[field] = func(data_entry, field)
        return collect

    def collect_func(
        self,
        func: Callable[[dict[str, Any], str], Any],
        data_entry: dict[str, Any],
        fields: tuple[str, ...],
        optional_fields: tuple[str, ...] = (),
    ) -> Union[list[Any], dict[str, Any]]:
        if not hasattr(self, "collection_type"):
            raise NotImplementedError(
                f"{self.__class__.__name__} has no attribute 'collection_type', "
                "please use collect_func_list or collect_func_dict instead."
            )

        collection_type = getattr(self, "collection_type")
        if collection_type == list:
            collect_func = self.collect_func_list
        elif collection_type == dict:
            collect_func = self.collect_func_dict
        else:
            raise ValueError(f"Unknown collection_type: {collection_type}")

        return collect_func(
            func,
            data_entry,
            fields,
            optional_fields=optional_fields,
        )


class CheckArrNDimMixin:
    def check_ndim(self, name: str, arr: np.ndarray, expected_ndim: int):
        if isinstance(arr, list):
            self.check_ndim(name, arr[0], expected_ndim - 1)
            return

        if arr.ndim != expected_ndim:
            raise AssertionError(
                f"Array '{name}' for {self.__class__.__name__} "
                f"has expected ndim: {expected_ndim}, "
                f"but got ndim: {arr.ndim} of shape {arr.shape}."
            )
