import abc
from typing import Any, Optional, Callable

import torch
from einops import rearrange, reduce
from torch.distributions import Distribution

from utils import safe_div, abstract_class_property

"""
-------- Base Loss --------
"""


class PackedLoss(abc.ABC):
    """
    Abstract base class for loss functions supporting packed inputs.
    Subclasses should implement the _loss_func method which computes the loss function per token.
    """

    def __call__(
        self,
        pred: Any,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        prediction_mask: Optional[torch.BoolTensor],  # (batch_size, seq_len)
        observed_mask: Optional[torch.BoolTensor] = None,  # (batch_size, seq_len, patch_size)
        sample_id: Optional[torch.IntTensor] = None,  # (batch_size, seq_len)
        variate_id: Optional[torch.IntTensor] = None,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:
        """
        :param pred: predictions
        :param target: target labels
        :param prediction_mask: 1 for predictions, 0 for non-predictions
        :param observed_mask: 1 for observed values, 0 for non-observed values
        :param sample_id: integer array representing the sample id
        :param variate_id: integer array representing the variate id
        :return: loss
        """
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros_like(prediction_mask, dtype=torch.long)
        if variate_id is None:
            variate_id = torch.zeros_like(prediction_mask, dtype=torch.long)

        loss = self._loss_func(
            pred, target, prediction_mask, observed_mask, sample_id, variate_id
        )
        return self.reduce_loss(
            loss, prediction_mask, observed_mask, sample_id, variate_id
        )

    @abc.abstractmethod
    def _loss_func(
        self,
        pred: Any,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        prediction_mask: torch.BoolTensor,  # (batch_size, seq_len)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor: ...  # (batch_size, seq_len, patch_size)

    def reduce_loss(
        self,
        loss: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        prediction_mask: Optional[torch.BoolTensor],  # (batch_size, seq_len)
        observed_mask: Optional[torch.BoolTensor],  # (batch_size, seq_len, patch_size)
        sample_id: Optional[torch.IntTensor],  # (batch_size, seq_len)
        variate_id: Optional[torch.IntTensor],  # (batch_size, seq_len)
    ) -> torch.FloatTensor:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        mask = prediction_mask.unsqueeze(-1) * observed_mask
        tobs = reduce(
            id_mask
            * reduce(
                mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        nobs = reduce(
            id_mask * rearrange(prediction_mask, "... seq -> ... 1 seq"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        ) * prediction_mask.unsqueeze(-1)
        nobs = torch.where(nobs == 0, nobs, 1 / nobs).sum()
        loss = safe_div(loss, tobs * nobs)
        return (loss * mask).sum()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


"""
-------- Distributional Loss --------
"""


class PackedDistributionLoss(PackedLoss):
    """Abstract base class for loss functions on probabilistic (distribution) forecasts."""

    @abc.abstractmethod
    def _loss_func(
        self,
        pred: Distribution,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        prediction_mask: torch.BoolTensor,  # (batch_size, seq_len)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor: ...  # (batch_size, seq_len, patch_size)


class PackedNLLLoss(PackedDistributionLoss):
    def _loss_func(
        self,
        pred: Distribution,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        prediction_mask: torch.BoolTensor,  # (batch_size, seq_len)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, patch_size)
        return -pred.log_prob(target)


"""
-------- Point Loss --------
"""


class PackedPointLoss(PackedLoss):
    """Abstract base class for loss functions on point forecasts."""

    @abc.abstractmethod
    def _loss_func(
        self,
        pred: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        prediction_mask: torch.BoolTensor,  # (batch_size, seq_len)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor: ...  # (batch_size, seq_len, patch_size)


@abstract_class_property("error_func")
class PackedPointNormalizedLoss(PackedPointLoss, abc.ABC):
    error_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = NotImplemented

    def __init__(
        self,
        normalize: str = "none",
        correction: int = 1,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.normalize = normalize
        self.correction = correction
        self.epsilon = epsilon

    def _loss_func(
        self,
        pred: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        prediction_mask: torch.BoolTensor,  # (batch_size, seq_len)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, patch_size)
        loss = self.error_func(pred, target)
        denominator = self.denominator_func(
            target, observed_mask, sample_id, variate_id
        )
        loss = safe_div(loss, denominator)
        return loss

    @property
    def denominator_func(self) -> Callable:
        func_map = {
            "none": self.none_denominator,
            "absolute_target": self.abs_target_denominator,  # normalize by mean abs_target for each obs
            "absolute_target_squared": self.abs_target_sq_denominator,  # matfact def of NRMSE/ND
            "target": self.target_denominator,  # normalize by mean target for each obs
            "target_squared": self.target_sq_denominator,  # classical def of NRMSE/NMAE
            "standard_deviation": self.std_dev_denominator,  # normalize by standard deviation of target for each obs
            "variance": self.var_denominator,  # normalize by variance of target for each obs
        }
        if self.normalize not in func_map:
            raise ValueError(f"Invalid normalize type '{self.normalize}'")
        return func_map[self.normalize]

    @staticmethod
    def none_denominator(
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, patch_size)
        return torch.ones_like(target)

    @staticmethod
    def reduce_denominator(
        value: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, patch_size)
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        value = reduce(
            id_mask * reduce(value * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        value = safe_div(value, tobs)
        return value

    def abs_target_denominator(
        self,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, patch_size)
        return self.reduce_denominator(
            target.abs(), observed_mask, sample_id, variate_id
        )

    def abs_target_sq_denominator(
        self,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, patch_size)
        return torch.pow(
            self.reduce_denominator(target.abs(), observed_mask, sample_id, variate_id),
            2,
        )

    def target_denominator(
        self,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, patch_size)
        return self.reduce_denominator(target, observed_mask, sample_id, variate_id)

    def target_sq_denominator(
        self,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, patch_size)
        return torch.pow(
            self.reduce_denominator(target, observed_mask, sample_id, variate_id), 2
        )

    def std_dev_denominator(
        self,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, patch_size)
        var = self.var_denominator(target, observed_mask, sample_id, variate_id)
        std_dev = torch.sqrt(var + self.epsilon)
        return std_dev

    def var_denominator(
        self,
        target: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        observed_mask: torch.BoolTensor,  # (batch_size, seq_len, patch_size)
        sample_id: torch.IntTensor,  # (batch_size, seq_len)
        variate_id: torch.IntTensor,  # (batch_size, seq_len)
    ) -> torch.FloatTensor:  # (batch_size, seq_len, patch_size)
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = reduce(
            id_mask * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = safe_div(loc, tobs)
        var = reduce(
            id_mask
            * reduce(
                ((target - loc) ** 2) * observed_mask, "... seq dim -> ... 1 seq", "sum"
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        var = safe_div(var, (tobs - self.correction))
        return var


class PackedNMSELoss(PackedPointNormalizedLoss):
    error_func = torch.nn.MSELoss(reduction="none")


class PackedMSELoss(PackedNMSELoss):
    def __init__(self, normalize):
        super().__init__(normalize=normalize)


class PackedNRMSELoss(PackedPointNormalizedLoss):
    error_func = torch.nn.MSELoss(reduction="none")

    def reduce_loss(
        self,
        loss: torch.FloatTensor,  # (batch_size, seq_len, patch_size)
        prediction_mask: Optional[torch.BoolTensor],   # (batch_size, seq_len)
        observed_mask: Optional[torch.BoolTensor],  # (batch_size, seq_len, patch_size)
        sample_id: Optional[torch.IntTensor],  # (batch_size, seq_len)
        variate_id: Optional[torch.IntTensor],  # (batch_size, seq_len)
    ) -> torch.FloatTensor:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        mask = prediction_mask.unsqueeze(-1) * observed_mask
        loss = reduce(
            id_mask
            * reduce(
                loss * mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loss = torch.sqrt(loss + self.epsilon)
        tobs = reduce(
            id_mask
            * reduce(
                mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loss = safe_div(loss, torch.sqrt(tobs))

        return super().reduce_loss(
            loss, prediction_mask, observed_mask, sample_id, variate_id
        )


class PackedRMSELoss(PackedNRMSELoss):
    def __init__(self, normalize):
        super().__init__(normalize=normalize)
