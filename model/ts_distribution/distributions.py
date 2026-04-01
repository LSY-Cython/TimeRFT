from typing import Callable, Optional, Union

import torch
from jaxtyping import PyTree
from torch.distributions import StudentT, Normal, Distribution, Gamma, constraints, LogNormal
from torch.distributions.utils import broadcast_all, lazy_property, logits_to_probs
from torch.nn import functional as F

from model.ts_distribution.base import DistributionOutput


class StudentTOutput(DistributionOutput):
    distr_cls = StudentT
    args_dim = dict(df=1, loc=1, scale=1)

    @property
    def domain_map(
        self,
    ) -> PyTree[
        Callable[[torch.FloatTensor],  torch.FloatTensor], "T"
    ]:
        return dict(df=self._df, loc=self._loc, scale=self._scale)

    @staticmethod
    def _df(df: torch.FloatTensor) -> torch.FloatTensor:  # (batch_size, 1) -> (batch_size, )
        return (2.0 + F.softplus(df)).squeeze(-1)

    @staticmethod
    def _loc(loc: torch.FloatTensor) -> torch.FloatTensor:
        return loc.squeeze(-1)

    @staticmethod
    def _scale(scale: torch.FloatTensor) -> torch.FloatTensor:
        epsilon = torch.finfo(scale.dtype).eps  # ensure variance > 0
        return F.softplus(scale).clamp_min(epsilon).squeeze(-1)


class NormalFixedScaleOutput(DistributionOutput):
    distr_cls = Normal
    args_dim = dict(loc=1)

    def __init__(self, scale: float = 1e-3):
        self.scale = scale

    @property
    def domain_map(
        self,
    ) -> PyTree[
        Callable[[torch.FloatTensor], torch.FloatTensor], "T"
    ]:
        return dict(loc=self._loc)

    @staticmethod
    def _loc(loc: torch.FloatTensor) -> torch.FloatTensor:
        return loc.squeeze(-1)

    def _distribution(
        self,
        distr_params: PyTree[torch.FloatTensor, "T"],
        validate_args: Optional[bool] = None,
    ) -> Normal:
        loc = distr_params["loc"]
        distr_params["scale"] = torch.as_tensor(
            self.scale, dtype=loc.dtype, device=loc.device
        )
        return self.distr_cls(**distr_params, validate_args=validate_args)


class NegativeBinomial(Distribution):
    arg_constraints = {
        "total_count": constraints.positive,
        "logits": constraints.real,
    }
    support = constraints.nonnegative
    has_rsample = False

    def __init__(
        self,
        total_count: Union[float, torch.Tensor],
        logits: Union[float, torch.Tensor],
        validate_args: Optional[bool] = None,
    ):
        (
            self.total_count,
            self.logits,
        ) = broadcast_all(total_count, logits)
        self.total_count = self.total_count.type_as(self.logits)
        batch_shape = self.logits.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape: torch.Size, _instance=None) -> "NegativeBinomial":
        new = self._get_checked_instance(NegativeBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new.logits = self.logits.expand(batch_shape)
        super(NegativeBinomial, new).__init__(
            batch_shape=batch_shape,
            validate_args=False,
        )
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            sample = torch.poisson(
                Gamma(
                    concentration=self.total_count,
                    rate=torch.exp(-self.logits),
                    validate_args=False,
                ).sample(sample_shape),
            )
        return sample

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        log_unnormalized_prob = (
            self.total_count * F.logsigmoid(-self.logits)
            + F.logsigmoid(self.logits) * value
        )
        log_normalization = self._lbeta(1 + value, self.total_count) + torch.log(
            self.total_count + value
        )
        return log_unnormalized_prob - log_normalization

    def _lbeta(self, x, y):
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)

    @property
    def mean(self) -> torch.Tensor:
        return self.total_count * torch.exp(self.logits)

    @property
    def variance(self) -> torch.Tensor:
        return self.mean / torch.sigmoid(-self.logits)


class NegativeBinomialOutput(DistributionOutput):
    distr_cls = NegativeBinomial
    args_dim = dict(total_count=1, logits=1)

    @property
    def domain_map(
        self,
    ) -> PyTree[
        Callable[[torch.FloatTensor], torch.FloatTensor], "T"
    ]:
        return dict(total_count=self._total_count, logits=self._logits)

    @staticmethod
    def _total_count(
        total_count: torch.FloatTensor
    ) -> torch.FloatTensor:
        return F.softplus(total_count).squeeze(-1)

    @staticmethod
    def _logits(
        logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        return logits.squeeze(-1)


class LogNormalOutput(DistributionOutput):
    distr_cls = LogNormal
    args_dim = dict(loc=1, scale=1)

    @property
    def domain_map(
        self,
    ) -> PyTree[
        Callable[[torch.FloatTensor], torch.FloatTensor], "T"
    ]:
        return dict(loc=self._loc, scale=self._scale)

    @staticmethod
    def _loc(loc: torch.FloatTensor) -> torch.FloatTensor:
        return loc.squeeze(-1)

    @staticmethod
    def _scale(scale: torch.FloatTensor) -> torch.FloatTensor:
        epsilon = torch.finfo(scale.dtype).eps
        return F.softplus(scale).clamp_min(epsilon).squeeze(-1)
