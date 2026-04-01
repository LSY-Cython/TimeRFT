import numpy as np
import properscoring as ps
from einops import rearrange
from typing import Union, Optional, Dict, Any


def compute_point_forecast(forecasts, forecast_type):  # (num_samples, prediction_length, target_dim)
    if forecast_type == "mean":
        prediction = np.mean(forecasts, axis=0)  # (prediction_length, target_dim)
    elif forecast_type == "quantile":
        prediction = np.quantile(forecasts, q=0.5, axis=0)  # (prediction_length, target_dim)
    else:
        raise ValueError(f"{forecast_type} is not supported")

    return prediction


def compute_seasonal_error(past, seasonality):  # (context_length, target_dim)
    seasonal_diff = past[seasonality:] - past[:-seasonality]
    is_nan = np.isnan(seasonal_diff)
    if is_nan.any():
        num_nan = np.sum(is_nan)
        print(f"Number of NaN in seasonal_diff: {num_nan}")

    abs_seasonal_error = np.nanmean(np.abs(seasonal_diff), axis=0, keepdims=True)  # (1, target_dim)

    return abs_seasonal_error


def compute_quantile_loss(forecasts, target, quantile_levels):
    """
    Compute quantile loss for each observation.
    """

    pred_per_quantile = []
    for q in quantile_levels:
        pred_per_quantile.append(np.quantile(forecasts, q, axis=0))
    q_pred = np.stack(pred_per_quantile, axis=-1)  # (prediction_length, target_dim, num_quantile)

    target = target[..., None]  # (prediction_length, target_dim, 1)
    assert target.shape[:-1] == q_pred.shape[:-1]

    quantile_loss = 2 * np.abs((target - q_pred) * ((target <= q_pred) - np.array(quantile_levels)))

    return quantile_loss  # (prediction_length, target_dim, num_quantile)


class MAE:
    """
    Mean absolute error.
    """

    def __init__(self, forecast_type: str = 'mean'):
        self.forecast_type = forecast_type

    def __call__(self,
                 forecasts: np.ndarray,  # (num_samples, prediction_length, target_dim)
                 target: np.ndarray,  # (prediction_length, target_dim),
                 past: Optional[np.ndarray] = None,
                 seasonality: Optional[int] = None,
                 axis: Optional[Union[int, tuple]] = None) -> Union[float, np.ndarray]:
        prediction = compute_point_forecast(forecasts, self.forecast_type)  # (prediction_length, target_dim)

        abs_error = np.abs(prediction - target)
        if axis is None:
            mae = np.nanmean(abs_error)
        else:
            mae = np.nanmean(abs_error, axis=axis)

        return float(mae) if np.isscalar(mae) else mae


class MSE:
    """
    Mean squared error.
    """

    def __init__(self, forecast_type: str = 'mean'):
        self.forecast_type = forecast_type

    def __call__(self,
                 forecasts: np.ndarray,  # (num_samples, prediction_length, target_dim)
                 target: np.ndarray,  # (prediction_length, target_dim),
                 past: Optional[np.ndarray] = None,
                 seasonality: Optional[int] = None,
                 axis: Optional[Union[int, tuple]] = None) -> Union[float, np.ndarray]:
        prediction = compute_point_forecast(forecasts, self.forecast_type)  # (prediction_length, target_dim)

        squared_error = (prediction - target) ** 2
        if axis is None:
            mse = np.nanmean(squared_error)
        else:
            mse = np.nanmean(squared_error, axis=axis)

        return float(mse) if np.isscalar(mse) else mse


class RMSE:
    """
    Root mean squared error.
    """

    def __init__(self, forecast_type: str = 'mean'):
        self.forecast_type = forecast_type

    def __call__(self,
                 forecasts: np.ndarray,  # (num_samples, prediction_length, target_dim)
                 target: np.ndarray,  # (prediction_length, target_dim),
                 past: Optional[np.ndarray] = None,
                 seasonality: Optional[int] = None,
                 axis: Optional[Union[int, tuple]] = None) -> Union[float, np.ndarray]:
        prediction = compute_point_forecast(forecasts, self.forecast_type)  # (prediction_length, target_dim)

        squared_error = (prediction - target) ** 2
        if axis is None:
            rmse = np.sqrt(np.nanmean(squared_error))
        else:
            rmse = np.sqrt(np.nanmean(squared_error, axis=axis))

        return float(rmse) if np.isscalar(rmse) else rmse


class MASE:
    """
    Mean absolute scaled error.
    """

    def __init__(self, forecast_type: str = 'mean', epsilon: float = 0.0):
        self.forecast_type = forecast_type
        self.epsilon = epsilon

    def __call__(self,
                 forecasts: np.ndarray,  # (num_samples, prediction_length, target_dim)
                 target: np.ndarray,  # (prediction_length, target_dim)
                 past: np.ndarray,  # (context_length, target_dim)
                 seasonality: int) -> Union[float, np.ndarray]:
        prediction = compute_point_forecast(forecasts, self.forecast_type)  # (prediction_length, target_dim)

        seasonal_error = compute_seasonal_error(past, seasonality)  # (1, target_dim)
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)

        mase = np.nanmean(np.abs(target - prediction) / seasonal_error)

        return float(mase) if np.isscalar(mase) else mase


class CRPS:
    """
    Continuous ranked probability score.
    """

    def __init__(self, epsilon: float = 0.0):
        self.epsilon = epsilon

    def __call__(self,
                 forecasts: np.ndarray,  # (num_samples, prediction_length, target_dim)
                 target: np.ndarray,  # (prediction_length, target_dim)
                 past: Optional[np.ndarray] = None,  # (context_length, target_dim)
                 seasonality: Optional[int] = None) -> Union[float, np.ndarray]:
        target = rearrange(target, "l c -> (l c)")
        forecasts = rearrange(forecasts, "n l c -> (l c) n")

        crps_per_time_step = ps.crps_ensemble(target, forecasts)
        crps = np.sum(crps_per_time_step) / forecasts.shape[0]

        return float(crps) if np.isscalar(crps) else crps


class MQL:
    """
    Mean quantile loss.
    """

    def __init__(self, quantile_levels: list[float], epsilon: float = 0.0):
        self.quantile_levels = quantile_levels
        self.epsilon = epsilon

    def __call__(self,
                 forecasts: np.ndarray,  # (num_samples, prediction_length, target_dim)
                 target: np.ndarray,  # (prediction_length, target_dim)
                 past: Optional[np.ndarray] = None,  # (context_length, target_dim)
                 seasonality: Optional[int] = None) -> Union[float, np.ndarray]:
        quantile_loss = compute_quantile_loss(forecasts, target, self.quantile_levels)  # (prediction_length, target_dim, num_quantile)
        ql_per_time_step = np.nanmean(quantile_loss, axis=-1)  # (prediction_length, target_dim)

        mql = np.nanmean(ql_per_time_step)

        return float(mql) if np.isscalar(mql) else mql


class SQL:
    """
    Scaled quantile loss.
    """

    def __init__(self, quantile_levels: list[float], epsilon: float = 0.0):
        self.quantile_levels = quantile_levels
        self.epsilon = epsilon

    def __call__(self,
                 forecasts: np.ndarray,  # (num_samples, prediction_length, target_dim)
                 target: np.ndarray,  # (prediction_length, target_dim)
                 past: np.ndarray,  # (context_length, target_dim)
                 seasonality: int) -> Union[float, np.ndarray]:
        quantile_loss = compute_quantile_loss(forecasts, target, self.quantile_levels)  # (prediction_length, target_dim, num_quantile)
        ql_per_time_step = np.nanmean(quantile_loss, axis=-1)  # (prediction_length, target_dim)

        seasonal_error = compute_seasonal_error(past, seasonality)  # (1, target_dim)
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)

        sql = np.nanmean(ql_per_time_step / seasonal_error)

        return float(sql) if np.isscalar(sql) else sql