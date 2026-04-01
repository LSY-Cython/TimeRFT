import torch
import numpy as np
from einops import rearrange
from antropy import spectral_entropy
from hurst import compute_Hc
import matplotlib.pyplot as plt

from utils import move_dict_to_device


def compute_coverage_rate(forecasts, target, prediction_length, num_var, confidence=0.5):
    """
    Compute Prediction Interval Coverage Probability (PICP),
    refer to TMDM: Transformer-modulated diffusion models for probabilistic multivariate time series forecasting.
    """
    forecasts = rearrange(forecasts, 'n (c l) p -> n (l p) c', c=num_var)[:, -prediction_length:, :]
    target = rearrange(target, 'b (c l) p -> b (l p) c', c=num_var)[0, -prediction_length:, ...]

    low = (1 - confidence) / 2
    high = 1 - low
    lower_bound = torch.quantile(forecasts, q=low, dim=0)  # (prediction_length, num_var)
    upper_bound = torch.quantile(forecasts, q=high, dim=0)  # (prediction_length, num_var)
    target_in_interval = (target >= lower_bound) & (target <= upper_bound)  # (prediction_length, num_var)
    picp = torch.mean(target_in_interval.to(torch.float32))

    return picp


def compute_anomaly_ratio(target, num_var, z_score=1.96):
    """
    Compute the proportion of outliers, which reflects the level of noise in time series,
    refer to BLAST: Balanced Sampling Time Series Corpus for Universal Forecasting Models.
    """
    target = rearrange(target, 'b (c l) p -> b (l p) c', c=num_var)

    target_mean = torch.mean(target, dim=1, keepdim=True)
    target_std = torch.std(target, dim=1, keepdim=True)
    target_norm = (target - target_mean) / (target_std + 1e-6)
    outliers = (torch.abs(target_norm) > z_score).to(torch.float32)
    anomaly_ratio = torch.mean(outliers)

    return anomaly_ratio


def compute_forecastability(target, num_var):
    """
    Refer to ForeCA: Forecastable component analysis, ICML 2013.
    """
    target = rearrange(target, 'b (c l) p -> b (l p) c', c=num_var)[0]
    if not isinstance(target, np.ndarray):
        target = target.detach().cpu().numpy()

    try:
        entropy_channel = spectral_entropy(target, sf=1, normalize=True, axis=0)  # (num_var, )
        entropy_foreca = np.mean(1.0 - entropy_channel)

        hurst_channel = []
        for i in range(num_var):
            hurst, _, _ = compute_Hc(target[:, i], kind='change', simplified=False)
            hurst_channel.append(hurst)
        hurst_foreca = np.mean(hurst_channel)

        total_foreca = 0.5 * entropy_foreca + 0.5 * hurst_foreca

    except Exception as e:
        total_foreca = 1.0

    return total_foreca


def training_data_selection(ts_model, grpo_trainer, dataloader, num_generations, prediction_length,
                            target_dim, target_dim_pred, device):
    easy_indices, medium_indices, hard_indices = [], [], []
    anomaly_indices, foreca_indices = [], []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = move_dict_to_device(batch_data, device)
            data_idx = batch_data["window"].item()  # (1, ), by adding window into SelectFields
            target = batch_data["target"]  # (1, seq_len, patch_size)
            forecasts = grpo_trainer.generate_forecasts(ts_model, batch_data, num_generations)  # (num_generations, seq_len, patch_size)

            # Cannot evaluate on covariates
            num_token_per_var = target.shape[1] // target_dim
            target = target[:, 0*num_token_per_var:target_dim_pred*num_token_per_var, :]
            forecasts = forecasts[:, 0*num_token_per_var:target_dim_pred*num_token_per_var, :]

            # Difficulty-based selection
            picp_50 = compute_coverage_rate(forecasts, target, prediction_length, target_dim_pred, confidence=0.5).item()
            picp_90 = compute_coverage_rate(forecasts, target, prediction_length, target_dim_pred, confidence=0.9).item()
            if picp_50 >= 0.7:
                easy_indices.append(data_idx)
            elif picp_90 <= 0.7:
                hard_indices.append(data_idx)
            else:
                medium_indices.append(data_idx)

            # Anomaly-based selection
            anomaly_ratio = compute_anomaly_ratio(target, target_dim_pred, z_score=1.96)  # 95% confidence interval
            if anomaly_ratio >= 0.5:
                anomaly_indices.append(data_idx)

            # Forecastability selection
            forecastability = compute_forecastability(target, target_dim_pred)
            if forecastability <= 0.5:
                foreca_indices.append(data_idx)

            print(f"[Batch {batch_idx}]: picp_50={picp_50}, picp_90={picp_90}, "
                  f"anomaly_ratio={anomaly_ratio}, forecastability={forecastability}")

        filter_indices = list(dict.fromkeys(easy_indices + hard_indices + anomaly_indices + foreca_indices))
        retain_indices = [x for x in medium_indices if x not in filter_indices]

        return torch.tensor(retain_indices).to(device), torch.tensor(filter_indices).to(device)