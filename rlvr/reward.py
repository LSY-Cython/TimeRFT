"""
Define outcome-supervised reward functions for time series forecasting.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


def sequence_normalization(prediction, target, prediction_length):  # (num_samples, series_length, num_var)
    # Channel-wise normalization
    target_mean = torch.mean(target[:, -prediction_length:, :], dim=1).unsqueeze(1)  # (num_samples, 1, num_var)
    target_std = torch.std(target[:, -prediction_length:, :], dim=1).unsqueeze(1)  # (num_samples, 1, num_var)
    target_norm = (target - target_mean) / (target_std + 1e-8)
    pred_norm = (prediction - target_mean) / (target_std + 1e-8)

    return pred_norm, target_norm


def sequence_accuracy_reward(prediction, target, prediction_length, patch_size):  # (num_samples, series_length, num_var)
    pred_norm, target_norm = sequence_normalization(prediction, target, prediction_length)
    num_samples, series_length, num_var = target.shape

    # Calculate channel-wise MSE reward
    mse_channel = torch.mean((target_norm - pred_norm) ** 2, dim=1)  # (num_samples, num_var)
    num_token_per_var = series_length // patch_size
    mse_channel = torch.repeat_interleave(mse_channel, repeats=num_token_per_var, dim=1)  # (num_samples, token_length)
    mse_reward = torch.exp(-mse_channel)

    # Calculate channel-wise MAE reward
    mae_channel = torch.mean(torch.abs(target_norm - pred_norm), dim=1)  # (num_samples, num_var)
    num_token_per_var = series_length // patch_size
    mae_channel = torch.repeat_interleave(mae_channel, repeats=num_token_per_var, dim=1)  # (num_samples, token_length)
    mae_reward = torch.exp(-mae_channel)

    return 1.0 * mse_reward + 0.0 * mae_reward  # (num_samples, token_length)


def sequence_frequency_reward(prediction, target, prediction_length, patch_size):  # (num_samples, series_length, num_var)
    pred_norm, target_norm = sequence_normalization(prediction, target, prediction_length)
    num_samples, series_length, num_var = target.shape

    # Calculate channel-wise spectral reward
    freq_channel = (torch.fft.rfft(pred_norm[:, -prediction_length:, :], dim=1, norm="backward") -
                    torch.fft.rfft(target_norm[:, -prediction_length:, :], dim=1, norm="backward")).abs()  # (num_samples, series_length//2+1, num_var)
    freq_bins = torch.fft.rfftfreq(prediction_length, 1.0).unsqueeze(0).unsqueeze(2).to(target.device)  # (1, series_length//2+1, 1)
    freq_weights = torch.softmax(freq_bins, dim=1)
    freq_channel = torch.mean((freq_channel ** 2) * freq_weights, dim=1)
    num_token_per_var = series_length // patch_size
    freq_channel = torch.repeat_interleave(freq_channel, repeats=num_token_per_var, dim=1)  # (num_samples, token_length)
    freq_reward = torch.exp(-freq_channel)

    return freq_reward  # (num_samples, token_length)


def sequence_structure_reward(prediction, target, prediction_length, patch_size, type="Variance"):  # (num_samples, series_length, num_var)
    """
    Refer to PSLoss: Sequence-wise Structural Loss for Time Series Forecasting.
    """
    pred_norm, target_norm = sequence_normalization(prediction, target, prediction_length)
    num_samples, series_length, num_var = target.shape

    target_mean = torch.mean(target_norm, dim=1, keepdim=True)
    pred_mean = torch.mean(pred_norm, dim=1, keepdim=True)
    target_std = torch.std(target_norm, dim=1, keepdim=True)
    pred_std = torch.std(pred_norm, dim=1, keepdim=True)

    if type == "Variance":  # Calculate sequence-wise variability reward
        kl_loss = nn.KLDivLoss(reduction='none')
        target_softmax = torch.softmax(target_norm, dim=1)
        pred_softmax = torch.log_softmax(pred_norm, dim=1)
        kl_seq = torch.sum(kl_loss(pred_softmax, target_softmax), dim=1, keepdim=True)  # (num_samples, 1, num_var)
        num_token_per_var = series_length // patch_size
        var_reward = torch.exp(-kl_seq).repeat(1, num_token_per_var, 1)
        var_reward = rearrange(var_reward, "n l c -> n (c l)")

        return var_reward  # (num_samples, token_length)

    else:
        raise NotImplementedError


def patch_accuracy_reward(prediction, target, prediction_length, patch_size):  # (num_samples, series_length, num_var)
    pred_norm, target_norm = sequence_normalization(prediction, target, prediction_length)
    num_samples, _, num_var = target.shape
    target_patch = rearrange(target_norm, 'n (l p) c -> n (c l) p', p=patch_size)
    pred_patch = rearrange(pred_norm, 'n (l p) c -> n (c l) p', p=patch_size)

    # Calculate patch-wise MSE reward
    mse_patch = torch.mean((target_patch - pred_patch) ** 2, dim=-1)  # (num_samples, token_length)
    mse_reward = torch.exp(-mse_patch)

    # Calculate patch-wise MAE reward
    mae_patch = torch.mean(torch.abs(target_patch - pred_patch), dim=-1)  # (num_samples, token_length)
    mae_reward = torch.exp(-mae_patch)

    return 1.0 * mse_reward + 0.0 * mae_reward


def patch_frequency_reward(prediction, target, prediction_length, patch_size):  # (num_samples, series_length, num_var)
    pred_norm, target_norm = sequence_normalization(prediction, target, prediction_length)
    num_samples, _, num_var = target.shape
    target_patch = rearrange(target_norm, 'n (l p) c -> n (c l) p', p=patch_size)
    pred_patch = rearrange(pred_norm, 'n (l p) c -> n (c l) p', p=patch_size)

    # Calculate patch-wise spectral reward
    freq_patch = (torch.fft.rfft(pred_patch, dim=-1, norm="backward") -
                  torch.fft.rfft(target_patch, dim=-1, norm="backward")).abs()  # (num_samples, token_length, patch_size//2+1)
    freq_bins = torch.fft.rfftfreq(patch_size, 1.0).unsqueeze(0).unsqueeze(1).to(target.device)  # (1, 1, patch_size//2+1)
    freq_weights = torch.softmax(freq_bins, dim=-1)
    freq_patch = torch.mean((freq_patch ** 2) * freq_weights, dim=-1)  # (num_samples, token_length)
    freq_reward = torch.exp(-freq_patch)

    return freq_reward


def patch_structure_reward(prediction, target, prediction_length, patch_size, type="Variance"):  # (num_samples, series_length, num_var)
    """
    Refer to PSLoss: Patch-wise Structural Loss for Time Series Forecasting.
    """
    pred_norm, target_norm = sequence_normalization(prediction, target, prediction_length)
    num_samples, _, num_var = target.shape
    pred_patch = rearrange(pred_norm, 'n (l p) c -> n l p c', p=patch_size)
    target_patch = rearrange(target_norm, 'n (l p) c -> n l p c', p=patch_size)

    target_patch_mean = torch.mean(target_patch, dim=2, keepdim=True)
    pred_patch_mean = torch.mean(pred_patch, dim=2, keepdim=True)
    target_patch_std = torch.std(target_patch, dim=2, keepdim=True)
    pred_patch_std = torch.std(pred_patch, dim=2, keepdim=True)

    if type == "Correlation":  # Calculate patch-wise Pearson Correlation Coefficient reward
        cov_patch = torch.mean((target_patch-target_patch_mean)*(pred_patch-pred_patch_mean), dim=2, keepdim=True)
        pcc_patch = (cov_patch + 1e-5) / (target_patch_std*pred_patch_std + 1e-5)
        pcc_reward = (pcc_patch.squeeze(2) + 1.0) * 0.5  # (num_samples, num_token_per_var, num_var)
        pcc_reward = rearrange(pcc_reward, 'n l c -> n (c l)')

        return pcc_reward  # (num_samples, token_length)

    elif type == "Variance":  # Calculate patch-wise variability reward
        kl_loss = nn.KLDivLoss(reduction='none')
        target_patch_softmax = torch.softmax(target_patch, dim=2)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=2)
        kl_patch = kl_loss(pred_patch_softmax, target_patch_softmax).sum(dim=2)
        var_reward = rearrange(torch.exp(-kl_patch), 'n l c -> n (c l)')

        return var_reward  # (num_samples, token_length)

    elif type == "Mean":  # Calculate patch-wise mean reward
        mean_patch = torch.abs(target_patch_mean-pred_patch_mean).squeeze(dim=2)  # (num_samples, num_token_per_var, num_var)
        mean_reward = rearrange(mean_patch, 'n l c -> n (c l)')

        return mean_reward  # (num_samples, token_length)

    else:
        raise NotImplementedError


def season_trend_decomposition(alpha: float, x):  # (num_samples, series_length, num_var)
    """
    Exponential Moving Average (EMA) decomposition, refer to baselines/utils.py in DBLoss.
    """
    _, T, _ = x.shape
    powers = torch.flip(torch.arange(T, dtype=torch.double), dims=(0,))
    weights = torch.pow((1 - alpha), powers).to(x.device)
    divisor = weights.clone()
    weights[1:] = weights[1:] * alpha
    weights = weights.reshape(1, T, 1)
    divisor = divisor.reshape(1, T, 1)
    trend = torch.div(torch.cumsum(x * weights, dim=1), divisor).to(torch.float32)
    season = x - trend

    return trend, season  # (num_samples, length, num_var)


def season_trend_reward(alpha: float, prediction, target, prediction_length, patch_size):  # (num_samples, series_length, num_var)
    pred_trend, pred_season = season_trend_decomposition(alpha, prediction)
    target_trend, target_season = season_trend_decomposition(alpha, target)

    trend_reward = sequence_accuracy_reward(pred_trend, target_trend, prediction_length, patch_size)
    season_reward = sequence_accuracy_reward(pred_season, target_season, prediction_length, patch_size)
    reward = 0.5 * trend_reward + 0.5 * season_reward

    return reward  # (num_samples, )


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import matplotlib.pyplot as plt

    """
    Compare alpha-sensitive EMA with time-consuming STL
    """
    # from statsmodels.tsa.seasonal import STL
    # import time

    # rs = np.random.RandomState(0xA4FD94BC)
    # tau = 2000
    # t = np.arange(tau)
    # period = int(0.05 * tau)
    # seasonal = period + ((period % 2) == 0)  # Ensure odd
    # e = 0.25 * rs.standard_normal(tau)
    # y = np.cos(t / tau * 2 * np.pi) + 0.25 * np.sin(t / period * 2 * np.pi) + e
    # plt.plot(y)
    # plt.title("Simulated Data")
    # xlim = plt.gca().set_xlim(0, tau)
    # plt.show()
    #
    # stl_st = time.time()
    # stl_fit = STL(y, period=period, seasonal=seasonal).fit()  # y must be squeezable to 1d
    # stl_trend = stl_fit.trend
    # stl_season = stl_fit.seasonal
    # stl_resid = stl_fit.resid
    # stl_et = time.time()
    #
    # alpha = 1 / (period + 1)
    # y_temp = torch.from_numpy(y).reshape(1, -1, 1)
    # ema_st = time.time()
    # ema_trend, ema_season = season_trend_decomposition(alpha, y_temp)
    # ema_trend = ema_trend.detach().cpu().numpy().reshape(-1)
    # ema_season = ema_season.detach().cpu().numpy().reshape(-1)
    # ema_et = time.time()
    #
    # print(f"STL time cost: {stl_et-stl_st}s; EMA time cost: {ema_et-ema_st}s")
    # plt.subplot(2, 2, 1)
    # plt.plot(stl_trend)
    # plt.title("STL trend")
    # plt.subplot(2, 2, 2)
    # plt.plot(ema_trend)
    # plt.title("EMA trend")
    # plt.subplot(2, 2, 3)
    # plt.plot(stl_season)
    # plt.title("STL season")
    # plt.subplot(2, 2, 4)
    # plt.plot(ema_season)
    # plt.title("EMA season")
    # plt.show()


