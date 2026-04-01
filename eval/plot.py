import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecasts_res(
    target: np.ndarray,  # (test_length, num_var)
    forecasts: np.ndarray,  # (num_samples, test_length, num_var)
    prediction: np.ndarray,  # (test_length, num_var)
    storage_path: str,
    prediction_length: int,
    test_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
):
    num_var = target.shape[1]
    test_windows = int(test_length / prediction_length)

    if prediction_length <= 48:
        for i in range(num_var):
            # Plot point forecasts
            plt.plot(target[:, i], color="red", label="Target")
            plt.plot(prediction[:, i], color="blue", label="Forecast")

            # Plot prediction intervals
            for interval in intervals:
                low = (1 - interval) / 2
                plt.fill_between(
                    x=np.arange(target.shape[0]),
                    y1=np.quantile(forecasts[..., i], q=low, axis=0),
                    y2=np.quantile(forecasts[..., i], q=1-low, axis=0),
                    alpha=0.5 - interval / 3,
                    facecolor="blue",
                    label=f"pred: {interval}"
                )

            img_path = os.path.join(storage_path, f"Variate {i}.png")
            plt.title(f"Variate {i} result")
            plt.legend(loc="upper right")
            plt.savefig(img_path)
            plt.close()
            plt.clf()

    else:
        for i in range(num_var):
            for j in range(test_windows):
                # Plot point forecasts
                plt.plot(target[j*prediction_length:(j+1)*prediction_length, i], color="red", label="Target")
                plt.plot(prediction[j*prediction_length:(j+1)*prediction_length, i], color="blue", label="Forecast")

                # Plot prediction intervals
                for interval in intervals:
                    low = (1 - interval) / 2
                    plt.fill_between(
                        x=np.arange(prediction_length),
                        y1=np.quantile(forecasts[:, j*prediction_length:(j+1)*prediction_length, i], q=low, axis=0),
                        y2=np.quantile(forecasts[:, j*prediction_length:(j+1)*prediction_length, i], q=1 - low, axis=0),
                        alpha=0.5 - interval / 3,
                        facecolor="blue",
                        label=f"pred: {interval}"
                    )

                img_path = os.path.join(storage_path, f"Variate {i} Window {j}.png")
                plt.title(f"Variate {i} Window {j} result")
                plt.legend(loc="upper right")
                plt.savefig(img_path)
                plt.close()
                plt.clf()
