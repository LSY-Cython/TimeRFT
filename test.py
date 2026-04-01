"""
Evaluate on GiftEval benchmark, refer to ../gift-eval/notebooks/moirai.ipynb
"""
import os
import yaml
import torch
import numpy as np
from safetensors.torch import load_file
import argparse
import time
import logging
import pickle as pkl
import copy

from model.ts_moe import MoiraiMoEModule
from model.ts_distribution.mixture import MixtureOutput
from model.ts_distribution.distributions import (
    StudentTOutput,
    NormalFixedScaleOutput,
    NegativeBinomialOutput,
    LogNormalOutput,
)
from eval.forecast import MoiraiMoEForecast
from eval.data import Dataset
from eval.plot import plot_forecasts_res
from eval.metrics import MSE, MAE, RMSE, MASE, CRPS, MQL, SQL
from utils import fix_seed

from gluonts.model import evaluate_model
from gluonts.itertools import batcher
from gluonts.time_feature import get_seasonality


def main(cfg, stage, model_scale="small"):
    # Instantiate the metrics
    metrics = {"MSE[mean]": MSE(forecast_type="mean"), "MAE[mean]": MAE(forecast_type="mean"),
               "CRPS": CRPS(), "MQL": MQL(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), }
    eval_res = {"MSE[mean]": [], "MAE[mean]": [], "CRPS": [], "MQL": []}

    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    eval_data_cfg = cfg["eval_dataset"]
    dataset = Dataset(name=eval_data_cfg["dataset_name"],
                      prediction_length=eval_data_cfg["prediction_length"],
                      test_length=eval_data_cfg["test_length"],
                      to_univariate=eval_data_cfg["to_univariate"],
                      dataset_path=eval_data_cfg['dataset_path'])

    # Instantiate Moirai-MoE model
    model_cfg = cfg["model"]
    distr_output = MixtureOutput(components=[StudentTOutput(),  # must instantiate a distribution object
                                             NormalFixedScaleOutput(),
                                             NegativeBinomialOutput(),
                                             LogNormalOutput()])
    ts_model = MoiraiMoEModule(
        distr_output=distr_output,
        d_model=model_cfg["d_model"],
        d_ff=model_cfg["d_ff"],
        num_layers=model_cfg["num_layers"],
        patch_sizes=model_cfg["patch_sizes"],
        max_seq_len=model_cfg["max_seq_len"],
        attn_dropout_p=model_cfg["attn_dropout_p"],
        dropout_p=model_cfg["dropout_p"],
        scaling=model_cfg["scaling"],
    ).to(device)

    # Load checkpoints
    if args.eval_stage == "pretrain":
        checkpoint = load_file(args.pretrained_path)
        print(f"Load checkpoint: {args.pretrained_path}")
    elif args.eval_stage == "finetune" and stage == "sft":
        checkpoint = load_file(args.sft_path)
        print(f"Load checkpoint: {args.sft_path}")
    elif args.eval_stage == "finetune" and stage == "rft":
        checkpoint = load_file(args.rft_path)
        print(f"Load checkpoint: {args.rft_path}")
    else:
        raise NotImplementedError
    ts_model.load_state_dict(checkpoint)

    # Instantiate evaluation model
    ts_model.eval()
    eval_model = MoiraiMoEForecast(
        prediction_length=eval_data_cfg["prediction_length"],
        target_dim=eval_data_cfg["target_dim"],
        feat_dynamic_real_dim=eval_data_cfg["feat_dynamic_real_dim"],
        past_feat_dynamic_real_dim=eval_data_cfg["past_feat_dynamic_real_dim"],
        context_length=eval_data_cfg["context_length"],
        module=ts_model,
        patch_size=eval_data_cfg["patch_size"],
        num_samples=eval_data_cfg["num_samples"])
    predictor = eval_model.create_predictor(eval_data_cfg["batch_size"], device.type)

    # Mark experiment settings
    if stage == "pretrain":
        eval_setting = f"eval-{stage}-cl{eval_data_cfg['context_length']}-pl{eval_data_cfg['prediction_length']}"
    elif stage == "sft":
        eval_setting = f"eval-{stage}-cl{eval_data_cfg['context_length']}-pl{eval_data_cfg['prediction_length']}"
    elif stage == "rft":
        rl_trainer_cfg = cfg["rl_trainer"]
        eval_setting = f"eval-{stage}-cl{eval_data_cfg['context_length']}-pl{eval_data_cfg['prediction_length']}-" + \
                       f"ng{int(rl_trainer_cfg['num_generations'])}-kl{rl_trainer_cfg['beta']}-" + \
                       f"ds{int(rl_trainer_cfg['data_selection'])}-rs{int(rl_trainer_cfg['reward_shaping'])}-" + \
                       f"lar{rl_trainer_cfg['lambda_acc_reward']}-lvr{rl_trainer_cfg['lambda_var_reward']}-" + \
                       f"lvs{rl_trainer_cfg['lambda_var_synergy']}-lfs{rl_trainer_cfg['lambda_freq_synergy']}"
    else:
        raise NotImplementedError

    # Configure logging
    log_cfg = cfg["logger"]
    log_path = f"{log_cfg['filename']}-{eval_setting}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(f"{log_path}/epoch{args.epoch}.log", mode='w')
                        ])
    logger = logging.getLogger(__name__)
    logger.info(f"Start test stage with setting: {log_cfg['filename'].lstrip('logging/')}-{eval_setting}-epoch{args.epoch}")

    # De-normalization
    norm_path = f"{eval_data_cfg['dataset_path']}_normalize.pkl"
    with open(norm_path, "rb") as f:
        norm_data = pkl.load(f)
    mean = norm_data["mean"].reshape(1, -1)  # (1, target_dim)
    std = norm_data["std"].reshape(1, -1)  # (1, target_dim)

    # Non-overlap rolling forecasting stage
    seasonality = get_seasonality(dataset.freq)
    # res = evaluate_model(
    #     predictor,
    #     test_data=dataset.test_data,
    #     metrics=metrics,
    #     batch_size=eval_data_cfg["batch_size"],
    #     axis=None,
    #     mask_invalid_label=True,
    #     allow_nan_forecast=False,
    #     seasonality=seasonality,
    # )
    # print(f"Results for {eval_data_cfg['dataset_name']}: {res}")

    forecasts = predictor.predict(dataset.test_data.input)
    forecast_batches = batcher(forecasts, batch_size=eval_data_cfg["batch_size"])
    input_batches = batcher(dataset.test_data.input, batch_size=eval_data_cfg["batch_size"])
    label_batches = batcher(dataset.test_data.label, batch_size=eval_data_cfg["batch_size"])
    forecasts_batches, prediction_batches, target_batches = [], [], []
    test_id = 0
    for input_batch, label_batch, forecast_batch in zip(input_batches, label_batches, forecast_batches):
        if eval_data_cfg["mode"] == "univariate":
            inp = input_batch[0]["target"][:, None]  # (dataset_length, 1)
            target = label_batch[0]["target"][:, None]  # (prediction_length, 1)
            forecasts = forecast_batch[0].samples[..., None]  # (num_samples, prediction_length, 1)
            prediction = forecast_batch[0].quantile(0.5)[:, None]  # (prediction_length, 1)
            context = inp[-eval_data_cfg["context_length"]:, ...]  # (context_length, 1)
        else:
            inp = input_batch[0]["target"].T  # (dataset_length, target_dim)
            target = label_batch[0]["target"].T  # (prediction_length, target_dim)
            forecasts = forecast_batch[0].samples  # (num_samples, prediction_length, target_dim)
            prediction = forecast_batch[0].quantile(0.5)  # (prediction_length, target_dim)
            context = inp[-eval_data_cfg["context_length"]:, ...]  # (context_length, target_dim)

        target_real = target * (std + 1e-8) + mean
        forecasts_real = forecasts * (std[None, ...] + 1e-8) + mean[None, ...]
        prediction_real = prediction * (std + 1e-8) + mean
        context_real = context * (std + 1e-8) + mean
        target_batches.append(target_real)
        forecasts_batches.append(forecasts_real)
        prediction_batches.append(prediction_real)

        # Calculate evaluation metrics
        for metric_name, metric_class in metrics.items():
            metric_value = metric_class(forecasts=forecasts_real, target=target_real,
                                        past=context_real, seasonality=seasonality)
            eval_res[metric_name].append(metric_value)
        test_sample_res = {metric: value[-1] for metric, value in eval_res.items()}

        logger.info(f"[Test sample {test_id}/{dataset.test_data.windows}]: seasonality={seasonality}, "
                    f"metrics={test_sample_res}")
        test_id += 1

    # Remove anomalous test samples
    if "loop_seattle_5T" in eval_data_cfg["dataset_name"] and "transfer" not in eval_data_cfg["dataset_name"]:
        for window_id in [9]:
            for metric, value in eval_res.items():
                value.pop(window_id)
    if "loop_seattle_5T_transfer" in eval_data_cfg["dataset_name"]:
        for window_id in [12, 9]:
            for metric, value in eval_res.items():
                value.pop(window_id)

    eval_res_mean = {metric: float(np.mean(value)) for metric, value in eval_res.items()}
    logger.info(f"[Epoch {args.epoch}] Evaluation results: {eval_res_mean}")

    # Visualize forecasting results
    target_batches = np.concatenate(target_batches, axis=0)  # (prediction_length*test_windows, target_dim)
    forecasts_batches = np.concatenate(forecasts_batches, axis=1)  # (num_samples, prediction_length*test_windows, target_dim)
    prediction_batches = np.concatenate(prediction_batches, axis=0)  # (prediction_length*test_windows, target_dim)
    img_storage_path = f"figures/{eval_data_cfg['dataset_name']}-moirai-moe-1.0-R-{model_scale}/{eval_setting}"
    if not os.path.exists(img_storage_path):
        os.makedirs(img_storage_path)
    plot_forecasts_res(target_batches, forecasts_batches, prediction_batches, img_storage_path,
                       eval_data_cfg["prediction_length"], eval_data_cfg["test_length"])


if __name__ == "__main__":
    # Input args
    parser = argparse.ArgumentParser(description="Moirai-MoE")
    parser.add_argument("--eval_stage", type=str, help="options=[pretrain, finetune]")

    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--sft_path", type=str)
    parser.add_argument("--rft_path", type=str)
    parser.add_argument("--epoch", type=int, default=0)

    args = parser.parse_args()
    print(args)

    for ratio in [2, 4, 16, 24]:
        # Load experiment configurations
        # cfg_path = f"configs/loop_seattle_5T_{ratio}%_moirai_moe_1.0_R_small.yaml"
        # cfg_path = f"configs/loop_seattle_5T_transfer_{ratio}%_moirai_moe_1.0_R_small.yaml"
        cfg_path = f"configs/ett_15T_{20}%_moirai_moe_1.0_R_small.yaml"
        # cfg_path = f"configs/ett_15T_transfer_{ratio}%_moirai_moe_1.0_R_base.yaml"
        # cfg_path = f"configs/boomlet_963_T_{ratio}%_moirai_moe_1.0_R_small.yaml"
        # cfg_path = f"configs/jena_weather_10T_{ratio}%_moirai_moe_1.0_R_small.yaml"
        # cfg_path = f"configs/ercot_1H_{ratio}%_moirai_moe_1.0_R_small.yaml"
        with open(cfg_path, "r") as f:
            ts_configs = yaml.load(f, Loader=yaml.FullLoader)

        # Test pretrained model
        fix_seed(seed=2025)  # Fix random seed for reproduction
        args.eval_stage = "pretrain"
        args.pretrained_path = "checkpoints/pretrain/moirai-moe-1.0-R-small.safetensors"
        # main(cfg=ts_configs, stage="pretrain", model_scale="small")
        logging.getLogger().handlers.clear()

        # Test SFT method
        args.eval_stage = "finetune"
        sft_configs = copy.deepcopy(ts_configs)
        sft_configs["rl_trainer"]["data_selection"] = False
        for epoch in list(range(25, 50, 1)) + list(range(75, 100, 1)):
            fix_seed(seed=2025)  # Fix random seed for reproduction
            args.epoch = epoch
            # args.sft_path = f"checkpoints/finetune/sft/loop_seattle_5T_{ratio}%-moirai-moe-1.0-R-small/cl1728-pl288/epoch{epoch}.safetensors"
            args.sft_path = f"checkpoints/finetune/sft/ett_15T_{ratio}%-moirai-moe-1.0-R-small/cl1056-pl96/epoch{epoch}.safetensors"
            # args.sft_path = f"checkpoints/finetune/sft/boomlet_963_T_{ratio}%-moirai-moe-1.0-R-small/cl1020-pl60/epoch{epoch}.safetensors"
            # args.sft_path = f"checkpoints/finetune/sft/jena_weather_10T_{ratio}%-moirai-moe-1.0-R-small/cl1008-pl144/epoch{epoch}.safetensors"
            # args.sft_path = f"checkpoints/finetune/sft/ercot_1H_{ratio}%-moirai-moe-1.0-R-small/cl1680-pl168/epoch{epoch}.safetensors"
            # main(cfg=sft_configs, stage="sft", model_scale="small")
            logging.getLogger().handlers.clear()

        # Test native GPRO method
        args.eval_stage = "finetune"
        rft_configs = copy.deepcopy(ts_configs)
        rft_configs["rl_trainer"]["beta"] = 0.001
        rft_configs["rl_trainer"]["num_generations"] = ratio
        rft_configs["rl_trainer"]["data_selection"] = False
        rft_configs["rl_trainer"]["reward_shaping"] = False
        rft_configs["rl_trainer"]["lambda_acc_reward"] = 1.0
        rft_configs["rl_trainer"]["lambda_var_reward"] = 0.0
        rft_configs["rl_trainer"]["lambda_var_synergy"] = 0.0
        rft_configs["rl_trainer"]["lambda_freq_synergy"] = 0.0
        for epoch in range(25, 50, 1):
            fix_seed(seed=2025)  # Fix random seed for reproduction
            args.epoch = epoch
            # args.rft_path = f"checkpoints/finetune/rft/loop_seattle_5T_{ratio}%-moirai-moe-1.0-R-small/cl1728-pl288-ng8-kl0.001-ds0-rs0-lar1.0-lvr0.0-lvs0.0-lfs0.0/epoch{epoch}.safetensors"
            args.rft_path = f"checkpoints/finetune/rft/ett_15T_{20}%-moirai-moe-1.0-R-small/cl1056-pl96-ng{ratio}-kl0.001-ds0-rs0-lar1.0-lvr0.0-lvs0.0-lfs0.0/epoch{epoch}.safetensors"
            # args.rft_path = f"checkpoints/finetune/rft/boomlet_963_T_{ratio}%-moirai-moe-1.0-R-small/cl1020-pl60-ng8-kl0.001-ds0-rs0-lar1.0-lvr0.0-lvs0.0-lfs0.0/epoch{epoch}.safetensors"
            # args.rft_path = f"checkpoints/finetune/rft/jena_weather_10T_{ratio}%-moirai-moe-1.0-R-small/cl1008-pl144-ng8-kl0.001-ds0-rs0-lar1.0-lvr0.0-lvs0.0-lfs0.0/epoch{epoch}.safetensors"
            # args.rft_path = f"checkpoints/finetune/rft/ercot_1H_{ratio}%-moirai-moe-1.0-R-small/cl1680-pl168-ng8-kl0.001-ds0-rs0-lar1.0-lvr0.0-lvs0.0-lfs0.0/epoch{epoch}.safetensors"
            main(cfg=rft_configs, stage="rft", model_scale="small")
            logging.getLogger().handlers.clear()

        # Test native GPRO method
        args.eval_stage = "finetune"
        rft_configs = copy.deepcopy(ts_configs)
        rft_configs["rl_trainer"]["beta"] = 0.001
        rft_configs["rl_trainer"]["num_generations"] = ratio
        rft_configs["rl_trainer"]["data_selection"] = True
        rft_configs["rl_trainer"]["reward_shaping"] = True
        rft_configs["rl_trainer"]["lambda_acc_reward"] = 0.9
        rft_configs["rl_trainer"]["lambda_var_reward"] = 0.1
        rft_configs["rl_trainer"]["lambda_var_synergy"] = 0.01
        rft_configs["rl_trainer"]["lambda_freq_synergy"] = 0.01
        for epoch in range(25, 50, 1):
            fix_seed(seed=2025)  # Fix random seed for reproduction
            args.epoch = epoch
            # args.rft_path = f"checkpoints/finetune/rft/loop_seattle_5T_{ratio}%-moirai-moe-1.0-R-small/cl1728-pl288-ng8-kl0.001-ds1-rs1-lar0.9-lvr0.0-lvs0.01-lfs0.01/epoch{epoch}.safetensors"
            args.rft_path = f"checkpoints/finetune/rft/ett_15T_{20}%-moirai-moe-1.0-R-small/cl1056-pl96-ng{ratio}-kl0.001-ds1-rs1-lar0.9-lvr0.1-lvs0.01-lfs0.01/epoch{epoch}.safetensors"
            # args.rft_path = f"checkpoints/finetune/rft/boomlet_963_T_{ratio}%-moirai-moe-1.0-R-small/cl1020-pl60-ng8-kl0.001-ds0-rs0-lar1.0-lvr0.0-lvs0.0-lfs0.0/epoch{epoch}.safetensors"
            # args.rft_path = f"checkpoints/finetune/rft/jena_weather_10T_{ratio}%-moirai-moe-1.0-R-small/cl1008-pl144-ng8-kl0.001-ds0-rs0-lar1.0-lvr0.0-lvs0.0-lfs0.0/epoch{epoch}.safetensors"
            # args.rft_path = f"checkpoints/finetune/rft/ercot_1H_{ratio}%-moirai-moe-1.0-R-small/cl1680-pl168-ng8-kl0.001-ds0-rs0-lar1.0-lvr0.0-lvs0.0-lfs0.0/epoch{epoch}.safetensors"
            main(cfg=rft_configs, stage="rft", model_scale="small")
            logging.getLogger().handlers.clear()

        # Test TimeRFT method
        args.eval_stage = "finetune"
        ts_configs["rl_trainer"]["beta"] = 0.001
        for epoch in range(25, 50, 1):
            fix_seed(seed=2025)  # Fix random seed for reproduction
            args.epoch = epoch
            # args.rft_path = f"checkpoints/finetune/rft/loop_seattle_5T_{ratio}%-moirai-moe-1.0-R-small/cl1728-pl288-ng8-kl0.001-ds1-rs1-lar0.9-lvr0.1-lvs0.01-lfs0.01/epoch{epoch}.safetensors"
            args.rft_path = f"checkpoints/finetune/rft/ett_15T_{ratio}%-moirai-moe-1.0-R-small/cl1056-pl96-ng8-kl0.001-ds1-rs1-lar0.9-lvr0.1-lvs0.01-lfs0.01/epoch{epoch}.safetensors"
            # args.rft_path = f"checkpoints/finetune/rft/boomlet_963_T_{ratio}%-moirai-moe-1.0-R-small/cl1020-pl60-ng8-kl0.001-ds1-rs1-lar0.9-lvr0.1-lvs0.01-lfs0.01/epoch{epoch}.safetensors"
            # args.rft_path = f"checkpoints/finetune/rft/jena_weather_10T_{ratio}%-moirai-moe-1.0-R-small/cl1008-pl144-ng8-kl0.001-ds1-rs1-lar0.9-lvr0.1-lvs0.01-lfs0.01/epoch{epoch}.safetensors"
            # args.rft_path = f"checkpoints/finetune/rft/ercot_1H_{ratio}%-moirai-moe-1.0-R-small/cl1680-pl168-ng8-kl0.001-ds1-rs1-lar0.9-lvr0.1-lvs0.01-lfs0.01/epoch{epoch}.safetensors"
            # main(cfg=ts_configs, stage="rft", model_scale="small")
            logging.getLogger().handlers.clear()