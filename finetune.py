import numpy as np
import yaml
import os
import pickle as pkl
import torch
import torch.nn as nn
from torch.distributions import Distribution
from einops import rearrange
from safetensors.torch import load_file, save_file
import time
import logging
import copy
from typing import Tuple

from model.ts_moe import MoiraiMoEModule
from model.ts_distribution.mixture import MixtureOutput
from model.ts_distribution.distributions import (
    StudentTOutput,
    NormalFixedScaleOutput,
    NegativeBinomialOutput,
    LogNormalOutput,
)
from model.loss import PackedNLLLoss
from model.optimizer import configure_optimizers

from rlvr.grpo import GRPOTrainer
from rlvr.data_filter import training_data_selection

from eval.metrics import MSE, MAE, CRPS, MQL

from data.ts_transform.mapping import finetune_transform_map, val_transform_map
from data.loader import DataLoader
from data_converter import generate_finetune_builder, generate_eval_builder
from utils import move_dict_to_device, fix_seed


def forward(
    model: nn.Module,
    target: torch.FloatTensor,  # (batch_size, seq_len, max_patch)
    observed_mask: torch.BoolTensor,  # (batch_size, seq_len, max_patch)
    sample_id: torch.IntTensor,  # (batch_size, seq_len)
    time_id: torch.IntTensor,  # (batch_size, seq_len)
    variate_id: torch.IntTensor,  # (batch_size, seq_len)
    prediction_mask: torch.BoolTensor,  # (batch_size, seq_len)
    patch_size: torch.IntTensor,  # (batch_size, seq_len)
) -> Distribution:
    distr = model(
        target=target,
        observed_mask=observed_mask,
        sample_id=sample_id,
        time_id=time_id,
        variate_id=variate_id,
        prediction_mask=prediction_mask,
        patch_size=patch_size,
    )
    return distr


def sft_training_step(batch: dict[str, torch.Tensor], model, loss_func, seq_fields) -> torch.Tensor:
    distr = forward(model,
                    **{field: batch[field] for field in list(seq_fields) + ["sample_id"]}
    )

    """
    Since the training objective of decoder-only Moirai-MoE is different from that of encoder-only Moirai,
    the sequence offset by one token is needed to satisfy standard next-token prediction.
    """
    if "MoE" in model.__class__.__name__:
        target_raw = batch["target"]  # (batch_size, seq_len, patch_size)
        pred_mask_raw = batch["prediction_mask"]  # (batch_size, seq_len)
        seq_len = target_raw.shape[1]
        num_var = batch["variate_id"].unique().shape[-1]
        num_patch_per_var = seq_len // num_var
        target_offset = torch.zeros_like(target_raw).to(target_raw.device)
        pred_mask_offset = torch.zeros_like(pred_mask_raw).to(pred_mask_raw.device)
        for i in range(num_var):
            target_per_var = target_raw[..., i*num_patch_per_var:(i+1)*num_patch_per_var, :]
            target_offset[..., i*num_patch_per_var:(i+1)*num_patch_per_var-1, :] = target_per_var[..., 1:, :]
            mask_per_var = pred_mask_raw[..., i*num_patch_per_var:(i+1)*num_patch_per_var]
            pred_mask_offset[..., i*num_patch_per_var:(i+1)*num_patch_per_var-1] = mask_per_var[..., 1:]
        batch["target"] = target_offset
        batch["prediction_mask"] = pred_mask_offset

    loss = loss_func(
        pred=distr, **{field: batch[field] for field in
                       ["target", "prediction_mask", "observed_mask", "sample_id", "variate_id"]},
    )
    return loss


def batch_data_selection(batch_data, filter_indices):
    data_indices = batch_data["window"]
    select_mask = ~torch.isin(data_indices, filter_indices)
    batch_data_selected = {field: data[select_mask] for field, data in batch_data.items()}

    return batch_data_selected


def main(cfg, stage, model_scale="small"):
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    )

    # Load device
    sft_trainer_cfg = cfg["sft_trainer"]
    if sft_trainer_cfg["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Determine SFT or RFT settings
    rl_trainer_cfg = cfg["rl_trainer"]
    if stage == "sft":
        num_epochs = sft_trainer_cfg["max_epochs"]
        accumulate_grad_steps = sft_trainer_cfg["accumulate_grad_steps"]
        checkpoint = load_file(sft_trainer_cfg["pretrained_path"])
        is_valid = sft_trainer_cfg["is_valid"]
        init_lr = float(sft_trainer_cfg["init_lr"])
        T_max = int(sft_trainer_cfg["T_max"])
        eta_min = float(sft_trainer_cfg["eta_min"])
        train_length = int(sft_trainer_cfg["train_length"])
        offset = int(sft_trainer_cfg["offset"])
    elif stage == "rft":
        num_epochs = rl_trainer_cfg["max_epochs"]
        accumulate_grad_steps = rl_trainer_cfg["accumulate_grad_steps"]
        checkpoint = load_file(rl_trainer_cfg["pretrained_path"])
        is_valid = rl_trainer_cfg["is_valid"]
        init_lr = float(rl_trainer_cfg["init_lr"])
        T_max = int(rl_trainer_cfg["T_max"])
        eta_min = float(rl_trainer_cfg["eta_min"])
        train_length = int(rl_trainer_cfg["train_length"])
        offset = int(rl_trainer_cfg["offset"])
    else:
        raise NotImplementedError

    # For fine-tuning w/o using sequence packing, create 'sample_id' for each sample by transformation.
    if "collate_fn" not in cfg["train_dataloader"]:
        seq_fields = seq_fields + ("sample_id",)

    # Using window to index training samples
    if rl_trainer_cfg["data_selection"]:
        seq_fields = seq_fields + ("window", )

    # Build training dataset
    train_data_cfg = cfg["train_dataset"]
    train_transform_map = finetune_transform_map(
        hparams=model_cfg,
        offset=offset,
        distance=train_data_cfg["distance"],
        prediction_length=train_data_cfg["prediction_length"],
        context_length=train_data_cfg["context_length"],
        patch_size=train_data_cfg["patch_size"],
        seq_fields=seq_fields)
    train_dataset = generate_finetune_builder(
        dataset=train_data_cfg["dataset"],
        offset=offset,
        train_length=train_length,
        prediction_length=train_data_cfg["prediction_length"],
        context_length=train_data_cfg["context_length"],
        patch_size=train_data_cfg["patch_size"],
        mode=train_data_cfg["mode"],
        storage_path=train_data_cfg["storage_path"],
        distance=train_data_cfg["distance"]
    ).load_dataset(train_transform_map)

    # Build validation dataset
    val_data_cfg = cfg["val_dataset"]
    val_transform_obj = val_transform_map(
        hparams=model_cfg,
        offset=val_data_cfg["offset"],
        distance=val_data_cfg["distance"],
        prediction_length=val_data_cfg["prediction_length"],
        context_length=val_data_cfg["context_length"],
        patch_size=val_data_cfg["patch_size"],
        seq_fields=seq_fields)
    val_dataset = generate_eval_builder(
        dataset=val_data_cfg["dataset"],
        offset=val_data_cfg["offset"],
        eval_length=val_data_cfg["val_length"],
        prediction_length=val_data_cfg["prediction_length"],
        context_length=val_data_cfg["context_length"],
        patch_size=val_data_cfg["patch_size"],
        mode=val_data_cfg["mode"],
        storage_path=val_data_cfg["storage_path"],
        distance=val_data_cfg["distance"]
    ).load_dataset(val_transform_obj)

    # Build training dataloader
    train_loader_cfg = cfg["train_dataloader"]
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_loader_cfg["batch_size"] // accumulate_grad_steps,
        batch_size_factor=train_loader_cfg["batch_size_factor"],
        cycle=train_loader_cfg["cycle"],
        num_batches_per_epoch=train_loader_cfg["num_batches_per_epoch"],
        shuffle=train_loader_cfg["shuffle"],
        num_workers=train_loader_cfg["num_workers"],
        pin_memory=train_loader_cfg["pin_memory"],
        drop_last=train_loader_cfg["drop_last"],
        fill_last=train_loader_cfg["fill_last"],
        worker_init_fn=train_loader_cfg["worker_init_fn"],
        prefetch_factor=train_loader_cfg["prefetch_factor"],
        persistent_workers=train_loader_cfg["persistent_workers"]
    )

    # Build filter dataloader
    filter_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1,  # must be 1
        batch_size_factor=1.0,
        cycle=train_loader_cfg["cycle"],
        num_batches_per_epoch=train_loader_cfg["num_batches_per_epoch"],
        shuffle=False,
        num_workers=train_loader_cfg["num_workers"],
        pin_memory=train_loader_cfg["pin_memory"],
        drop_last=train_loader_cfg["drop_last"],
        fill_last=train_loader_cfg["fill_last"],
        worker_init_fn=train_loader_cfg["worker_init_fn"],
        prefetch_factor=train_loader_cfg["prefetch_factor"],
        persistent_workers=train_loader_cfg["persistent_workers"]
    )

    # Build validation dataloader
    val_loader_cfg = cfg["val_dataloader"]
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,  # must be 1
        batch_size_factor=1.0,
        cycle=val_loader_cfg["cycle"],
        num_batches_per_epoch=val_loader_cfg["num_batches_per_epoch"],
        shuffle=val_loader_cfg["shuffle"],
        num_workers=val_loader_cfg["num_workers"],
        pin_memory=val_loader_cfg["pin_memory"],
        drop_last=val_loader_cfg["drop_last"],
        fill_last=val_loader_cfg["fill_last"],
        worker_init_fn=val_loader_cfg["worker_init_fn"],
        prefetch_factor=val_loader_cfg["prefetch_factor"],
        persistent_workers=val_loader_cfg["persistent_workers"]
    )

    # Configure optimizer and scheduler
    optim_cfg = cfg["optimizer"]
    optim_cfg["init_lr"] = init_lr
    optim_cfg["T_max"] = T_max
    optim_cfg["eta_min"] = eta_min
    optim_dict = configure_optimizers(model=ts_model, hparams=optim_cfg)
    optimizer = optim_dict["optimizer"]
    lr_scheduler = optim_dict["lr_scheduler"]["scheduler"]

    # Instantiate loss function
    ts_loss_func = PackedNLLLoss()

    # Instantiate SFT or RFT trainer
    ts_model.load_state_dict(checkpoint)
    grpo_trainer = GRPOTrainer(
        model=ts_model,
        seq_fields=seq_fields[0:-1] if rl_trainer_cfg["data_selection"] else seq_fields,  # remove window
        num_generations=rl_trainer_cfg["num_generations"],
        beta=rl_trainer_cfg["beta"],
        context_length=train_data_cfg["context_length"],
        prediction_length=train_data_cfg["prediction_length"],
        target_dim=train_data_cfg["target_dim"],
        patch_size=train_data_cfg["patch_size"],
        reward_shaping=rl_trainer_cfg["reward_shaping"],
        lambda_acc_reward=rl_trainer_cfg["lambda_acc_reward"],
        lambda_var_reward=rl_trainer_cfg["lambda_var_reward"],
        lambda_var_synergy=rl_trainer_cfg["lambda_var_synergy"],
        lambda_freq_synergy=rl_trainer_cfg["lambda_freq_synergy"],
    )

    # Mark experiment settings
    if stage == "rft":
        ft_setting = f"finetune-{stage}-cl{train_data_cfg['context_length']}-pl{train_data_cfg['prediction_length']}-" + \
                     f"ng{int(rl_trainer_cfg['num_generations'])}-kl{rl_trainer_cfg['beta']}-" + \
                     f"ds{int(rl_trainer_cfg['data_selection'])}-rs{int(rl_trainer_cfg['reward_shaping'])}-" + \
                     f"lar{rl_trainer_cfg['lambda_acc_reward']}-lvr{rl_trainer_cfg['lambda_var_reward']}-" + \
                     f"lvs{rl_trainer_cfg['lambda_var_synergy']}-lfs{rl_trainer_cfg['lambda_freq_synergy']}"
    else:
        ft_setting = f"finetune-{stage}-cl{train_data_cfg['context_length']}-pl{train_data_cfg['prediction_length']}"

    # Configure logging
    log_cfg = cfg["logger"]
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(f"{log_cfg['filename']}-{ft_setting}.log", mode='w')
                        ])
    logger = logging.getLogger(__name__)
    logger.info(f"Start training stage with setting: {log_cfg['filename'].lstrip('logging/')}-{ft_setting}")

    # Offline data selection
    if rl_trainer_cfg["data_selection"]:
        ts_model.eval()
        retain_indices, filter_indices = training_data_selection(ts_model, grpo_trainer, filter_dataloader,
            cfg["eval_dataset"]["num_samples"], train_data_cfg["prediction_length"], train_data_cfg["target_dim"], device)
        logger.info(f"Retain samples: {retain_indices}, Filter samples: {filter_indices}")

    # Training loop
    num_batches_per_epoch = len(train_dataloader.dataloader)
    num_batches_val = len(val_dataloader.dataloader)
    for epoch in range(num_epochs):
        ts_model.train()
        for batch_idx, batch_data in enumerate(train_dataloader):
            start_time = time.time()
            batch_data = move_dict_to_device(batch_data, device)
            batch_size_raw = batch_data["target"].shape[0]
            batch_size_select = batch_size_raw

            if stage == "sft":
                loss = sft_training_step(batch_data, ts_model, ts_loss_func, seq_fields[0:-1])
                reward, kl = None, None
            elif stage == "rft":
                if rl_trainer_cfg["data_selection"]:
                    batch_data = batch_data_selection(batch_data, filter_indices)
                    batch_size_select = batch_data["target"].shape[0]
                    if batch_size_select == 0:  # bypass void batch
                        continue

                # RFT-zero without SFT warmup
                loss = grpo_trainer.compute_loss(ts_model, batch_data, epoch)
                reward = grpo_trainer.metrics['reward'][-1]
                kl = grpo_trainer.metrics['kl'][-1]
            else:
                raise NotImplementedError

            # Consider effect of data filtering on batch size, default: loss = loss / accumulate_grad_steps
            loss = (loss * batch_size_select) / (batch_size_raw * accumulate_grad_steps)
            loss.backward()  # accumulating gradients by default

            # gradient accumulation
            if ((batch_idx+1) % accumulate_grad_steps == 0) or ((batch_idx+1) == num_batches_per_epoch):
                optimizer.step()
                optimizer.zero_grad()
                end_time = time.time()

                logger.info(f"[epoch {epoch}/{num_epochs} | batch {batch_idx}/{num_batches_per_epoch}]: "
                            f"loss={(loss.item())*accumulate_grad_steps}, reward={reward}, kl={kl}, "
                            f"batch_time={end_time-start_time}s")
        lr_scheduler.step()  # epoch-wise lr scheduling
        logger.info(f"scheduler state: {lr_scheduler.state_dict()}")

        # Validation set rewards
        if is_valid:
            # Instantiate validation metrics
            mse = MSE(forecast_type="mean")
            mae = MAE(forecast_type="mean")
            crps = CRPS()
            mql = MQL(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            val_mse, val_mae, val_crps, val_mql = [], [], [], []

            # Series de-normalization
            norm_path = f"{cfg['eval_dataset']['dataset_path']}_normalize.pkl"
            with open(norm_path, "rb") as f:
                norm_data = pkl.load(f)
            mean = norm_data["mean"].reshape(1, -1)  # (1, num_var)
            std = norm_data["std"].reshape(1, -1)  # (1, num_var)

            ts_model.eval()
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(val_dataloader):
                    batch_data = move_dict_to_device(batch_data, device)
                    num_generations = cfg["eval_dataset"]["num_samples"]
                    forecasts = grpo_trainer.generate_forecasts(ts_model, batch_data, num_generations)

                    target = batch_data["target"]  # (batch_size, seq_len, patch_size)
                    batch_size = target.shape[0]
                    num_var = train_data_cfg["target_dim"]
                    prediction_length = train_data_cfg["prediction_length"]
                    context_length = train_data_cfg["context_length"]
                    forecasts = rearrange(forecasts, '(b n) (c l) p -> b n (l p) c', b=batch_size, c=num_var)[0]
                    target = rearrange(target, 'b (c l) p -> b (l p) c', c=num_var)[0]
                    forecasts = forecasts.detach().cpu().numpy()
                    target_raw = target.detach().cpu().numpy()
                    forecasts = forecasts[:, -prediction_length:, :]
                    target = target_raw[-prediction_length:, :]
                    context = target_raw[-(context_length+prediction_length):-prediction_length, :]
                    target_real = target * (std + 1e-8) + mean
                    forecasts_real = forecasts * (std[None, ...] + 1e-8) + mean[None, ...]
                    context_real = context * (std + 1e-8) + mean
                    mse_value = mse(forecasts=forecasts_real, target=target_real)
                    mae_value = mae(forecasts=forecasts_real, target=target_real)
                    crps_value = crps(forecasts=forecasts_real, target=target_real)
                    mql_value = mql(forecasts=forecasts_real, target=target_real)
                    val_mse.append(mse_value)
                    val_mae.append(mae_value)
                    val_crps.append(crps_value)
                    val_mql.append(mql_value)

            logger.info(
                f"[epoch {epoch}/{num_epochs}]: val_MSE={np.mean(val_mse), val_mse}")
            logger.info(
                f"[epoch {epoch}/{num_epochs}]: val_MAE={np.mean(val_mae), val_mae}")
            logger.info(
                f"[epoch {epoch}/{num_epochs}]: val_CRPS={np.mean(val_crps), val_crps}")
            logger.info(
                f"[epoch {epoch}/{num_epochs}]: val_MQL={np.mean(val_mql), val_mql}")

        # Save checkpoint
        ft_setting_new = ft_setting.lstrip(f"finetune-{stage}-")
        ckpt_path = f"checkpoints/finetune/{stage}/{train_data_cfg['dataset']}-moirai-moe-1.0-R-{model_scale}/{ft_setting_new}"
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        save_file(ts_model.state_dict(), f"{ckpt_path}/epoch{epoch}.safetensors")


if __name__ == "__main__":
    # Load experiment configurations
    cfg_path = "configs/ett_15T_100%_moirai_moe_1.0_R_small.yaml"
    with open(cfg_path, "r") as f:
        ts_configs = yaml.load(f, Loader=yaml.FullLoader)

    # Supervised finetuning
    fix_seed(seed=2025)  # Fix random seed for reproduction
    sft_configs = copy.deepcopy(ts_configs)
    sft_configs["rl_trainer"]["data_selection"] = False
    main(cfg=sft_configs, stage="sft", model_scale="small")
    logging.getLogger().handlers.clear()

    # RL finetuning by TimeRFT
    fix_seed(seed=2025)  # Fix random seed for reproduction
    ts_configs["rl_trainer"]["beta"] = 0.001
    main(cfg=ts_configs, stage="rft", model_scale="small")
    logging.getLogger().handlers.clear()