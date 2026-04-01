import math
from functools import partial

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from model.ts_transformer.norm import RMSNorm
from model.ts_transformer.position import BinaryAttentionBias
from model.ts_embed import FeatLinear, MultiInSizeLinear, MultiOutSizeLinear


def _get_constant_lambda(_=None):
    return 1


def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1):
    return LambdaLR(optimizer, _get_constant_lambda, last_epoch=last_epoch)


def get_cosine_schedule(optimizer: Optimizer, T_max: int, eta_min: float):
    return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    if progress >= 1.0:
        return 0.0
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))
    )


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    "cosine_with_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "constant": get_constant_schedule,
    "cosine": get_cosine_schedule,
}


def configure_optimizers(model, hparams) -> dict:
    """
    :return: dictionary of optimizers and learning rate schedulers
    """
    decay = set()
    no_decay = set()

    if hparams["finetune_pattern"] == "full":
        pass
    elif hparams["finetune_pattern"] == "freeze_ffn":
        for pn, p in model.named_parameters():
            if "ffn" in pn:
                p.requires_grad = False
    elif hparams["finetune_pattern"] == "head_only":
        for pn, p in model.named_parameters():
            if "param_proj" not in pn:
                p.requires_grad = False
    else:
        raise ValueError(
            "Unsupported finetune pattern {}".format(hparams["finetune_pattern"])
        )

    whitelist_params = (
        FeatLinear,
        MultiInSizeLinear,
        MultiOutSizeLinear,
        nn.Linear,
    )
    blacklist_params = (
        BinaryAttentionBias,
        RMSNorm,
        nn.Embedding,
        nn.LayerNorm,
    )

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if not p.requires_grad:
                continue

            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_params):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_params):
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
            len(inter_params) == 0
    ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert (
            len(param_dict.keys() - union_params) == 0
    ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    optim_groups = [
        {
            "params": filter(
                lambda p: p.requires_grad,
                [param_dict[pn] for pn in sorted(list(decay))],
            ),
            "weight_decay": hparams["weight_decay"],
        },
        {
            "params": filter(
                lambda p: p.requires_grad,
                [param_dict[pn] for pn in sorted(list(no_decay))],
            ),
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=hparams["init_lr"],
        betas=(hparams["beta1"], hparams["beta2"]),
        eps=1e-6,
    )

    scheduler_name = hparams["scheduler_type"]
    scheduler_func = TYPE_TO_SCHEDULER_FUNCTION[scheduler_name]
    if scheduler_name == "constant":
        scheduler = scheduler_func(optimizer)
    elif scheduler_name == "cosine":
        scheduler = scheduler_func(optimizer, T_max=hparams["T_max"], eta_min=hparams["eta_min"])
    else:
        scheduler = scheduler_func(
            optimizer,
            num_warmup_steps=hparams["num_warmup_steps"],
            num_training_steps=hparams["num_training_steps"],
        )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "train_loss",
            "interval": "step",  # update the learning rate every training step instead of epoch
        },
    }