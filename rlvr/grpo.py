import torch
import torch.nn as nn
import math
from einops import rearrange
from copy import deepcopy
from collections import defaultdict
from functools import partial

from rlvr.reward import sequence_accuracy_reward, sequence_frequency_reward, sequence_structure_reward, \
    patch_accuracy_reward, patch_frequency_reward, patch_structure_reward


class GRPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        seq_fields: tuple[str, ...],
        num_generations: int,
        beta: float,
        context_length: int,
        prediction_length: int,
        target_dim: int,
        patch_size: int,
        reward_shaping: bool,
        lambda_acc_reward: float,
        lambda_var_reward: float,
        lambda_var_synergy: float,
        lambda_freq_synergy: float,
    ):
        super().__init__()
        self.seq_fields = seq_fields
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.patch_size = patch_size

        self.num_generations = num_generations  # number of group samples
        self.beta = beta  # weight of KL regularization
        self.ref_model = self.create_reference_model(model)

        self.reward_shaping = reward_shaping
        self.lambda_acc_reward = lambda_acc_reward
        self.lambda_var_reward = lambda_var_reward
        self.lambda_var_synergy = lambda_var_synergy
        self.lambda_freq_synergy = lambda_freq_synergy

        self.metrics = defaultdict(list)

    def create_reference_model(self, model):
        ref_model = deepcopy(model)
        return ref_model.eval()

    def generate_forecasts(self, model, batch, num_generations):
        distr = model(**{field: batch[field] for field in list(self.seq_fields)})  # patch-wise conditional distribution p_\theta(patch_i|patch_{1:i-1})
        preds = distr.sample(torch.Size((num_generations,)))  # (num_generations, batch_size, seq_len, patch_size)
        # entropy = distr.entropy()  # Not implemented in Mixture Distribution

        predict_step = math.ceil(self.prediction_length / self.patch_size)
        context_step = math.ceil(self.context_length / self.patch_size)
        predict_token = predict_step * self.target_dim
        context_token = context_step * self.target_dim

        pred_index = torch.arange(
            start=context_step - 1,
            end=context_token + predict_step * (self.target_dim - 1),
            step=context_step + predict_step
        )
        assign_index = pred_index + 1

        expand_target = batch["target"].unsqueeze(0).repeat(num_generations, 1, 1, 1)  # zero-padding at the head with (context_token+predict_token-context_length-prediction_length) time steps
        expand_prediction_mask = batch["prediction_mask"].unsqueeze(0).repeat(num_generations, 1, 1)
        expand_observed_mask = batch["observed_mask"].unsqueeze(0).expand(num_generations, -1, -1, -1)
        expand_sample_id = batch["sample_id"].unsqueeze(0).expand(num_generations, -1, -1)
        expand_time_id = batch["time_id"].unsqueeze(0).expand(num_generations, -1, -1)
        expand_variate_id = batch["variate_id"].unsqueeze(0).expand(num_generations, -1, -1)
        expand_patch_size = batch["patch_size"].unsqueeze(0).expand(num_generations, -1, -1)

        expand_target[..., assign_index, :] = preds[..., pred_index, :]
        expand_prediction_mask[..., assign_index] = False

        remain_step = predict_step - 1
        while remain_step > 0:
            distr = model(
                target=expand_target,
                observed_mask=expand_observed_mask,
                sample_id=expand_sample_id,
                time_id=expand_time_id,
                variate_id=expand_variate_id,
                prediction_mask=expand_prediction_mask,
                patch_size=expand_patch_size
            )
            preds = distr.sample(torch.Size((1,)))
            _, _, batch_size, num_token, patch_size = preds.shape
            preds = preds.view(-1, batch_size, num_token, patch_size)  # (num_generations, batch_size, seq_len, patch_size)

            pred_index = assign_index
            assign_index = assign_index + 1
            expand_target[..., assign_index, :] = preds[..., pred_index, :]
            expand_prediction_mask[..., assign_index] = False

            remain_step -= 1

        forecast_samples = rearrange(expand_target, "n b l p -> (b n) l p")
        return forecast_samples  # (batch_size*num_generations, seq_len, patch_size)

    def get_per_token_logps(self, model, batch, forecasts):
        expand_prediction_mask = batch["prediction_mask"].repeat_interleave(self.num_generations, dim=0)
        expand_observed_mask = batch["observed_mask"].repeat_interleave(self.num_generations, dim=0)
        expand_sample_id = batch["sample_id"].repeat_interleave(self.num_generations, dim=0)
        expand_time_id = batch["time_id"].repeat_interleave(self.num_generations, dim=0)
        expand_variate_id = batch["variate_id"].repeat_interleave(self.num_generations, dim=0)
        expand_patch_size = batch["patch_size"].repeat_interleave(self.num_generations, dim=0)

        distr = model(
            target=forecasts,
            observed_mask=expand_observed_mask,
            sample_id=expand_sample_id,
            time_id=expand_time_id,
            variate_id=expand_variate_id,
            prediction_mask=expand_prediction_mask,
            patch_size=expand_patch_size
        )

        num_token = forecasts.shape[1]
        num_var = expand_variate_id.unique().shape[-1]
        num_token_per_var = math.ceil(num_token / num_var)
        forecasts_offset = torch.zeros_like(forecasts).to(forecasts.device)
        pred_mask_offset = torch.zeros_like(expand_prediction_mask).to(expand_prediction_mask.device)
        for i in range(num_var):
            forecasts_per_var = forecasts[..., i*num_token_per_var:(i+1)*num_token_per_var, :]
            forecasts_offset[..., i*num_token_per_var:(i+1)*num_token_per_var-1, :] = forecasts_per_var[..., 1:, :]
            mask_per_var = expand_prediction_mask[..., i*num_token_per_var:(i+1)*num_token_per_var]
            pred_mask_offset[..., i*num_token_per_var:(i+1)*num_token_per_var-1] = mask_per_var[..., 1:]

        pred_mask_offset = pred_mask_offset.unsqueeze(-1)
        per_token_logps = distr.log_prob(forecasts_offset) * pred_mask_offset  # (batch_size*num_generations, seq_len, patch_size)

        return per_token_logps, pred_mask_offset.squeeze(-1)

    def offset_rewards(self, rewards, pred_mask, num_var, num_token_per_var):
        rewards_offset = torch.zeros_like(rewards).to(rewards.device)  # (batch_size*num_generations, seq_len)
        for i in range(num_var):
            rewards_per_var = rewards[..., i*num_token_per_var:(i+1)*num_token_per_var]
            rewards_offset[..., i*num_token_per_var:(i+1)*num_token_per_var-1] = rewards_per_var[..., 1:]
        rewards_offset = rewards_offset * pred_mask  # mask rewards over context window

        return rewards_offset

    def shape_rewards(self, rewards, per_token_logps, unit, type, threshold=0.8, alpha=0.01):
        if type == "Nonlinear-shaping":
            rewards_shaped = torch.where(rewards >= 0.8, 0.8 + 0.01 * torch.log((rewards - 0.8) + 1),
                                         0.8 - (torch.exp(0.8 - rewards) - 1) / (torch.e - 1))

        elif type == "Piecewise-shaping":
            rewards_shaped = torch.where(rewards >= threshold, threshold + alpha * torch.log((rewards - threshold) + 1),
                                         rewards)

        elif type == "Likelihood-shaping":
            rewards = rearrange(rewards, "(b n) l -> n b l", n=self.num_generations)
            per_token_logps = rearrange(per_token_logps, "(b n) l p -> n b l p", n=self.num_generations)
            if unit == "patch":
                per_token_probs = torch.sum(per_token_logps, dim=-1)  # (num_generations, batch_size, seq_len)
            elif unit == "sequence":
                per_token_probs = torch.sum(per_token_logps, dim=[-1, -2]).unsqueeze(-1).repeat(1, 1, rewards.shape[-1])  # (num_generations, batch_size, seq_len)
            else:
                raise NotImplementedError
            sorted_rewards, sorted_indices = torch.sort(rewards, dim=0, descending=True)  # (num_generations, batch_size, seq_len)
            second_max_rewards, second_max_indices = sorted_rewards[1, :, :], sorted_indices[1, :, :]  # (batch_size, seq_len)
            second_max_probs = torch.gather(per_token_probs, dim=0, index=second_max_indices.unsqueeze(0)).squeeze(0)  # (batch_size, seq_len)
            max_rewards, max_probs = rewards[-1], per_token_probs[-1]  # (batch_size, seq_len)
            downscaling = torch.clamp(torch.exp(max_probs - second_max_probs), min=0.0, max=1.0)
            downscaling = torch.nan_to_num(downscaling, nan=1.0, posinf=1.0, neginf=1.0)  # inf computation can induce nan
            max_rewards_shaped = second_max_rewards + (max_rewards - second_max_rewards) * downscaling
            max_rewards_shaped = torch.clamp(max_rewards_shaped, min=0.0, max=1.0)
            rewards[-1] = max_rewards_shaped
            rewards_shaped = rearrange(rewards, "n b l -> (b n) l")

        else:
            raise NotImplementedError

        return rewards_shaped

    def compute_normalized_advantages(self, rewards, mean_grouped_rewards, std_grouped_rewards, pred_mask):
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages = advantages.view(rewards.shape[0], -1)  # (batch_size*num_generations, seq_len)
        advantages = advantages * pred_mask  # mask advantages over context window

        return advantages

    def compute_sequence_advantages(self, rewards, pred_mask, num_var):  # Corresponding to outcome supervision
        rewards = rearrange(rewards, "(b n) (c l) -> b n c l", n=self.num_generations, c=num_var)
        mean_grouped_rewards = rewards.mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = rewards.std(dim=1).repeat_interleave(self.num_generations, dim=0)
        rewards = rearrange(rewards, "b n c l -> (b n) c l")
        advantages = self.compute_normalized_advantages(rewards, mean_grouped_rewards, std_grouped_rewards, pred_mask)

        return advantages

    def compute_patch_advantages(self, rewards, pred_mask, num_var, num_token_per_var, type="PRIME", is_cumsum=False):  # Corresponding to process supervision
        if type == "PRIME":
            rewards = rearrange(rewards, "(b n) (c l) -> b c n l", n=self.num_generations, c=num_var)
            predict_step = math.ceil(self.prediction_length / self.patch_size)
            grouped_rewards = torch.sum(rewards, dim=-1) / predict_step
            mean_grouped_rewards = torch.mean(grouped_rewards, dim=-1, keepdim=True).repeat(1, 1, self.num_generations*num_token_per_var)
            std_grouped_rewards = torch.std(grouped_rewards, dim=-1, keepdim=True).repeat(1, 1, self.num_generations*num_token_per_var)
            mean_grouped_rewards = rearrange(mean_grouped_rewards, "b c (n l) -> (b n) c l", n=self.num_generations)
            std_grouped_rewards = rearrange(std_grouped_rewards, "b c (n l) -> (b n) c l", n=self.num_generations)
            rewards = rearrange(rewards, "b c n l -> (b n) c l")

        elif type == "DeepSeek":
            rewards = rearrange(rewards, "(b n) (c l) -> b c (n l)", n=self.num_generations, c=num_var)
            predict_step = math.ceil(self.prediction_length / self.patch_size)
            mean_grouped_rewards = torch.sum(rewards, dim=-1, keepdim=True) / (self.num_generations*predict_step)
            mean_grouped_rewards = mean_grouped_rewards.repeat(1, 1, self.num_generations*num_token_per_var)
            pred_mask_new = rearrange(pred_mask, "(b n) (c l) -> b c (n l)", n=self.num_generations, c=num_var)
            var_grouped_rewards = torch.sum((rewards - mean_grouped_rewards) ** 2 * pred_mask_new,
                                            dim=-1, keepdim=True) / (self.num_generations*predict_step-1)
            std_grouped_rewards = torch.sqrt(var_grouped_rewards).repeat(1, 1, self.num_generations*num_token_per_var)
            mean_grouped_rewards = rearrange(mean_grouped_rewards, "b c (n l) -> (b n) c l", n=self.num_generations)
            std_grouped_rewards = rearrange(std_grouped_rewards, "b c (n l) -> (b n) c l", n=self.num_generations)
            rewards = rearrange(rewards, "b c (n l) -> (b n) c l", n=self.num_generations)

        elif type == "Naive":
            rewards = rearrange(rewards, "(b n) (c l) -> b n c l", n=self.num_generations, c=num_var)
            mean_grouped_rewards = torch.mean(rewards, dim=1, keepdim=True).repeat(1, self.num_generations, 1, 1)
            std_grouped_rewards = torch.std(rewards, dim=1, keepdim=True).repeat(1, self.num_generations, 1, 1)
            mean_grouped_rewards = rearrange(mean_grouped_rewards, "b n c l -> (b n) c l")
            std_grouped_rewards = rearrange(std_grouped_rewards, "b n c l -> (b n) c l")
            rewards = rearrange(rewards, "b n c l -> (b n) c l")

        else:
            raise NotImplementedError

        # Step-wise advantage accumulation
        advantages = self.compute_normalized_advantages(rewards, mean_grouped_rewards, std_grouped_rewards, pred_mask)
        advantages = rearrange(advantages, "n (c l) -> n c l", c=num_var)
        if is_cumsum:
            advantages_reversed = torch.flip(advantages, dims=[-1])
            advantages_reversed = torch.cumsum(advantages_reversed, dim=-1)
            advantages = torch.flip(advantages_reversed, dims=[-1]).view(advantages.shape[0], -1) * pred_mask
        else:
            advantages = advantages.view(advantages.shape[0], -1) * pred_mask

        return advantages

    def clip_advantages(self, advantages, type="NSR"):
        if type == "NSR":  # Solely NSR training
            advantages_raw = rearrange(advantages, "(b n) l -> n b l", n=self.num_generations)
            advantages = torch.clamp(advantages_raw, max=0)
            advantages[-1, ...] = advantages_raw[-1, ...]  # preserve positive off-policy advantage
            advantages = rearrange(advantages, "n b l -> (b n) l")

        elif type == "PSR":  # Solely PSR training
            advantages = rearrange(advantages, "(b n) l -> n b l", n=self.num_generations)
            advantages = torch.clamp_min(advantages, min=0)
            advantages = rearrange(advantages, "n b l -> (b n) l")

        elif type == "W-Reinforce":  # W-Reinforce training
            advantages = torch.where(advantages > 0, advantages * 0.1, advantages)

        else:
            raise NotImplementedError

        return advantages

    def compute_loss(self, model, batch, epoch):
        # Generate forecasts (completions)
        forecasts = self.generate_forecasts(model, batch, self.num_generations-1)

        # Incorporate off-policy guidance samples
        target = batch["target"].unsqueeze(0)  # (1, batch_size, seq_len, patch_size)
        forecasts = rearrange(forecasts, "(b n) l p -> n b l p", n=self.num_generations-1)
        forecasts = torch.concatenate([forecasts, target], dim=0)  # (num_generations, batch_size, seq_len, patch_size)
        forecasts = rearrange(forecasts, "n b l p -> (b n) l p")

        # Get per-token log probability for both native and reference model
        per_token_logps, pred_mask_offset = self.get_per_token_logps(model, batch, forecasts)
        ref_per_token_logps, _ = self.get_per_token_logps(self.ref_model, batch, forecasts)

        # Compute KL divergence between native reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Prepare inputs
        expand_target = batch["target"].repeat_interleave(self.num_generations, dim=0)
        num_token = expand_target.shape[1]
        num_var = batch["variate_id"].unique().shape[-1]
        num_token_per_var = math.ceil(num_token / num_var)
        patch_size = expand_target.shape[-1]
        expand_target = rearrange(expand_target, 'n (c l) p -> n (l p) c', c=num_var)
        forecasts = rearrange(forecasts, 'n (c l) p -> n (l p) c', c=num_var)

        # Compute sequence-wise rewards
        acc_rewards_seq = sequence_accuracy_reward(prediction=forecasts, target=expand_target,  # (batch_size*num_generations, seq_len)
                                                   prediction_length=self.prediction_length, patch_size=patch_size)
        var_rewards_seq = sequence_structure_reward(prediction=forecasts, target=expand_target,  # (batch_size*num_generations, seq_len)
                                                    prediction_length=self.prediction_length, patch_size=patch_size, type="Variance")
        freq_rewards = sequence_frequency_reward(prediction=forecasts, target=expand_target,  # (batch_size*num_generations, seq_len)
                                                 prediction_length=self.prediction_length, patch_size=patch_size)

        # Compute patch-wise rewards
        acc_rewards = patch_accuracy_reward(prediction=forecasts, target=expand_target,  # (batch_size*num_generations, seq_len)
                                            prediction_length=self.prediction_length, patch_size=patch_size)
        var_rewards = patch_structure_reward(prediction=forecasts, target=expand_target,  # (batch_size*num_generations, seq_len)
                                             prediction_length=self.prediction_length, patch_size=patch_size, type="Variance")

        # Aggregate rewards with synergy bonus
        if epoch < 30:  # learning dominant low-frequency patterns firstly
            lambda_freq_synergy = 0.0
        else:
            lambda_freq_synergy = self.lambda_freq_synergy
        patch_rewards_total = self.lambda_acc_reward * acc_rewards + \
                              self.lambda_var_reward * var_rewards + \
                              self.lambda_var_synergy * (acc_rewards * var_rewards) + \
                              lambda_freq_synergy * (acc_rewards * freq_rewards)
        seq_rewards_total = self.lambda_acc_reward * acc_rewards_seq + \
                            self.lambda_var_reward * var_rewards_seq + \
                            self.lambda_var_synergy * (acc_rewards_seq * var_rewards_seq) + \
                            lambda_freq_synergy * (acc_rewards_seq * freq_rewards)

        # Rewards shaping
        if self.reward_shaping:
            seq_rewards_total = self.shape_rewards(seq_rewards_total, per_token_logps.detach(), unit="sequence",
                                                   type="Piecewise-shaping", threshold=0.8, alpha=0.01)
            patch_rewards_total = self.shape_rewards(patch_rewards_total, per_token_logps.detach(), unit="patch",
                                                     type="Piecewise-shaping", threshold=0.8, alpha=0.01)

        # Offset rewards
        seq_rewards_offset = self.offset_rewards(seq_rewards_total, pred_mask_offset, num_var, num_token_per_var)
        patch_rewards_offset = self.offset_rewards(patch_rewards_total, pred_mask_offset, num_var, num_token_per_var)

        # Compute sequence-wise advantages
        seq_advantages = self.compute_sequence_advantages(seq_rewards_offset, pred_mask_offset, num_var)  # (batch_size*num_generations, seq_len)

        # Compute patch-wise advantages
        patch_advantages = self.compute_patch_advantages(patch_rewards_offset, pred_mask_offset, num_var, num_token_per_var,
                                                         type="PRIME", is_cumsum=True)  # (batch_size*num_generations, seq_len)

        # Compute combined rewards and advantages
        rewards_offset = 0.0 * seq_rewards_offset + 1.0 * patch_rewards_offset
        advantages = 0.0 * seq_advantages + 1.0 * patch_advantages

        # # Clip advantages
        # advantages = self.clip_advantages(advantages, type="NSR")

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(-1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl) * pred_mask_offset.unsqueeze(-1)
        per_token_loss = per_token_loss.view(per_token_loss.shape[0], -1)
        num_pred_token = num_var * num_token_per_var
        loss = (per_token_loss.sum(dim=1) / num_pred_token).mean()

        # Record monitoring metrics
        reward_mean = rewards_offset.mean().item()
        self.metrics["reward"].append(reward_mean)
        per_token_kl = per_token_kl.view(per_token_kl.shape[0], -1)
        kl_mean = (per_token_kl.sum(dim=1) / num_pred_token).mean().item()
        self.metrics["kl"].append(kl_mean)

        return loss






