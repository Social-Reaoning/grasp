import re

import torch
from transformers.modeling_utils import unwrap_model

from socialmllm.train.metrics_callback import GlobalMetricsCallback
from socialmllm.utils.train_utils import BaseTrainer, rank0_print

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>")
_GAZE_TAG_RE = re.compile(r"<gaze>(.*?)</gaze>", re.DOTALL)
_GESTURE_TAG_RE = re.compile(r"<gesture>(.*?)</gesture>", re.DOTALL)
_PERSON_ID_RE = re.compile(r"(?:Person\s+|\bP)(\d+)")


def compute_grounding_score(text, gt_events, question_text=""):
    """Social Grounding Reward: participant-level grounding quality.

    Participants mentioned in the question are excluded from the targets.
    Gaze-event participants must appear inside <gaze> tags and gesture-event
    participants inside <gesture> tags. Score = 0.7 * recall + 0.3 * precision.
    """
    if not gt_events:
        return 0.0

    gaze_text = " ".join(_GAZE_TAG_RE.findall(text))
    gesture_text = " ".join(_GESTURE_TAG_RE.findall(text))
    if not gaze_text.strip() and not gesture_text.strip():
        return 0.0

    gaze_persons = set(int(m) for m in _PERSON_ID_RE.findall(gaze_text))
    gesture_persons = set(int(m) for m in _PERSON_ID_RE.findall(gesture_text))

    q_persons = (
        set(int(m) for m in _PERSON_ID_RE.findall(question_text))
        if question_text
        else set()
    )

    gaze_gt = set()
    gesture_gt = set()
    for event in gt_events:
        novel = set(event.get("persons", [])) - q_persons
        if event["source"] == "gaze":
            gaze_gt.update(novel)
        else:
            gesture_gt.update(novel)

    if not gaze_gt and not gesture_gt:
        return 0.0

    recalls = []
    for event in gt_events:
        ep = set(event.get("persons", [])) - q_persons
        if not ep:
            continue
        gen = gaze_persons if event["source"] == "gaze" else gesture_persons
        recalls.append(len(gen & ep) / len(ep))
    recall = sum(recalls) / len(recalls) if recalls else 0.0

    precisions = []
    gaze_gen_novel = gaze_persons - q_persons
    gesture_gen_novel = gesture_persons - q_persons
    if gaze_gen_novel and gaze_gt:
        precisions.append(len(gaze_gen_novel & gaze_gt) / len(gaze_gen_novel))
    if gesture_gen_novel and gesture_gt:
        precisions.append(len(gesture_gen_novel & gesture_gt) / len(gesture_gen_novel))
    precision = sum(precisions) / len(precisions) if precisions else 0.0

    return 0.7 * recall + 0.3 * precision


class ReasoningQATrainer(BaseTrainer):
    def __init__(self, ref_model=None, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model

    @torch.no_grad()
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        K = self.args.num_generations

        all_micro_batches = []
        for _ in range(num_batches):
            try:
                inputs = next(epoch_iterator)
            except StopIteration:
                break
            inputs = self._prepare_inputs(inputs)
            all_micro_batches.extend(self._generate_rollouts(inputs, K, device))

        return all_micro_batches, None

    @torch.no_grad()
    def _generate_rollouts(self, inputs, K, device):
        """Generate K rollouts for one input, compute advantages, return K micro-batches."""
        prompt_len = inputs["input_ids"].shape[1]
        target_words = inputs.pop("target_words")
        gt_events_list = inputs.pop("gt_events", None)
        B = inputs["input_ids"].shape[0]
        N = B * K

        self.model.eval()
        unwrapped = unwrap_model(self.model)
        gc_was_enabled = getattr(unwrapped, "is_gradient_checkpointing", False)
        if gc_was_enabled:
            unwrapped.gradient_checkpointing_disable()

        inputs.pop("mm_token_type_ids", None)
        gen_kwargs = dict(
            max_new_tokens=self.args.max_new_tokens,
            num_return_sequences=K,
            do_sample=True,
            temperature=self.args.generation_temperature,
            top_p=1.0,
            top_k=0,
            use_cache=True,
            pad_token_id=self.processing_class.tokenizer.pad_token_id,
        )
        try:
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        except (ValueError, RuntimeError) as e:
            if "do not match" in str(e) or "invalid for input of size" in str(e):
                rank0_print(f"[WARN] Skipping batch in generation: {e}")
                if gc_was_enabled:
                    unwrapped.gradient_checkpointing_enable()
                self.model.train()
                return []
            raise
        if gc_was_enabled:
            unwrapped.gradient_checkpointing_enable()
        self.model.train()

        rewards = self._calculate_raw_rewards(
            generated_ids, prompt_len, device, target_words, K, gt_events_list
        )
        advantages = self._aggregate_advantages(rewards).reshape(N)

        micro_batches = []
        for k in range(K):
            indices = torch.arange(k, N, K, device=device)
            micro_ids = generated_ids[indices]
            micro_adv = advantages[indices]
            attention_mask = micro_ids != self.processing_class.tokenizer.pad_token_id
            response_mask = attention_mask.clone()
            response_mask[:, :prompt_len] = False
            labels = torch.where(response_mask, micro_ids.clone(), -100)

            micro_batch = {
                "input_ids": micro_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "advantages": micro_adv,
            }
            for key, value in inputs.items():
                if key not in [
                    "input_ids",
                    "attention_mask",
                    "target_words",
                    "gt_events",
                ]:
                    micro_batch[key] = value

            micro_batches.append(micro_batch)

        return micro_batches

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        adv = inputs.pop("advantages")
        assert adv.ndim == 1
        seq_len = (inputs["labels"] != -100).sum(1)

        if self.ref_model is not None:
            ref_device = next(self.ref_model.parameters()).device
            target_device = inputs["input_ids"].device
            if ref_device != target_device:
                self.ref_model = self.ref_model.to(target_device)

            try:
                batched_nll, cur_token_nll, batch_indices = model(
                    **inputs, return_per_token_nll=True
                )
            except (ValueError, RuntimeError) as e:
                if "do not match" in str(e) or "invalid for input of size" in str(e):
                    rank0_print(f"[WARN] Skipping batch in compute_loss: {e}")
                    return inputs["input_ids"].new_tensor(
                        0.0, dtype=torch.float, requires_grad=True
                    )
                raise
            with torch.no_grad():
                _, ref_token_nll, _ = self.ref_model(
                    **inputs, return_per_token_nll=True
                )

            r_t = (cur_token_nll - ref_token_nll).clamp(-20, 20)
            kl_per_token = (r_t.exp() - r_t - 1).clamp(-10, 10)

            n_response_tokens = seq_len.sum()
            kl_mean = kl_per_token.sum() / n_response_tokens

            policy_loss = (adv * batched_nll / seq_len).mean()
            loss = policy_loss + self.args.kl_coeff * kl_mean

            GlobalMetricsCallback.record_metric(
                self, "kl", kl_per_token.sum(), n_response_tokens
            )
        else:
            try:
                batched_nll = model(**inputs)
            except (ValueError, RuntimeError) as e:
                if "do not match" in str(e) or "invalid for input of size" in str(e):
                    rank0_print(f"[WARN] Skipping batch in compute_loss: {e}")
                    return inputs["input_ids"].new_tensor(
                        0.0, dtype=torch.float, requires_grad=True
                    )
                raise
            assert batched_nll.ndim == 1
            loss = (adv * batched_nll / seq_len).mean()

        return (loss, None) if return_outputs else loss

    @torch.no_grad()
    def _calculate_raw_rewards(
        self,
        generated_ids,
        prompt_len,
        device,
        target_words,
        num_generations,
        gt_events_list=None,
    ):
        N = generated_ids.size(0)
        K = num_generations
        B = N // K

        is_correct = torch.zeros(B, K, device=device, dtype=torch.bool)
        has_format = torch.zeros(B, K, device=device, dtype=torch.bool)
        has_grounding = torch.zeros(B, K, device=device, dtype=torch.bool)
        grounding_quality = torch.zeros(B, K, device=device, dtype=torch.float)
        seq_length = (
            (
                generated_ids[:, prompt_len:]
                != self.processing_class.tokenizer.pad_token_id
            )
            .sum(dim=-1)
            .reshape(B, K)
        )

        decoded_texts = self.processing_class.batch_decode(
            generated_ids[:, prompt_len:], skip_special_tokens=True
        )
        for i, text in enumerate(decoded_texts):
            b_idx, k_idx = divmod(i, K)
            target = target_words[b_idx]
            match = _ANSWER_RE.search(text)
            if match:
                has_format[b_idx, k_idx] = True
                if match.group(1).strip() == target:
                    is_correct[b_idx, k_idx] = True

            if _GAZE_TAG_RE.search(text) or _GESTURE_TAG_RE.search(text):
                has_grounding[b_idx, k_idx] = True
            if gt_events_list and gt_events_list[b_idx]:
                info = gt_events_list[b_idx]
                if isinstance(info, dict):
                    events = info["events"]
                    q_text = info.get("question_text", "")
                else:
                    events = info
                    q_text = ""
                grounding_quality[b_idx, k_idx] = compute_grounding_score(
                    text, events, q_text
                )

        tot_cor = is_correct.sum()
        tot_fmt = has_format.sum()
        tot_len = seq_length.sum()
        cor_len = seq_length[is_correct].sum()
        GlobalMetricsCallback.record_metric(self, "accuracy", tot_cor, N)
        GlobalMetricsCallback.record_metric(self, "format_rate", tot_fmt, N)
        GlobalMetricsCallback.record_metric(self, "average_length", tot_len, N)
        GlobalMetricsCallback.record_metric(self, "correct_length", cor_len, tot_cor)
        GlobalMetricsCallback.record_metric(
            self, "grounding_rate", has_grounding.sum(), N
        )
        GlobalMetricsCallback.record_metric(
            self, "grounding_quality", grounding_quality.sum(), N
        )

        return dict(
            is_correct=is_correct,
            has_format=has_format,
            seq_length=seq_length,
            has_grounding=has_grounding,
            grounding_quality=grounding_quality,
        )

    def _aggregate_advantages(self, rewards):
        processed_rewards = dict(
            correct=rewards["is_correct"].float(),
            has_format=rewards["has_format"].float(),
            has_grounding=rewards["has_grounding"].float(),
            grounding_quality=rewards["grounding_quality"],
        )
        total_advantage = torch.zeros_like(rewards["is_correct"], dtype=torch.float)
        for name, beta in self.args.betas.items():
            if name in processed_rewards:
                adv = self._compute_normalized_advantages(processed_rewards[name])
                total_advantage += adv * beta

        if self.args.overlong_buffer_len > 0:
            soft_limit = self.args.max_new_tokens - self.args.overlong_buffer_len
            exceed = rewards["seq_length"].float() - soft_limit
            penalty = (
                -exceed
                / self.args.overlong_buffer_len
                * self.args.overlong_penalty_factor
            ).clamp(max=0)
            total_advantage += penalty

        return total_advantage.clamp(
            -self.args.advantage_clip, self.args.advantage_clip
        )

    @torch.no_grad()
    def _compute_normalized_advantages(self, rewards):
        adv = rewards - torch.nanmean(rewards, dim=1, keepdim=True)
        return torch.where(torch.isnan(adv), 0, adv)
