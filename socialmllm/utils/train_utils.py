import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
import yaml
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.trainer import is_sagemaker_mp_enabled

from socialmllm.utils.data_processing import LengthGroupedSampler
from socialmllm.utils.input_processing import (
    load_content,
    prepare_spec,
    time_synchronize,
)


def chunk_forward_fn(x_c, y_c, w, b, out_c):
    logits = F.linear(x_c, w, b)
    loss = (
        logits.logsumexp(dim=-1)
        - logits[torch.arange(x_c.size(0), device=x_c.device), y_c]
    )
    out_c.copy_(loss)


def chunk_backward_fn(x_c, y_c, grad_out_c, w, b, grad_x_c, grad_w, grad_b):
    logits = F.linear(x_c, w, b)
    p_c = torch.softmax(logits, dim=-1)
    p_c[torch.arange(x_c.size(0), device=x_c.device), y_c] -= 1.0
    grad_logits = p_c * grad_out_c.unsqueeze(1)
    torch.mm(grad_logits, w, out=grad_x_c)
    grad_w.addmm_(grad_logits.T, x_c)
    if grad_b is not None:
        grad_b.add_(grad_logits.sum(0))


class FusedNLL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, weight, bias=None, chunk_size=256):
        assert x.ndim == 2
        assert weight.ndim == 2
        assert target.ndim == 1
        assert target.size(0) == x.size(0)
        assert bias is None or (bias.ndim == 1 and bias.size(0) == weight.size(0))

        ctx.save_for_backward(x, target, weight, bias)
        ctx.chunk_size = chunk_size

        N = x.size(0)
        per_sample_loss = torch.empty(N, device=x.device, dtype=x.dtype)

        for i in range(0, N, chunk_size):
            ed = min(i + chunk_size, N)
            x_c = x[i:ed]
            y_c = target[i:ed]
            out_c = per_sample_loss[i:ed]

            if x_c.size(0) < chunk_size:
                pad_len = chunk_size - x_c.size(0)
                x_pad = F.pad(x_c, (0, 0, 0, pad_len), value=0)
                y_pad = F.pad(y_c, (0, pad_len), value=0)
                out_pad = torch.empty(chunk_size, device=x.device, dtype=x.dtype)
                chunk_forward_fn(x_pad, y_pad, weight, bias, out_pad)
                out_c.copy_(out_pad[: x_c.size(0)])
            else:
                chunk_forward_fn(x_c, y_c, weight, bias, out_c)

        return per_sample_loss

    @staticmethod
    def backward(ctx, grad_output):
        x, target, weight, bias = ctx.saved_tensors
        chunk_size = ctx.chunk_size

        N = x.size(0)
        grad_x = torch.empty_like(x)
        grad_w = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        for i in range(0, N, chunk_size):
            ed = min(i + chunk_size, N)
            x_c = x[i:ed]
            y_c = target[i:ed]
            grad_out_c = grad_output[i:ed]
            grad_x_c = grad_x[i:ed]

            if x_c.size(0) < chunk_size:
                pad_len = chunk_size - x_c.size(0)
                x_p = F.pad(x_c, (0, 0, 0, pad_len))
                y_p = F.pad(y_c, (0, pad_len))
                grad_out_p = F.pad(grad_out_c, (0, pad_len), value=0)
                grad_x_p = torch.empty(
                    chunk_size, x.size(1), device=x.device, dtype=x.dtype
                )
                chunk_backward_fn(
                    x_p, y_p, grad_out_p, weight, bias, grad_x_p, grad_w, grad_bias
                )
                grad_x_c.copy_(grad_x_p[: x_c.size(0)])
            else:
                chunk_backward_fn(
                    x_c, y_c, grad_out_c, weight, bias, grad_x_c, grad_w, grad_bias
                )

        return grad_x, None, grad_w, grad_bias, None


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)
    else:
        print(*args)


class BaseDataset(Dataset):
    """Dataset for supervised fine-tuning supporting multiple model types."""

    def __init__(
        self,
        data_path: str,
        processor,
        data_args,
        model_type: str = "qwen3_vl",
        spec_processor=None,
        model_config=None,
    ):
        super().__init__()
        self.processor = processor
        self.data_args = data_args
        self.model_type = model_type
        self.spec_processor = spec_processor
        self.model_config = model_config
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                datasets = yaml_data.get("datasets", [])
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    self._load_json(json_path)
        elif data_path.endswith((".json", ".jsonl")):
            self._load_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}.")

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")

    def _load_json(self, json_path):
        if json_path.endswith(".jsonl"):
            with open(json_path, "r") as f:
                for line in f:
                    if line.strip():
                        self.list_data_dict.append(json.loads(line.strip()))
        else:
            with open(json_path, "r") as f:
                self.list_data_dict.extend(json.load(f))

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for item in tqdm.tqdm(
            self.list_data_dict,
            desc="Data Sampler Preprocessing",
            disable=dist.is_initialized() and dist.get_rank() != 0,
            ncols=70,
            mininterval=1,
        ):
            specs = prepare_spec(item)
            specs = self.spec_processor(
                specs,
                processor=self.processor,
                data_args=self.data_args,
                model_config=self.model_config,
            )
            length_list.append(sum(s.num_tokens for s in specs))
        return length_list

    @property
    def difficulties(self):
        diff_list = []
        for item in self.list_data_dict:
            diff = "medium"
            for entry in item:
                if isinstance(entry, dict) and entry.get("type") == "meta":
                    diff = entry.get("difficulty", "medium")
                    break
            diff_list.append(diff)
        return diff_list

    def _extract_meta_field(self, idx, field, default=None):
        """Extract a field from the meta block of the raw data entry."""
        for entry in self.list_data_dict[idx]:
            if isinstance(entry, dict) and entry.get("type") == "meta":
                return entry.get(field, default)
        return default

    def __getitem__(self, idx):
        specs = prepare_spec(self.list_data_dict[idx])
        specs = self.spec_processor(
            specs,
            processor=self.processor,
            data_args=self.data_args,
            model_config=self.model_config,
        )
        specs = [load_content(s) for s in specs]
        specs = time_synchronize(specs)
        gt_events = self._extract_meta_field(idx, "gt_events")
        if gt_events is not None:
            grounding_info = {
                "events": gt_events,
                "question_text": self._extract_meta_field(idx, "question_text", ""),
            }
            return specs, grounding_info
        return specs


@dataclass
class BaseDataCollator:
    """Collate examples for supervised fine-tuning."""

    processor: Any
    model_type: str = "qwen3_vl"
    apply_chat_template: Optional[Any] = None

    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        gt_events_list = None
        if instances and isinstance(instances[0], tuple):
            specs_list, gt_events_list = zip(*instances)
            instances = list(specs_list)
        result = self.apply_chat_template(instances, processor=self.processor)
        if gt_events_list is not None:
            result["gt_events"] = list(gt_events_list)
        return result


def configure_model_for_training(model, model_args, training_args):
    """Configure which parts of the model to train."""
    model.config.use_cache = False
    model.requires_grad_(False)

    if getattr(model_args, "tune_lang", False):
        for p in model.language_parameters:
            p.requires_grad = True
        rank0_print("Trainable: language model")

    if getattr(model_args, "tune_proj", False):
        for p in model.projection_parameters:
            p.requires_grad = True
        rank0_print("Trainable: visual.merger (projector)")

    if getattr(model_args, "tune_vis", False):
        for p in model.vision_parameters:
            p.requires_grad = True
        rank0_print("Trainable: visual (full vision tower)")

    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        p_numel = getattr(p, "ds_numel", p.numel())
        total_params += p_numel
        if p.requires_grad:
            trainable_params += p_numel

    rank0_print(f"Total parameters: {total_params:,}")
    rank0_print(
        f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    return model


class BaseTrainer(Trainer):
    """Base trainer with common functionality for SocialMLLM."""

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        generator = torch.Generator()
        generator.manual_seed(
            self.args.data_seed if self.args.data_seed is not None else self.args.seed
        )
        return LengthGroupedSampler(
            global_batch_size=self.args.train_batch_size * self.args.world_size,
            lengths=train_dataset.lengths,
            difficulties=train_dataset.difficulties,
            generator=generator,
            curriculum=getattr(self.args, "curriculum", True),
        )

    def create_optimizer(self):
        """Create optimizer with different learning rates for different components."""
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)

            unwrapped_model = unwrap_model(opt_model)

            param_sets = {
                "language": set(getattr(unwrapped_model, "language_parameters", [])),
                "projector": set(getattr(unwrapped_model, "projection_parameters", [])),
                "vision": set(getattr(unwrapped_model, "vision_parameters", [])),
            }

            lr_lang = (
                self.args.lr_lang
                if self.args.lr_lang is not None
                else self.args.learning_rate
            )
            lr_proj = (
                self.args.lr_proj
                if self.args.lr_proj is not None
                else self.args.learning_rate
            )
            lr_vis = (
                self.args.lr_vis
                if self.args.lr_vis is not None
                else self.args.learning_rate
            )

            groups = {
                "language": {"lr": lr_lang, "decay": [], "no_decay": [], "count": 0},
                "projector": {"lr": lr_proj, "decay": [], "no_decay": [], "count": 0},
                "vision": {"lr": lr_vis, "decay": [], "no_decay": [], "count": 0},
                "other": {"lr": lr_lang, "decay": [], "no_decay": [], "count": 0},
            }

            for name, param in opt_model.named_parameters():
                keys = [n for n, p in param_sets.items() if param in p] or ["other"]
                assert len(keys) == 1, (
                    f"Parameter {name} overlaps or is unassigned: {keys}"
                )
                key = keys[0]

                groups[key]["count"] += getattr(param, "ds_numel", param.numel())

                if param.requires_grad:
                    if name in decay_parameters:
                        groups[key]["decay"].append(param)
                    else:
                        groups[key]["no_decay"].append(param)

            if self.is_world_process_zero():
                print(f"{'=' * 20} Optimizer Groups {'=' * 20}")
                for key, info in groups.items():
                    if info["count"] > 0:
                        trainable_decay = sum(
                            getattr(p, "ds_numel", p.numel()) for p in info["decay"]
                        )
                        trainable_no_decay = sum(
                            getattr(p, "ds_numel", p.numel()) for p in info["no_decay"]
                        )
                        trainable_total = trainable_decay + trainable_no_decay
                        print(
                            f"Group '{key}': {trainable_total:,} trainable / {info['count']:,} total params. LR: {info['lr']}"
                        )
                        print(
                            f"Group '{key}' with weight decay: {trainable_decay:,} trainable params. LR: {info['lr']}"
                        )
                        print(
                            f"Group '{key}' without weight decay: {trainable_no_decay:,} trainable params. LR: {info['lr']}"
                        )
                print(f"{'=' * 58}")

            optimizer_grouped_parameters = []
            for key, info in groups.items():
                if info["decay"]:
                    optimizer_grouped_parameters.append(
                        {
                            "params": info["decay"],
                            "weight_decay": self.args.weight_decay,
                            "lr": info["lr"],
                        }
                    )
                if info["no_decay"]:
                    optimizer_grouped_parameters.append(
                        {
                            "params": info["no_decay"],
                            "weight_decay": 0.0,
                            "lr": info["lr"],
                        }
                    )

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer
