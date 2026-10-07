from dataclasses import dataclass, field
from typing import List, Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-VL-8B-Instruct")
    model_type: str = field(
        default="qwen3_vl",
        metadata={"help": "Model type: qwen3_vl, qwen3_5"},
    )
    tune_lang: bool = field(default=False)
    tune_proj: bool = field(default=False)
    tune_vis: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to training data JSON/JSONL/YAML file"}
    )
    video_min_frames_per_clip: int = field(default=2)
    video_max_total_frames: int = field(default=512)
    video_max_fps: float = field(default=0.5)
    video_frame_multiple: int = field(default=2)

    frame_time_patch: int = field(default=2)
    frame_spatial_patch: int = field(default=32)
    frame_max_tokens: int = field(default=256)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=4096)
    dataloader_num_workers: int = field(default=4)

    lr_lang: Optional[float] = field(default=None)
    lr_proj: Optional[float] = field(default=None)
    lr_vis: Optional[float] = field(default=None)

    report_to: List[str] = field(default_factory=lambda: ["wandb"])

    attn_implementation: str = field(default="flash_attention_2")
