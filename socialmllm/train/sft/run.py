import pathlib
import warnings

import torch
import transformers
from transformers import AutoProcessor
from transformers.trainer_utils import set_seed

from socialmllm.model import Qwen3_5ForSFT, Qwen3VLForSFT
from socialmllm.train.metrics_callback import GlobalMetricsCallback
from socialmllm.train.sft.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from socialmllm.train.sft.trainer import SFTTrainer
from socialmllm.utils.train_utils import (
    BaseDataCollator,
    BaseDataset,
    configure_model_for_training,
    rank0_print,
)

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def make_supervised_data_module(processor, data_args, model_type, model):
    """Make dataset and collator for supervised fine-tuning."""
    assert model is not None
    spec_processor = model.preprocess_input_spec
    apply_chat_template = model.apply_chat_template

    train_dataset = BaseDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_type=model_type,
        spec_processor=spec_processor,
        model_config=model.config,
    )
    data_collator = BaseDataCollator(
        processor=processor,
        model_type=model_type,
        apply_chat_template=apply_chat_template,
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def get_model(model_args, training_args):
    """Load model based on model type."""
    rank0_print(f"Loading model: {model_args.model_name_or_path}")
    rank0_print(f"Model type: {model_args.model_type}")

    model_kwargs = {
        "cache_dir": training_args.cache_dir,
        "dtype": torch.bfloat16 if training_args.bf16 else torch.float16,
        "attn_implementation": training_args.attn_implementation,
    }
    processor_kwargs = {
        "videos_kwargs": {"do_resize": False},
        "do_sample_frames": False,
        "model_max_length": training_args.model_max_length,
        "pad_to_multiple_of": 256,
        "padding": True,
        "padding_side": "left",
    }

    if model_args.model_type == "qwen3_vl":
        model = Qwen3VLForSFT.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif model_args.model_type == "qwen3_5":
        model = Qwen3_5ForSFT.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_args.model_type}")

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, **processor_kwargs
    )

    return model, processor


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    rank0_print("=" * 50)
    rank0_print("SocialMLLM SFT Training")
    rank0_print("=" * 50)
    rank0_print(f"Model: {model_args.model_name_or_path}")
    rank0_print(f"Model type: {model_args.model_type}")
    rank0_print(f"Data: {data_args.data_path}")
    rank0_print("=" * 50)

    if training_args.local_rank <= 0:
        from huggingface_hub import snapshot_download

        rank0_print(f"Pre-downloading: {model_args.model_name_or_path}")
        snapshot_download(model_args.model_name_or_path)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model, processor = get_model(model_args, training_args)

    model = configure_model_for_training(model, model_args, training_args)

    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args,
        model_type=model_args.model_type,
        model=model,
    )

    trainer = SFTTrainer(
        model=model, args=training_args, processing_class=processor, **data_module
    )
    trainer.add_callback(GlobalMetricsCallback(trainer))

    if model_args.model_type == "qwen3_5":
        import transformers.modeling_flash_attention_utils as fa_utils

        if not getattr(fa_utils, "_mrope_patched", False):
            orig = fa_utils._is_packed_sequence

            def _safe_is_packed(position_ids, batch_size):
                if position_ids is not None and position_ids.ndim == 3:
                    return False
                return orig(position_ids, batch_size)

            fa_utils._is_packed_sequence = _safe_is_packed
            fa_utils._mrope_patched = True
            rank0_print("Patched flash attention for MROPE 3D position_ids")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    trainer.save_model(training_args.output_dir)
    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
