import torch

from socialmllm.utils.train_utils import BaseTrainer, rank0_print


class SFTTrainer(BaseTrainer):
    """Custom trainer for SFT multimodal models."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute loss for multimodal models."""
        inputs.pop("mm_token_type_ids", None)
        try:
            outputs = model(**inputs)
        except ValueError as e:
            if "do not match" in str(e):
                rank0_print(f"[WARN] Skipping batch in compute_loss: {e}")
                zero = inputs["input_ids"].new_tensor(
                    0.0, dtype=torch.float, requires_grad=True
                )
                return (zero, None) if return_outputs else zero
            raise
        loss = outputs.loss

        if loss is None:
            raise ValueError("Model did not return loss. Ensure labels are provided.")

        return (loss, outputs) if return_outputs else loss
