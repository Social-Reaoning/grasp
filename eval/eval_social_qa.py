"""Evaluate Qwen3-VL / Qwen3.5 models on GRASP-Bench.

Usage (from the repository root):
    python -m eval.eval_social_qa \
        --model_type qwen3_vl \
        --model_path interlive/GRASP-Qwen3-VL-8B \
        --data_path /path/to/grasp_bench \
        --output eval/results/grasp_qwen3_vl.json \
        --reasoning

--data_path points to the extracted GRASP-Bench directory:
    json/{name}.json: {"clip_file": "name.mp4", "qa": {"category": "T1", "question": "...", ...}}
    video/{name}.mp4: corresponding clip
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

MOCK_FPS = 100
VIDEO_FPS = 2


class StopOnAnswer(StoppingCriteria):
    """Stop generation when </answer> is decoded."""

    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0, self.prompt_len :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return "</answer>" in text


def load_video(video_path, max_frames=240):
    """Load video frames as numpy array [T, H, W, 3]."""
    import decord

    vr = decord.VideoReader(video_path, num_threads=1)
    n = min(max_frames, len(vr))
    indices = np.linspace(0, len(vr) - 1, n, dtype=int)
    return vr.get_batch(indices).asnumpy(), n


def preprocess_frames_qwen(
    frames, frame_time_patch=2, frame_spatial_patch=32, frame_max_tokens=256
):
    """Resize frames to fit the Qwen patch grid."""
    from socialmllm.utils.input_processing import (
        VideoSpec,
        resize_image,
        resolve_resolution,
    )

    src_resolution = (frames.shape[1], frames.shape[2])
    spec = VideoSpec(
        path="",
        fps=1.0,
        start_seconds=0.0,
        end_seconds=1.0,
        src_resolution=src_resolution,
        num_frames=frame_time_patch,
    )
    dst_resolution, _ = resolve_resolution(
        spec,
        frame_time_patch=frame_time_patch,
        frame_spatial_patch=frame_spatial_patch,
        frame_max_tokens=frame_max_tokens,
    )
    dst_resolution = tuple(dst_resolution)
    if frames.shape[1] == dst_resolution[0] and frames.shape[2] == dst_resolution[1]:
        return frames
    return np.stack([resize_image(f, dst_resolution) for f in frames])


def extract_reasoning(text):
    """Split output into (<think> block, rest)."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end() :].strip()
    if "</think>" in text:
        parts = text.split("</think>", 1)
        return parts[0].strip(), parts[1].strip()
    return "", text.strip()


def extract_answer(text):
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text)
    if m:
        ans = m.group(1).strip()
        if ans and ans[0].upper() in "ABCD":
            return ans[0].upper()
    m = re.search(r"\\boxed\{\s*([^}]*?)\s*\}", text)
    if m:
        ans = m.group(1).strip()
        if ans and ans[0].upper() in "ABCD":
            return ans[0].upper()
    text = re.sub(r"^(Answer:\s*|The answer is\s*)", "", text, flags=re.IGNORECASE)
    for letter in "ABCD":
        if f"{letter})" in text or f"{letter}." in text:
            return letter
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    return ""


def load_model(model_type, model_path):
    from transformers import AutoProcessor

    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    elif model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration

        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    processor = AutoProcessor.from_pretrained(model_path)
    return model.eval(), processor


def compose_prompt(item, reasoning=False):
    """Compose prompt from question + options."""
    question = item["question"]
    if item.get("options"):
        opts = "\n".join(item["options"])
        if reasoning:
            return (
                f"{question}\n{opts}\n"
                "Think step by step about the gaze and gesture interactions you observe. "
                "Use <gaze> and <gesture> tags to describe what you see, then select the "
                "best answer wrapped in <answer> tags, e.g. <answer>A</answer>."
            )
        return f"{question}\n{opts}\nAnswer with just the letter."
    return question


def _patch_flash_attn_for_mrope():
    """Handle Qwen3.5's 3D MROPE position_ids in flash attention."""
    import transformers.modeling_flash_attention_utils as fa_utils

    if getattr(fa_utils, "_mrope_patched", False):
        return
    orig = fa_utils._is_packed_sequence

    def _safe_is_packed(position_ids, batch_size):
        if position_ids is not None and position_ids.ndim == 3:
            return False
        return orig(position_ids, batch_size)

    fa_utils._is_packed_sequence = _safe_is_packed
    fa_utils._mrope_patched = True


def _generate_qwen35(model, inputs, max_new_tokens, eos_token_id):
    """Greedy generation for Qwen3.5 via the inner model forward."""
    _patch_flash_attn_for_mrope()

    input_ids = inputs["input_ids"]
    attn_mask = inputs.get("attention_mask")
    pixel_values_videos = inputs.get("pixel_values_videos")
    video_grid_thw = inputs.get("video_grid_thw")

    generated = []
    for _ in range(max_new_tokens):
        model.model.rope_deltas = None
        hidden_states = model.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            use_cache=False,
        )[0]
        logits = model.lm_head(hidden_states[:, -1:, :])
        next_token = logits.argmax(-1)
        if next_token.item() == eos_token_id:
            break
        generated.append(next_token)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=-1)
        pixel_values_videos = None
        video_grid_thw = None

    if not generated:
        return torch.zeros((1, 0), dtype=torch.long, device=input_ids.device)
    return torch.cat(generated, dim=-1)


@torch.inference_mode()
def generate(
    model,
    processor,
    model_type,
    video_path,
    prompt,
    max_new_tokens=64,
    max_frames=240,
    reasoning=False,
):
    frames, n_frames = load_video(video_path, max_frames)
    frame_indices = [int(i * (MOCK_FPS / VIDEO_FPS)) for i in range(n_frames)]
    frames = preprocess_frames_qwen(frames)

    messages = []
    if reasoning:
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Answer multiple-choice questions with only the option letter (A, B, C, or D) inside <answer> tags. Example: <answer>A</answer>",
                    }
                ],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ],
        }
    )

    template_kwargs = {}
    if model_type == "qwen3_5":
        template_kwargs["enable_thinking"] = reasoning

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        return_mm_token_type_ids=False,
        videos_kwargs={
            "do_sample_frames": False,
            "do_resize": False,
            "video_metadata": [
                {
                    "fps": MOCK_FPS,
                    "frames_indices": frame_indices,
                    "total_num_frames": n_frames,
                }
            ],
        },
        **template_kwargs,
    )
    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    if model_type == "qwen3_vl" and reasoning:
        think_ids = processor.tokenizer.encode("<think>\n", add_special_tokens=False)
        think_tensor = torch.tensor([think_ids], device=inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], think_tensor], dim=-1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    inputs["attention_mask"].new_ones((1, len(think_ids))),
                ],
                dim=-1,
            )

    if model_type == "qwen3_5" and not reasoning:
        eos_id = processor.tokenizer.eos_token_id
        out = _generate_qwen35(model, inputs, max_new_tokens, eos_id)
        return processor.batch_decode(out, skip_special_tokens=True)[0]

    if model_type == "qwen3_5":
        _patch_flash_attn_for_mrope()
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    if reasoning:
        stopper = StopOnAnswer(processor.tokenizer, inputs["input_ids"].shape[1])
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stopper])
    out = model.generate(**inputs, **gen_kwargs)
    out = out[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(out, skip_special_tokens=True)[0]


def load_eval_data(path):
    """Load GRASP-Bench items from <path>/json and <path>/video."""
    items = []
    json_dir = os.path.join(path, "json")
    video_dir = os.path.join(path, "video")
    for fp in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        with open(fp) as f:
            data = json.load(f)
        qa = data["qa"]
        qa["video_path"] = os.path.join(video_dir, data["clip_file"])
        items.append(qa)
    return items


def compute_metrics(results):
    import tempfile

    from eval.metric_report import evaluate as _report

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        tmp = f.name
    _report(tmp)
    os.remove(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["qwen3_vl", "qwen3_5"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_frames", type=int, default=240)
    parser.add_argument(
        "--reasoning", action="store_true", help="Enable thinking/reasoning mode"
    )
    args = parser.parse_args()

    model, processor = load_model(args.model_type, args.model_path)
    data = load_eval_data(args.data_path)

    def _key(it):
        return (
            it.get("video_path", ""),
            it.get("question", "")[:100],
            it.get("category", ""),
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = []
    done_keys = set()
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                results = json.load(f)
            for r in results:
                done_keys.add(_key(r))
            print(f"Resume: {len(results)} items already done, skipping")
        except (json.JSONDecodeError, ValueError):
            print("WARN: existing output corrupt, starting fresh")
            results = []
            done_keys = set()

    SAVE_EVERY = 10
    new_since_save = 0
    for item in tqdm(data, desc="Evaluating"):
        if _key(item) in done_keys:
            continue
        vp = item["video_path"]
        if not os.path.exists(vp):
            print(f"Skip: {vp}")
            continue

        prompt = compose_prompt(item, reasoning=args.reasoning)
        is_mcq = bool(item.get("options"))
        if args.reasoning:
            max_tokens = 2048
        else:
            max_tokens = 64 if is_mcq else 256

        output = generate(
            model,
            processor,
            args.model_type,
            vp,
            prompt,
            max_tokens,
            args.max_frames,
            reasoning=args.reasoning,
        )
        thinking, answer_text = extract_reasoning(output)
        pred = extract_answer(output) if is_mcq else answer_text
        result = {**item, "pred": pred, "raw_output": output, "reasoning": thinking}
        results.append(result)
        done_keys.add(_key(result))
        new_since_save += 1
        if new_since_save >= SAVE_EVERY:
            tmp = args.output + ".tmp"
            with open(tmp, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            os.replace(tmp, args.output)
            new_since_save = 0

    tmp = args.output + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, args.output)

    compute_metrics(results)
    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
