from typing import List

import torch
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
)

from socialmllm.utils.input_processing import (
    ImageSpec,
    InputSpec,
    TextSpec,
    VideoSpec,
    distribute_frames,
    extract_bounded_spans,
    resolve_resolution,
)
from socialmllm.utils.train_utils import FusedNLL


class Qwen3_5ForSFT(Qwen3_5ForConditionalGeneration):
    accepts_loss_kwargs = False

    @property
    def vision_parameters(self):
        exclude = set(self.projection_parameters)
        for p in self.model.visual.parameters():
            if p not in exclude:
                yield p

    @property
    def projection_parameters(self):
        yield from self.model.visual.merger.parameters()

    @property
    def language_parameters(self):
        yield from self.model.language_model.parameters()
        yield from self.lm_head.parameters()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        if labels is not None:
            shift_labels = labels[..., 1:]
            valid_mask = shift_labels != -100
            valid_labels = shift_labels[valid_mask]

            valid_hidden_states = hidden_states[..., :-1, :][valid_mask]
            valid_logits = self.lm_head(valid_hidden_states)

            loss = torch.nn.functional.cross_entropy(valid_logits, valid_labels)
            logits = None
        else:
            loss = None
            logits = self.lm_head(hidden_states)

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    @classmethod
    def preprocess_input_spec(
        cls, stream: List[InputSpec], processor, data_args, **kwargs
    ):
        videospecs = []
        for item in stream:
            if isinstance(item, VideoSpec):
                videospecs.append(item)
            elif isinstance(item, ImageSpec):
                raise NotImplementedError()
            elif isinstance(item, TextSpec):
                item.num_tokens = len(processor.tokenizer(item.content).input_ids)

        for videospec, n_frame in zip(
            videospecs,
            distribute_frames(
                videospecs,
                min_frames_per_clip=data_args.video_min_frames_per_clip,
                frame_multiple=data_args.video_frame_multiple,
                max_total_frames=data_args.video_max_total_frames,
                max_fps=data_args.video_max_fps,
            ),
        ):
            videospec.num_frames = n_frame
            resolution, num_tokens = resolve_resolution(
                videospec,
                frame_time_patch=data_args.frame_time_patch,
                frame_spatial_patch=data_args.frame_spatial_patch,
                frame_max_tokens=data_args.frame_max_tokens,
            )
            videospec.dst_resolution = resolution
            videospec.num_tokens = num_tokens
        return stream

    @classmethod
    def apply_chat_template(
        cls, batch_stream: List[List[InputSpec]], processor, **kwargs
    ):
        def add_content(messages, role, content):
            if len(messages) > 0 and messages[-1]["role"] == role:
                messages[-1]["content"].append(content)
            else:
                messages.append(dict(role=role, content=[content]))

        MOCK_FPS = 100
        video_metadatas = []
        messages = []
        for stream in batch_stream:
            messages.append([])
            for spec in stream:
                if isinstance(spec, TextSpec):
                    add_content(
                        messages[-1],
                        role=["user", "assistant"][spec.output],
                        content=dict(type="text", text=spec.content),
                    )
                elif isinstance(spec, VideoSpec):
                    assert spec.content_time is not None
                    add_content(
                        messages[-1],
                        role="user",
                        content=dict(type="video", video=spec.content),
                    )
                    indices = (spec.content_time * MOCK_FPS).tolist()
                    video_metadatas.append(
                        {
                            "fps": MOCK_FPS,
                            "frames_indices": indices,
                            "total_num_frames": len(indices),
                        }
                    )
                else:
                    raise ValueError(f"Unsupported spec type: {type(spec)}")

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            enable_thinking=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            videos_kwargs={
                "do_sample_frames": False,
                "do_resize": False,
                "video_metadata": video_metadatas,
            },
        )
        inputs["labels"] = extract_bounded_spans(
            inputs["input_ids"],
            start_seq=(248045, 74455, 198),
            end_seq=(248046, 198),
            start_offset=3,
            end_offset=1,
            fill_value=-100,
        )
        return inputs


class Qwen3_5ForReasoningQA(Qwen3_5ForSFT):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        return_per_token_nll=False,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        if labels is not None:
            shift_labels = labels[..., 1:]
            valid_mask = shift_labels != -100
            valid_labels = shift_labels[valid_mask]

            valid_hidden_states = hidden_states[..., :-1, :][valid_mask]
            valid_nll = FusedNLL.apply(
                valid_hidden_states,
                valid_labels,
                self.lm_head.weight,
                self.lm_head.bias,
            )
            valid_batch_indices = valid_mask.nonzero()[:, 0]
            batched_nll = valid_nll.new_zeros(hidden_states.size(0))
            batched_nll.index_add_(0, valid_batch_indices, valid_nll)
            if return_per_token_nll:
                return batched_nll, valid_nll, valid_batch_indices
            return batched_nll
        else:
            loss = None
            logits = self.lm_head(hidden_states)

            return Qwen3_5CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                rope_deltas=outputs.rope_deltas,
            )

    @classmethod
    def apply_chat_template(
        cls, batch_stream: List[List[InputSpec]], processor, **kwargs
    ):
        def add_content(messages, role, content):
            if len(messages) > 0 and messages[-1]["role"] == role:
                messages[-1]["content"].append(content)
            else:
                messages.append(dict(role=role, content=[content]))

        MOCK_FPS = 100
        video_metadatas = []
        messages = []
        target_words = []
        for stream in batch_stream:
            messages.append([])
            for spec in stream:
                if isinstance(spec, TextSpec):
                    add_content(
                        messages[-1],
                        role=["user", "assistant"][spec.output],
                        content=dict(type="text", text=spec.content),
                    )
                elif isinstance(spec, VideoSpec):
                    assert spec.content_time is not None
                    add_content(
                        messages[-1],
                        role="user",
                        content=dict(type="video", video=spec.content),
                    )
                    indices = (spec.content_time * MOCK_FPS).tolist()
                    video_metadatas.append(
                        {
                            "fps": MOCK_FPS,
                            "frames_indices": indices,
                            "total_num_frames": len(indices),
                        }
                    )
                else:
                    raise ValueError(f"Unsupported spec type: {type(spec)}")

            last = messages[-1].pop(-1)
            assert last["role"] == "assistant"
            assert len(last["content"]) == 1
            target_words.append(last["content"][0]["text"].strip())

            messages[-1].insert(
                0,
                dict(
                    role="system",
                    content=[
                        dict(
                            type="text",
                            text="Answer multiple-choice questions with only the option letter (A, B, C, or D) inside <answer> tags. Example: <answer>A</answer>",
                        )
                    ],
                ),
            )

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            videos_kwargs={
                "do_sample_frames": False,
                "do_resize": False,
                "video_metadata": video_metadatas,
            },
        )
        inputs["target_words"] = target_words
        return inputs
