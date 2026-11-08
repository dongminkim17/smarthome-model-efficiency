"""Qwen2.5-Omni audiovisual classification for bounded feasibility pilots.

The processor cache stores CPU tensors, never task labels. Media are sampled
across their full duration, with the complete audio track kept in alignment.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid
from typing import Any, Mapping, Sequence

import av
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from .evaluation import normalize_modes, parse_generated_label
from .pruning import parameter_counts, processor_cache_identity


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
CACHE_VERSION = 1


def build_prompt(record: Mapping[str, Any], labels: Sequence[str], task: str) -> str:
    """Build a fixed closed-set prompt without reading the example's target label."""
    if not 2 <= len(labels) <= 26:
        raise ValueError("Classification requires 2 to 26 labels")
    is_intent = "mintrec" in task.lower() or "intent" in task.lower()
    if is_intent:
        instruction = (
            "Identify the speaker's communicative intent in this audiovisual clip. "
            "Use the spoken utterance, vocal delivery, and visible context."
        )
    else:
        instruction = (
            "Identify the speaker's emotion in this audiovisual clip. "
            "Use the spoken utterance, vocal delivery, and facial expression."
        )
    display_labels = {
        "emotions_attitudes": "Express emotions or attitudes (complain, praise, apologise, thank, criticize, care, agree, taunt, flaunt, joke, oppose)",
        "goals": "Achieve goals (comfort, inform, advise, arrange, introduce, leave, prevent, greet, ask for help)",
    }
    options = "\n".join(
        f"{chr(65 + i)}. {display_labels.get(label, label)}" for i, label in enumerate(labels)
    )
    transcript = record.get("transcript", record.get("text", ""))
    if is_intent and transcript:
        instruction += "\nUtterance transcript: " + str(transcript)
    return instruction + "\nAllowed labels:\n" + options + (
        "\nAnswer with exactly one option letter and no explanation."
    )


def build_input_prompt(record: Mapping[str, Any], labels: Sequence[str], task: str) -> str:
    """Extend the frozen historical prompt only for MER records containing text.

    Keeping build_prompt's source unchanged preserves existing AV/intent teacher
    cache hashes. The actual prompt text also keys processor caches.
    """
    prompt = build_prompt(record, labels, task)
    if task == "mer6" and record.get("transcript") is not None:
        text = record["transcript"]
        if not isinstance(text, str):
            raise ValueError("MER transcript must be a string")
        line = "Utterance transcript: " + (text if text else "[No transcript available]")
        prompt = prompt.replace("\nAllowed labels:\n", "\n" + line + "\nAllowed labels:\n", 1)
    return prompt


def trim_audio_padding(batch):
    """Drop only masked trailing audio frames; preserve every selected feature.

    Qwen selects features with feature_attention_mask before its audio encoder.
    Compact copies also avoid serializing a view's original padded storage.
    """
    if "input_features" not in batch or "feature_attention_mask" not in batch:
        return batch
    mask = batch["feature_attention_mask"]
    features = batch["input_features"]
    if mask.ndim != 2 or features.ndim != 3 or features.shape[0] != mask.shape[0] or features.shape[-1] != mask.shape[-1]:
        raise ValueError("Audio features and mask have incompatible shapes")
    positions = mask.bool().any(dim=0).nonzero(as_tuple=True)[0]
    if not len(positions):
        raise ValueError("Audio contains no unmasked frames")
    length = int(positions[-1]) + 1
    if length == mask.shape[-1]:
        return batch
    result = dict(batch)
    result["input_features"] = features[..., :length].clone(memory_format=torch.contiguous_format)
    result["feature_attention_mask"] = mask[..., :length].clone(memory_format=torch.contiguous_format)
    return result


def collate_prepared_batches(batches, pad_token_id):
    """Left-pad text/audio and concatenate variable-size visual patch sequences.

    Each cached input describes one audio/video clip. Concatenation preserves
    clip order for Qwen's multimodal token-to-feature assignment and RoPE.
    """
    if not batches:
        raise ValueError("Cannot collate an empty batch")
    keys = set(batches[0])
    known = {"input_ids", "attention_mask", "input_features", "feature_attention_mask",
             "pixel_values_videos", "video_grid_thw", "video_second_per_grid",
             "pixel_values", "image_grid_thw", "audio_feature_lengths"}
    if keys - known or any(set(b) != keys for b in batches):
        raise ValueError(f"Unsupported or inconsistent processor fields: {keys}")
    for b in batches:
        if b["input_ids"].shape[0] != 1 or not torch.all(b["attention_mask"] == 1):
            raise ValueError("Collation expects individual unpadded cached clips")
    result = {}
    for key in keys:
        tensors = [b[key] for b in batches]
        if key in ("input_ids", "attention_mask"):
            length = max(t.shape[-1] for t in tensors)
            fill = pad_token_id if key == "input_ids" else 0
            result[key] = torch.cat([torch.nn.functional.pad(t, (length-t.shape[-1], 0), value=fill)
                                     for t in tensors], dim=0)
        elif key in ("input_features", "feature_attention_mask"):
            # Audio feature padding is on the right; the feature mask tracks real frames.
            length = max(t.shape[-1] for t in tensors)
            result[key] = torch.cat([torch.nn.functional.pad(t, (0, length-t.shape[-1]))
                                     for t in tensors], dim=0)
        else:
            result[key] = torch.cat(tensors, dim=0)
    return result


class PilotModel:
    """Lazy-loaded BF16 Thinker with scoped LoRA and cached processor inputs."""

    def __init__(
        self,
        model_path: str | Path,
        task: str,
        labels: Sequence[str],
        cache_dir: str | Path,
        max_frames: int = 8,
        max_pixels: int = 224**2,
        use_transcript: bool = True,
        device_index: int = 0,
        gradient_checkpointing: bool = True,
    ) -> None:
        if not 2 <= max_frames <= 32 or max_frames % 2:
            raise ValueError("max_frames must be an even integer between 2 and 32")
        if max_pixels < 56**2:
            raise ValueError("max_pixels must be at least 56**2")
        if not 2 <= len(labels) <= 26 or len(set(labels)) != len(labels):
            raise ValueError("labels must contain 2 to 26 distinct strings")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is missing from PATH; activate the project conda environment")
        self.model_path = str(Path(model_path).resolve())
        if not Path(self.model_path).is_dir():
            raise ValueError("Use an already downloaded local model snapshot")
        if not isinstance(device_index, int) or device_index < 0:
            raise ValueError("device_index must be a nonnegative visible CUDA device index")
        self.device_index = device_index
        self.gradient_checkpointing = gradient_checkpointing
        self.task = task
        self.labels = list(labels)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames
        self.max_pixels = max_pixels
        self.use_transcript = use_transcript
        self.max_audio_seconds = 60.0 if task == "mer6" else 30.0
        self.versions = {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "peft", "qwen-omni-utils", "av", "torchvision", "librosa")
        }
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.label_token_ids = []
        for i in range(len(labels)):
            letter = chr(65 + i)
            tokens = self.processor.tokenizer.encode(letter, add_special_tokens=False)
            if len(tokens) != 1 or self.processor.tokenizer.decode(tokens) != letter:
                raise ValueError(f"Option {letter!r} is not exactly one tokenizer token: {tokens}")
            self.label_token_ids.append(tokens[0])
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        if self.eos_token_id is None:
            raise ValueError("Tokenizer has no EOS token")
        self.model = None
        self.parameters_info: dict[str, Any] = {}
        self.load_seconds = 0.0
        self.lora_enabled = False
        self.cache_hits = 0
        self.cache_misses = 0
        self.model_fingerprint = processor_cache_identity(Path(self.model_path))

    @staticmethod
    def _media_path(record: Mapping[str, Any]) -> Path:
        for key in ("video", "media_path", "media", "path"):
            if record.get(key):
                path = Path(record[key]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                return path
        raise ValueError("Record requires a local video path")

    def _check_media(self, path: Path) -> None:
        with av.open(str(path)) as container:
            if not container.streams.video or not container.streams.audio:
                raise ValueError(f"Both audio and video tracks are required: {path}")
            durations = []
            if container.duration is not None:
                durations.append(float(container.duration / av.time_base))
            for stream in [container.streams.video[0], container.streams.audio[0]]:
                if stream.duration is not None and stream.time_base is not None:
                    durations.append(float(stream.duration * stream.time_base))
            durations = [x for x in durations if math.isfinite(x) and x > 0]
            if not durations:
                raise ValueError(f"Cannot determine media duration before processing: {path}")
            if max(durations) > self.max_audio_seconds + 0.02:
                raise ValueError(
                    f"Clip duration {max(durations):.3f}s exceeds the pilot limit of "
                    f"{self.max_audio_seconds:g}s: {path}. The clip was not silently truncated."
                )

    def prepare(self, record: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Prepare/cache one whole clip; returned tensors stay on CPU and are mutable."""
        path = self._media_path(record)
        prompt_record = dict(record)
        if not self.use_transcript:
            prompt_record.pop("transcript", None)
            prompt_record.pop("text", None)
        prompt = build_input_prompt(prompt_record, self.labels, self.task)
        stat = path.stat()
        cache_spec = {
            "version": CACHE_VERSION,
            "model": self.model_fingerprint,
            "task": self.task,
            "labels": self.labels,
            "system_prompt": SYSTEM_PROMPT,
            "prompt": prompt,
            "media": str(path),
            "media_size": stat.st_size,
            "media_mtime_ns": stat.st_mtime_ns,
            "max_frames": self.max_frames,
            "fps": 2.0,
            "max_pixels": self.max_pixels,
            "min_pixels": 56**2,
            "max_audio_seconds": self.max_audio_seconds,
            "use_transcript": self.use_transcript,
            "versions": self.versions,
        }
        key = hashlib.sha256(json.dumps(cache_spec, sort_keys=True).encode()).hexdigest()
        cache_path = self.cache_dir / f"{key}.pt"
        if cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
            if payload.get("cache_spec") != cache_spec:
                raise RuntimeError(f"Cache metadata mismatch: {cache_path}")
            batch = trim_audio_padding(payload["batch"])
            if not all(isinstance(value, torch.Tensor) and value.device.type == "cpu" for value in batch.values()):
                raise RuntimeError(f"Cache contains non-CPU-tensor values: {cache_path}")
            self.cache_hits += 1
            return batch
        self._check_media(path)
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "video", "video": str(path), "fps": 2.0,
                 "min_frames": 2, "max_frames": self.max_frames,
                 "min_pixels": 56**2, "max_pixels": self.max_pixels},
                {"type": "text", "text": prompt},
            ]},
        ]
        # Explicitly avoid creating inference tensors, which cannot be saved for backward.
        with torch.inference_mode(False), torch.no_grad():
            audios, images, videos, video_kwargs = process_mm_info(
                conversation, use_audio_in_video=True, return_video_kwargs=True,
            )
            if not audios or not videos or len(audios) != 1 or len(videos) != 1:
                raise RuntimeError(f"Expected one native audio track and one video: {path}")
            sample_rate = self.processor.feature_extractor.sampling_rate
            if len(audios[0]) / sample_rate > self.max_audio_seconds + 0.02:
                raise ValueError(f"Decoded audio exceeds {self.max_audio_seconds:g}s: {path}")
            video_kwargs.pop("do_sample_frames", None)
            if isinstance(video_kwargs.get("fps"), list):
                if len(video_kwargs["fps"]) != 1:
                    raise RuntimeError("Expected a single video sampling rate")
                video_kwargs["fps"] = video_kwargs["fps"][0]
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
            )
            encoded = self.processor(
                text=text, audio=audios, images=images, videos=videos,
                use_audio_in_video=True, min_pixels=56**2, max_pixels=self.max_pixels,
                return_tensors="pt", padding=True, **video_kwargs,
            )
            batch = {name: value.detach().cpu().clone() for name, value in encoded.items()}
        if batch["input_ids"].shape[0] != 1 or not torch.all(batch["attention_mask"] == 1):
            raise RuntimeError("Single unpadded example expected")
        batch = trim_audio_padding(batch)
        tmp_path = cache_path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save({"cache_spec": cache_spec, "batch": batch}, tmp_path)
            os.replace(tmp_path, cache_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        self.cache_misses += 1
        return batch

    def load_model(self) -> None:
        """Load only the Thinker (language body, language head, and AV encoders)."""
        if self.model is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("A visible CUDA GPU is required for the pilot")
        if self.device_index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {self.device_index} is not visible")
        torch.cuda.set_device(self.device_index)
        started = time.perf_counter()
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": self.device_index}, local_files_only=True,
        )
        counts: dict[str, int] = defaultdict(int)
        for name, parameter in self.model.named_parameters():
            counts[name.split(".")[0]] += parameter.numel()
        self.parameters_info = {
            "components": dict(counts), "total": sum(counts.values()),
            "scope": "Thinker: language body/head plus audio and vision encoders; no speech generator",
        }
        self.parameters_info["language_parameters"] = parameter_counts(self.model)["language"]
        compact_manifest = Path(self.model_path) / "compression.json"
        if compact_manifest.is_file():
            manifest = json.loads(compact_manifest.read_text())
            actual = parameter_counts(self.model)
            if actual != manifest["parameters_after"]:
                raise RuntimeError("Compact checkpoint parameter counts disagree with its manifest")
            self.parameters_info["compression"] = {
                "method": manifest["method"], "source_checkpoint": manifest["source_checkpoint"],
                "parameters_before": manifest["parameters_before"], "parameters_after": actual,
                "manifest": str(compact_manifest),
            }
        self.load_seconds = time.perf_counter() - started
        self.model.config.use_cache = False
        self.model.eval()

    def _fresh_gpu_batch(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.load_model()
        return {
            key: value.to(device=self.model.device,
                          dtype=self.model.dtype if value.is_floating_point() else value.dtype,
                          copy=True)
            for key, value in batch.items()
        }

    def encode_media(self, batch):
        """Frozen native encoder outputs only; no LM forward or answer tokens."""
        gpu_batch = self._fresh_gpu_batch(batch)
        base = self.model.get_base_model() if self.lora_enabled else self.model
        base.eval()
        with torch.no_grad():
            audio = base.get_audio_features(gpu_batch['input_features'],
                                           feature_attention_mask=gpu_batch['feature_attention_mask'])
            vision = base.get_video_features(gpu_batch['pixel_values_videos'], gpu_batch['video_grid_thw'])
        from .feature_distillation import validate_features
        features = {'audio': audio.detach().cpu().to(torch.bfloat16),
                    'vision': vision.detach().cpu().to(torch.bfloat16)}
        validate_features(features)
        return features

    def _reset_positions(self) -> None:
        base = self.model.get_base_model() if self.lora_enabled else self.model
        base.rope_deltas = None

    def score(self, batch: Mapping[str, torch.Tensor]) -> list[float]:
        """Return next-token logits for the allowed letters in the supplied label order."""
        gpu_batch = self._fresh_gpu_batch(batch)
        self.model.eval()
        self._reset_positions()
        with torch.no_grad():
            output = self.model(**gpu_batch, use_audio_in_video=True, use_cache=False)
            scores = output.logits[0, -1, self.label_token_ids].float().cpu().tolist()
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError("Nonfinite classification scores")
        return scores

    def _configure_adapter_training(self) -> None:
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        else:
            self.model.gradient_checkpointing_disable()
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        if any(p.requires_grad and "lora_" not in name for name, p in self.model.named_parameters()):
            raise RuntimeError("Unexpected trainable base-model parameter")

    def enable_lora(
        self, rank: int = 16, alpha: int | None = None, dropout: float = 0.0,
        target_modules: str | Sequence[str] | None = None,
    ) -> None:
        """Train configurable LoRA adapters; the default targets decoder q/v only."""
        if not isinstance(rank, int) or rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        alpha = 2 * rank if alpha is None else alpha
        if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be positive and finite")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.load_model()
        if self.lora_enabled:
            raise RuntimeError("LoRA has already been enabled")
        targets = target_modules if target_modules is not None else r"model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
        if not targets:
            raise ValueError("LoRA target modules must not be empty")
        if not isinstance(targets, str):
            targets = list(targets)
        self.model = get_peft_model(self.model, LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets,
            task_type=TaskType.CAUSAL_LM, bias="none",
        ))
        self.lora_enabled = True
        self._configure_adapter_training()
        self.parameters_info.update({
            "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "lora_rank": rank, "lora_alpha": alpha, "lora_dropout": dropout,
            "lora_target_modules": targets,
        })

    def load_adapter(self, path: str | Path, trainable: bool = True) -> None:
        """Reload a local adapter into a fresh base; retain the unwrapped PEFT model."""
        path = Path(path).resolve()
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(path / "adapter_config.json")
        self.load_model()
        if self.lora_enabled:
            raise RuntimeError("Load adapters into a fresh PilotModel")
        self.model = PeftModel.from_pretrained(
            self.model, str(path), is_trainable=trainable, local_files_only=True,
        )
        self.lora_enabled = True
        if trainable:
            self._configure_adapter_training()
        else:
            self.model.eval()
        config = self.model.peft_config["default"]
        targets = config.target_modules
        self.parameters_info.update({
            "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "lora_rank": config.r, "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_target_modules": targets if isinstance(targets, str) else sorted(targets),
            "adapter_path": str(path),
        })

    def loss_batch(self, batch, labelindex, forward_model=None, train=True, return_class_gradients=False, return_encoder_features=False, return_per_example_sft=False):
        """Original single-example API, also used by validation."""
        if batch["input_ids"].shape[0] != 1 or not torch.all(batch["attention_mask"] == 1):
            raise ValueError("loss_batch expects one unpadded example")
        result = self._loss_with_targets(batch, [labelindex], forward_model, train, return_class_gradients, return_encoder_features, return_per_example_sft)
        result["class_logits"] = result["class_logits"][0]
        return result

    def loss_microbatch(self, batches, labelindices, forward_model=None, train=True, return_class_gradients=False, return_encoder_features=False, return_per_example_sft=False):
        """Average class/EOS SFT loss over a batch of variable-length clips."""
        if len(batches) != len(labelindices):
            raise ValueError("One target label is required per clip")
        batch = collate_prepared_batches(batches, self.processor.tokenizer.pad_token_id)
        return self._loss_with_targets(batch, labelindices, forward_model, train, return_class_gradients, return_encoder_features, return_per_example_sft)

    def _loss_with_targets(self, batch, labelindices, forward_model, train, return_class_gradients=False, return_encoder_features=False, return_per_example_sft=False):
        if return_per_example_sft and not train:
            raise ValueError('Per-example recovery SFT losses are training-only')
        if train and not self.lora_enabled:
            raise RuntimeError("Enable or load a trainable adapter before training")
        if not labelindices or any(not 0 <= i < len(self.labels) for i in labelindices):
            raise ValueError(f"Invalid label indices: {labelindices}")
        gpu_batch = self._fresh_gpu_batch(batch)
        prefix_length = gpu_batch["input_ids"].shape[1]
        if gpu_batch["input_ids"].shape[0] != len(labelindices):
            raise ValueError("Target count must match the batch size")
        target = torch.tensor([[self.label_token_ids[i], self.eos_token_id] for i in labelindices],
                              device=self.model.device, dtype=gpu_batch["input_ids"].dtype)
        gpu_batch["input_ids"] = torch.cat([gpu_batch["input_ids"], target], dim=1)
        gpu_batch["attention_mask"] = torch.cat([
            gpu_batch["attention_mask"], torch.ones_like(target),
        ], dim=1)
        targets = gpu_batch["input_ids"].clone()
        targets[:, :prefix_length] = -100
        model = self.model if forward_model is None else forward_model
        model.train(train)
        self._reset_positions()
        from contextlib import nullcontext
        capture = None
        if return_encoder_features:
            from .feature_distillation import EncoderFeatureCapture
            capture = EncoderFeatureCapture(self.model)
        with torch.set_grad_enabled(train):
            with capture if capture is not None else nullcontext():
                output = model(**gpu_batch, labels=targets, use_audio_in_video=True, use_cache=False)
            loss = output.loss
            # Opt-in branch: keep the historical scalar SFT path exactly intact.
            # Each clip has exactly two supervised tokens, letter and EOS.
            sft_per_example = None
            if return_per_example_sft:
                letter = torch.nn.functional.cross_entropy(
                    output.logits[:, prefix_length - 1, :].float(), target[:, 0], reduction='none',
                )
                eos = torch.nn.functional.cross_entropy(
                    output.logits[:, prefix_length, :].float(), target[:, 1], reduction='none',
                )
                sft_per_example = (letter + eos) / 2
            kd_logits = output.logits[:, prefix_length - 1, self.label_token_ids].float() if return_class_gradients else None
            with torch.no_grad():
                class_loss = torch.nn.functional.cross_entropy(
                    output.logits[:, prefix_length - 1, :].float(), target[:, 0],
                )
                eos_loss = torch.nn.functional.cross_entropy(
                    output.logits[:, prefix_length, :].float(), target[:, 1],
                )
                class_logits = output.logits[:, prefix_length - 1, self.label_token_ids].float().detach()
        if not all(torch.isfinite(value).all().item() for value in (loss, class_loss, eos_loss, class_logits)):
            raise RuntimeError("Nonfinite loss or classification logits")
        result = {"loss": loss, "sft_loss": loss, "class_loss": class_loss,
                  "eos_loss": eos_loss, "class_logits": class_logits}
        if return_encoder_features:
            result['encoder_features_for_kd'] = capture.result()
        if return_class_gradients:
            result["class_logits_for_kd"] = kd_logits
        if return_per_example_sft:
            result["sft_per_example_for_kd"] = sft_per_example
        return result

    def train_batch(
        self, batch: Mapping[str, torch.Tensor], labelindex: int, forward_model=None,
    ) -> torch.Tensor:
        """Return differentiable SFT loss; caller owns backward and optimizer."""
        return self.loss_batch(batch, labelindex, forward_model=forward_model, train=True)["loss"]

    def evaluate_batch(
        self, batch: Mapping[str, torch.Tensor], labelindex: int,
        modes: str | Sequence[str] = "both", max_new_tokens: int = 8,
    ) -> dict[str, Any]:
        """Compute loss diagnostics and configured closed-set/generative predictions.

        Logit scoring uses the final prompt position of the teacher-forced
        forward, before the gold answer can be attended to under the causal mask.
        Greedy generation always receives the original prompt-only batch and has
        no valid-answer token restriction or fallback to the logit prediction.
        """
        modes = normalize_modes(modes)
        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be between 1 and 128")
        losses = self.loss_batch(batch, labelindex, train=False)
        result: dict[str, Any] = {
            name: float(losses[name].item()) for name in ("sft_loss", "class_loss", "eos_loss")
        }
        if "logit" in modes:
            scores = losses["class_logits"].cpu().tolist()
            result["logit_scores"] = scores
            result["logit_predicted_label"] = self.labels[max(range(len(scores)), key=scores.__getitem__)]
        del losses
        if "greedy" in modes:
            gpu_batch = self._fresh_gpu_batch(batch)
            prefix_length = gpu_batch["input_ids"].shape[1]
            self.model.eval()
            self._reset_positions()
            with torch.no_grad():
                generated = self.model.generate(
                    **gpu_batch, use_audio_in_video=True, use_cache=True,
                    max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    return_dict_in_generate=False,
                )
            new_ids = generated[0, prefix_length:].detach().cpu().tolist()
            text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True)
            parsed = parse_generated_label(text, self.labels)
            result.update({
                "generated_text": text, "generated_token_ids": new_ids,
                "greedy_predicted_label": parsed["predicted_label"],
                "invalid_reason": parsed["invalid_reason"],
                "generation_reached_eos": self.eos_token_id in new_ids,
            })
        return result

