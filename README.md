# Smart-home model efficiency

Structured pruning and knowledge distillation for Qwen2.5-Omni-3B on multimodal intent and emotion recognition.

The final method uses activation-based FFN channel pruning and two recovery stages. All three tasks use the same architecture and numerical final-recovery settings, with separate task-specific teachers and weights.

## Final results

| Task | Inputs | Teacher | Final compact model | Rank in selected comparison cohort |
| --- | --- | --- | --- | --- |
| MIntRec 2-class | Audio, video, provided transcript | 93.93% accuracy | **92.13% accuracy** | 1 / 15 |
| MIntRec 20-class | Audio, video, provided transcript | 78.88% accuracy | **78.43% accuracy** | 1 / 19 |
| MER2023-SEMI | Audio, video | 86.53% weighted F1 | **88.63% weighted F1** | 4 / 17 |

Scores use validation-selected training checkpoints on complete official test sets. Model cards report separate checks of the merged release weights. Rankings use the published comparison configurations in [MIntRec](https://arxiv.org/abs/2209.04355), [ARL](https://proceedings.neurips.cc/paper_files/paper/2025/file/cbed27fa7f3b112caebf3ba6e8b3cd8c-Paper-Conference.pdf) and [Emotion-LLaMA](https://arxiv.org/abs/2406.11161). The comparison scope is described in [the method document](docs/METHOD.md).

| Component | Original parameters | Final parameters |
| --- | ---: | ---: |
| Language model, including embeddings and output head | 3,397,103,616 | **1,924,902,912** |
| Audio encoder | 637,676,544 | 637,676,544 |
| Vision encoder | 668,684,288 | 668,684,288 |

Counts cover the understanding components used for classification. Speech-generation modules Talker and Token2Wav are outside this pipeline. The final language model meets the 2B boundary and each encoder meets the 1B boundary.

## Setup

```sh
git clone https://github.com/dongminkim17/smarthome-model-efficiency.git
cd smarthome-model-efficiency
conda env create -f environment.yml
conda activate smarthome_efficiency
```

Run the commands below from the repository root. They use the selected task's YAML configs and write outputs under `artifacts/reproduce/<task>/`.

### Prepare model and datasets

Download the pretrained snapshot used in the experiments.

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Omni-3B', revision='f75b40e3da2003cdd6e1829b1f420ca70797c34e', local_dir='models/Qwen2.5-Omni-3B')"
```

Prepare MIntRec for either intent task.

```sh
mkdir -p data
python scripts/download_mintrec.py --output data/MIntRec
```

For MER2023-SEMI, obtain official dataset access and arrange the supplied archives as shown in [the MER preparation instructions](docs/REPRODUCE.md#prepare-model-and-data), then run the preparation command.

```sh
python scripts/prepare_mer2023.py --data-root data/MER2023 --extract-media
```

Existing model and prepared dataset directories can be linked as `models/Qwen2.5-Omni-3B`, `data/MIntRec` and `data/MER2023`.

## Download and evaluate the released models

Each Hugging Face repository contains a complete merged model, its processor/tokenizer, task settings, license and verified test metrics.

| Task | Hugging Face model | Merged release test result |
| --- | --- | --- |
| MIntRec 2-class | [smarthome-omni-mintrec2](https://huggingface.co/dongminkim17/smarthome-omni-mintrec2) | 92.13% accuracy |
| MIntRec 20-class | [smarthome-omni-mintrec20](https://huggingface.co/dongminkim17/smarthome-omni-mintrec20) | 78.65% accuracy |
| MER2023-SEMI | [smarthome-omni-mer2023-semi](https://huggingface.co/dongminkim17/smarthome-omni-mer2023-semi) | 88.14% weighted F1 |

After preparing the matching dataset, choose a released model and evaluate it directly. The evaluator downloads the Hub snapshot and uses the saved task prompts, label order and preprocessing settings.

```sh
MODEL=dongminkim17/smarthome-omni-mintrec2

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model "$MODEL" --data-root data \
  --split test --modes both \
  --output artifacts/evaluation/mintrec2_released
```

Use a new output directory for each evaluation. `--revision <commit>` pins a Hub snapshot. `--model /path/to/downloaded/model` evaluates a local release. A native Transformers loading example is included in each model card.

## Train and evaluate the original model

Choose a task. Use `mintrec2` for binary intent, `mintrec20` for fine-grained intent or `mer6` for emotion recognition.

```sh
TASK=mintrec2
RUN=artifacts/reproduce/$TASK
mkdir -p "$RUN"
```

Keep this output layout when using the supplied recovery configs, which refer to these paths. Select available GPUs through `CUDA_VISIBLE_DEVICES`.

### Train the task teacher

The teacher configs train for five epochs with rank-16 LoRA. This command uses four GPUs with effective batch size four.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train.py \
  --config configs/$TASK.yaml \
  --output "$RUN/teacher"
```

For single-GPU teacher training, set `micro_batch_size: 1` and `gradient_accumulation_steps: 4` in the teacher YAML and run the same script with `CUDA_VISIBLE_DEVICES=0 python` in place of `torchrun --standalone --nproc_per_node=4`.

### Evaluate the teacher on the official test set

The training run selects its best checkpoint using validation data. Evaluate that checkpoint to obtain the unpruned baseline result.

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --checkpoint "$RUN/teacher/checkpoints/best" \
  --split test --modes both --output "$RUN/teacher_test"
```

## Reproduce the compact-model results

Continue with the same `TASK` and `RUN` after teacher training. The following stages each use one GPU. Both recovery stages train for five epochs.

### 1. Prune the language model

Merge the teacher adapter and prune the decoder FFN width from 11,008 to 4,352 using 128 training clips for activation calibration.

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/prune_model.py \
  --source-checkpoint "$RUN/teacher/checkpoints/best" \
  --method ffn_activation --ffn-width 4352 \
  --calibration-samples 128 --calibration-seed 42 \
  --calibration-cache "$RUN/calibration" \
  --output "$RUN/pruned"
```

### 2. Cache teacher targets

Compute class logits and encoder features on training samples for the two recovery stages.

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/cache_teacher_logits.py \
  --teacher-checkpoint "$RUN/teacher/checkpoints/best" \
  --output "$RUN/teacher_targets/classes"

CUDA_VISIBLE_DEVICES=0 python scripts/cache_encoder_features.py \
  --teacher-checkpoint "$RUN/teacher/checkpoints/best" \
  --output "$RUN/teacher_targets/features"
```

### 3. Recover the pruned language model

Apply rank-16 decoder LoRA with SFT and class distillation, then merge the validation-selected adapter into the pruned base.

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/reproduce/$TASK/lm_recovery.yaml \
  --output "$RUN/lm_recovery"

CUDA_VISIBLE_DEVICES=0 python scripts/export_recovered_model.py \
  --source-checkpoint "$RUN/lm_recovery/checkpoints/best" \
  --output "$RUN/recovered_lm"
```

### 4. Jointly recover the language model and encoders

Apply rank-64 LoRA with SFT, class distillation and audio/vision feature distillation.

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/reproduce/$TASK/final_recovery.yaml \
  --output "$RUN/final"
```

The recovery configs use microbatch two with two accumulation steps. For a lower-memory setting, use microbatch one with four accumulation steps to preserve effective batch size four.

### 5. Evaluate the final model on the official test set

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --checkpoint "$RUN/final/checkpoints/best" \
  --split test --modes both --output "$RUN/test"
```

To evaluate the intermediate LM recovery, use `--checkpoint "$RUN/lm_recovery/checkpoints/best"` and a separate output directory such as `"$RUN/lm_recovery_test"`.

Repeat the workflow with `TASK=mintrec20` and `TASK=mer6` for the other two settings. [METHOD.md](docs/METHOD.md) describes the data splits, architecture and losses, and [REPRODUCE.md](docs/REPRODUCE.md) provides additional preparation details.

## Evaluate an existing checkpoint

A saved checkpoint can also be evaluated independently. Set `CHECKPOINT` to an existing checkpoint directory containing `metadata.json` and `adapter/`.

```sh
CHECKPOINT=artifacts/reproduce/mintrec2/final/checkpoints/best

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --checkpoint "$CHECKPOINT" \
  --split test --modes both \
  --output artifacts/evaluation/mintrec2_final
```

Use `--modes logit`, `--modes greedy` or `--modes both`. Use `--split dev` to evaluate validation data. Each evaluation output directory must be new, and the base model referenced by the saved checkpoint must be available locally.

## Resume training and inspect outputs

Resume an interrupted final-recovery run with its original config and output directory.

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/reproduce/$TASK/final_recovery.yaml \
  --output "$RUN/final" \
  --resume "$RUN/final/checkpoints/latest"
```

Training outputs include `metrics.jsonl`, `scalars.csv`, `curves.png`, `curves.svg` and best/latest checkpoint pointers. Official evaluation writes `report.json` and `predictions.jsonl` in the requested output directory. Regenerate a run's curves with the plotting command.

```sh
python scripts/plot_training.py "$RUN/final"
```

## Main entry points

| Script | Purpose |
| --- | --- |
| `scripts/train.py` | SFT or SFT plus distillation, validation, resumable checkpoints and local curves |
| `scripts/prune_model.py` | Merge a teacher adapter and export a structurally pruned model |
| `scripts/cache_teacher_logits.py` | Cache teacher class logits on training samples |
| `scripts/cache_encoder_features.py` | Cache teacher audio and vision output features |
| `scripts/export_recovered_model.py` | Merge the first recovery into its pruned base |
| `scripts/export_release_model.py` | Merge and verify the final decoder/encoder adapters for distribution |
| `scripts/evaluate.py` | Evaluate a training checkpoint or released Hub/local model on dev or official test |
| `scripts/plot_training.py` | Generate training and validation plots from saved metrics |

The repository also contains experimental pruning variants and scheduler scripts used in the wider study. The reproduction instructions identify the entry points for the chosen method.

## Configuration and outputs

- `configs/{mintrec2,mintrec20,mer6}.yaml` configure the original task teachers.
- `configs/reproduce/<task>/` configure both compact-model recovery stages.
- `configs/mer6_avt.yaml` is an optional text-input experiment with a separate data root.
- `eval_modes` enables `logit`, `greedy` or both. `selection_mode` chooses the validation metric used for checkpoint selection.
- Runs save JSONL/CSV metrics, predictions, PNG/SVG curves, and best/latest checkpoint pointers under their output directory.
- Local model weights, datasets, experiment outputs and internal project records are covered by `.gitignore`.

For a full optimizer/RNG resume, reuse the original config and output directory with `--resume <run>/checkpoints/latest`. Use `--init-adapter <checkpoint>/adapter` for a fresh optimizer and schedule starting from existing adapter weights.

## License

Original source code and documentation in this repository are licensed under the [MIT License](LICENSE).

## Upstream model and attribution

Built with Qwen. The pretrained Qwen2.5-Omni-3B snapshot uses the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/LICENSE). Its model-use and redistribution terms include a non-commercial scope and upstream notice requirements. Distribution of adapted weights must retain the applicable upstream terms and notices.

Pruning references include [NVIDIA Model Optimizer / Minitron](https://github.com/NVIDIA/Model-Optimizer) and [ShortGPT](https://github.com/icip-cas/ShortGPT). Dataset files are obtained through their providers and retain the providers' access and distribution terms.
