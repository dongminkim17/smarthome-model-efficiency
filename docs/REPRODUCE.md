# Reproduce the final pipeline

Run commands from the repository root in the environment specified by `environment.yml`. Training and model-export commands require CUDA. Use available GPUs through `CUDA_VISIBLE_DEVICES`.

## Prepare model and data

The reported experiments used the following pretrained snapshot.

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Omni-3B', revision='f75b40e3da2003cdd6e1829b1f420ca70797c34e', local_dir='models/Qwen2.5-Omni-3B')"
```

Alternatively, link an existing snapshot into `models/Qwen2.5-Omni-3B`.

```sh
mkdir -p data
python scripts/download_mintrec.py --output data/MIntRec
```

For MER2023, obtain the dataset through the official access process and place the supplied archives, access README and preparation inputs under the MER2023 data directory. The preparation tool reads the supplied archive credentials locally. The full-archive preparation command expects this layout.

```text
data/MER2023/archives/
  README_AFTER_APPROVAL.md
  test-labels.zip
  mer2023train.z01
  mer2023train.z02
  mer2023train.z03
  mer2023train.z04
  mer2023train.z05
  mer2023train.z06
  mer2023train.zip
```

Use the access README supplied after approval as `README_AFTER_APPROVAL.md`. Archive credentials stay in the local dataset directory.

```sh
python scripts/prepare_mer2023.py --data-root data/MER2023 --extract-media
```

Existing prepared datasets can be linked as `data/MIntRec` and `data/MER2023`. The data loader expects MIntRec split TSVs and raw MP4 clips, and MER's `manifests/trainval.jsonl` and `manifests/semi_test.jsonl` with the corresponding media. Check the preparation tool's `--help` for selective-download options.

## Choose a task

Use `mintrec2`, `mintrec20` or `mer6`. The matching configuration files already specify each task's input modalities and validation split.

```sh
TASK=mintrec2
RUN=artifacts/reproduce/$TASK
mkdir -p "$RUN"
```

## Train the task teacher

The teacher config uses microbatch one and one accumulation step. Four GPU ranks give effective batch size four.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train.py --config configs/$TASK.yaml --output "$RUN/teacher"
```

## Prune and cache training targets

The following commands each use a single available GPU. They use the validation-selected teacher checkpoint and training samples.

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/prune_model.py \
  --source-checkpoint "$RUN/teacher/checkpoints/best" \
  --method ffn_activation --ffn-width 4352 --calibration-samples 128 \
  --calibration-seed 42 --calibration-cache "$RUN/calibration" \
  --output "$RUN/pruned"

CUDA_VISIBLE_DEVICES=0 python scripts/cache_teacher_logits.py \
  --teacher-checkpoint "$RUN/teacher/checkpoints/best" \
  --output "$RUN/teacher_targets/classes"

CUDA_VISIBLE_DEVICES=0 python scripts/cache_encoder_features.py \
  --teacher-checkpoint "$RUN/teacher/checkpoints/best" \
  --output "$RUN/teacher_targets/features"
```

## Recover the pruned language model

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/reproduce/$TASK/lm_recovery.yaml \
  --output "$RUN/lm_recovery"

CUDA_VISIBLE_DEVICES=0 python scripts/export_recovered_model.py \
  --source-checkpoint "$RUN/lm_recovery/checkpoints/best" \
  --output "$RUN/recovered_lm"
```

## Jointly recover the language model and encoders

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/reproduce/$TASK/final_recovery.yaml \
  --output "$RUN/final"
```

Both recovery configs use microbatch two with two accumulation steps. If memory requires microbatch one, set accumulation to four to preserve the effective batch size.

## Official test evaluation

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --checkpoint "$RUN/final/checkpoints/best" \
  --split test --modes both --output "$RUN/test"
```

Each evaluation output directory must be new. The trainer saves validation metrics, predictions, losses, timing information and PNG/SVG curves. The evaluator saves a report and full official-test predictions.

## Resume

For interrupted training, reuse the stage's config and output directory and add `--resume <stage-output>/checkpoints/latest`. Teacher resume uses the original four-rank topology and recovery resume uses one rank. Cache creation accepts `--resume` to reuse validated targets or complete a partial cache.

Input preprocessing caches are regenerated on demand. Teacher class and encoder-feature targets are validated against the task, teacher and training samples before use.
