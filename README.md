# Smart-home model efficiency

Structured pruning and knowledge distillation for Qwen2.5-Omni-3B on multimodal intent and emotion recognition.

The final method uses activation-based FFN channel pruning and two recovery stages. All three tasks use the same architecture and numerical final-recovery settings, with separate task-specific teachers and weights.

## Final results

| Task | Inputs | Teacher | Final compact model | Rank in selected comparison cohort |
| --- | --- | --- | --- | --- |
| MIntRec 2-class | Audio, video, provided transcript | 93.93% accuracy | **92.13% accuracy** | 1 / 15 |
| MIntRec 20-class | Audio, video, provided transcript | 78.88% accuracy | **78.43% accuracy** | 1 / 19 |
| MER2023-SEMI | Audio, video | 86.53% weighted F1 | **88.63% weighted F1** | 4 / 17 |

Scores use complete official test sets and validation-selected checkpoints. Rankings use the published comparison configurations in [MIntRec](https://arxiv.org/abs/2209.04355), [ARL](https://proceedings.neurips.cc/paper_files/paper/2025/file/cbed27fa7f3b112caebf3ba6e8b3cd8c-Paper-Conference.pdf) and [Emotion-LLaMA](https://arxiv.org/abs/2406.11161). The comparison scope is described in [the method document](docs/METHOD.md).

| Component | Original parameters | Final parameters |
| --- | ---: | ---: |
| Language model, including embeddings and output head | 3,397,103,616 | **1,924,902,912** |
| Audio encoder | 637,676,544 | 637,676,544 |
| Vision encoder | 668,684,288 | 668,684,288 |

Counts cover the understanding components used for classification. Speech-generation modules Talker and Token2Wav are outside this pipeline. The final language model meets the 2B boundary and each encoder meets the 1B boundary.

## Setup and reproduction

```sh
git clone https://github.com/dongminkim17/smarthome-model-efficiency.git
cd smarthome-model-efficiency
conda env create -f environment.yml
conda activate smarthome_efficiency
```

Place a local Qwen2.5-Omni-3B snapshot at `models/Qwen2.5-Omni-3B`, and prepare the datasets under `data/`. Existing model and dataset directories can be linked into these locations.

Follow [REPRODUCE.md](docs/REPRODUCE.md) for data preparation and the full teacher, pruning, recovery and official-test commands. The commands use the repository root as the working directory. [METHOD.md](docs/METHOD.md) records the architecture, losses, splits and shared training recipe.

## Main entry points

| Script | Purpose |
| --- | --- |
| `scripts/train.py` | SFT or SFT plus distillation, validation, resumable checkpoints and local curves |
| `scripts/prune_model.py` | Merge a teacher adapter and export a structurally pruned model |
| `scripts/cache_teacher_logits.py` | Cache teacher class logits on training samples |
| `scripts/cache_encoder_features.py` | Cache teacher audio and vision output features |
| `scripts/export_recovered_model.py` | Merge the first recovery into its pruned base |
| `scripts/evaluate.py` | Evaluate a saved validation-selected checkpoint on dev or official test |
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

## CPU tests

```sh
PYTHONPATH=src python -B -m unittest discover -s tests
```

## License

Original source code and documentation in this repository are licensed under the [MIT License](LICENSE).

## Upstream model and attribution

Built with Qwen. The pretrained Qwen2.5-Omni-3B snapshot uses the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/LICENSE). Its model-use and redistribution terms include a non-commercial scope and upstream notice requirements. Distribution of adapted weights must retain the applicable upstream terms and notices.

Pruning references include [NVIDIA Model Optimizer / Minitron](https://github.com/NVIDIA/Model-Optimizer) and [ShortGPT](https://github.com/icip-cas/ShortGPT). Dataset files are obtained through their providers and retain the providers' access and distribution terms.
