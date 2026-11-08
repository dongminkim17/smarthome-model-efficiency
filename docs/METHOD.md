# Final method

## Tasks and data

| Setting | Train | Validation | Official test | Metric |
| --- | ---: | ---: | ---: | --- |
| MIntRec 2-class | 1,334 | 445 | 445 | Accuracy |
| MIntRec 20-class | 1,334 | 445 | 445 | Accuracy |
| MER2023-SEMI, six emotions | 2,698 | 675 | 834 | Weighted F1 |

MIntRec uses its official splits with audio, video and the provided utterance transcript. MER uses audio and video, with a stratified 80/20 split of its 3,373 labeled training/validation samples at split seed 42. The official MER-SEMI test split contains 834 samples.

Video processing uses 2 frames per second, at most eight frames and at most 50,176 pixels per frame. Audio uses the clip's audio track. Fixed task prompts provide answer-letter choices. SFT supervises the answer letter and EOS. Allowed-letter logit scoring and greedy decoding are both implemented, and validation selection uses logit accuracy or weighted F1.

## Teacher preparation

Fine-tune Qwen2.5-Omni-3B separately for each task for five epochs. Apply rank-16, alpha-32 LoRA to decoder query/value projections. Select the teacher checkpoint with the best validation metric.

## Activation-based FFN pruning

Merge the teacher adapter into the original base, retain all 36 decoder layers and reduce every decoder FFN intermediate width from 11,008 to 4,352. Importance uses 128 training clips. Compute mean absolute gated-FFN activation over valid tokens within each clip and aggregate across clips with an L2 norm. Retain the highest-scoring channels and remove matching gate/up rows and down columns.

The activation criterion follows the approach examined in [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679). The implementation uses the shared pruning/export code in this repository. Original audio and vision encoder dimensions are retained.

## Two recovery stages

| Setting | LM recovery | Final joint recovery |
| --- | --- | --- |
| Epochs | 5 | 5 |
| LoRA rank / alpha | 16 / 32 | 64 / 128 |
| Scope | Decoder attention and FFN projections | Decoder and encoder attention, FFN and output projections |
| LM learning rate | 1e-4 | 1e-4 |
| Encoder learning rate | Frozen | 1e-5 |
| SFT weight | 0.25 | 0.25 |
| Class KD weight | 0.75 | 0.75 |
| KD temperature | 2 | 2 |
| Audio / vision feature weights | 0 / 0 | 0.1 / 0.1 |

Class KD uses KL divergence between teacher and student probabilities over the task answer letters, scaled by temperature squared. Encoder feature KD minimizes tokenwise cosine distance to the original task teacher. Targets use training data only.

Merge the validation-selected LM recovery adapter before starting a fresh final-recovery adapter. Both stages use the same original task teacher. The final recovery updates encoder adapters while retaining the original encoder architecture.

All stages use AdamW, a cosine schedule, 3% warmup, effective batch size four, zero weight decay/dropout and gradient clipping at 1.0. Seed 42 is used throughout. Recovery configs use a single GPU with microbatch two and two accumulation steps. Intent LM-only recovery disables gradient checkpointing, while MER LM recovery and all final recoveries enable it.

Validation runs approximately every half epoch. Select each stage's checkpoint on its complete validation split and evaluate the selected final checkpoint on the complete official test split.

## Comparison scope

MIntRec comparisons use the relevant configurations in MIntRec Table 4 and ARL Table 1, with ARL appendix Table 9a optimization controls included for 20-class intent. MER comparisons use Emotion-LLaMA Table 3. Distinct modalities and methods are separate configurations, identical repeated configurations are merged, and human reference scores are outside the model rankings.

The reported ranks are within these selected published comparison cohorts. The final checkpoints have independent task weights under one common architecture and recovery recipe.
