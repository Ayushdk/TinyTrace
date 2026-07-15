# TinyTrace Implementation Plan

## Purpose

This document is the execution roadmap for the B.Tech project version of TinyTrace. It is designed to keep implementation aligned with the frozen architecture in [trace_lightwieght.md](/home/vikaspal/Desktop/Traceall/TinyTrace/trace_lightwieght.md), while also pushing the codebase toward research-grade quality and later publication readiness.

This plan removes week-based scheduling and instead focuses on phase gates, deliverables, and exit criteria.

## Current Reality Check

The current workspace already contains a runnable TinyTrace prototype, but it is not yet a fully architecture-faithful TinyTrace implementation.

### Already complete

- standalone TinyTrace codebase exists
- event schema `(timestamp, score, caption)` exists
- `time -> score -> caption` generation flow exists
- decoder-only LCEM-style prototype exists
- real video loading exists
- QVHighlights subset conversion exists
- TRACE-style QVHighlights highlight metrics exist
- one-sample prediction inspection exists
- downloader script for QVHighlights subset exists
- README and repo hygiene are in place

### Still incomplete / important gaps

- real MobileCLIP integration is not yet implemented in this workspace
- current visual encoder is still a lightweight conv-based placeholder
- visual token compression is prototype-level
- prompt serialization is simplified
- decoder generation on real data is still unstable
- training pipeline is working, but not yet research-grade

## Guiding Rules

1. The frozen architecture takes priority over convenience.
2. TRACE-master is a reference only, never a runtime dependency.
3. No large-scale real training begins before architecture validation succeeds.
4. Synthetic overfit is a required gate before serious real-data experiments.
5. Every phase should leave behind reproducible artifacts, not just code changes.

## M0 — Architecture and Tensor Contract

### Goal

Lock down the exact implementation contract before further architecture work.

### Tasks

- audit each implemented module against `trace_lightwieght.md`
- define required tensor contracts between:
  - frame loader
  - visual encoder
  - compression module
  - time encoder
  - prompt builder
  - LCEM decoder
  - parser
- document expected shapes and token ordering
- identify which parts are:
  - complete
  - partial
  - placeholder
  - missing

### Deliverable

- architecture/tensor contract document committed in repo

### Status

- `partial`

Notes:
- `trace_lightwieght.md` already freezes the high-level design
- a compact execution contract document for code-level tensor flow is still missing

## Phase 1 — Architecture Alignment

### Goal

Make the implementation match the frozen TinyTrace architecture.

### Phase 1A — MobileCLIP integration

#### Tasks

- replace the custom conv visual encoder with MobileCLIP-S0
- keep MobileCLIP fully standalone inside TinyTrace
- freeze MobileCLIP parameters
- force frozen normalization behavior correctly during training
- implement official preprocessing:
  - resize
  - normalization
  - tensor shape expectations
- expose spatial features before global pooling
- adapt MobileCLIP output into TinyTrace visual token flow
- verify the output shape matches compression input requirements

#### Deliverable

- `Video -> MobileCLIP -> spatial features` works end to end

#### Status

- `pending`

### Phase 1B — TRACE-style token pipeline

#### Tasks

- keep the learned lightweight compression module for now
- replace continuous frame-time MLP approach with TRACE-style discrete 13-token numeric time encoding where required by the frozen design
- ensure prompt order matches:
  - `Visual Tokens + Time Tokens + Instruction Tokens -> LCEM`
- verify event parser still reconstructs:
  - timestamp
  - score
  - caption
- verify head switching remains `time -> score -> caption`

#### Deliverable

- full forward pass matching architecture specification

#### Status

- `partial`

Notes:
- token flow and parser exist
- full TRACE-style time-token alignment is not complete yet

## Phase 2 — Engineering Stabilization

### Goal

Make the system reliable and maintainable.

### Tasks

- fix and standardize configuration loading
- restore config cleanly from checkpoints
- centralize serialization code paths
- add safe malformed-generation handling
- improve batch handling
- add padding and masking for variable-length videos
- clean duplicate logic between training, eval, and parser paths
- verify dependency list is complete

### Deliverable

- stable training and inference pipeline

### Status

- `partial`

Notes:
- baseline training and inference work
- pipeline is not yet polished enough for sustained experiments

## Phase 3 — Tests Integrated With Development

### Goal

Build confidence continuously instead of postponing all tests.

### Required tests

- numeric tokenizer encode/decode
- event serialization/parsing
- MobileCLIP output shapes
- compression output shapes
- LCEM forward pass
- `time -> score -> caption` switching
- checkpoint save/load
- variable-length batch collation
- malformed generation safety

### Deliverable

- passing test suite for all core invariants

### Exit criterion

- all tests pass

### Status

- `pending`

## Phase 4 — Synthetic Validation Gate

### Goal

Prove the architecture can learn before spending time on real-video scale-up.

### Tasks

- create a tiny synthetic dataset with 4 to 8 samples
- train until the model nearly memorizes them
- inspect timestamp, score, and caption predictions directly
- if overfitting fails:
  - stop
  - debug architecture/training
  - do not proceed to real-data scale-up

### Deliverable

- clear evidence that TinyTrace can overfit tiny synthetic data

### Exit criterion

- near-perfect memorization on the tiny synthetic set

### Status

- `completed`

Notes:
- synthetic baseline already overfits successfully in the current workspace

## Phase 5 — Dataset Preparation

### Goal

Prepare a clean real-data subset only after architecture validation.

### Tasks

- validate dataset schema before large downloads
- validate timestamp and score ranges
- download 50 to 100 valid QVHighlights videos
- verify each video is readable
- regenerate TinyTrace JSON from clean valid clips
- create explicit train/validation splits
- keep initial experiments small and controlled

### Deliverable

- clean 50 to 100 video TinyTrace-ready subset

### Exit criterion

- train/val subsets exist and all files are decodable

### Status

- `partial`

Notes:
- downloader exists
- tiny real subset exists
- 50 to 100 valid clean subset is not prepared yet

## Phase 6 — Initial Training

### Goal

Validate the real-data pipeline on a controlled subset.

### Tasks

- train on 50 to 100 valid QVHighlights videos
- run 10 to 20 epochs, adjusted by convergence
- save checkpoints every epoch
- track training and validation loss
- save sample predictions after each epoch
- inspect failure patterns in:
  - timestamps
  - scores
  - captions

### Deliverable

- initial real-data training run with checkpoints and prediction examples

### Exit criterion

- training completes cleanly and produces interpretable prediction outputs

### Status

- `partial`

Notes:
- smoke training on a tiny real subset is already done
- proper 50 to 100 video training has not started

## Phase 7 — TRACE-Style Training Strategy

### Goal

Implement staged optimization closer to TRACE training logic.

### Tasks

- add stage controls before larger real-data training
- Stage 1:
  - freeze MobileCLIP and LCEM
  - train compression, time/score embeddings, and task heads
- Stage 2:
  - keep MobileCLIP frozen
  - jointly fine-tune LCEM and task modules
- evaluate after each stage
- compare Stage 1 and Stage 2 behavior

### Deliverable

- two-stage TinyTrace training pipeline

### Exit criterion

- both stages run reproducibly and improve prediction quality relative to smoke baseline

### Status

- `pending`

## Phase 8 — Evaluation and Thesis Artifacts

### Goal

Produce reproducible thesis-quality outputs.

### Tasks

- save final checkpoints
- save exact configuration files
- save dataset splits
- save training logs
- save sample predictions
- save QVHighlights metrics
- record parameter count
- record memory usage
- record inference latency
- produce qualitative examples for thesis figures
- prepare limitations and reproducibility notes

### Deliverable

- thesis-ready artifact bundle

### Exit criterion

- a third party can reproduce the reported runs using repo artifacts

### Status

- `pending`

## Milestones

| Milestone | Success Criteria | Status |
|---|---|---|
| M0 | Architecture/tensor contract documented | Partial |
| M1a | MobileCLIP integrated and frozen; spatial feature extraction works | Pending |
| M1b | MobileCLIP feature shapes verified against compression path | Pending |
| M2 | TRACE-style token pipeline implemented | Partial |
| M3 | Core tests pass | Pending |
| M4 | Model overfits 4–8 synthetic samples | Completed |
| M5 | 50–100 valid QVHighlights videos prepared | Pending |
| M6 | Initial real-data training completes with checkpoints and predictions | Partial |
| M7 | Two-stage training implemented | Pending |
| M8 | Evaluation metrics and thesis-ready artifacts generated | Pending |

## Immediate Next Actions

1. Create the explicit architecture/tensor contract file.
2. Replace the placeholder conv visual encoder with MobileCLIP-S0.
3. Add tests alongside the MobileCLIP integration work.
4. Align the frame-time representation with the frozen TRACE-style token pipeline.
5. Only after that, scale the dataset from tiny subset to 50+ valid videos.

## Important Warning

Do not treat the current prototype as final TinyTrace.

The current codebase is useful because:

- it proves the pipeline can run
- it exposes the real-data failure modes
- it provides a stable base for the next architecture-faithful implementation steps

But before serious BTP experiments and before publication claims, MobileCLIP integration and architecture tightening are mandatory.
