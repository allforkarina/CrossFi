# Repository Guidelines

## Project Structure & Module Organization
Compact Python utility for MM-Fi pose data.

- `dataloader.py` contains dataset discovery, HDF5 packing, `MMFiPoseDataset`, loaders, and an inspection CLI.
- `models/` contains trainable model modules, including CSI-Net for Query-Key CSI similarity scoring.
- `scripts/build_h5_dataset.py` converts the raw MM-Fi directory tree into one `.h5` file.
- `tests/` contains lightweight pytest coverage for model and data utility behavior.
- `data/`, `outputs/`, `runs/`, and `checkpoints/` are ignored local artifact paths.

## Build, Test, and Development Commands
Create a virtual environment, then install dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install h5py numpy scipy tqdm torch torchvision sympy pytest
```

Build an HDF5 dataset from raw MM-Fi files:

```powershell
python scripts\build_h5_dataset.py --dataset-root D:\path\to\dataset --output-path data\mmfi_pose.h5
```

Inspect split counts and sample shapes:

```powershell
python dataloader.py --dataset-root data\mmfi_pose.h5 --preview
```

Run tests once added:

```powershell
python -m pytest
```

## Coding Style & Naming Conventions
Use Python 3.10+ type hints, `pathlib.Path`, and `dataclass` for records. Follow PEP 8 with 4-space indentation. Use `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants. Add concise English comments only for non-obvious logic; keep adjacent comments aligned.

## Testing Guidelines
Use `pytest`. Name files `tests/test_*.py` and functions `test_*`. Cover path resolution, split creation, normalization, HDF5 schema, models, and `MMFiPoseDataset.__getitem__`. Use small fixtures instead of the full raw dataset.

## Architecture Notes
CSI-Net receives HDF5 `csi_amplitude` and `csi_phase_cos`, prepares `b x 2 x 10 x 342`, and returns a `b1 x b2` Query-Key similarity matrix. Weight-Net consumes template-stage `b x k x k` CSI-Net matrices and returns `b x k` sample quality confidence scores.

## Commit & Pull Request Guidelines
History uses short, imperative, lowercase subjects such as `add dataloader .h5 file`. PRs should describe the change, commands run, assumptions, and related issues. Include CLI output when changing HDF5 structure or splits.

## Security & Configuration Tips
Do not commit raw datasets, `.h5` outputs, logs, checkpoints, or runs. Keep machine-specific paths configurable through CLI arguments.

## Agent-Specific Instructions
Reply to the user in Chinese. Keep this `AGENTS.md` file and all code/script comments in English. After each project modification, update this guide when core structure, workflow, commands, or operating assumptions change. Keep additions short and actionable.

## Required Code-Change Principles
Before coding, state assumptions, surface unclear tradeoffs, and ask only when exploration cannot resolve the issue. Prefer the minimum implementation that solves the requested problem; do not add speculative features, abstractions, or flexibility. Make surgical edits only in files required by the task, match existing style, and avoid unrelated refactors or cleanup. Define verifiable success criteria, add or update focused tests when practical, and run the relevant checks before reporting completion.

## Local and Server Workflow
This Windows repository is for local editing only, not model training. Training scripts and long-running experiments run on the Linux server. After each completed update, commit changes and push them to GitHub `origin`; the Linux server receives updates with `git pull`.
