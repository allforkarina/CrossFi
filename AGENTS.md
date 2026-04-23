# Repository Guidelines

## Project Structure & Module Organization
This is a compact Python utility for MM-Fi pose data.

- `dataloader.py` contains raw dataset discovery, HDF5 packing helpers, `MMFiPoseDataset`, PyTorch `DataLoader` factories, and a CLI for HDF5 inspection.
- `scripts/build_h5_dataset.py` converts the raw MM-Fi directory tree into one `.h5` file.
- `data/`, `outputs/`, `runs/`, and `checkpoints/` are ignored local artifact paths.
- No `tests/` directory exists yet. Add tests under `tests/`.

## Build, Test, and Development Commands
Create a virtual environment before installing dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install h5py numpy scipy tqdm torch sympy
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
Use Python 3.10+ type hints, `pathlib.Path` for paths, and `dataclass` for simple records. Follow PEP 8 with 4-space indentation. Use `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants such as `SPLIT_NAMES`. Add concise English comments only for non-obvious logic; keep adjacent comments neatly aligned.

## Testing Guidelines
Use `pytest` for new tests. Name files `tests/test_*.py` and test functions `test_*`. Cover path resolution, sample/environment mapping, split creation, normalization, HDF5 schema, and `MMFiPoseDataset.__getitem__`. Use small temporary HDF5 fixtures instead of the full raw dataset.

## Commit & Pull Request Guidelines
Current history uses short, imperative, lowercase commit subjects such as `add dataloader .h5 file`. Keep subjects concise and action-oriented. Pull requests should describe the data-loading change, commands run, dataset assumptions, and related issues. Include CLI output when changing HDF5 structure or split behavior.

## Security & Configuration Tips
Do not commit raw datasets, `.h5` outputs, logs, checkpoints, or experiment runs. Keep machine-specific dataset paths configurable through CLI arguments instead of hard-coding new local paths.

## Agent-Specific Instructions
Reply to the user in Chinese. Keep this `AGENTS.md` file and all code/script comments in English. After each project modification, update this guide when core structure, workflow, commands, or operating assumptions change. Keep additions short and actionable.

## Local and Server Workflow
This Windows repository is for local editing only, not model training. Training scripts and long-running experiments run on the Linux server. After each completed update, commit changes and push them to GitHub `origin`; the Linux server receives updates with `git pull`.
