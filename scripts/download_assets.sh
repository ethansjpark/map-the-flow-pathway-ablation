#!/usr/bin/env bash
set -euo pipefail

# Edit these patterns after reading the printed Hub file lists, if needed.
MODEL_INCLUDE_PATTERNS=("*")
TVBENCH_JSON_INCLUDE_PATTERNS=("json/*")
# Order: action_sequence, scene_transition, moving_direction, object_count.
TVBENCH_ZIP_INCLUDE_PATTERNS=(
  "*action_sequence*.zip"
  "*scene_transition*.zip"
  "*moving_direction*.zip"
  "*object_count*.zip"
)
LOG_FILE="/workspace/logs/download_assets.log"

mkdir -p /workspace/logs
exec > >(tee -a "$LOG_FILE") 2>&1
log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
trap 'status=$?; log "FAIL download_assets (exit $status)"; exit "$status"' ERR
log "Starting Hub file listing and selected downloads"

python3 - "${MODEL_INCLUDE_PATTERNS[@]}" -- "${TVBENCH_JSON_INCLUDE_PATTERNS[@]}" -- "${TVBENCH_ZIP_INCLUDE_PATTERNS[@]}" <<'PY'
from fnmatch import fnmatchcase
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

from huggingface_hub import HfApi, hf_hub_download

MODEL_REPO = "byminji/LLaVA-NeXT-7B-Video-FT"
DATA_REPO = "FunAILab/TVBench"
MODEL_DIR = Path("/workspace/models/LLaVA-NeXT-7B-Video-FT")
DATA_DIR = Path("/workspace/datasets/TVBench")
TASKS = ("action_sequence", "scene_transition", "moving_direction", "object_count")

def log(message):
    from datetime import datetime
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)

args = sys.argv[1:]
first = args.index("--")
second = args.index("--", first + 1)
model_patterns = args[:first]
json_patterns = args[first + 1:second]
zip_patterns = args[second + 1:]
if len(zip_patterns) != len(TASKS):
    raise ValueError("Expected one editable zip pattern per TVBench task")

# False prevents use of a cached credential when HF_TOKEN is unset.
token = os.environ.get("HF_TOKEN") or False
api = HfApi(token=token)
model_files = sorted(api.list_repo_files(MODEL_REPO, repo_type="model", token=token))
data_files = sorted(api.list_repo_files(DATA_REPO, repo_type="dataset", token=token))
log(f"MODEL REPO FILES: {MODEL_REPO}")
for name in model_files:
    print(name, flush=True)
log(f"DATASET REPO FILES: {DATA_REPO}")
for name in data_files:
    print(name, flush=True)

def matches(name, patterns):
    return any(fnmatchcase(name.lower(), pattern.lower()) for pattern in patterns)

model_selected = [name for name in model_files if matches(name, model_patterns)]
json_selected = [name for name in data_files if matches(name, json_patterns)]
if not model_selected or not json_selected:
    raise RuntimeError("Model or TVBench json/ pattern matched no files")
zip_selected = {}
for task, pattern in zip(TASKS, zip_patterns):
    found = [name for name in data_files if matches(name, [pattern])]
    if len(found) != 1:
        raise RuntimeError(f"{task}: expected one zip for {pattern}, found {found}; edit patterns at top")
    zip_selected[task] = found[0]

def download(repo_id, repo_type, directory, name):
    target = directory / name
    if target.is_file() and target.stat().st_size > 0:
        log(f"SKIP existing {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {repo_id}/{name}")
    hf_hub_download(
        repo_id=repo_id, filename=name, repo_type=repo_type,
        local_dir=str(directory), local_dir_use_symlinks=False, token=token,
    )
    return target

for name in model_selected:
    download(MODEL_REPO, "model", MODEL_DIR, name)
for name in json_selected:
    download(DATA_REPO, "dataset", DATA_DIR, name)
for task, name in zip_selected.items():
    archive_path = download(DATA_REPO, "dataset", DATA_DIR, name)
    destination = DATA_DIR / "video" / task
    with zipfile.ZipFile(archive_path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if not members:
            raise RuntimeError(f"Empty archive: {archive_path}")
        for member in members:
            raw = member.filename.replace("\\", "/")
            parts = PurePosixPath(raw).parts
            if raw.startswith("/") or ".." in parts:
                raise RuntimeError(f"Unsafe archive member: {raw}")
            task_positions = [i for i, part in enumerate(parts) if part.lower() == task]
            relative_parts = parts[task_positions[-1] + 1:] if task_positions else parts
            if not relative_parts:
                continue
            target = destination.joinpath(*relative_parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    if not destination.is_dir() or not any(destination.iterdir()):
        raise RuntimeError(f"No extracted video files in {destination}")
    log(f"Extracted {archive_path} to {destination}")
log("PASS download_assets")
PY
