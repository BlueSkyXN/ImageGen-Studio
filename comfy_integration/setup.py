"""Install the pinned ComfyUI runtime without overwriting application code."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from core.settings import CATEGORY_TO_DIR_MAP, INPUT_DIR, OUTPUT_DIR


APP_DIR = Path(__file__).resolve().parents[1]
LOCK_FILE = APP_DIR / "vendor.lock.yaml"
VENDOR_DIR = APP_DIR / "_vendor"
CUSTOM_NODES_DIR = APP_DIR / "custom_nodes"


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


GIT_TIMEOUT_SECONDS = _bounded_env_int(
    "IMAGEGEN_GIT_TIMEOUT_SECONDS", 180, 30, 900
)
GIT_NETWORK_ATTEMPTS = _bounded_env_int("IMAGEGEN_GIT_ATTEMPTS", 2, 1, 5)


def _run_git(*args: str, cwd: Path | None = None) -> str:
    attempts = GIT_NETWORK_ATTEMPTS if args and args[0] == "fetch" else 1
    for attempt in range(1, attempts + 1):
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(cwd) if cwd else None,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=GIT_TIMEOUT_SECONDS,
            )
            return completed.stdout.strip()
        except subprocess.TimeoutExpired as exc:
            if attempt == attempts:
                raise RuntimeError(
                    f"Git 操作超过 {GIT_TIMEOUT_SECONDS} 秒：git {args[0]}。"
                    "可稍后重启，或设置 COMFYUI_PATH 使用本地 checkout。"
                ) from exc
            print(f"⚠️ Git {args[0]} 超时，正在重试（{attempt}/{attempts}）…")
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            print(f"⚠️ Git {args[0]} 失败，正在重试（{attempt}/{attempts}）…")
    raise RuntimeError("Unreachable git retry state")


def _ensure_pinned_repo(name: str, url: str, revision: str, destination: Path) -> None:
    if destination.exists() and not (destination / ".git").is_dir():
        raise RuntimeError(
            f"{name} 目录已存在但不是 Git 仓库：{destination}。"
            "请移走该目录后重新启动。"
        )

    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"--- [Vendor] Cloning pinned {name} ---")
        partial = destination.with_name(f"{destination.name}.partial")
        last_error = None
        for attempt in range(1, GIT_NETWORK_ATTEMPTS + 1):
            if partial.exists():
                shutil.rmtree(partial)
            try:
                _run_git(
                    "clone", "--filter=blob:none", "--no-checkout", url, str(partial)
                )
                partial.replace(destination)
                last_error = None
                break
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                last_error = exc
                if attempt < GIT_NETWORK_ATTEMPTS:
                    print(f"⚠️ {name} clone 失败，正在重试（{attempt}/{GIT_NETWORK_ATTEMPTS}）…")
        if last_error is not None:
            raise RuntimeError(
                f"无法拉取 {name}。可稍后重启；本地离线运行可设置 "
                "COMFYUI_PATH，并按需设置 IMAGEGEN_SKIP_CUSTOM_NODES=1。"
            ) from last_error

    current = ""
    try:
        current = _run_git("rev-parse", "HEAD", cwd=destination)
    except (subprocess.CalledProcessError, RuntimeError):
        pass

    if current != revision:
        print(f"--- [Vendor] Checking out {name} @ {revision[:12]} ---")
        _run_git("fetch", "--depth", "1", "origin", revision, cwd=destination)
        _run_git("checkout", "--detach", "--force", "FETCH_HEAD", cwd=destination)

    actual = _run_git("rev-parse", "HEAD", cwd=destination)
    if actual != revision:
        raise RuntimeError(
            f"{name} 版本不匹配：期望 {revision}，实际 {actual}。"
        )
    print(f"✅ {name} ready @ {actual[:12]}")


def _load_lock() -> dict:
    with LOCK_FILE.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "comfyui" not in data:
        raise RuntimeError(f"Missing comfyui entry in {LOCK_FILE}")
    return data


def initialize_comfyui() -> Path:
    """Prepare pinned sources and make ComfyUI importable.

    Set ``COMFYUI_PATH`` to use an existing local checkout. This is the
    recommended offline/local-development route.
    """

    lock = _load_lock()
    configured_path = os.getenv("COMFYUI_PATH", "").strip()

    if configured_path:
        comfyui_path = Path(configured_path).expanduser().resolve()
        if not (comfyui_path / "nodes.py").is_file():
            raise RuntimeError(f"COMFYUI_PATH is not a ComfyUI checkout: {comfyui_path}")
    else:
        comfyui_path = VENDOR_DIR / "ComfyUI"
        comfy = lock["comfyui"]
        _ensure_pinned_repo("ComfyUI", comfy["url"], comfy["revision"], comfyui_path)

    CUSTOM_NODES_DIR.mkdir(parents=True, exist_ok=True)
    if os.getenv("IMAGEGEN_SKIP_CUSTOM_NODES", "0").lower() not in {"1", "true", "yes"}:
        for name, spec in (lock.get("custom_nodes") or {}).items():
            _ensure_pinned_repo(name, spec["url"], spec["revision"], CUSTOM_NODES_DIR / name)

    # ComfyUI contains a top-level `utils` package. The application uses the
    # collision-free `imagegen_utils` package, so ComfyUI can safely come first.
    comfyui_str = str(comfyui_path)
    if comfyui_str not in sys.path:
        sys.path.insert(0, comfyui_str)

    for relative_path in CATEGORY_TO_DIR_MAP.values():
        (APP_DIR / relative_path).mkdir(parents=True, exist_ok=True)
    (APP_DIR / INPUT_DIR).mkdir(parents=True, exist_ok=True)
    (APP_DIR / OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    import comfy.model_management  # noqa: F401

    print(f"✅ ComfyUI initialized from isolated path: {comfyui_path}")
    return comfyui_path
