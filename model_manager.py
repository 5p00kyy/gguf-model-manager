#!/usr/bin/env python3
"""
GGUF Model Manager - Professional Model Management Tool
A clean, efficient tool for downloading and managing GGUF models from Hugging Face.
"""

import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import queue as _queue
from getpass import getpass
from typing import List, Tuple, Dict, Optional

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub import get_token as hf_get_token
import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    TransferSpeedColumn, DownloadColumn, SpinnerColumn
)
from rich.align import Align
from rich import box
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MODELS = 50
MULTIPART_REGEX = r'(.+)-\d{1,5}-of-\d{1,5}\.gguf$'
APP_NAME = "GGUF Model Manager"
APP_VERSION = "4.0"
PRESETS_FILE = os.path.expanduser("~/presets.ini")
LLAMA_SERVER_API_KEY = "Smiledog69!"
LLAMA_SERVER_URL = "http://localhost:8080"
LXC_HOST = "192.168.0.113"

# Download engine configuration -- must be set before importing huggingface_hub internals
os.environ["HF_HUB_DISABLE_XET"] = "1"            # Use HTTP downloader (resume + Ctrl+C work; Xet breaks both)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"      # 60s socket timeout (default 10s too short for 10MB chunks)
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None) # Deprecated, remove if set externally
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)    # Not needed with Xet disabled

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename='model_manager.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------------------------------------------------------------------
# Rich console + HF API (initialized after token setup)
# ---------------------------------------------------------------------------
console = Console()
api = HfApi()  # Re-initialized with token in _init_token()

# ---------------------------------------------------------------------------
# Download queue
# ---------------------------------------------------------------------------
_dl_queue: _queue.Queue = _queue.Queue()
_dl_status = {
    'current': None,
    'pending': [],
    'completed': [],
    'failed': [],
    'paused': False,
}
_dl_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Shutdown handling
# ---------------------------------------------------------------------------
_in_download = False


def _sigint_handler(signum, frame):
    """Handle Ctrl+C. Exits immediately during downloads, gracefully otherwise."""
    if _in_download:
        os._exit(0)
    else:
        raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------
_active_token: Optional[str] = None


def get_active_token() -> Optional[str]:
    """Return the currently active HF token (cached in memory)."""
    return _active_token


def _resolve_token() -> Optional[str]:
    """Resolve the HF token from environment / cache file."""
    return hf_get_token()


def _get_token_username(token: str) -> Optional[str]:
    """Get the HuggingFace username for a token, or None if invalid."""
    try:
        info = HfApi(token=token).whoami()
        return info.get("name") or info.get("fullname") or "authenticated"
    except Exception:
        return None


def configure_token() -> None:
    """Interactive token configuration flow."""
    console.print(f"\n[bold {Colors.PRIMARY}]Token Configuration[/bold {Colors.PRIMARY}]")
    console.print("-" * 50)

    current = get_active_token()
    if current:
        username = _get_token_username(current)
        if username:
            print_success(f"Currently authenticated as: {username}")
        else:
            print_warning("A token is set but could not be verified")
        console.print()
        console.print(f"  [{Colors.PRIMARY}]1[/] - Set a new token")
        console.print(f"  [{Colors.PRIMARY}]2[/] - Remove token")
        console.print(f"  [{Colors.PRIMARY}]q[/] - Back")
        choice = _safe_input(f"\n[{Colors.PRIMARY}]Select option[/]: ")
        if choice is None or choice == 'q':
            return
        if choice == '2':
            _remove_token()
            return
        if choice != '1':
            print_error("Invalid option")
            return
    else:
        print_info("No HuggingFace token configured")
        print_info("A token allows higher download rate limits and access to gated models")
        console.print(f"[dim]Get your token at: https://huggingface.co/settings/tokens[/dim]")
        console.print()
        console.print(f"  [{Colors.PRIMARY}]1[/] - Enter token")
        console.print(f"  [{Colors.PRIMARY}]q[/] - Back (continue without token)")
        choice = _safe_input(f"\n[{Colors.PRIMARY}]Select option[/]: ")
        if choice is None or choice != '1':
            return

    _set_new_token()


def _set_new_token() -> None:
    """Prompt user for a new token and save it."""
    global _active_token, api

    console.print()
    console.print(f"[dim]Get your token at: https://huggingface.co/settings/tokens[/dim]")
    console.print(f"[dim]Paste your token below (input will be hidden):[/dim]")

    try:
        token = getpass("Token: ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print()
        return

    if not token:
        print_warning("No token entered")
        return

    with console.status(f"[{Colors.PRIMARY}]Validating token...[/]"):
        username = _get_token_username(token)

    if not username:
        print_error("Invalid token -- could not authenticate with HuggingFace")
        return

    try:
        token_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        os.makedirs(token_path, exist_ok=True)
        token_file = os.path.join(token_path, "token")
        with open(token_file, "w") as f:
            f.write(token)
        os.chmod(token_file, 0o600)
    except Exception as e:
        print_warning(f"Could not save token to cache: {e}")
        print_info("Token will be used for this session only")

    _active_token = token
    api = HfApi(token=token)
    print_success(f"Authenticated as: {username}")
    logging.info(f"Token configured for user: {username}")


def _remove_token() -> None:
    """Remove the saved token."""
    global _active_token, api

    confirm = _safe_input("Remove saved token? [y/N]: ")
    if confirm and confirm.lower() == 'y':
        token_file = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")
        try:
            if os.path.exists(token_file):
                os.remove(token_file)
        except Exception as e:
            print_warning(f"Could not remove token file: {e}")

        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

        _active_token = None
        api = HfApi()
        print_success("Token removed")
        logging.info("Token removed by user")
    else:
        print_info("Token kept")


def _init_token() -> None:
    """Initialize token on startup (non-interactive)."""
    global _active_token, api

    token = _resolve_token()
    if token:
        _active_token = token
        api = HfApi(token=token)
        logging.info("Token loaded from environment/cache")
    else:
        api = HfApi()


def _token_status_str() -> str:
    """Return a short string for the header showing token status."""
    token = get_active_token()
    if not token:
        return "Not set"
    try:
        info = api.whoami()
        name = info.get("name") or info.get("fullname") or "authenticated"
        return name
    except Exception:
        return "Set (unverified)"


# ---------------------------------------------------------------------------
# Color theme
# ---------------------------------------------------------------------------
class Colors:
    """Color constants for consistent theming."""
    PRIMARY = "cyan"
    SUCCESS = "green"
    WARNING = "yellow"
    ERROR = "red"
    INFO = "blue"
    MUTED = "dim"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def format_size(size_bytes: int) -> str:
    """Format byte size into human readable string."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def get_disk_space() -> Tuple[float, float]:
    """Get disk space information (free, total in GB)."""
    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, "df", "/root/models"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 4:
                    total = float(parts[1]) / 1024 / 1024  # KB to GB
                    free = float(parts[3]) / 1024 / 1024
                    return free, total
    except Exception:
        pass
    return 0.0, 0.0


def get_gpu_info() -> List[Dict[str, str]]:
    """Get GPU memory info."""
    gpus = []
    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, "nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpus.append({
                            'index': parts[0],
                            'used': parts[1],
                            'total': parts[2]
                        })
    except Exception:
        pass
    return gpus


def get_server_status() -> Tuple[bool, str]:
    """Check if llama-server is running and get loaded model."""
    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, "curl", "-s", "-m", "2", f"{LLAMA_SERVER_URL}/health"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and '{"status":"ok"}' in result.stdout:
            slots_result = subprocess.run(
                ["ssh", LXC_HOST, "curl", "-s", "-m", "2", "-H", f"Authorization: Bearer {LLAMA_SERVER_API_KEY}", f"{LLAMA_SERVER_URL}/slots"],
                capture_output=True, text=True, timeout=5
            )
            if slots_result.returncode == 0:
                import json
                try:
                    slots_data = json.loads(slots_result.stdout)
                    if slots_data and len(slots_data) > 0:
                        model_name = slots_data[0].get('model', '')
                        if model_name:
                            model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                            return True, model_short
                except json.JSONDecodeError:
                    pass
            return True, ""
    except Exception:
        pass
    return False, ""


def count_downloaded_models() -> int:
    """Count number of downloaded models."""
    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, "ls", "-la", "/root/models/"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            count = 0
            for line in result.stdout.split('\n'):
                if line.startswith('d'):
                    parts = line.split()
                    if len(parts) >= 9:
                        dirname = parts[-1]
                        if not dirname.startswith('.') and dirname not in ('models',):
                            count += 1
            return count
    except Exception:
        pass
    return 0


def _safe_input(prompt: str) -> Optional[str]:
    """Wrapper around console.input that handles KeyboardInterrupt cleanly."""
    try:
        return console.input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        console.print()
        return None


# ---------------------------------------------------------------------------
# Startup maintenance
# ---------------------------------------------------------------------------
STALE_THRESHOLD_HOURS = 24


def _clean_stale_incomplete_files() -> None:
    """Remove .incomplete files older than STALE_THRESHOLD_HOURS from models/."""
    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, "find", "/root/models/", "-name", "*.incomplete", "-mtime", "+24", "-type", "f"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            for fpath in result.stdout.strip().split('\n'):
                if fpath:
                    subprocess.run(["ssh", LXC_HOST, "rm", "-f", fpath], capture_output=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def show_status_panel():
    """Render status panel using Rich Panel/Table. Called at start of each main menu loop."""
    free_gb, total_gb = get_disk_space()
    gpus = get_gpu_info()
    server_running, server_model = get_server_status()
    
    with _dl_lock:
        pending_count = len(_dl_status['pending'])
        current = _dl_status['current']
        paused = _dl_status['paused']

    used_gb = total_gb - free_gb
    used_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
    
    if free_gb > 100:
        disk_color = Colors.SUCCESS
    elif free_gb >= 50:
        disk_color = Colors.WARNING
    else:
        disk_color = Colors.ERROR
    
    disk_text = f"Disk: {free_gb:.0f} GB free ({used_pct:.0f}% used)"
    if free_gb < 50:
        disk_text += " ⚠"
    
    gpu_text = ""
    if gpus:
        gpu_parts = []
        for gpu in gpus:
            gpu_parts.append(f"GPU{gpu['index']}: {gpu['used']}/{gpu['total']} MB")
        gpu_text = "  ".join(gpu_parts)
    
    if server_running:
        server_color = Colors.SUCCESS
        server_text = "● running"
        if server_model:
            server_text += f" [{server_model}]"
    else:
        server_color = Colors.ERROR
        server_text = "● offline"
    
    queue_text = ""
    if current:
        queue_text = f"⬇ Downloading: {current}"
        if pending_count > 0:
            queue_text += f" | Queue: {pending_count} pending"
    elif paused and pending_count > 0:
        queue_text = f"⏸ Queue paused | {pending_count} pending"
    elif pending_count > 0:
        queue_text = f"⬇ Queue: {pending_count} pending"
    
    table = Table(box=None, padding=(0, 0), show_header=False)
    table.add_column(style=Colors.PRIMARY)
    
    table.add_row(f"[{disk_color}]{disk_text}[/{disk_color}]")
    if gpu_text:
        table.add_row(gpu_text)
    table.add_row(f"[{server_color}]{server_text}[/{server_color}]")
    if queue_text:
        table.add_row(queue_text)
    
    panel = Panel(
        table,
        border_style=Colors.PRIMARY,
        padding=(1, 2),
        title=f"[bold {Colors.PRIMARY}]{APP_NAME} v{APP_VERSION}[/bold {Colors.PRIMARY}]"
    )
    console.print(panel)


def print_header():
    """Legacy header function - now shows status panel."""
    show_status_panel()


def print_success(message: str):
    """Print a success message."""
    console.print(f"[bold {Colors.SUCCESS}][OK][/bold {Colors.SUCCESS}] {message}")


def print_error(message: str):
    """Print an error message."""
    console.print(f"[bold {Colors.ERROR}][ERROR][/bold {Colors.ERROR}] {message}")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"[bold {Colors.WARNING}][WARN][/bold {Colors.WARNING}] {message}")


def print_info(message: str):
    """Print an info message."""
    console.print(f"[bold {Colors.INFO}][INFO][/bold {Colors.INFO}] {message}")


# ---------------------------------------------------------------------------
# Rich <-> tqdm bridge for download progress
# ---------------------------------------------------------------------------
class RichTqdm:
    """A tqdm-compatible class that bridges huggingface_hub's download progress to Rich Progress bars."""

    _progress: Optional[Progress] = None
    _lock = None

    @classmethod
    def set_progress(cls, progress: Optional[Progress]):
        cls._progress = progress

    @classmethod
    def get_lock(cls):
        if cls._lock is None:
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock

    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs.pop("unit_scale", None)
        kwargs.pop("unit_divisor", None)

        self.total = kwargs.get("total", 0) or 0
        self.n = kwargs.get("initial", 0) or 0
        self.desc = kwargs.get("desc", "")
        self.unit = kwargs.get("unit", "it")
        self.disable = kwargs.get("disable", False)
        self.task_id = None
        self._iterable = args[0] if args else kwargs.get("iterable", None)

        if self._progress is not None and not self.disable:
            is_bytes = self.unit == "B"
            self.task_id = self._progress.add_task(
                description=self.desc[:60] if self.desc else "Downloading",
                total=self.total if self.total > 0 else None,
                completed=self.n,
                visible=is_bytes
            )

    def __iter__(self):
        if self._iterable is None:
            return
        for obj in self._iterable:
            yield obj
            self.update(1)

    def __len__(self):
        if self._iterable is not None:
            return len(self._iterable)
        return 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def update(self, n=1):
        if n is None or self.disable:
            return
        self.n += n
        if self.task_id is not None and self._progress is not None:
            self._progress.update(self.task_id, advance=n)

    def refresh(self):
        if self.task_id is not None and self._progress is not None:
            self._progress.update(self.task_id, total=self.total, completed=self.n)

    def set_description(self, desc="", refresh=True):
        self.desc = desc
        if self.task_id is not None and self._progress is not None:
            self._progress.update(self.task_id, description=desc[:60])

    def set_description_str(self, desc="", refresh=True):
        self.set_description(desc, refresh)

    def close(self):
        if self.task_id is not None and self._progress is not None:
            if self.total and self.total > 0:
                self._progress.update(self.task_id, completed=self.total, total=self.total)

    def set_postfix(self, *args, **kwargs):
        pass

    def set_postfix_str(self, *args, **kwargs):
        pass

    def clear(self):
        pass

    def reset(self, total=None):
        if total is not None:
            self.total = total
        self.n = 0
        self.refresh()


# ---------------------------------------------------------------------------
# Download engine
# ---------------------------------------------------------------------------
def download_with_progress(repo_id: str, local_dir: str, patterns: List[str],
                          total_size: int = 0, action: str = "Downloading",
                          force: bool = False) -> bool:
    """Download files using snapshot_download with Rich progress bars."""
    global _in_download

    console.print(f"\n[bold {Colors.PRIMARY}]{action}:[/bold {Colors.PRIMARY}] {repo_id}")

    if total_size > 0:
        console.print(f"[dim]Total size: {format_size(total_size)}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False
    ) as progress:

        RichTqdm.set_progress(progress)
        _in_download = True
        signal.signal(signal.SIGINT, _sigint_handler)

        try:
            start_time = time.time()

            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=patterns,
                tqdm_class=RichTqdm,
                force_download=force,
                token=get_active_token(),
                max_workers=4,
            )

            elapsed = time.time() - start_time

            if total_size > 0 and elapsed > 0:
                speed = total_size / elapsed
                console.print(
                    f"[dim]Average speed: {format_size(int(speed))}/s[/dim]"
                )

            print_success(f"{action} complete in {elapsed:.1f}s")
            logging.info(f"{action} complete for {repo_id} in {elapsed:.1f}s")
            return True

        except KeyboardInterrupt:
            logging.info(f"{action} interrupted by user for {repo_id}")
            return False
        except Exception as e:
            print_error(f"{action} failed: {e}")
            logging.error(f"{action} failed for {repo_id}: {e}")
            return False
        finally:
            _in_download = False
            RichTqdm.set_progress(None)


# ---------------------------------------------------------------------------
# Download queue functions
# ---------------------------------------------------------------------------
def _download_worker():
    """Background worker thread for download queue."""
    while True:
        try:
            item = _dl_queue.get(timeout=1)
        except _queue.Empty:
            continue

        repo_id, files, force = item

        with _dl_lock:
            if _dl_status['paused']:
                _dl_queue.put(item)
                time.sleep(1)
                continue
            _dl_status['current'] = repo_id

        local_dir = f"/root/models/{repo_id}"
        
        total_size = 0
        try:
            repo_info = api.model_info(repo_id, files_metadata=True, token=get_active_token())
            if repo_info.siblings:
                for sibling in repo_info.siblings:
                    if sibling.size and sibling.rfilename in files:
                        total_size += sibling.size
        except Exception:
            pass

        patterns = [f"*{os.path.basename(f)}*" for f in files]
        action = "Downloading"
        
        success = download_with_progress(repo_id, local_dir, patterns, total_size, action, force=force)

        with _dl_lock:
            _dl_status['current'] = None
            if success:
                _dl_status['completed'].append(repo_id)
            else:
                _dl_status['failed'].append(repo_id)


def queue_download(repo_id: str, files: List[str], force: bool = False):
    """Add to download queue."""
    with _dl_lock:
        _dl_status['pending'].append(repo_id)
    _dl_queue.put((repo_id, files, force))


def pause_queue():
    """Pause the download queue."""
    with _dl_lock:
        _dl_status['paused'] = True


def resume_queue():
    """Resume the download queue."""
    with _dl_lock:
        _dl_status['paused'] = False


def clear_queue():
    """Clear pending downloads from queue."""
    with _dl_lock:
        _dl_status['pending'].clear()
    while True:
        try:
            _dl_queue.get_nowait()
        except _queue.Empty:
            break


def show_queue_menu():
    """Display and manage download queue."""
    while True:
        console.print(f"\n[bold {Colors.PRIMARY}]Download Queue[/bold {Colors.PRIMARY}]")
        console.print("-" * 50)

        with _dl_lock:
            current = _dl_status['current']
            pending = _dl_status['pending'].copy()
            completed = _dl_status['completed'].copy()
            failed = _dl_status['failed'].copy()
            paused = _dl_status['paused']

        if current:
            console.print(f"[{Colors.PRIMARY}]Current:[/] {current}")
        else:
            console.print(f"[{Colors.MUTED}]Current:[/] idle")

        console.print(f"[{Colors.PRIMARY}]Pending:[/] {len(pending)}")
        if pending:
            for p in pending[:5]:
                console.print(f"  - {p}")
            if len(pending) > 5:
                console.print(f"  ... and {len(pending) - 5} more")

        console.print(f"[{Colors.SUCCESS}]Completed:[/] {len(completed)}")
        if completed:
            for c in completed[-3:]:
                console.print(f"  - {c}")

        console.print(f"[{Colors.ERROR}]Failed:[/] {len(failed)}")
        if failed:
            for f in failed[-3:]:
                console.print(f"  - {f}")

        if paused:
            console.print(f"[{Colors.WARNING}]Status: PAUSED[/]")

        console.print()
        console.print(f"  [{Colors.PRIMARY}]p[/] - {'Resume' if paused else 'Pause'}")
        console.print(f"  [{Colors.PRIMARY}]c[/] - Clear pending")
        console.print(f"  [{Colors.PRIMARY}]q[/] - Back")

        choice = _safe_input(f"\n[{Colors.PRIMARY}]Select option[/]: ")

        if choice is None or choice == 'q':
            return
        elif choice == 'p':
            if paused:
                resume_queue()
                print_info("Queue resumed")
            else:
                pause_queue()
                print_info("Queue paused")
        elif choice == 'c':
            clear_queue()
            print_info("Queue cleared")


# ---------------------------------------------------------------------------
# llama-fit-params integration
# ---------------------------------------------------------------------------
def find_max_ctx_size(model_path: str, n_gpu_layers: int = 99, tensor_split: str = "1,1") -> Optional[int]:
    """Binary search for maximum ctx-size that fits in VRAM using llama-fit-params."""
    fit_binary = "/usr/local/bin/llama-fit-params"
    
    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, f"test -x {fit_binary}"],
            capture_output=True, timeout=5
        )
        if result.returncode != 0:
            print_warning("llama-fit-params not found, using heuristic")
            return None
    except Exception:
        print_warning("llama-fit-params not accessible, using heuristic")
        return None

    ctx_sizes = [131072, 65536, 32768, 16384, 8192, 4096]
    
    for ctx_size in ctx_sizes:
        try:
            result = subprocess.run(
                ["ssh", LXC_HOST, fit_binary, "-m", model_path, "--ctx-size", str(ctx_size), 
                 "--n-gpu-layers", str(n_gpu_layers), "--tensor-split", tensor_split],
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                output = result.stdout + result.stderr
                if "successfully fit params" in output.lower():
                    match = re.search(r'-c\s+(\d+)', result.stdout)
                    if match:
                        return int(match.group(1))
                    return ctx_size
        except Exception:
            pass

    return None


def _detect_tensor_split(repo_id: str, preset_name: str) -> str:
    """Detect tensor-split from existing presets for model family."""
    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, "cat", "/root/presets.ini"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            preset_name_lower = preset_name.lower()
            for line in result.stdout.split('\n'):
                if 'tensor-split' in line.lower() and '=' in line:
                    parts = line.split('=')
                    if len(parts) == 2:
                        family_match = re.match(r'\[([^\]]+)\]', parts[0].strip())
                        if family_match:
                            family = family_match.group(1).lower()
                            if family in preset_name_lower or preset_name_lower in family:
                                return parts[1].strip()
    except Exception:
        pass
    return "1,1"


# ---------------------------------------------------------------------------
# Preset organization
# ---------------------------------------------------------------------------
def _detect_family(preset_name: str) -> str:
    """Extract base family from preset name."""
    name = preset_name
    name = re.sub(r'-\d+B.*$', '', name)
    name = re.sub(r'-Q\d+$', '', name)
    name = re.sub(r'-Q[45]?_?X?L?$', '', name)
    name = re.sub(r'-High$', '', name)
    name = re.sub(r'-Thinking$', '', name)
    name = re.sub(r'-Flash$', '', name)
    name = re.sub(r'-Instruct$', '', name)
    name = re.sub(r'-Chat$', '', name)
    return name.strip()


def _extract_param_count(preset_name: str) -> float:
    """Extract parameter count for sorting."""
    match = re.search(r'(\d+(?:\.\d+)?)B', preset_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 999.0


def write_preset_organized(preset_name: str, params: dict) -> None:
    """Insert preset into presets.ini in the correct position."""
    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, "cat", PRESETS_FILE],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            write_preset_simple(preset_name, params)
            return
        
        existing_content = result.stdout
        family = _detect_family(preset_name)
        preset_params = _extract_param_count(preset_name)
        
        lines = existing_content.split('\n')
        new_lines = []
        family_lines = []
        in_family = False
        last_family = None
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            section_match = re.match(r'\[([^\]]+)\]', line.strip())
            if section_match:
                if family_lines:
                    new_lines.extend(family_lines)
                    new_lines.append('')
                    family_lines = []
                
                current_name = section_match.group(1)
                current_family = _detect_family(current_name)
                current_params = _extract_param_count(current_name)
                
                if current_family == family:
                    in_family = True
                    last_family = current_family
                    if current_params <= preset_params:
                        family_lines.append(line)
                    else:
                        insert_params = True
                        if not family_lines:
                            new_lines.append('')
                        for pl in family_lines:
                            new_lines.append(pl)
                        family_lines = []
                        new_lines.append(f"[{preset_name}]")
                        for k, v in params.items():
                            new_lines.append(f"{k} = {v}")
                        new_lines.append('')
                        family_lines.append(line)
                        in_family = False
                else:
                    in_family = False
                    if in_family and family_lines:
                        family_lines.append(line)
                    else:
                        new_lines.append(line)
            elif in_family and line.strip():
                family_lines.append(line)
            elif line.strip() or (in_family and line == ''):
                if in_family:
                    family_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
            
            i += 1
        
        if family_lines:
            new_lines.extend(family_lines)
        
        has_preset = any(f"[{preset_name}]" in l for l in new_lines)
        if not has_preset:
            new_lines.append('')
            new_lines.append(f"[{preset_name}]")
            for k, v in params.items():
                new_lines.append(f"{k} = {v}")
            new_lines.append('')
        
        new_content = '\n'.join(new_lines)
        
        subprocess.run(
            ["ssh", LXC_HOST, f"cat > {PRESETS_FILE}"],
            input=new_content,
            text=True,
            timeout=10
        )
        
        print_success(f"Preset [{preset_name}] written to {PRESETS_FILE}")
        logging.info(f"Auto-generated preset [{preset_name}]")
        
    except Exception as e:
        print_warning(f"Could not organize presets: {e}")
        write_preset_simple(preset_name, params)


def write_preset_simple(preset_name: str, params: dict) -> None:
    """Simple append preset to file."""
    lines = [f"\n[{preset_name}]"]
    for k, v in params.items():
        lines.append(f"{k} = {v}")
    block = "\n".join(lines) + "\n"
    
    try:
        subprocess.run(
            ["ssh", LXC_HOST, f"cat >> {PRESETS_FILE}"],
            input=block,
            text=True,
            timeout=10
        )
        print_success(f"Preset [{preset_name}] written to {PRESETS_FILE}")
    except Exception as e:
        print_error(f"Failed to write preset: {e}")


# ---------------------------------------------------------------------------
# Model card snippet
# ---------------------------------------------------------------------------
def show_model_card_snippet(repo_id: str) -> None:
    """Fetch and display model card excerpt."""
    try:
        readme_url = f"https://huggingface.co/{repo_id}/raw/main/README.md"
        result = subprocess.run(
            ["curl", "-s", "-m", "10", readme_url],
            capture_output=True, text=True, timeout=15
        )
        
        if result.returncode != 0 or not result.stdout:
            return
        
        content = result.stdout
        lines = content.split('\n')
        excerpt_lines = []
        in_paragraph = False
        
        for line in lines:
            if line.startswith('#'):
                if excerpt_lines:
                    break
                in_paragraph = True
                continue
            
            line = line.strip()
            if not line:
                if in_paragraph and excerpt_lines:
                    break
                in_paragraph = True
                continue
            
            if any(kw in line.lower() for kw in ['ctx', 'context', 'n_ctx', 'temperature', 'recommended']):
                excerpt_lines.append(line)
            elif len(excerpt_lines) < 2 and len(line) > 20:
                excerpt_lines.append(line)
        
        if excerpt_lines:
            snippet = ' '.join(excerpt_lines[:3])[:300]
            panel = Panel(
                f"[dim]{snippet}[/dim]",
                title="[dim]Model Info[/dim]",
                border_style=Colors.MUTED,
                padding=(1, 1)
            )
            console.print(panel)
            
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Maintenance functions
# ---------------------------------------------------------------------------
def get_llamacpp_version() -> Optional[str]:
    """Get installed llama.cpp version."""
    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, "llama-server", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            output = result.stdout + result.stderr
            match = re.search(r'build:\s*(\d+)', output)
            if match:
                return match.group(1)
    except Exception:
        pass
    return None


def get_llamacpp_latest() -> Optional[str]:
    """Fetch latest llama.cpp release from GitHub."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-m", "5", "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            tag = data.get('tag_name', '')
            match = re.search(r'b(\d+)', tag)
            if match:
                return match.group(1)
    except Exception:
        pass
    return None


def get_whisper_version() -> Optional[str]:
    """Get installed whisper.cpp version."""
    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, "whisper-server", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            output = result.stdout + result.stderr
            match = re.search(r'v?(\d+\.\d+\.\d+)', output)
            if match:
                return match.group(1)
    except Exception:
        pass
    return "unknown"


def get_whisper_latest() -> Optional[str]:
    """Fetch latest whisper.cpp release from GitHub."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-m", "5", "https://api.github.com/repos/ggerganov/whisper.cpp/releases/latest"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            tag = data.get('tag_name', '')
            return tag
    except Exception:
        pass
    return None


def show_maintenance_menu():
    """Display maintenance options."""
    while True:
        console.print(f"\n[bold {Colors.PRIMARY}]Maintenance[/bold {Colors.PRIMARY}]")
        console.print("-" * 50)

        llama_installed = get_llamacpp_version()
        llama_latest = get_llamacpp_latest()
        
        whisper_installed = get_whisper_version()
        whisper_latest = get_whisper_latest()

        def format_version(name: str, installed: Optional[str], latest: Optional[str]) -> str:
            if installed is None:
                return f"{name}: installed: [dim]unknown[/] latest: {latest or '[dim]check failed[/]'} [dim][check failed][/dim]"
            if latest is None:
                return f"{name}: installed: {installed} latest: [dim]check failed[/dim] [dim][check failed][/dim]"
            if installed == latest:
                return f"{name}: installed: {installed} latest: {latest} [green]up to date[/green]"
            return f"{name}: installed: {installed} latest: {latest} [yellow]UPDATE AVAILABLE[/yellow]"

        console.print(format_version("llama.cpp", llama_installed, llama_latest))
        console.print(format_version("whisper.cpp", whisper_installed, whisper_latest))

        console.print("-" * 50)
        console.print(f"  [{Colors.PRIMARY}]1[/] - Update llama.cpp")
        console.print(f"  [{Colors.PRIMARY}]2[/] - Update whisper.cpp")
        console.print(f"  [{Colors.PRIMARY}]q[/] - Back")

        choice = _safe_input(f"\n[{Colors.PRIMARY}]Select option[/]: ")

        if choice is None or choice == 'q':
            return
        elif choice == '1':
            confirm = _safe_input("⚠ This will stop the running llama-server and rebuild (~10 min). Continue? [y/N]: ")
            if confirm and confirm.lower() == 'y':
                run_update_script("update-llama.sh")
        elif choice == '2':
            confirm = _safe_input("⚠ This will rebuild whisper.cpp (~10 min). Continue? [y/N]: ")
            if confirm and confirm.lower() == 'y':
                run_update_script("update-whisper.sh")


def run_update_script(script_name: str):
    """Run update script and stream output."""
    try:
        console.print(f"[{Colors.PRIMARY}]Running update...[/]")
        process = subprocess.Popen(
            ["ssh", LXC_HOST, f"bash /root/{script_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            if line:
                console.print(f"[dim]{line.rstrip()}[/dim]")
        
        process.wait()
        
        if process.returncode == 0:
            print_success("Update completed")
        else:
            print_error("Update failed")
            
    except Exception as e:
        print_error(f"Update failed: {e}")


# ---------------------------------------------------------------------------
# Preset auto-generation
# ---------------------------------------------------------------------------
def _detect_preset_params(repo_id: str, selected_files: List[str], ctx_override: Optional[int] = None) -> dict:
    """Infer sensible preset parameters from repo_id and selected files."""
    all_text = (repo_id + " " + " ".join(selected_files)).lower()

    is_moe = bool(re.search(r'-a\d+b|moe|_moe', all_text))

    size_match = re.search(r'(\d+(?:\.\d+)?)b', all_text)
    total_b = float(size_match.group(1)) if size_match else 7.0

    is_reasoning = bool(re.search(r'thinking|reason|qwq|deepseek-r|skywork-o', all_text))

    first_file = sorted(selected_files)[0] if selected_files else ''
    multipart_match = re.search(r'(.+)-(\d{5})-of-(\d{5})\.gguf$', first_file, re.IGNORECASE)

    if multipart_match:
        model_path = f"/root/models/{repo_id}/{first_file}"
    else:
        model_path = f"/root/models/{repo_id}/{first_file}"

    params = {
        'model': model_path,
        'n-gpu-layers': 99,
        'temp': 0.7,
        'batch-size': 4096,
        'ubatch-size': 2048,
        'jinja': 'on',
    }

    if ctx_override:
        params['ctx-size'] = ctx_override
    elif is_moe:
        params['ctx-size'] = 8192
    else:
        params['ctx-size'] = 65536

    tensor_split = _detect_tensor_split(repo_id, repo_id.split('/')[-1])
    
    if is_moe:
        params['tensor-split'] = '4,1' if total_b >= 100 else '1,1.2'
        params['top-k'] = 20
        params['top-p'] = 0.8
        params['min-p'] = 0
        params['presence-penalty'] = 1.5
        if total_b >= 60:
            params['n-cpu-moe'] = 22
            params['cache-type-k'] = 'q8_0'
            params['cache-type-v'] = 'q8_0'
        elif total_b >= 30:
            params['n-cpu-moe'] = 18
    else:
        params['tensor-split'] = tensor_split
        params['cache-type-k'] = 'q8_0'
        params['cache-type-v'] = 'q8_0'
        params['top-k'] = 20
        params['top-p'] = 0.8
        params['min-p'] = 0
        params['presence-penalty'] = 1.5

    if is_reasoning:
        params['temp'] = 0.6
        params['reasoning-format'] = 'auto'
        params['top-k'] = 20
        params['top-p'] = 0.95

    return params


def _preset_name_from_repo(repo_id: str) -> str:
    """Generate a clean preset name from repo_id."""
    name = repo_id.split('/')[-1]
    for suffix in ['-GGUF', '-gguf', '-Instruct', '-instruct', '-Chat', '-chat']:
        name = name.replace(suffix, '')
    return name


def offer_preset_generation(repo_id: str, selected_files: List[str]) -> None:
    """After a successful download, offer to write a preset entry to presets.ini."""
    result = subprocess.run(
        ["ssh", LXC_HOST, "test", "-f", PRESETS_FILE],
        capture_output=True
    )
    if result.returncode != 0:
        print_warning(f"presets.ini not found at {PRESETS_FILE} — skipping preset generation")
        return

    preset_name = _preset_name_from_repo(repo_id)

    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, "grep", f"[{preset_name}]", PRESETS_FILE],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print_info(f"Preset [{preset_name}] already exists in presets.ini — skipping")
            return
    except Exception:
        pass

    model_path = f"/root/models/{repo_id}"
    first_file = sorted(selected_files)[0] if selected_files else ''
    full_model_path = f"{model_path}/{first_file}"

    with console.status(f"[{Colors.PRIMARY}]Finding max ctx-size...[/]"):
        max_ctx = find_max_ctx_size(full_model_path, n_gpu_layers=99)

    params = _detect_preset_params(repo_id, selected_files, ctx_override=max_ctx)

    console.print(f"\n[bold green]Preset preview — [{preset_name}][/bold green]")
    if max_ctx:
        console.print(f"  [dim]Max context:[/dim] {max_ctx} tokens (llama-fit-params)")
    for k, v in params.items():
        console.print(f"  [dim]{k}[/dim] = {v}")

    answer = _safe_input("\nWrite this preset to presets.ini? [Y/n/e(dit name)]: ")
    if answer is None or answer.lower() == 'n':
        return
    if answer.lower() == 'e':
        new_name = _safe_input("Preset name: ")
        if new_name:
            preset_name = new_name.strip()

    write_preset_organized(preset_name, params)


def download_model(repo_id: str, selected_files: List[str],
                   is_update: bool = False, force: bool = False) -> bool:
    """Download selected GGUF files for a model with progress display."""
    subprocess.run(["ssh", LXC_HOST, "mkdir", "-p", f"/root/models/{repo_id}"], check=True)
    local_dir = f"/root/models/{repo_id}"

    total_size = 0
    try:
        repo_info = api.model_info(repo_id, files_metadata=True, token=get_active_token())
        if repo_info.siblings:
            for sibling in repo_info.siblings:
                if sibling.size and sibling.rfilename in selected_files:
                    total_size += sibling.size
    except Exception as e:
        logging.warning(f"Could not fetch size info for {repo_id}: {e}")

    if total_size > 0:
        try:
            free_gb, _ = get_disk_space()
            required_gb = total_size * 1.1 / 1e9
            if free_gb < required_gb:
                print_warning(f"Low disk space: {free_gb:.1f} GB free, {required_gb:.1f} GB required")
                return False
        except Exception as e:
            logging.warning(f"Disk space check failed: {e}")

    if force:
        print_info(f"Force redownload: removing {len(selected_files)} file(s)...")
        for f in selected_files:
            subprocess.run(
                ["ssh", LXC_HOST, "rm", "-f", f"{local_dir}/{f}"],
                capture_output=True
            )

    patterns = [f"*{os.path.basename(f)}*" for f in selected_files]
    action = "Redownloading" if is_update else "Downloading"

    logging.info(f"Starting {action.lower()} for {repo_id}: {selected_files} (force={force})")
    return download_with_progress(repo_id, local_dir, patterns, total_size, action, force=force)


# ---------------------------------------------------------------------------
# Direct download
# ---------------------------------------------------------------------------
def direct_download() -> None:
    """Direct repo ID/URL input flow."""
    console.print(f"\n[bold {Colors.PRIMARY}]Direct Download[/bold {Colors.PRIMARY}]")
    console.print("-" * 50)

    repo_input = _safe_input("Enter HuggingFace repo ID or URL: ")
    if not repo_input:
        print_warning("No input")
        return

    repo_input = repo_input.strip()
    
    if repo_input.startswith('https://huggingface.co/'):
        repo_input = repo_input.replace('https://huggingface.co/', '').strip()
        if '/resolve/' in repo_input:
            repo_input = repo_input.split('/resolve/')[0].strip()
        if '/tree/' in repo_input:
            repo_input = repo_input.split('/tree/')[0].strip()
    
    repo_id = repo_input

    try:
        with console.status(f"[{Colors.PRIMARY}]Fetching model files...[/]"):
            repo_info = api.model_info(repo_id, files_metadata=True, token=get_active_token())
            all_files = {sibling.rfilename: sibling for sibling in repo_info.siblings} if repo_info.siblings else {}
            files = list(all_files.keys())

        gguf_files = [f for f in files if f.lower().endswith('.gguf')]

        if not gguf_files:
            print_warning("No GGUF files found in this repository")
            return

        show_model_card_snippet(repo_id)

        quant_groups: Dict[str, List[str]] = {}
        quant_sizes: Dict[str, int] = {}

        for f in gguf_files:
            match = re.match(MULTIPART_REGEX, f)
            file_size = 0
            if f in all_files:
                size_val = all_files[f].size
                file_size = size_val if size_val is not None else 0

            if match:
                base = match.group(1)
                if base not in quant_groups:
                    quant_groups[base] = []
                    quant_sizes[base] = 0
                quant_groups[base].append(f)
                quant_sizes[base] = quant_sizes[base] + file_size
            else:
                quant_groups[f] = [f]
                quant_sizes[f] = file_size

        quant_list = sorted(quant_groups.keys())

        table = Table(
            title=f"Available Quantizations: {repo_id}",
            box=box.ROUNDED,
            border_style=Colors.PRIMARY
        )
        table.add_column("#", style=Colors.PRIMARY, justify="right", width=4)
        table.add_column("Quant Name", style="white")
        table.add_column("Files", justify="right", style=Colors.INFO)
        table.add_column("Size", justify="right", style=Colors.SUCCESS)

        for i, base in enumerate(quant_list, 1):
            num_files = len(quant_groups[base])
            size = quant_sizes.get(base, 0)
            size_str = format_size(size) if size > 0 else "Unknown"
            table.add_row(str(i), base, str(num_files), size_str)

        console.print(table)

        while True:
            quant_choice = _safe_input(
                f"\n[{Colors.PRIMARY}]Select quant #[/]1-{len(quant_list)} or '[q]uit': "
            )

            if quant_choice is None or quant_choice == 'q':
                return

            try:
                idx = int(quant_choice) - 1
                if 0 <= idx < len(quant_list):
                    base = quant_list[idx]
                    selected_files = quant_groups[base]

                    total_size = quant_sizes.get(base, 0)
                    size_str = format_size(total_size) if total_size > 0 else "Unknown"

                    console.print()
                    console.print(f"  [{Colors.PRIMARY}]d[/] - Download now")
                    console.print(f"  [{Colors.PRIMARY}]q[/] - Queue for later")
                    console.print(f"  [{Colors.PRIMARY}]c[/] - Cancel")

                    action_choice = _safe_input(f"\n[{Colors.PRIMARY}]Select action[/]: ")

                    if action_choice is None or action_choice.lower() == 'c':
                        return
                    elif action_choice.lower() == 'q':
                        queue_download(repo_id, selected_files, force=False)
                        print_success(f"Queued: {repo_id}")
                        return
                    elif action_choice.lower() == 'd':
                        if download_model(repo_id, selected_files):
                            offer_preset_generation(repo_id, selected_files)
                        return
                    else:
                        print_error("Invalid option")
                else:
                    print_error(f"Please enter a number between 1 and {len(quant_list)}")
            except ValueError:
                print_error("Invalid input. Enter a number or 'q'")

    except KeyboardInterrupt:
        return
    except Exception as e:
        print_error(f"Error: {e}")
        logging.error(f"Direct download failed for {repo_id}: {e}")


# ---------------------------------------------------------------------------
# Search and download flow
# ---------------------------------------------------------------------------
def search_and_download() -> None:
    """Search for models and download selected ones."""
    console.print(f"\n[bold {Colors.PRIMARY}]Search Models[/bold {Colors.PRIMARY}]")
    console.print("-" * 50)

    while True:
        query = _safe_input(f"[{Colors.PRIMARY}]Enter search keywords[/] (e.g., 'Llama', 'GPT') or 'q' to quit: ")

        if query is None or query.lower() == 'q':
            return

        if not query:
            print_warning("Query cannot be empty")
            continue

        try:
            with console.status(f"[{Colors.PRIMARY}]Searching Hugging Face Hub...[/]"):
                models = list(api.list_models(
                    search=query,
                    filter=["gguf"],
                    limit=MAX_MODELS,
                    sort="downloads",
                    direction=-1,
                    token=get_active_token()
                ))

            if not models:
                print_warning(f"No models found for '{query}'")
                continue

        except KeyboardInterrupt:
            return
        except Exception as e:
            print_error(f"Search failed: {e}")
            logging.error(f"Search failed for '{query}': {e}")
            continue

        table = Table(
            title=f"Search Results: {len(models)} models found",
            box=box.ROUNDED,
            border_style=Colors.PRIMARY
        )
        table.add_column("#", style=Colors.PRIMARY, justify="right", width=4)
        table.add_column("Model Name", style="white")
        table.add_column("Author", style=Colors.MUTED)
        table.add_column("Downloads", justify="right", style=Colors.INFO)

        for i, model in enumerate(models[:20], 1):
            author = model.author or model.id.split('/')[0]
            downloads = model.downloads or 0

            if downloads >= 1000000:
                download_str = f"{downloads/1000000:.1f}M"
            elif downloads >= 1000:
                download_str = f"{downloads/1000:.1f}k"
            else:
                download_str = str(downloads)

            table.add_row(str(i), model.id, author, download_str)

        if len(models) > 20:
            table.add_row("", f"[dim]... and {len(models) - 20} more[/]", "", "")

        console.print(table)

        selected_model = None
        while True:
            choice = _safe_input(
                f"\n[{Colors.PRIMARY}]Select model #[/]1-{min(len(models), 20)}, '[r]esearch', or '[q]uit': "
            )

            if choice is None or choice == 'q':
                return
            if choice == 'r':
                break

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected_model = models[idx]
                    break
                else:
                    print_error(f"Please enter a number between 1 and {min(len(models), 20)}")
            except ValueError:
                print_error("Invalid input. Enter a number, 'r', or 'q'")

        if choice == 'r' or selected_model is None:
            continue

        try:
            with console.status(f"[{Colors.PRIMARY}]Fetching model files...[/]"):
                repo_info = api.model_info(selected_model.id, files_metadata=True, token=get_active_token())
                all_files = {sibling.rfilename: sibling for sibling in repo_info.siblings} if repo_info.siblings else {}
                files = list(all_files.keys())

            gguf_files = [f for f in files if f.lower().endswith('.gguf')]

            if not gguf_files:
                print_warning("No GGUF files found in this repository")
                continue

            show_model_card_snippet(selected_model.id)

            quant_groups: Dict[str, List[str]] = {}
            quant_sizes: Dict[str, int] = {}

            for f in gguf_files:
                match = re.match(MULTIPART_REGEX, f)
                file_size = 0
                if f in all_files:
                    size_val = all_files[f].size
                    file_size = size_val if size_val is not None else 0

                if match:
                    base = match.group(1)
                    if base not in quant_groups:
                        quant_groups[base] = []
                        quant_sizes[base] = 0
                    quant_groups[base].append(f)
                    quant_sizes[base] = quant_sizes[base] + file_size
                else:
                    quant_groups[f] = [f]
                    quant_sizes[f] = file_size

            quant_list = sorted(quant_groups.keys())

            table = Table(
                title=f"Available Quantizations: {selected_model.id}",
                box=box.ROUNDED,
                border_style=Colors.PRIMARY
            )
            table.add_column("#", style=Colors.PRIMARY, justify="right", width=4)
            table.add_column("Quant Name", style="white")
            table.add_column("Files", justify="right", style=Colors.INFO)
            table.add_column("Size", justify="right", style=Colors.SUCCESS)

            for i, base in enumerate(quant_list, 1):
                num_files = len(quant_groups[base])
                size = quant_sizes.get(base, 0)
                size_str = format_size(size) if size > 0 else "Unknown"
                table.add_row(str(i), base, str(num_files), size_str)

            console.print(table)

            while True:
                quant_choice = _safe_input(
                    f"\n[{Colors.PRIMARY}]Select quant #[/]1-{len(quant_list)} or '[q]uit': "
                )

                if quant_choice is None or quant_choice == 'q':
                    break

                try:
                    idx = int(quant_choice) - 1
                    if 0 <= idx < len(quant_list):
                        base = quant_list[idx]
                        selected_files = quant_groups[base]

                        total_size = quant_sizes.get(base, 0)
                        size_str = format_size(total_size) if total_size > 0 else "Unknown"

                        console.print()
                        console.print(f"  [{Colors.PRIMARY}]d[/] - Download now")
                        console.print(f"  [{Colors.PRIMARY}]q[/] - Queue for later")
                        console.print(f"  [{Colors.PRIMARY}]c[/] - Cancel")

                        action_choice = _safe_input(f"\n[{Colors.PRIMARY}]Select action[/]: ")

                        if action_choice is None or action_choice.lower() == 'c':
                            return
                        elif action_choice.lower() == 'q':
                            queue_download(selected_model.id, selected_files, force=False)
                            print_success(f"Queued: {selected_model.id}")
                            return
                        elif action_choice.lower() == 'd':
                            if download_model(selected_model.id, selected_files):
                                offer_preset_generation(selected_model.id, selected_files)
                            return
                        else:
                            print_error("Invalid option")
                    else:
                        print_error(f"Please enter a number between 1 and {len(quant_list)}")
                except ValueError:
                    print_error("Invalid input. Enter a number or 'q'")

        except KeyboardInterrupt:
            return
        except Exception as e:
            print_error(f"Error fetching files: {e}")
            logging.error(f"File fetch failed for {selected_model.id}: {e}")


# ---------------------------------------------------------------------------
# Local model management
# ---------------------------------------------------------------------------
def list_downloaded_models() -> None:
    """List and manage downloaded models."""
    console.print(f"\n[bold {Colors.PRIMARY}]Local Models[/bold {Colors.PRIMARY}]")
    console.print("-" * 50)

    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, "ls", "-la", "/root/models/"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print_info("No models directory found")
            return
    except Exception:
        print_info("Cannot access models directory")
        return

    all_models = []
    total_size = 0

    try:
        result = subprocess.run(
            ["ssh", LXC_HOST, "ls", "/root/models/"],
            capture_output=True, text=True, timeout=10
        )
        authors = result.stdout.strip().split('\n') if result.returncode == 0 else []
        
        for author in authors:
            if not author or author.startswith('.'):
                continue
            
            result = subprocess.run(
                ["ssh", LXC_HOST, "ls", f"/root/models/{author}/"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                continue
                
            models = result.stdout.strip().split('\n')
            for model in models:
                if not model or model.startswith('.'):
                    continue
                
                model_path = f"/root/models/{author}/{model}"
                repo_id = f"{author}/{model}"
                
                result = subprocess.run(
                    ["ssh", LXC_HOST, "ls", model_path],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    continue
                    
                local_ggufs = sorted([f for f in result.stdout.strip().split('\n') if f.endswith('.gguf')])
                if not local_ggufs:
                    continue
                
                model_size = 0
                for f in local_ggufs:
                    result = subprocess.run(
                        ["ssh", LXC_HOST, "stat", "-c", "%s", f"{model_path}/{f}"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        try:
                            model_size += int(result.stdout.strip())
                        except ValueError:
                            pass
                
                total_size += model_size
                all_models.append((repo_id, model_path, local_ggufs, model_size))

    except Exception as e:
        print_error(f"Error listing models: {e}")
        return

    if not all_models:
        print_info("No models found in models/")
        return

    all_models.sort(key=lambda x: x[3], reverse=True)

    table = Table(
        title=f"Downloaded Models ({len(all_models)} models, {format_size(total_size)} total)",
        box=box.ROUNDED,
        border_style=Colors.PRIMARY
    )
    table.add_column("#", style=Colors.PRIMARY, justify="right", width=4)
    table.add_column("Model", style="white")
    table.add_column("Files", style=Colors.MUTED)
    table.add_column("Size", justify="right", style=Colors.SUCCESS)

    for i, (repo_id, _, local_ggufs, size) in enumerate(all_models, 1):
        files_str = f"{len(local_ggufs)} file(s)"
        size_str = format_size(size)
        table.add_row(str(i), repo_id, files_str, size_str)

    console.print(table)

    console.print(f"\n[{Colors.PRIMARY}]Actions:[/]")
    console.print(f"  [{Colors.PRIMARY}]r[/] - Redownload model(s) (resumes partial downloads)")
    console.print(f"  [{Colors.PRIMARY}]f[/] - Force redownload model(s) (fresh start)")
    console.print(f"  [{Colors.PRIMARY}]d[/] - Delete model(s)")
    console.print(f"  [{Colors.PRIMARY}]q[/] - Return to main menu")

    action = _safe_input(f"\n[{Colors.PRIMARY}]Select action[/]: ")

    if action is None or action == 'q':
        return

    if action not in ('r', 'f', 'd'):
        print_error("Invalid action")
        return

    model_choices = [f"{i+1}. {repo_id}" for i, (repo_id, _, _, _) in enumerate(all_models)]

    try:
        selected = inquirer.checkbox(
            "Select models (space to select, enter to confirm):",
            choices=model_choices
        )

        if not selected:
            print_info("No models selected")
            return

        for sel in selected:
            idx = int(sel.split('.')[0]) - 1
            repo_id, model_path, local_ggufs, _ = all_models[idx]

            if action in ('r', 'f'):
                force = (action == 'f')
                label = "Force redownloading" if force else "Redownloading"
                print_info(f"{label} {repo_id}...")
                logging.info(f"{label} {repo_id}: {local_ggufs}")
                download_model(repo_id, local_ggufs, is_update=True, force=force)
            elif action == 'd':
                confirm = _safe_input(
                    f"Delete {repo_id}? This cannot be undone. [y/N]: "
                )

                if confirm and confirm.lower() == 'y':
                    try:
                        subprocess.run(
                            ["ssh", LXC_HOST, "rm", "-rf", model_path],
                            capture_output=True, timeout=30
                        )
                        print_success(f"Deleted {repo_id}")
                        logging.info(f"Deleted model: {repo_id}")
                    except Exception as e:
                        print_error(f"Failed to delete: {e}")

    except KeyboardInterrupt:
        return
    except Exception as e:
        print_error(f"Selection failed: {e}")


# ---------------------------------------------------------------------------
# Main menu + entry point
# ---------------------------------------------------------------------------
def show_main_menu() -> Optional[str]:
    """Display main menu and return user choice."""
    show_status_panel()

    console.print(f"  [{Colors.PRIMARY}]1[/] - Search and Download")
    console.print(f"  [{Colors.PRIMARY}]2[/] - Direct Download (paste repo ID/URL)")
    console.print(f"  [{Colors.PRIMARY}]3[/] - Manage Local Models")
    console.print(f"  [{Colors.PRIMARY}]4[/] - Download Queue")
    console.print(f"  [{Colors.PRIMARY}]5[/] - Maintenance")
    console.print(f"  [{Colors.PRIMARY}]6[/] - Configure Token")
    console.print(f"  [{Colors.PRIMARY}]7[/] - Exit")

    choice = _safe_input(f"\n[{Colors.PRIMARY}]Select option[/] (1-7): ")
    return choice


def main():
    """Main application entry point."""
    signal.signal(signal.SIGINT, _sigint_handler)

    _init_token()

    _clean_stale_incomplete_files()

    download_thread = threading.Thread(target=_download_worker, daemon=True)
    download_thread.start()

    try:
        while True:
            choice = show_main_menu()

            if choice is None:
                console.print(f"\n[{Colors.SUCCESS}]Goodbye![/]\n")
                logging.info("Application exited via Ctrl+C at main menu")
                break
            elif choice == '7' or choice.lower() == 'q':
                console.print(f"\n[{Colors.SUCCESS}]Goodbye![/]\n")
                logging.info("Application exited normally")
                break
            elif choice == '1':
                search_and_download()
            elif choice == '2':
                direct_download()
            elif choice == '3':
                list_downloaded_models()
            elif choice == '4':
                show_queue_menu()
            elif choice == '5':
                show_maintenance_menu()
            elif choice == '6':
                configure_token()
            else:
                print_error("Invalid option. Please enter 1-7.")

    except KeyboardInterrupt:
        console.print()
        logging.info("User exited with Ctrl+C")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
