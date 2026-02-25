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
import sys
import threading
import time
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
APP_VERSION = "3.0"
PRESETS_FILE = os.path.expanduser("~/presets.ini")


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
# Shutdown handling
# ---------------------------------------------------------------------------
_in_download = False


def _sigint_handler(signum, frame):
    """Handle Ctrl+C. Exits immediately during downloads, gracefully otherwise."""
    if _in_download:
        # During a download: kill instantly. No I/O here -- Rich's live
        # renderer may hold the console lock, so writing would deadlock.
        # The HTTP downloader leaves .incomplete files that resume next run.
        os._exit(0)
    else:
        # Outside a download: raise normal KeyboardInterrupt so _safe_input
        # and menu loops can handle it gracefully.
        raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------
_active_token: Optional[str] = None


def get_active_token() -> Optional[str]:
    """Return the currently active HF token (cached in memory)."""
    return _active_token


def _resolve_token() -> Optional[str]:
    """Resolve the HF token from environment / cache file.

    Precedence:
      1. HF_TOKEN environment variable
      2. ~/.cache/huggingface/token file (written by `huggingface-cli login`)
    """
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

    # Validate
    with console.status(f"[{Colors.PRIMARY}]Validating token...[/]"):
        username = _get_token_username(token)

    if not username:
        print_error("Invalid token -- could not authenticate with HuggingFace")
        return

    # Save to HF cache (~/.cache/huggingface/token)
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

        # Also clear env var for this session
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
    # Try to get username (cached call -- keep it fast)
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
        usage = shutil.disk_usage("models") if os.path.exists("models") else shutil.disk_usage(".")
        return usage.free / 1e9, usage.total / 1e9
    except Exception:
        return 0.0, 0.0


def count_downloaded_models() -> int:
    """Count number of downloaded models."""
    if not os.path.exists('models'):
        return 0
    count = 0
    for author in os.listdir('models'):
        author_path = os.path.join('models', author)
        if os.path.isdir(author_path):
            count += len([d for d in os.listdir(author_path)
                         if os.path.isdir(os.path.join(author_path, d))])
    return count


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
    """Remove .incomplete files older than STALE_THRESHOLD_HOURS from models/.

    These are leftover partial downloads (especially from Xet which cannot
    resume them). The HTTP downloader creates new .incomplete files that it
    CAN resume, so only stale ones are removed.
    """
    if not os.path.exists("models"):
        return

    threshold = time.time() - (STALE_THRESHOLD_HOURS * 3600)
    cleaned = 0
    freed = 0

    for dirpath, dirnames, filenames in os.walk("models"):
        for fname in filenames:
            if not fname.endswith(".incomplete"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                stat = os.stat(fpath)
                if stat.st_mtime < threshold:
                    # Use actual disk usage (handles sparse files correctly)
                    freed += stat.st_blocks * 512 if hasattr(stat, 'st_blocks') else stat.st_size
                    os.remove(fpath)
                    cleaned += 1
            except OSError:
                continue

    if cleaned > 0:
        logging.info(f"Cleaned {cleaned} stale .incomplete file(s), freed {format_size(freed)}")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def print_header():
    """Print application header with stats."""
    free_gb, total_gb = get_disk_space()
    model_count = count_downloaded_models()
    token_status = _token_status_str()

    header_text = Text()
    header_text.append(f"{APP_NAME} v{APP_VERSION}\n", style=f"bold {Colors.PRIMARY}")
    header_text.append("-" * 40 + "\n", style=Colors.MUTED)
    header_text.append(f"Disk: {format_size(int(free_gb * 1e9))} free", style=Colors.SUCCESS)
    header_text.append(f"  |  Models: {model_count} cached", style=Colors.INFO)
    header_text.append(f"  |  Token: {token_status}", style=Colors.SUCCESS if get_active_token() else Colors.WARNING)

    panel = Panel(
        Align.center(header_text),
        border_style=Colors.PRIMARY,
        padding=(1, 2)
    )
    console.print(panel)
    console.print()


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
    """
    A tqdm-compatible class that bridges huggingface_hub's download progress
    to Rich Progress bars. Used as tqdm_class in snapshot_download().

    snapshot_download creates two bars using this class:
      1. A bytes_progress bar (total starts at 0, grows as files are discovered)
      2. A file-count bar via thread_map (iterable over filenames)

    The internal _AggregatedTqdm class mutates bytes_progress.total directly
    then calls bytes_progress.refresh(), so we must handle mutable .total
    and sync it back to Rich on refresh().
    """

    # Shared Rich Progress instance (set before download, cleared after)
    _progress: Optional[Progress] = None
    _lock = None

    @classmethod
    def set_progress(cls, progress: Optional[Progress]):
        cls._progress = progress

    @classmethod
    def get_lock(cls):
        """Required by tqdm.contrib.concurrent.thread_map."""
        if cls._lock is None:
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        """Required by tqdm.contrib.concurrent.thread_map."""
        cls._lock = lock

    def __init__(self, *args, **kwargs):
        # Pop kwargs that tqdm accepts but we handle differently
        kwargs.pop("name", None)
        kwargs.pop("unit_scale", None)
        kwargs.pop("unit_divisor", None)

        self.total = kwargs.get("total", 0) or 0
        self.n = kwargs.get("initial", 0) or 0
        self.desc = kwargs.get("desc", "")
        self.unit = kwargs.get("unit", "it")
        self.disable = kwargs.get("disable", False)
        self.task_id = None

        # Handle iterable (used by thread_map for file-count bar)
        self._iterable = args[0] if args else kwargs.get("iterable", None)

        if self._progress is not None and not self.disable:
            is_bytes = self.unit == "B"
            self.task_id = self._progress.add_task(
                description=self.desc[:60] if self.desc else "Downloading",
                total=self.total if self.total > 0 else None,
                completed=self.n,
                visible=is_bytes  # Only show the bytes bar, hide the file-count bar
            )

    def __iter__(self):
        """Iterate and yield items (used by thread_map for file-count progress)."""
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
        """Advance the progress bar by n units."""
        if n is None or self.disable:
            return
        self.n += n
        if self.task_id is not None and self._progress is not None:
            self._progress.update(self.task_id, advance=n)

    def refresh(self):
        """Sync .total changes back to the Rich task (called after .total is mutated)."""
        if self.task_id is not None and self._progress is not None:
            self._progress.update(self.task_id, total=self.total, completed=self.n)

    def set_description(self, desc="", refresh=True):
        """Update the description label."""
        self.desc = desc
        if self.task_id is not None and self._progress is not None:
            self._progress.update(self.task_id, description=desc[:60])

    def set_description_str(self, desc="", refresh=True):
        self.set_description(desc, refresh)

    def close(self):
        """Mark the task as finished."""
        if self.task_id is not None and self._progress is not None:
            # Ensure bar shows 100% on completion
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
    """Download files using snapshot_download with Rich progress bars.

    Args:
        repo_id: HuggingFace repository ID.
        local_dir: Local directory to download into.
        patterns: File patterns to match (allow_patterns).
        total_size: Expected total size in bytes (for display).
        action: Label for log/display (e.g. "Downloading", "Resuming").
        force: If True, force fresh download (ignore partial files).

    Returns:
        True if successful, False otherwise.
    """
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

        # Re-register signal handler in case any library import (e.g. Rust
        # extensions) overwrote it with SA_RESTART, which would prevent
        # Python from ever seeing the signal while blocked in syscalls.
        signal.signal(signal.SIGINT, _sigint_handler)

        try:
            start_time = time.time()

            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=patterns,
                tqdm_class=RichTqdm,  # type: ignore[arg-type]
                force_download=force,
                token=get_active_token(),
                max_workers=4,
            )

            elapsed = time.time() - start_time

            # Calculate average speed
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
# Preset auto-generation
# ---------------------------------------------------------------------------
def _detect_preset_params(repo_id: str, selected_files: List[str]) -> dict:
    """Infer sensible preset parameters from repo_id and selected files."""
    all_text = (repo_id + " " + " ".join(selected_files)).lower()

    # Detect MoE (active-param suffix like A3B, A10B, A22B or keyword moe)
    is_moe = bool(re.search(r'-a\d+b|moe|_moe', all_text))

    # Extract total param size (e.g. 35B, 27B, 122B)
    size_match = re.search(r'(\d+(?:\.\d+)?)b', all_text)
    total_b = float(size_match.group(1)) if size_match else 7.0

    # Detect if model needs reasoning format (thinking/reasoning models)
    is_reasoning = bool(re.search(r'thinking|reason|qwq|deepseek-r|skywork-o', all_text))

    # Detect if multi-part (needs first shard path)
    first_file = sorted(selected_files)[0] if selected_files else ''
    multipart_match = re.search(r'(.+)-(\d{5})-of-(\d{5})\.gguf$', first_file, re.IGNORECASE)

    # Build model path
    if multipart_match:
        model_path = f"models/{repo_id}/{first_file}"
    else:
        model_path = f"models/{repo_id}/{first_file}"

    # Parameter defaults based on type/size
    params = {
        'model': model_path,
        'n-gpu-layers': 99,
        'temp': 0.7,
        'batch-size': 4096,
        'ubatch-size': 2048,
        'jinja': 'on',
    }

    if is_moe:
        params['ctx-size'] = 8192
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
        params['ctx-size'] = 65536
        params['tensor-split'] = '1,1'
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
    # Take last path component, strip common suffixes
    name = repo_id.split('/')[-1]
    for suffix in ['-GGUF', '-gguf', '-Instruct', '-instruct', '-Chat', '-chat']:
        name = name.replace(suffix, '')
    return name


def offer_preset_generation(repo_id: str, selected_files: List[str]) -> None:
    """After a successful download, offer to write a preset entry to presets.ini."""
    if not os.path.exists(PRESETS_FILE):
        print_warning(f"presets.ini not found at {PRESETS_FILE} — skipping preset generation")
        return

    preset_name = _preset_name_from_repo(repo_id)

    # Check if preset already exists
    with open(PRESETS_FILE, 'r') as f:
        existing = f.read()
    if f'[{preset_name}]' in existing:
        print_info(f"Preset [{preset_name}] already exists in presets.ini — skipping")
        return

    params = _detect_preset_params(repo_id, selected_files)

    # Show preview
    console.print(f"\n[bold green]Preset preview — [{preset_name}][/bold green]")
    for k, v in params.items():
        console.print(f"  [dim]{k}[/dim] = {v}")

    answer = _safe_input("\nWrite this preset to presets.ini? [Y/n/e(dit name)]: ")
    if answer is None or answer.lower() == 'n':
        return
    if answer.lower() == 'e':
        new_name = _safe_input("Preset name: ")
        if new_name:
            preset_name = new_name.strip()

    # Build preset block
    lines = [f"\n[{preset_name}]"]
    for k, v in params.items():
        lines.append(f"{k} = {v}")
    block = "\n".join(lines) + "\n"

    with open(PRESETS_FILE, 'a') as f:
        f.write(block)

    print_success(f"Preset [{preset_name}] written to {PRESETS_FILE}")
    logging.info(f"Auto-generated preset [{preset_name}] for {repo_id}")

def download_model(repo_id: str, selected_files: List[str],
                   is_update: bool = False, force: bool = False) -> bool:
    """
    Download selected GGUF files for a model with progress display.

    When force=False (default), partial downloads from previous interrupted
    sessions are automatically resumed via huggingface_hub's built-in
    .incomplete file mechanism.

    Args:
        repo_id: Repository ID (e.g., 'author/model-name')
        selected_files: List of files to download
        is_update: Whether this is a redownload from local models view
        force: If True, delete existing files and start fresh (no resume)

    Returns:
        True if successful, False otherwise
    """
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    local_dir = f"models/{repo_id}"
    os.makedirs(local_dir, exist_ok=True)

    # Calculate total size
    total_size = 0
    try:
        repo_info = api.model_info(repo_id, files_metadata=True, token=get_active_token())
        if repo_info.siblings:
            for sibling in repo_info.siblings:
                if sibling.size and sibling.rfilename in selected_files:
                    total_size += sibling.size
    except Exception as e:
        logging.warning(f"Could not fetch size info for {repo_id}: {e}")

    # Check disk space
    if total_size > 0:
        try:
            free_gb, _ = get_disk_space()
            required_gb = total_size * 1.1 / 1e9  # 10% buffer
            if free_gb < required_gb:
                print_warning(f"Low disk space: {free_gb:.1f} GB free, {required_gb:.1f} GB required")
                return False
        except Exception as e:
            logging.warning(f"Disk space check failed: {e}")

    # Handle force redownload -- delete existing files AND .incomplete partials
    # so snapshot_download starts fresh. Without force, existing .incomplete
    # files are resumed automatically via HTTP Range headers.
    if force:
        print_info(f"Force redownload: removing {len(selected_files)} file(s)...")
        for f in selected_files:
            local_file = os.path.join(local_dir, f)
            if os.path.exists(local_file):
                os.remove(local_file)
                console.print(f"[dim]  Removed: {os.path.basename(f)}[/dim]")
        # Also clean .incomplete files in the local cache
        cache_dir = os.path.join(local_dir, ".cache", "huggingface", "download")
        if os.path.exists(cache_dir):
            for fname in os.listdir(cache_dir):
                if fname.endswith(".incomplete"):
                    try:
                        os.remove(os.path.join(cache_dir, fname))
                    except OSError:
                        pass

    patterns = [f"*{os.path.basename(f)}" for f in selected_files]
    action = "Redownloading" if is_update else "Downloading"

    logging.info(f"Starting {action.lower()} for {repo_id}: {selected_files} (force={force})")
    return download_with_progress(repo_id, local_dir, patterns, total_size, action, force=force)


# ---------------------------------------------------------------------------
# Search and download flow
# ---------------------------------------------------------------------------
def search_and_download() -> None:
    """Search for models and download selected ones."""
    console.print(f"\n[bold {Colors.PRIMARY}]Search Models[/bold {Colors.PRIMARY}]")
    console.print("-" * 50)

    while True:
        # Get search query
        query = _safe_input(f"[{Colors.PRIMARY}]Enter search keywords[/] (e.g., 'Llama', 'GPT') or 'q' to quit: ")

        if query is None or query.lower() == 'q':
            return

        if not query:
            print_warning("Query cannot be empty")
            continue

        # Search for models
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

        # Display results
        table = Table(
            title=f"Search Results: {len(models)} models found",
            box=box.ROUNDED,
            border_style=Colors.PRIMARY
        )
        table.add_column("#", style=Colors.PRIMARY, justify="right", width=4)
        table.add_column("Model Name", style="white")
        table.add_column("Author", style=Colors.MUTED)
        table.add_column("Downloads", justify="right", style=Colors.INFO)

        for i, model in enumerate(models[:20], 1):  # Show top 20
            author = model.author or model.id.split('/')[0]
            downloads = model.downloads or 0

            # Format large download numbers
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

        # Get user selection
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

        # Fetch and display available files
        try:
            with console.status(f"[{Colors.PRIMARY}]Fetching model files...[/]"):
                repo_info = api.model_info(selected_model.id, files_metadata=True, token=get_active_token())
                all_files = {sibling.rfilename: sibling for sibling in repo_info.siblings} if repo_info.siblings else {}
                files = list(all_files.keys())

            gguf_files = [f for f in files if f.lower().endswith('.gguf')]

            if not gguf_files:
                print_warning("No GGUF files found in this repository")
                continue

            # Group by quant
            quant_groups: Dict[str, List[str]] = {}
            quant_sizes: Dict[str, int] = {}

            for f in gguf_files:
                match = re.match(MULTIPART_REGEX, f)

                # Get file size with proper None handling
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

            # Display quants
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

            # Select quant
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

                        # Confirm download
                        total_size = quant_sizes.get(base, 0)
                        size_str = format_size(total_size) if total_size > 0 else "Unknown"

                        confirm = _safe_input(
                            f"\nDownload [bold]{base}[/] ({len(selected_files)} file(s), {size_str})? [Y/n]: "
                        )

                        if confirm is None:
                            break
                        if confirm.lower() in ('', 'y', 'yes'):
                            logging.info(f"Selected quant for {selected_model.id}: {selected_files}")
                            if download_model(selected_model.id, selected_files):
                                offer_preset_generation(selected_model.id, selected_files)
                        break
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

    if not os.path.exists('models'):
        print_info("No models directory found")
        return

    # Collect all models
    all_models = []
    total_size = 0

    for author in os.listdir('models'):
        author_path = os.path.join('models', author)
        if os.path.isdir(author_path):
            for model in os.listdir(author_path):
                model_path = os.path.join(author_path, model)
                if os.path.isdir(model_path):
                    repo_id = f"{author}/{model}"
                    local_ggufs = sorted([f for f in os.listdir(model_path) if f.endswith('.gguf')])
                    if local_ggufs:
                        # Calculate total size
                        model_size = 0
                        for f in local_ggufs:
                            fpath = os.path.join(model_path, f)
                            if os.path.exists(fpath):
                                model_size += os.path.getsize(fpath)
                        total_size += model_size
                        all_models.append((repo_id, model_path, local_ggufs, model_size))

    if not all_models:
        print_info("No models found in models/")
        return

    # Sort by size (largest first)
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

    # Model management options
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

    # Select models
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
                        shutil.rmtree(model_path)
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
    console.print(f"\n[bold {Colors.PRIMARY}]Main Menu[/bold {Colors.PRIMARY}]")
    console.print("-" * 50)

    console.print(f"  [{Colors.PRIMARY}]1[/] - Search and Download Models")
    console.print(f"  [{Colors.PRIMARY}]2[/] - Manage Local Models")
    console.print(f"  [{Colors.PRIMARY}]3[/] - Configure Token")
    console.print(f"  [{Colors.PRIMARY}]4[/] - Exit")

    choice = _safe_input(f"\n[{Colors.PRIMARY}]Select option[/] (1-4): ")
    return choice


def main():
    """Main application entry point."""
    # Install signal handler for clean Ctrl+C
    signal.signal(signal.SIGINT, _sigint_handler)

    # Initialize token (non-interactive, reads env/cache)
    _init_token()

    # Clean stale partial downloads from previous sessions
    _clean_stale_incomplete_files()

    try:
        print_header()

        while True:
            choice = show_main_menu()

            if choice is None:
                # Ctrl+C at the main menu -- exit cleanly
                console.print(f"\n[{Colors.SUCCESS}]Goodbye![/]\n")
                logging.info("Application exited via Ctrl+C at main menu")
                break
            elif choice == '4' or choice.lower() == 'q':
                console.print(f"\n[{Colors.SUCCESS}]Goodbye![/]\n")
                logging.info("Application exited normally")
                break
            elif choice == '1':
                search_and_download()
            elif choice == '2':
                list_downloaded_models()
            elif choice == '3':
                configure_token()
                # Refresh header after token change
                print_header()
            else:
                print_error("Invalid option. Please enter 1, 2, 3, or 4.")

    except KeyboardInterrupt:
        console.print()
        logging.info("User exited with Ctrl+C")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
