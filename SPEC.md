# gguf-model-manager v4 — Development Spec

## Overview
Single-file Python CLI tool that runs on the llama LXC (`/root/model_manager.py`).
Manages GGUF model downloads, presets, and server status for a llama.cpp router setup.

**Constraints:**
- Single file (`model_manager.py`) — no additional modules
- Dependencies: `huggingface-hub`, `rich`, `inquirer` (no new deps)
- Must run on the LXC directly at `/root/` (not over SSH)
- Python 3.10+

**Do not touch:**
- HuggingFace search and quant selection flow — works well
- Resume/`.incomplete` file logic — works correctly
- Force redownload logic
- Token management (`configure_token`, `_init_token`, etc.)
- Delete model flow
- Rich UI color scheme / `Colors` class

---

## Change 1 — Status Panel (main menu header)

Replace the static `print_header()` with a live status panel shown at the top of every main menu render.

### Show:
1. **Disk** — `df /root/models` → free GB / total GB, color coded:
   - Green if >100GB free
   - Yellow if 50–100GB free
   - Red if <50GB free

2. **GPU VRAM** — run `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits`
   - Show GPU0 and GPU1 as mini bars: `GPU0: 754 / 16311 MB`

3. **Server** — `curl -s http://localhost:8080/health` (timeout 2s)
   - If `{"status":"ok"}`: show green "● running", then fetch loaded model from `curl -s -H "Authorization: Bearer Smiledog69!" http://localhost:8080/slots` → extract `model` field from first slot
   - If unreachable: show red "● offline"

4. **Download queue** — if queue is non-empty, show: `⬇ Downloading: <repo_id> | Queue: N pending`
   - If paused: `⏸ Queue paused | N pending`
   - If idle: omit this line

### Implementation:
```python
def show_status_panel():
    """Render status panel using Rich Panel/Table. Called at start of each main menu loop."""
```

Keep it fast — use `subprocess` with 2s timeout for nvidia-smi and curl. Fail silently per item if any command errors.

---

## Change 2 — Background Download Queue

Current architecture: synchronous, blocks main thread during download.

### New architecture:
Add a background worker thread that processes a queue of `(repo_id, files, force)` tuples.

```python
import threading
import queue as _queue

_dl_queue: _queue.Queue = _queue.Queue()
_dl_status = {
    'current': None,      # repo_id string or None
    'pending': [],        # list of repo_id strings
    'completed': [],      # list of repo_id strings
    'failed': [],         # list of repo_id strings
    'paused': False,
}
_dl_lock = threading.Lock()
```

**Worker thread** (`_download_worker`):
- Runs as daemon thread, started in `main()` before the menu loop
- Checks `_dl_status['paused']` before dequeuing; if paused, sleeps 1s and retries
- Pops item, sets `_dl_status['current']`, calls existing `download_with_progress()`
- On completion: clears `current`, appends to `completed` or `failed`
- When queue is empty: sets `current = None`, waits

**Queue management functions:**
```python
def queue_download(repo_id, files, force=False):
    """Add to queue. Appends repo_id to _dl_status['pending']."""

def pause_queue():
    """Set paused=True. Current in-progress download finishes, next won't start."""

def resume_queue():
    """Set paused=False."""

def clear_queue():
    """Drain _dl_queue and clear _dl_status['pending']. Does not cancel current download."""
```

**Interrupt behaviour**: Ctrl+C during a download exits the download (existing `_sigint_handler` / `os._exit(0)` path). The `.incomplete` file stays on disk and will resume next time that repo_id is queued. This is the existing behaviour — preserve it.

### Menu changes:
- After quant selection + confirm, show:
  ```
  [D]ownload now  [Q]ueue  [C]ancel
  ```
  "Download now" blocks as before. "Queue" adds to background queue and returns to menu.

- Add new main menu option **"Download Queue"** (option 3, shift existing 3→Configure Token to 4, Exit to 5):
  Shows: current download, pending list, completed, failed
  Actions: `[P]ause`, `[R]esume`, `[C]lear pending`, `[Q]uit`

---

## Change 3 — Direct Repo Input

Add main menu option **"Direct Download"** (option 2, shift others down):

```
Enter HuggingFace repo ID or URL:
> unsloth/Qwen3.5-27B-GGUF
  OR
> https://huggingface.co/unsloth/Qwen3.5-27B-GGUF
```

Parse URL to extract `author/repo` format. Then go directly to the existing file listing / quant selection flow (skip search). This is the most common real-world workflow — you already know which repo you want.

---

## Change 4 — llama-fit-params Integration

Called after every successful download, before `offer_preset_generation`.

### Binary search for max ctx-size:

```python
def find_max_ctx_size(model_path: str, n_gpu_layers: int = 99,
                      tensor_split: str = "1,1") -> Optional[int]:
    """
    Binary search for maximum ctx-size that fits in VRAM.
    Calls /usr/local/bin/llama-fit-params.
    Returns max fitting ctx-size, or None if even minimum fails.
    """
```

**Strategy:**
1. Try `ctx-size = 131072` first
2. If fails: try `65536`, `32768`, `16384`, `8192` in sequence
3. "Fits" = llama-fit-params exits 0 AND output does not contain any of: `"exceeds"`, `"not enough"`, `"insufficient"`, `"OOM"` (check actual binary output first — see note below)
4. Return highest fitting value

**Command:**
```bash
/usr/local/bin/llama-fit-params \
  -m <model_path> \
  --ctx-size <N> \
  --n-gpu-layers <n_gpu_layers> \
  --tensor-split <tensor_split>
```

**⚠️ Agent must check actual llama-fit-params output format before writing the parser.**
Run: `llama-fit-params --help` and `llama-fit-params -m /root/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf --ctx-size 131072 --n-gpu-layers 99` and examine stdout/stderr and exit code to determine how to detect "fits" vs "doesn't fit".

**After finding max ctx-size:**
- Pass it to `_detect_preset_params()` as an override — don't let the heuristic override the measured value
- Show the user: `Max context: 131072 tokens (llama-fit-params)`

**UX:**
- Show a spinner while testing
- If `llama-fit-params` not found at `/usr/local/bin/llama-fit-params`: warn and fall back to heuristic
- Add a `--tensor-split` detection step: check existing presets for this model family to guess the right split

---

## Change 5 — Preset Organization

Replace the simple `append` in `offer_preset_generation` with an organized insert.

### Family detection:
```python
def _detect_family(preset_name: str) -> str:
    """
    Extract base family from preset name.
    Examples:
      'Qwen3.5-27B' → 'Qwen3.5'
      'Qwen3.5-27B-Q5' → 'Qwen3.5'
      'GPT-OSS-120B-High' → 'GPT-OSS'
      'GLM-4.7-Flash' → 'GLM'
      'Nemotron-3-Nano' → 'Nemotron'
    Strip: trailing -Q4, -Q5, -Q6, -High, -Thinking, param counts (27B, 120B, etc.)
    """
```

### Param count extraction:
```python
def _extract_param_count(preset_name: str) -> float:
    """
    Extract parameter count for sorting.
    'Qwen3.5-0.8B' → 0.8, 'Qwen3.5-27B' → 27.0, 'Qwen3.5-35B-A3B' → 35.0
    Uses the total param count (first number before 'B'), not active params.
    Returns 999.0 if not parseable (sorts to end).
    """
```

### Organized write:
```python
def write_preset_organized(preset_name: str, params: dict) -> None:
    """
    Insert preset into presets.ini in the correct position:
    1. Parse entire presets.ini into ordered list of (name, content) blocks
    2. Find the family group for this preset
    3. Insert after last member of same family, ordered by param count ascending
    4. If no family members exist yet: append at end
    5. Write entire file back out
    """
```

### Separator comments:
Between family groups, add a blank line. Within a family group, presets are contiguous. No other changes to existing preset content.

---

## Change 6 — Model Card Snippet

In the quant selection flow, after the user selects a model but before quant selection, fetch and display a brief excerpt:

```python
def show_model_card_snippet(repo_id: str) -> None:
    """
    Fetch model card (README.md) via HF API.
    Extract and display:
    - First non-header paragraph (skip # headings, up to 300 chars)
    - Any lines containing: ctx, context, n_ctx, temperature, recommended
    Show in a dim panel. Fail silently if fetch fails.
    """
```

This is informational only — the user still edits the preset manually or accepts the auto-generated one.

---

## Change 7 — Maintenance (llama.cpp + Whisper updates)

Add main menu option **"Maintenance"**.

### Version check:

```python
def get_llamacpp_version() -> Optional[str]:
    """Run `llama-server --version`, parse and return version string (e.g. 'b5234')."""

def get_llamacpp_latest() -> Optional[str]:
    """
    Fetch latest release from GitHub API:
    GET https://api.github.com/repos/ggml-org/llama.cpp/releases/latest
    Return tag_name (e.g. 'b5234'). Timeout 5s. Fail silently → return None.
    """
```

### Maintenance screen:

```
═══════════════════════════════════
 Maintenance
───────────────────────────────────
 llama.cpp    installed: b5180   latest: b5234   [UPDATE AVAILABLE]
 whisper.cpp  installed: unknown latest: unknown  [check failed]
───────────────────────────────────
  [1] Update llama.cpp
  [2] Update whisper.cpp
  [q] Back
```

- If installed == latest: show green "up to date"
- If latest > installed: show yellow "UPDATE AVAILABLE"
- If check failed: show dim "check failed"

### Update flow:

When user selects update:
1. Show warning: `"⚠ This will stop the running llama-server and rebuild (~10 min). Continue? [y/N]"`
2. If confirmed: stream output of `bash /root/update-llama.sh` (or `update-whisper.sh`) line by line using `subprocess.Popen` with `stdout=PIPE, stderr=STDOUT`
3. Show each line as it arrives using `console.print`
4. On completion: show success or error
5. Do NOT restart the server — user does that manually

**whisper.cpp version check**: `whisper-server --version` (check if flag exists, fall back to "unknown")
**whisper GitHub**: `https://api.github.com/repos/ggerganov/whisper.cpp/releases/latest`

---

## Final Menu Structure

```
═══════════════════════════════════
 GGUF Model Manager v4
───────────────────────────────────
 Disk: 95 GB free (88% used) ⚠
 GPU0: 754/16311 MB  GPU1: 1817/16311 MB
 Server: ● running [GLM-4.7-Flash]
 ⬇ Queue: idle
═══════════════════════════════════

  1 - Search and Download
  2 - Direct Download (paste repo ID/URL)
  3 - Manage Local Models
  4 - Download Queue
  5 - Maintenance
  6 - Configure Token
  7 - Exit
```

---

## Testing Checklist (agent must verify before closing)

- [ ] `python3 model_manager.py` launches without error on the LXC (`/root/`)
- [ ] Status panel renders correctly (test with server running and server stopped)
- [ ] Direct download: `unsloth/Qwen3.5-0.8B-GGUF` → quant select → queue → appears in queue view
- [ ] llama-fit-params runs and produces a ctx-size (test with smallest model in `/root/models/`)
- [ ] Preset written to `/root/presets.ini` in correct family group position
- [ ] Download queue: queue 2 models, pause, verify second doesn't start until resumed
- [ ] Existing flows unchanged: search, redownload, force, delete
- [ ] `Ctrl+C` during download exits cleanly, `.incomplete` file present for resume

---

## Files

- **Edit**: `/root/projects/gguf-model-manager/model_manager.py`
- **Do not create** additional files
- **Test on**: `ssh root@192.168.0.113` — run from `/root/` directory
- **Presets file**: `/root/presets.ini`
- **Models dir**: `/root/models/`
- **llama-fit-params**: `/usr/local/bin/llama-fit-params`
- **llama-server API**: `http://localhost:8080` with key `Smiledog69!`
- **nvidia-smi**: available on PATH
