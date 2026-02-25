# GGUF Model Manager

A CLI tool for downloading and managing GGUF models from Hugging Face, built for llama.cpp setups.

## Features

- Search and browse models from Hugging Face
- Download GGUF files with Rich progress bars
- Handles multi-part models automatically
- HuggingFace token management
- Lists and manages downloaded models
- Resume interrupted downloads

## Requirements

```
pip install -r requirements.txt
```

## Usage

```bash
python3 model_manager.py
```

Interactive menu — search, select, download.

## Hardware

Built and tested on:
- 2× RTX 5060 Ti 16GB (CUDA)
- Dual Xeon E5-2680 v4
- 128GB DDR4 2400MHz
- llama.cpp (CUDA build)

## Notes

- Models download to `./models/` by default
- Multipart GGUFs (e.g. 122B split across 3 files) are handled automatically
- Pairs well with [llama-server](https://github.com/ggml-org/llama.cpp) and a `presets.ini` setup
