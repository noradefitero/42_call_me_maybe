# llm_sdk

`llm_sdk` wraps Hugging Face `transformers` to run causal language models
locally, no API and no GPU required. It exposes a single class,
`Small_LLM_Model`, built for quick experiments on modest memory. The code lives
in `lib/llm_sdk`.

## Requirements

- Python >= 3.10
- torch >= 2.0
- transformers >= 4.40
- huggingface-hub >= 0.20

## Installation

The main project already depends on it by path (`lib/llm_sdk`, see
`[tool.uv.sources]` in `pyproject.toml`). To install it on its own:

```bash
uv pip install -e lib/llm_sdk
```

## Usage

```python
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()  # Qwen/Qwen3-0.6B by default
```

Building the model downloads the tokenizer and weights from Hugging Face Hub on
first run; after that they stay in cache.

### Tokenize and decode

| Method | What it does |
|---|---|
| `encode(text: str) -> torch.Tensor` | Tokenizes `text` and returns a 2-D `input_ids` tensor on the model's device. |
| `decode(ids) -> str` | The inverse. Accepts a tensor or a list of ints and returns the text without special tokens. |

### Getting logits

`get_logits_from_input_ids(input_ids: list[int]) -> list[float]` returns the
raw logits (no softmax) for the last token of the sequence. The model runs in
inference mode, with no gradients.

### Paths to the tokenizer files

These methods return the local path of a tokenizer file. If the file is not
already cached, it is downloaded with `hf_hub_download`.

- `get_path_to_vocab_file()`: path to `vocab.json`
- `get_path_to_merges_file()`: path to `merges.txt`
- `get_path_to_tokenizer_file()`: path to `tokenizer.json`

Useful when another tool needs to read those files directly, rather than going
through the transformers tokenizer.

## Default device and precision

If you don't pass `device`, the library picks one: `mps` on a Mac, `cuda` when a
GPU is available, and `cpu` otherwise. On `mps` or `cuda` it uses `float16` to
keep memory in check; on `cpu` it stays on `float32` for compatibility.

Both are keyword-only arguments, so you can force them:

```python
model = Small_LLM_Model(device="cpu", dtype=torch.float32)
```

## Implementation notes

- The model loads in eval mode (`eval()`) and with gradients disabled.
- `trust_remote_code=True` by default, for models that need their own code.
- If the tokenizer has no `pad_token`, the `eos_token` is used instead.
- `transformers` logging is silenced to error level.
