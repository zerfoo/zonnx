# zonnx

This tool is a standalone command-line utility responsible for converting machine learning models between the ONNX format and the Zerfoo Model Format (ZMF). It also provides functionality to download ONNX models directly from HuggingFace Hub.

## Features

- **ONNX → ZMF conversion (fast, deterministic)**: Produce portable ZMF artifacts fully decoupled from the `zerfoo` runtime.
- **Model inspection (ONNX and ZMF)**: Introspect model metadata, IOs, nodes and tensor stats. Output is JSON-friendly; `--pretty` planned.
- **HuggingFace integration**: Download ONNX models and common tokenizer files in one step.
- **CGO-free builds**: Ships as a single static binary. Easy to distribute and run in minimal containers.
- **Clean separation of concerns**: Converter lives outside the training/runtime stack. No `github.com/zerfoo/zerfoo` imports in conversion code.
- **ZMF → ONNX export (planned)**: Round-trip conversion is on the roadmap.

## Architectural Principles

`zonnx` is designed as a standalone model converter, strictly decoupled from the `zerfoo` runtime. Its primary responsibility is to transform ONNX models into the Zerfoo Model Format (ZMF), which serves as the universal intermediate representation for `zerfoo`.

Key principles:

- **ZMF-Only Emission**: `zonnx` emits only ZMF models. It does not contain any `zerfoo` runtime code, graph building logic, or direct dependencies on `zerfoo`'s internal components (e.g., `compute`, `graph`, `model`, `numeric`, `tensor`).
- **Explicit ZMF Schema**: The ZMF schema is designed to be explicit, capturing all necessary model attributes and shapes directly, without relying on runtime inference of ONNX rules.
- **No `zerfoo` Imports**: The `zonnx` codebase (outside of documentation, tests, and examples) must not import any packages from `github.com/zerfoo/zerfoo`.
- **No ONNX in `zerfoo`**: Conversely, the `zerfoo` runtime must not contain any ONNX-specific code or dependencies. It consumes only ZMF models.

This strict separation ensures modularity, independent development, and maintainability of both the converter and the runtime.

## Usage

### Installation

Install the CLI directly:

```bash
go install github.com/zerfoo/zonnx/cmd/zonnx@latest
```

Or build from source at the repo root:

```bash
go build -o zonnx ./cmd/zonnx
```

Notes:
- Requires Go specified in `go.mod` (currently `go 1.25`).
- CGO is not required; the module is tested to build with `CGO_ENABLED=0`.

### Quickstart

```bash
# 1) Download an ONNX model and tokenizer files from HuggingFace
zonnx download --model google/gemma-2-2b-it --output ./models

# 2) Convert ONNX → ZMF (flags must come before positional args)
zonnx convert -output ./models/model.zmf ./models/model.onnx

# 3) Inspect either format (flags before input)
zonnx inspect -pretty ./models/model.onnx
zonnx inspect -pretty ./models/model.zmf
```

### Commands

#### `download`

Downloads an ONNX model and its associated tokenizer files from HuggingFace Hub.

**Syntax:**

```bash
./zonnx download --model <huggingface-model-id> [--output <output-directory>] [--api-key <your-api-key>]
```

**Arguments:**

- `--model <huggingface-model-id>`: (Required) The ID of the HuggingFace model to download (e.g., `openai/whisper-tiny.en`).
- `--output <output-directory>`: (Optional) The directory where the model and tokenizer files will be saved. Defaults to the current directory (`.`).
- `--api-key <your-api-key>`: (Optional) Your HuggingFace API key for authenticated downloads.

**API Key Configuration:**

For models that require authentication (e.g., private models or models with restricted access), you can provide your HuggingFace API key in one of two ways:

1.  **Using the `--api-key` flag:**
    Pass your API key directly as a command-line argument:
    ```bash
    ./zonnx download --model google/gemma-2-2b-it --api-key hf_YOUR_API_KEY
    ```
    Replace `hf_YOUR_API_KEY` with your actual HuggingFace API key.

2.  **Using the `HF_API_KEY` environment variable:**
    Set the `HF_API_KEY` environment variable before running the `zonnx` command:
    ```bash
    export HF_API_KEY=hf_YOUR_API_KEY
    ./zonnx download --model google/gemma-2-2b-it
    ```
    The `--api-key` flag takes precedence over the `HF_API_KEY` environment variable if both are provided.

When a model is downloaded, `zonnx` will automatically attempt to identify and download common tokenizer-related files (like `tokenizer.json`, `vocab.txt`, etc.) found in the same HuggingFace repository. These files will be saved alongside the ONNX model in the specified output directory.

#### `import`

Import ONNX and emit ZMF. This is a future-friendly alias for `convert`.

Status: planned; use `convert` today.

#### `export`

Export ZMF back to ONNX.

Status: planned; coming soon.

#### `inspect`

Inspect either ONNX or ZMF. Type can be inferred from extension or set explicitly.

Syntax:

```bash
zonnx inspect [-type onnx|zmf] [-pretty] <input-file>
```

Examples:

```bash
zonnx inspect -pretty ./path/to/model.onnx
zonnx inspect -type zmf -pretty ./path/to/model.zmf
```

Notes:
- `--pretty` human-friendly printing is planned; JSON schema output is the target.

#### `inspect-zmf`

Deprecated. Use `inspect <file.zmf>` or `inspect --type zmf`.

#### `convert`

Convert ONNX → ZMF. This is the primary conversion command.

Syntax:

```bash
zonnx convert [-output <output-file.zmf>] <input-file.onnx>
```

Example:

```bash
zonnx convert -output ./models/encoder.zmf ./models/encoder.onnx
```

Notes:
- Flags must appear before the first positional argument when using Go's standard `flag` package.
- The `convert` command accepts an alias `--output` in addition to `-output`.
- If no output is specified, the default is `<input-dir>/<input-base>.zmf`.
- Parent directories for the output path are created automatically.

## Why ZMF?

Zerfoo Model Format (ZMF) is a compact, explicit representation designed for fast loading and deterministic execution by the Zerfoo runtime. Benefits:

- Explicit shapes and attributes; no reliance on ONNX runtime semantics at load time.
- Portable files, amenable to signing and caching.
- Decouples model authoring/conversion from runtime execution.

## Development

- Test: `make test` (runs `go test ./...`)
- Lint: `make lint` (runs `golangci-lint run`)
- Lint (auto-fix): `make lint-fix`
- Format: `make format` (gofmt + goimports + gofumpt if available)

The codebase is intentionally free of `github.com/zerfoo/zerfoo` imports in conversion paths to preserve a strict boundary between conversion and runtime.
