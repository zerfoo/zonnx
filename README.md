# zonnx

A standalone command-line tool for converting machine learning models to GGUF format. Supports ONNX and SafeTensors inputs, with built-in HuggingFace Hub integration for downloading models.

## Features

- **ONNX / SafeTensors → GGUF conversion**: Produce portable GGUF files compatible with the `zerfoo` runtime and llama.cpp.
- **Model inspection**: Introspect model metadata, IOs, nodes and tensor stats for ONNX and GGUF files. JSON output with `--pretty` planned.
- **HuggingFace integration**: Download ONNX models and tokenizer files in one step.
- **Post-conversion quantization**: Quantize weights to Q4_0 or Q8_0 during conversion.
- **CGO-free builds**: Ships as a single static binary. Easy to distribute and run in minimal containers.
- **Architecture-aware mappings**: Tensor name and metadata mappings tuned per model family.

## Supported Models

zonnx maps tensor names and metadata to GGUF conventions for each architecture family. The `--arch` flag selects the mapping.

| Architecture | `--arch` value | Input Formats | Tensor Mapping | Notes |
|-------------|----------------|---------------|----------------|-------|
| Llama | `llama` (default) | ONNX | Decoder layers (`model.layers.N.*`) | Llama 3, Code Llama, etc. |
| Gemma | `gemma` | ONNX | Decoder layers (`model.layers.N.*`) | Gemma, Gemma 2, Gemma 3 |
| BERT | `bert` | ONNX, SafeTensors | Encoder layers (`bert.encoder.layer.N.*`) | Classification, embeddings, pooler |
| RoBERTa | `roberta` | ONNX, SafeTensors | Encoder layers (`roberta.encoder.layer.N.*`) | Same layer structure as BERT |

Any architecture string can be passed via `--arch`. The metadata mapping is generic (maps `hidden_size`, `num_hidden_layers`, etc. to `{arch}.*` GGUF keys). However, tensor name mapping currently covers Llama-style decoder models and BERT/RoBERTa encoder models. Unsupported tensor name patterns pass through unchanged.

### Metadata Mapped

These HuggingFace `config.json` fields are mapped to GGUF metadata for all architectures:

| config.json field | GGUF key |
|-------------------|----------|
| `hidden_size` | `{arch}.embedding_length` |
| `num_hidden_layers` | `{arch}.block_count` |
| `num_attention_heads` | `{arch}.attention.head_count` |
| `num_key_value_heads` | `{arch}.attention.head_count_kv` |
| `intermediate_size` | `{arch}.feed_forward_length` |
| `vocab_size` | `{arch}.vocab_size` |
| `max_position_embeddings` | `{arch}.context_length` |
| `rms_norm_eps` | `{arch}.attention.layer_norm_rms_epsilon` |
| `rope_theta` | `{arch}.rope.freq_base` |

BERT/RoBERTa additionally map `layer_norm_eps`, `num_labels`, and `pooler_type`.

## Usage

### Installation

```bash
go install github.com/zerfoo/zonnx/cmd/zonnx@latest
```

Or build from source:

```bash
go build -o zonnx ./cmd/zonnx
```

Requires Go 1.26+. CGO is not required (`CGO_ENABLED=0` works).

### Quickstart

```bash
# 1) Download an ONNX model and tokenizer files from HuggingFace
zonnx download --model google/gemma-2-2b-it --output ./models

# 2) Convert ONNX → GGUF
zonnx convert --arch gemma --output ./models/model.gguf ./models/model.onnx

# 3) Convert SafeTensors → GGUF (pass directory containing config.json + model.safetensors)
zonnx convert --format safetensors --arch bert --output ./models/model.gguf ./models/bert-dir/

# 4) Convert with quantization
zonnx convert --quantize q4_0 --output ./models/model-q4.gguf ./models/model.onnx

# 5) Inspect either format
zonnx inspect --pretty ./models/model.onnx
zonnx inspect --pretty ./models/model.gguf
```

### Commands

#### `convert`

Convert ONNX or SafeTensors models to GGUF.

```bash
zonnx convert [flags] <input>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `<input-dir>/<input-base>.gguf` | Output GGUF file path |
| `--arch` | `llama` | Model architecture for metadata/tensor mapping |
| `--format` | `onnx` | Input format: `onnx` or `safetensors` |
| `--quantize` | (none) | Quantize weights: `q4_0` or `q8_0` |

For ONNX input, `<input>` is a `.onnx` model file. For SafeTensors, `<input>` is a directory containing `config.json` and `model.safetensors`.

#### `download`

Download an ONNX model and tokenizer files from HuggingFace Hub.

```bash
zonnx download --model <huggingface-model-id> [--output <dir>] [--api-key <key>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | HuggingFace model ID (e.g., `google/gemma-2-2b-it`) |
| `--output` | `.` | Output directory |
| `--api-key` | `$HF_API_KEY` | HuggingFace API key for authenticated downloads |

The `--api-key` flag takes precedence over the `HF_API_KEY` environment variable.

#### `inspect`

Inspect ONNX or GGUF model files.

```bash
zonnx inspect [--type onnx|gguf] [--pretty] <input-file>
```

Type is inferred from file extension when not specified.

#### `import` / `export`

Future-friendly aliases. `import` is an alias for `convert`. `export` (GGUF → ONNX) is planned.

## Architectural Principles

zonnx is strictly decoupled from the `zerfoo` runtime:

- **GGUF-only output**: Emits only GGUF files. No runtime code.
- **No `zerfoo` imports**: The zonnx codebase does not import `github.com/zerfoo/zerfoo`.
- **No ONNX in `zerfoo`**: The zerfoo runtime consumes only GGUF models.
- **Explicit schema**: GGUF output captures all model attributes directly, without relying on ONNX runtime semantics.

## Development

```bash
make test       # go test ./...
make lint       # golangci-lint run
make lint-fix   # golangci-lint run --fix
make format     # gofmt + goimports
```

## License

Apache 2.0
