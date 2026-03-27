# zonnx

[![Go Reference](https://pkg.go.dev/badge/github.com/zerfoo/zonnx.svg)](https://pkg.go.dev/github.com/zerfoo/zonnx)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Standalone CLI for converting ONNX and SafeTensors models to GGUF format. Ships as a single static binary — zero CGo.

Part of the [Zerfoo](https://github.com/zerfoo) ML ecosystem.

## Features

- **ONNX / SafeTensors to GGUF** — produce portable GGUF files compatible with [zerfoo](https://github.com/zerfoo/zerfoo) and llama.cpp
- **Post-conversion quantization** — quantize weights to Q4_0 or Q8_0 during conversion
- **HuggingFace integration** — download ONNX models and tokenizer files in one step
- **Model inspection** — introspect metadata, IOs, nodes, and tensor stats for ONNX and GGUF files
- **Architecture-aware mappings** — tensor name and metadata mappings tuned per model family
- **CGo-free** — single static binary, easy to distribute and run in minimal containers

## Installation

```bash
go install github.com/zerfoo/zonnx/cmd/zonnx@latest
```

Or build from source:

```bash
go build -o zonnx ./cmd/zonnx
```

Requires Go 1.26+. `CGO_ENABLED=0` works.

## Quick Start

```bash
# Download an ONNX model from HuggingFace
zonnx download --model google/gemma-2-2b-it --output ./models

# Convert ONNX to GGUF
zonnx convert --arch gemma --output ./models/model.gguf ./models/model.onnx

# Convert SafeTensors to GGUF
zonnx convert --format safetensors --arch bert --output ./models/model.gguf ./models/bert-dir/

# Convert with quantization
zonnx convert --quantize q4_0 --output ./models/model-q4.gguf ./models/model.onnx

# Inspect a model file
zonnx inspect --pretty ./models/model.gguf
```

## Supported Architectures

| Architecture | `--arch` | Input Formats | Notes |
|-------------|----------|---------------|-------|
| Llama | `llama` (default) | ONNX | Llama 3, Code Llama |
| Gemma | `gemma` | ONNX | Gemma, Gemma 2, Gemma 3 |
| BERT | `bert` | ONNX, SafeTensors | Classification, embeddings |
| RoBERTa | `roberta` | ONNX, SafeTensors | Same layer structure as BERT |

Any architecture string can be passed via `--arch`. Metadata mapping is generic; tensor name mapping currently covers decoder (Llama-style) and encoder (BERT/RoBERTa) models.

## Commands

### `convert`

```
zonnx convert [flags] <input>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `<input>.gguf` | Output GGUF file path |
| `--arch` | `llama` | Model architecture for metadata/tensor mapping |
| `--format` | `onnx` | Input format: `onnx` or `safetensors` |
| `--quantize` | (none) | Quantize weights: `q4_0` or `q8_0` |

### `download`

```
zonnx download --model <huggingface-model-id> [--output <dir>] [--api-key <key>]
```

The `--api-key` flag takes precedence over the `HF_API_KEY` environment variable.

### `inspect`

```
zonnx inspect [--type onnx|gguf] [--pretty] <input-file>
```

Type is inferred from file extension when not specified.

## Metadata Mapped

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

## Design Principles

- **GGUF-only output** — emits only GGUF files, no runtime code
- **No `zerfoo` imports** — strictly decoupled from the inference runtime
- **Explicit schema** — GGUF output captures all model attributes directly

## Development

```bash
make test       # go test ./...
make lint       # golangci-lint run
make format     # gofmt + goimports
```

## License

Apache 2.0
