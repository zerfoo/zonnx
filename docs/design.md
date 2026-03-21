# zonnx Design Document

## Overview

zonnx is a standalone CLI tool for converting ML models from ONNX and SafeTensors formats to GGUF, inspecting model metadata, and downloading ONNX models from HuggingFace Hub. It is strictly decoupled from the zerfoo runtime -- no zerfoo imports in conversion code, and no ONNX code in zerfoo.

## Architecture

### Core Components

- **cmd/zonnx/**: CLI entry point. Subcommands: `convert`, `inspect`, `download`, `import` (alias for convert), `export` (planned).
- **pkg/gguf/**: GGUF v3 binary writer, architecture-aware metadata mapping, and tensor name mapping.
- **pkg/converter/**: SafeTensors-to-GGUF conversion (reads `config.json` + `model.safetensors`, writes GGUF directly).
- **pkg/downloader/**: Model download logic. Defines the `ModelSource` interface for extensible source support. Currently implements `HuggingFaceSource`.
- **pkg/importer/**: ONNX model parsing and intermediate representation for the ONNX → GGUF path.
- **pkg/quantize/**: Post-conversion weight quantization (Q4_0, Q8_0). Skips norm, embed, bias, 1D, and small tensors.
- **pkg/inspector/**: Model inspection for ONNX and GGUF formats.
- **internal/onnx/**: ONNX protobuf definitions.

### Conversion Paths

Two conversion pipelines exist:

1. **ONNX → GGUF**: `pkg/importer` parses ONNX into an intermediate representation, `pkg/gguf` maps metadata and tensor names, `pkg/quantize` optionally quantizes weights, then `pkg/gguf.Writer` emits the GGUF binary.

2. **SafeTensors → GGUF**: `pkg/converter` reads a directory containing `config.json` and `model.safetensors`, maps metadata and tensor names via `pkg/gguf`, and writes GGUF directly without an intermediate representation.

### GGUF Writer

`pkg/gguf/writer.go` implements the GGUF v3 binary format:
- Magic bytes, version, tensor/metadata counts.
- Metadata entries (string, uint32, float32 types).
- Tensor descriptors (name, dimensions, dtype, offset).
- Tensor data block with 32-byte alignment.

### Architecture Mappings

`pkg/gguf/metadata.go` maps HuggingFace `config.json` fields to GGUF metadata keys using an `{arch}` placeholder:
- Generic mappings: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, `vocab_size`, `max_position_embeddings`, `rms_norm_eps`, `rope_theta`, etc.
- BERT-specific mappings: `layer_norm_eps`, `num_labels` (derived from `id2label`), static `pooler_type`.

`pkg/gguf/tensornames.go` maps source tensor names to GGUF conventions:
- **Llama-style** (decoder models): `model.layers.N.<suffix>` → `blk.N.<gguf_suffix>`.
- **BERT/RoBERTa** (encoder models): `bert.encoder.layer.N.<suffix>` → `blk.N.<gguf_suffix>`.
- Static mappings for embeddings, norms, LM heads, poolers, and classifiers.

### Downloader Architecture

Interface-based design for extensibility (see docs/adr/001-modelsource-interface.md):

- `ModelSource` interface: single method `DownloadModel(modelID, destination) (*DownloadResult, error)`.
- `DownloadResult`: contains `ModelPath` (string) and `TokenizerPaths` ([]string).
- `Downloader`: orchestrator that delegates to a `ModelSource`.
- `HuggingFaceSource`: implementation that queries the HF API for model siblings, downloads `.onnx` files and tokenizer-related files.

Authentication: optional API key via `--api-key` flag or `HF_API_KEY` env var. Flag takes precedence.

### Key Design Principles

- **GGUF-only emission**: zonnx emits only GGUF files, no runtime code.
- **No zerfoo imports**: the zonnx codebase must not import `github.com/zerfoo/zerfoo`.
- **No ONNX in zerfoo**: the zerfoo runtime consumes only GGUF.
- **CGO-free**: ships as a single static binary.
- **Go standard library only**: no third-party HTTP or CLI libraries.

## Conventions

- **Language**: Go 1.26+.
- **Testing**: table-driven tests using the standard `testing` package. No testify. Target 90%+ coverage.
- **Linting**: `go fmt`, `go vet`, `golangci-lint run` before every commit.
- **Commits**: Conventional Commits format. Small, logical commits. No multi-directory commits.
- **Definition of Done**: code written, tests pass (including `-race`), docs updated, lint clean.

## Key File Paths

- `cmd/zonnx/main.go` -- CLI entry point and subcommand routing
- `pkg/gguf/writer.go` -- GGUF v3 binary writer
- `pkg/gguf/metadata.go` -- Architecture-aware metadata mapping
- `pkg/gguf/tensornames.go` -- Tensor name mapping (Llama, BERT, RoBERTa)
- `pkg/converter/safetensors.go` -- SafeTensors-to-GGUF converter
- `pkg/downloader/downloader.go` -- ModelSource interface, Downloader, HuggingFaceSource
- `pkg/importer/` -- ONNX model parsing
- `pkg/quantize/quantize.go` -- Quantization logic (Q4_0, Q8_0)

## References

- GGUF specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- HuggingFace Hub API: https://huggingface.co/docs/hub/index
