# zonnx Design Document

## Overview

zonnx is a standalone CLI tool for converting ML models between ONNX and Zerfoo Model Format (ZMF), inspecting model metadata, and downloading ONNX models from HuggingFace Hub. It is strictly decoupled from the zerfoo runtime -- no zerfoo imports in conversion code, and no ONNX code in zerfoo.

## Architecture

### Core Components

- **cmd/zonnx/**: CLI entry point. Subcommands: `convert`, `inspect`, `download`, `import` (planned alias), `export` (planned).
- **pkg/downloader/**: Model download logic. Defines the `ModelSource` interface for extensible source support. Currently implements `HuggingFaceSource`.
- **pkg/importer/**: ONNX-to-ZMF conversion via `ConvertOnnxToZmf()`.
- **pkg/quantize/**: Post-conversion weight quantization (Q4_0, Q8_0). Skips norm, embed, bias, 1D, and small tensors.
- **pkg/inspector/**: Model inspection for both ONNX and ZMF formats.
- **pkg/converter/**: Low-level conversion utilities.
- **internal/onnx/**: ONNX protobuf definitions.

### Downloader Architecture

The downloader uses an interface-based design for extensibility (see docs/adr/001-modelsource-interface.md):

- `ModelSource` interface: single method `DownloadModel(modelID, destination) (*DownloadResult, error)`.
- `DownloadResult`: contains `ModelPath` (string) and `TokenizerPaths` ([]string).
- `Downloader`: orchestrator that delegates to a `ModelSource`.
- `HuggingFaceSource`: implementation that queries the HF API for model siblings, downloads `.onnx` files and tokenizer-related files (`.json`, `.txt`, files containing "tokenizer").

Authentication: optional API key via `--api-key` flag or `HF_API_KEY` env var. Flag takes precedence.

### Key Design Principles

- **ZMF-only emission**: zonnx emits only ZMF models, no runtime code.
- **No zerfoo imports**: the zonnx codebase must not import `github.com/zerfoo/zerfoo`.
- **No ONNX in zerfoo**: the zerfoo runtime consumes only ZMF.
- **CGO-free**: ships as a single static binary.
- **Go standard library only**: no third-party HTTP or CLI libraries.

## Conventions

- **Language**: Go (version specified in go.mod, currently 1.25).
- **Testing**: table-driven tests using the standard `testing` package. No testify. Target 90%+ coverage.
- **Linting**: `go fmt`, `go vet`, `golangci-lint run` before every commit.
- **Commits**: Conventional Commits format. Small, logical commits. No multi-directory commits.
- **Definition of Done**: code written, tests pass (including `-race`), docs updated, lint clean.

## Key File Paths

- `cmd/zonnx/main.go` -- CLI entry point and subcommand routing
- `pkg/downloader/downloader.go` -- ModelSource interface, Downloader, HuggingFaceSource
- `pkg/downloader/downloader_test.go` -- unit tests for downloader (mock source + httptest)
- `cmd/zonnx/zonnx_test.go` -- integration tests for CLI commands
- `pkg/quantize/quantize.go` -- quantization logic (Q4_0, Q8_0)
- `pkg/importer/` -- ONNX-to-ZMF conversion

## References

- HuggingFace Hub API: https://huggingface.co/docs/hub/index
- Go httptest package: https://pkg.go.dev/net/http/httptest
