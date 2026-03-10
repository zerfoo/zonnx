# ADR 001: Interface-Based Downloader Architecture

## Status
Accepted

## Date
2025-08-23

## Context
zonnx needed a model download feature to fetch ONNX models and tokenizer files from external repositories. The initial target was HuggingFace Hub, but future sources (e.g., model zoos, custom registries) were anticipated. The design needed to support multiple file downloads per model (ONNX model + tokenizer files) while keeping the implementation modular.

## Decision
Implement the downloader using a `ModelSource` interface with a single method:

```go
type ModelSource interface {
    DownloadModel(modelID string, destination string) (*DownloadResult, error)
}
```

A `Downloader` struct wraps the interface and delegates to the configured source. The first (and currently only) implementation is `HuggingFaceSource`, which queries the HuggingFace API for model file listings and downloads ONNX and tokenizer files.

Authentication is optional and supported via an API key passed at construction time. The CLI accepts keys via `--api-key` flag or `HF_API_KEY` environment variable.

## Consequences

**Positive:**
- Adding new model sources requires only implementing the `ModelSource` interface.
- The core `Downloader` and CLI code are source-agnostic.
- Testing is straightforward: mock implementations for unit tests, httptest servers for integration tests.

**Negative:**
- The interface currently has a single implementation, which could be seen as premature abstraction.
- The tokenizer file detection heuristic (filename contains "tokenizer" or ends in ".json"/".txt") may be too broad or too narrow for some repositories.
