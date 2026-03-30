# ADR 002: API Stability Contract for v1.0.0

## Status
Accepted

## Date
2026-03-29

## Context
zonnx is approaching v1.0.0. Consumers need to know which exported types and functions are covered by semantic versioning guarantees and which may change in minor releases. This ADR documents the stability contract so that importers can depend on zonnx with confidence and maintainers know what constitutes a breaking change.

## Decision

### Stability Tiers

**Stable** -- covered by Go module semantic versioning from v1.0.0 onward. Removing, renaming, or changing the signature of these symbols requires a major version bump.

**Extensible** -- new fields, methods, or enum values may be added in minor releases. Callers should not rely on exhaustive switches or struct literal initialization without field names.

**Internal** -- not part of the public API. May change or be removed in any release.

### Package-by-Package Guarantees

#### `safetensors` (Stable)

| Symbol | Kind | Stability | Notes |
|--------|------|-----------|-------|
| `Open(path string) (*File, error)` | func | Stable | |
| `File` | struct | Stable | Opaque; interact via methods only |
| `File.Close() error` | method | Stable | |
| `File.TensorNames() []string` | method | Stable | |
| `File.TensorInfo(name string) (TensorInfo, bool)` | method | Stable | |
| `File.ReadTensor(name string) ([]byte, error)` | method | Stable | |
| `File.ReadFloat32(name string) ([]float32, error)` | method | Stable | New dtype support may be added |
| `TensorInfo` | struct | Extensible | New fields may be added; use named fields in literals |

#### `pkg/downloader` (Stable)

| Symbol | Kind | Stability | Notes |
|--------|------|-----------|-------|
| `ModelSource` | interface | Stable | Will not add methods; safe to implement externally |
| `ModelSource.DownloadModel(...)` | method | Stable | |
| `Downloader` | struct | Stable | Opaque |
| `NewDownloader(source ModelSource) *Downloader` | func | Stable | |
| `Downloader.Download(...)` | method | Stable | |
| `DownloadResult` | struct | Extensible | New fields may be added |
| `HuggingFaceSource` | struct | Stable | Opaque |
| `NewHuggingFaceSource(apiKey string) *HuggingFaceSource` | func | Stable | |
| `HuggingFaceSource.DownloadModel(...)` | method | Stable | |
| `HuggingFaceModelInfo` | struct | Extensible | Mirrors upstream API; fields may be added |

#### `pkg/gguf` (Stable)

| Symbol | Kind | Stability | Notes |
|--------|------|-----------|-------|
| `MapTensorName(name string) string` | func | Stable | New mappings may be added |
| `MapMetadata(arch string, config map[string]interface{}) []MetadataEntry` | func | Stable | New metadata keys may be emitted |
| `MetadataEntry` | struct | Extensible | New fields may be added |

#### `pkg/converter` (Stable)

| Symbol | Kind | Stability | Notes |
|--------|------|-----------|-------|
| `ConvertSafetensorsToGGUF(inputDir, outputPath, arch string) error` | func | Stable | |
| `ONNXToZMF(model *onnx.ModelProto) (*zmf.Model, error)` | func | Stable | |
| `ONNXToZMFWithPath(model *onnx.ModelProto, modelPath string) (*zmf.Model, error)` | func | Stable | |

#### `pkg/importer` (Stable)

| Symbol | Kind | Stability | Notes |
|--------|------|-----------|-------|
| `LoadOnnxModel(path string) (*onnx.ModelProto, error)` | func | Stable | |
| `ConvertOnnxToZmf(path string) (*zmf.Model, error)` | func | Stable | |

#### `pkg/inspector` (Stable)

| Symbol | Kind | Stability | Notes |
|--------|------|-----------|-------|
| `InspectONNX(inputFile string) error` | func | Stable | Output format may change |
| `InspectZMF(inputFile string) error` | func | Stable | Output format may change |

#### `pkg/quantize` (Stable)

| Symbol | Kind | Stability | Notes |
|--------|------|-----------|-------|
| `QuantType` | type | Extensible | New values (e.g. Q5_1) may be added |
| `Q4_0`, `Q8_0` | const | Stable | |
| `Model(m *zmf.Model, qt QuantType) error` | func | Stable | |

#### `pkg/registry` (Stable)

| Symbol | Kind | Stability | Notes |
|--------|------|-----------|-------|
| `Register(opType string, constructor LayerConstructor)` | func | Stable | |
| `Get(opType string) (LayerConstructor, bool)` | func | Stable | |
| `LayerConstructor` | type | Stable | Signature will not change |
| `ConversionContext` | struct | Extensible | New fields may be added for richer conversion |

#### `pkg/zmf_inspector` (Stable)

| Symbol | Kind | Stability | Notes |
|--------|------|-----------|-------|
| `Load(file string) (*zmf.Model, error)` | func | Stable | |
| `Inspect(model *zmf.Model)` | func | Stable | Output format may change |

#### `internal/onnx` (Internal)

Generated protobuf code. Not part of the public API. May be regenerated or restructured in any release.

#### `cmd/zonnx`, `cmd/granite2gguf` (Internal)

CLI entry points. Command-line flags and output format are not covered by the stability contract.

### Safe Extensions (no major version bump required)

The following changes are permitted in minor releases:

- Adding new exported functions, methods, or types to any package.
- Adding new fields to Extensible structs.
- Adding new `QuantType` constants.
- Adding new ONNX op support in `pkg/registry` and `pkg/importer/layers/`.
- Adding new tensor name or metadata mappings in `pkg/gguf`.
- Changing human-readable output format of inspector functions.

### Breaking Changes (require major version bump)

- Removing or renaming any Stable symbol.
- Changing the signature of any Stable function or method.
- Adding methods to the `ModelSource` interface.
- Changing the semantics of an existing function in a way that breaks existing callers.

## Consequences

**Positive:**
- Importers can depend on zonnx v1.x with clear expectations about what will and will not change.
- The Extensible tier gives maintainers room to evolve structs without a major version bump.
- The `ModelSource` interface is explicitly frozen, so third-party implementations are safe.

**Negative:**
- Freezing the `ModelSource` interface means new capabilities must be added via new interfaces or wrapper types rather than extending the existing one.
- CLI output is excluded from stability guarantees, which may surprise users who script against it.
