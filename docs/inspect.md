# Enhance zonnx inspect (Architecture-aligned)

- [ ] CLI UX
  - [ ] Modify `zonnx inspect` to accept a `--type` flag: `onnx` or `zmf`.
  - [ ] If `--type` is omitted, infer from extension (`.onnx` or `.zmf`).

- [ ] Architecture compliance
  - [ ] Implement inspection in `zonnx` with no `github.com/zerfoo/zerfoo` imports.
  - [ ] For ONNX: use native protobuf structs in `zonnx/internal/onnx` (pure Go).
  - [ ] For ZMF: use `zmf` generated structs; never require runtime execution to inspect.

- [ ] Output format
  - [ ] Standardize a single JSON schema for inspect output (model metadata, graph IOs, node ops, tensor stats).
  - [ ] Include quantization info if present (dtype, `quant.scale`, `quant.zero_point`, `axis`).
  - [ ] Provide `--pretty` for human-friendly printing.

- [ ] Implementation
  - [ ] Create `zonnx/cmd/inspect/inspect.go` with subroutines: `inspectONNX`, `inspectZMF`.
  - [ ] Add unit tests for both paths using small fixtures.

- [ ] Documentation
  - [ ] Update `zonnx/README.md` with usage, examples, and JSON schema.
  - [ ] Remove/cleanup any legacy `inspect-zmf` option after the unified command is in place.