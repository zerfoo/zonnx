# Contributing to zonnx

Thank you for your interest in contributing to zonnx, the ONNX-to-GGUF converter CLI for the Zerfoo ML ecosystem. This guide will help you get started.

## Table of Contents

- [Development Setup](#development-setup)
- [Building from Source](#building-from-source)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Good First Issues](#good-first-issues)
- [Key Conventions](#key-conventions)

## Development Setup

### Prerequisites

- **Go 1.25+**
- **Git**
- **protoc** (Protocol Buffers compiler, for regenerating ONNX proto bindings)

No GPU or C compiler is required. zonnx is a CGo-free standalone binary.

### Clone and Verify

```bash
git clone https://github.com/zerfoo/zonnx.git
cd zonnx
go mod tidy
go test ./...
```

zonnx depends on:

- [`github.com/zerfoo/zmf`](https://github.com/zerfoo/zmf) — Zerfoo Model Format (used internally for GGUF writing)
- [`google.golang.org/protobuf`](https://pkg.go.dev/google.golang.org/protobuf) — Protocol Buffers runtime for ONNX model parsing

These are fetched automatically by `go mod tidy`. Note: the `zmf` dependency uses a `replace` directive pointing to `../zmf`, so you need the `zmf` repo cloned as a sibling directory.

## Building from Source

```bash
go build ./...

# Build the CLI binary
go build -o zonnx ./cmd/zonnx
```

zonnx compiles as a standalone binary with no CGo dependencies.

## Running Tests

```bash
# Run all tests
go test ./...

# Run tests with race detector
go test -race ./...

# Run conversion tests (requires ONNX test fixtures in testdata/)
go test -run TestConvert -count=1 ./...

# Run tests with coverage
go test -cover ./...
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html
```

All new code must have tests. Aim for at least 80% coverage on new packages. When adding support for new ONNX operators, include test fixtures in `testdata/`.

## Code Style

### Formatting and Linting

- **`gofmt`** — all code must be formatted with `gofmt`
- **`goimports`** — imports must be organized (stdlib, external, internal)
- **`golangci-lint`** — run `golangci-lint run` before submitting

### Go Conventions

- Follow standard Go naming: PascalCase for exported symbols, camelCase for unexported
- Use table-driven tests with `t.Run` subtests
- Write documentation comments for all exported functions, types, and methods

## Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/) for automated versioning with release-please.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `perf` | A performance improvement |
| `docs` | Documentation only changes |
| `test` | Adding or correcting tests |
| `chore` | Maintenance tasks, CI, dependencies |
| `refactor` | Code change that neither fixes a bug nor adds a feature |

### Examples

```
feat(gguf): add support for converting attention weights
fix(onnx): handle dynamic batch dimension in reshape op
perf(convert): reduce memory allocation during tensor copy
docs: update supported ONNX operator list
test(gguf): add round-trip conversion tests for Gemma weights
```

## Pull Request Process

1. **One logical change per PR** — keep PRs focused and reviewable
2. **Branch from `main`** and keep your branch up to date with rebase
3. **All CI checks must pass** — tests, linting, formatting
4. **Rebase and merge** — we do not use squash merges or merge commits
5. **Reference related issues** — use `Fixes #123` or `Closes #123` in the PR description
6. **Respond to review feedback** promptly

### Before Submitting

```bash
go test ./...
go test -race ./...
go vet ./...
golangci-lint run
```

## Issue Reporting

### Bug Reports

Please include:

- **Description**: Clear summary of the bug
- **Steps to reproduce**: The ONNX model and command used
- **Expected behavior**: Expected GGUF output
- **Actual behavior**: Error message or incorrect output
- **Environment**: Go version, OS
- **Model details**: ONNX model source, opset version, architecture

### Feature Requests

Please include:

- **Problem statement**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you thought about
- **Use case**: Which ONNX models would this enable converting?

## Good First Issues

Look for issues labeled [`good first issue`](https://github.com/zerfoo/zonnx/labels/good%20first%20issue) on GitHub. These are scoped, well-defined tasks suitable for new contributors.

Good areas for first contributions:

- Adding support for new ONNX operators
- Documentation improvements
- Improving error messages for unsupported model features
- Adding test fixtures for new model architectures

## Key Conventions

These conventions are critical to maintaining consistency across the codebase:

### ONNX-to-GGUF conversion pipeline

The conversion follows a clear pipeline:

1. **Parse** — Read the ONNX protobuf model
2. **Map** — Translate ONNX tensor names to GGUF tensor names
3. **Convert** — Transform tensor data (transpose, quantize, repack as needed)
4. **Write** — Output a valid GGUF file

New operator support should fit into this pipeline. Do not add side channels or special cases outside the main conversion path.

### GGUF output correctness

The output GGUF file must be loadable by Zerfoo's inference engine (`github.com/zerfoo/zerfoo`). When adding new conversion features, verify the output by running inference with the converted model.

### CGo-free

zonnx must compile without CGo. Do not add C dependencies. The binary should be a single static executable that works on any platform.

### Test fixtures in testdata/

ONNX test models and expected outputs go in `testdata/`. Keep fixtures small (use tiny models with 1-2 layers) to avoid bloating the repository.
