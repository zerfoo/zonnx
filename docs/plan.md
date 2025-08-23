# Project zonnx: ONNX Model Downloader

## 1. Context

### Problem Statement
The `zonnx` tool currently requires users to manually download ONNX models before they can be used. This process is cumbersome and can be a barrier to entry for new users. There is a need for a streamlined way to fetch models directly from popular online repositories.

### Objectives
- Implement a new feature in the `zonnx` CLI to download ONNX models directly from HuggingFace Hub.
- Design the downloader with a flexible, interface-based architecture to allow for the addition of other model sources in the future (e.g., direct URL, other model zoos).
- Ensure the new feature is well-tested, documented, and easy to use.

### Non-Goals
- A graphical user interface (GUI) for model selection.
- Support for authenticated or private repositories in the initial version.
- Complex model versioning or dependency management.

### Constraints and Assumptions
- The implementation will be in Go, using only the standard library for HTTP requests and file system operations, adhering to project conventions.
- The feature will initially target public, anonymous downloads from HuggingFace.
- Users are expected to have an active internet connection to use the download functionality.

### Success Metrics
- A user can download a public ONNX model from HuggingFace via a `zonnx download` command.
- The code implementing the downloader is modular, with a clear `ModelSource` interface and a separate HuggingFace implementation, verified by code review.
- The new feature has at least 90% unit test coverage.

## 2. Scope and Deliverables

### In Scope
- A new `download` subcommand in the `zonnx` CLI.
- A `pkg/downloader` package containing the core download logic and source interface.
- A HuggingFace-specific implementation of the model source interface.
- Comprehensive unit and integration tests for the new functionality.
- Updated `README.md` and `docs/zonnx.md` to document the new feature.

### Out of Scope
- Support for any model repository other than HuggingFace in this iteration.
- Interactive model search or selection within the CLI.
- Resumable downloads.

### Deliverables
| ID | Description | Owner | Acceptance Criteria |
|---|---|---|---|
| D1 | Core downloader module with `ModelSource` interface | TBD | A `ModelSource` interface is defined and a generic download function that uses it is implemented and tested. |
| D2 | HuggingFace model source implementation | TBD | A struct that implements the `ModelSource` interface for HuggingFace is created and tested. |
| D3 | CLI integration | TBD | A `zonnx download` command is added, accepting flags for model ID and output path. |
| D4 | Unit and Integration Tests | TBD | Test coverage for all new packages and functions is at or above 90%. |
| D5 | Documentation Update | TBD | `README.md` and `docs/zonnx.md` are updated with clear instructions for the download feature. |

## 3. Work Breakdown Structure

### E1: Design and Core Implementation
*   **T1.1: Design Downloader Architecture** (Est: 2h)
    *   S1.1.1: Define `ModelSource` interface in `pkg/downloader/source.go`. (Est: 1h)
    *   S1.1.2: Research HuggingFace Hub file download URLs and API. (Est: 1h)
*   **T1.2: Implement Core Downloader** (Est: 4h)
    *   S1.2.1: Implement generic download logic in `pkg/downloader/downloader.go`. (Est: 2h)
    *   S1.2.2: Add unit tests for the generic downloader. (Est: 2h)

### E2: HuggingFace Integration
*   **T2.1: Implement HuggingFace Source** (Est: 4h)
    *   S2.1.1: Implement the `ModelSource` interface for HuggingFace in `pkg/downloader/huggingface/huggingface.go`. (Est: 2h)
    *   S2.1.2: Add unit tests for the HuggingFace source using `httptest`. (Est: 2h)

### E3: CLI and User Interface
*   **T3.1: Integrate into CLI** (Est: 3h)
    *   S3.1.1: Add `download` command to `cmd/zonnx/main.go`. (Est: 1.5h)
    *   S3.1.2: Add integration test for the `download` command. (Est: 1.5h)

### E4: Documentation and Finalization
*   **T4.1: Update Documentation** (Est: 2h)
    *   S4.1.1: Update `README.md` with usage examples. (Est: 1h)
    *   S4.1.2: Update `docs/zonnx.md` with detailed documentation. (Est: 1h)
*   **T4.2: Code Quality and Review** (Est: 2h)
    *   S4.2.1: Run `go fmt ./...` and `go vet ./...` across the project. (Est: 0.5h)
    *   S4.2.2: Run `golangci-lint run` and fix any issues. (Est: 1h)
    *   S4.2.3: Prepare and submit pull request. (Est: 0.5h)

## 4. Checkable Todo Board

### Not Started
- [ ] T1.1 Design Downloader Architecture  Owner: TBD  Est: 2h
- [ ] T1.2 Implement Core Downloader  Owner: TBD  Est: 4h
- [ ] T2.1 Implement HuggingFace Source  Owner: TBD  Est: 4h
- [ ] T3.1 Integrate into CLI  Owner: TBD  Est: 3h
- [ ] T4.1 Update Documentation  Owner: TBD  Est: 2h
- [ ] T4.2 Code Quality and Review  Owner: TBD  Est: 2h

### In Progress
- (None)

### Blocked
- (None)

### Done
- (None)

## 5. Timeline and Milestones

| ID | Task | Start Date | End Date | Duration | Dependencies |
|---|---|---|---|---|---|
| M1 | Milestone 1: Core Logic Complete | 2025-08-25 | 2025-08-26 | 2 days | |
| | T1.1, T1.2 | | | | |
| M2 | Milestone 2: HuggingFace Integration | 2025-08-27 | 2025-08-28 | 2 days | M1 |
| | T2.1 | | | | |
| M3 | Milestone 3: Feature Complete | 2025-08-29 | 2025-08-30 | 2 days | M2 |
| | T3.1, T4.1, T4.2 | | | | |

## 6. Operating Procedure

- **Daily Rhythm**: Check the board, pick a task, update status.
- **Definition of Done**: Code is written, unit and integration tests pass, documentation is updated, and the code is formatted and linted with no errors.
- **Testing**: All new implementation code must be accompanied by tests in the same commit.
- **Linting**: Run `go fmt`, `go vet`, and `golangci-lint run` before committing changes.

## 7. Progress Log

### Change Summary
- **2025-08-23**: Initial plan created to implement ONNX model downloading from HuggingFace (Tasks E1-E4).

### Log
- **2025-08-23**: No progress yet. Plan initialized.

## 8. Hand off Notes

This plan outlines the work to add a model downloader to `zonnx`. A new engineer should start by looking at the `Work Breakdown Structure` (Section 3) and the `Checkable Todo Board` (Section 4). The core design revolves around the `ModelSource` interface, which is the key to future extensibility. All necessary documentation and project links should be available in the repository.

## 9. Appendix

- **HuggingFace Hub Docs**: [https://huggingface.co/docs/hub/index](https://huggingface.co/docs/hub/index)
- **Go `httptest` package**: [https://pkg.go.dev/net/http/httptest](https://pkg.go.dev/net/http/httptest)
