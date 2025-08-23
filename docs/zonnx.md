# Design Specification & Learnings: `zerfoo/zonnx`

This document provides a comprehensive design for the `zerfoo/zonnx` tool and documents the key learnings from its development.

---

### 1. Philosophy and Guiding Principles

1.  **Standalone and Pure Go:** `zonnx` is a separate, standalone tool. A primary principle is to have **zero CGo dependencies**. This ensures simple, fast, and portable builds, avoiding the complexities of linking against external C++ libraries like `onnxruntime`.
2.  **Strict and Explicit:** The tool's primary responsibility is translation. It performs a strict conversion of supported operators. If an ONNX model contains features not supported by `zerfoo`, `zonnx` will fail with a clear error.
3.  **Stateless CLI:** `zonnx` is a stateless command-line tool. It reads an input file, produces an output file, and exits.

---

### 2. Core Architecture: Native Protobuf Parsing & Model Downloading

The initial approach of using the official C++ `onnxruntime` library via a CGo wrapper was **abandoned**. This decision was made due to:
*   **Extreme Build Complexity:** Managing the `onnxruntime` dependency, its specific versioning, and the associated build flags was fragile and platform-dependent.
*   **Versioning Conflicts:** The ONNX format is itself versioned. A specific build of `onnxruntime` is tied to a specific ONNX opset version. This created a high risk of version mismatches between the library and the models we needed to parse.

The new, successful architecture is based on a **native Go protobuf parser** and a **flexible model downloading system**.

1.  **ONNX is a Protobuf:** An `.onnx` file is fundamentally a serialized Google Protocol Buffer (protobuf) message.
2.  **Official `.proto` Definition:** The official ONNX repository provides the `onnx.proto` file that formally defines the structure of an ONNX model.
3.  **Native Go Compilation:** By compiling this `onnx.proto` file using `protoc` and the standard Go protobuf plugin, we generate native Go structs that perfectly mirror the ONNX model structure.
4.  **Pure Go Deserialization:** The `zonnx` tool can now read the bytes of any `.onnx` file and deserialize them directly into these Go structs using the standard `google.golang.org/protobuf/proto` library.
5.  **Extensible Model Downloader:** A new `pkg/downloader` package provides an interface-based system for downloading models from various sources. This allows for easy extension to new model hubs in the future. The initial implementation targets HuggingFace Hub.

---

### 3. CLI Design

The CLI is implemented using the standard Go `flag` package for simplicity.

#### 3.1. `convert` Command

Converts an ONNX model into the ZMF format.

```shell
zonnx convert <input-file.onnx> [flags]
```
*   `--output` (string, optional): Path for the converted `.zmf` file. Defaults to `<input-file>.zmf`.

#### 3.2. `inspect` Command

Prints a summary of an ONNX model's structure.

```shell
zonnx inspect <input-file.onnx>
```

#### 3.3. `download` Command

Downloads an ONNX model and its associated tokenizer files from HuggingFace Hub.

```shell
zonnx download --model <huggingface-model-id> [--output <output-directory>]
```
*   `--model` (string, required): The ID of the HuggingFace model to download (e.g., `openai/whisper-tiny.en`).
*   `--output` (string, optional): The directory where the model and tokenizer files will be saved. Defaults to the current directory (`.`).

The `download` command automatically identifies and fetches common tokenizer-related files (e.g., `tokenizer.json`, `vocab.txt`) found in the same HuggingFace repository as the specified model.

---

### 4. Conversion Logic: ONNX-to-ZMF

The conversion process is now a straightforward traversal of the native Go structs.

1.  **Loading:** The `.onnx` file is loaded and unmarshaled into an `onnx.ModelProto` struct.
2.  **Parameter Extraction:** All ONNX `initializers` (weights, biases) are extracted first, converted into the `zmf.Tensor` format, and stored in the `zmf.Graph`'s `parameters` map.
3.  **Node Conversion:** The tool iterates through the nodes of the ONNX graph. A `switch` statement on the `op_type` maps each ONNX operator to a corresponding `zmf.Node`, translating its inputs, outputs, and attributes.
4.  **Serialization:** The complete `zmf.Model` protobuf is assembled and serialized to the output `.zmf` file.

---

### 5. Model Downloading Logic

The model downloading functionality is encapsulated within the `pkg/downloader` package, designed for extensibility.

1.  **`ModelSource` Interface:** Defines a generic interface for any model source, abstracting away the specifics of different platforms.
2.  **`HuggingFaceSource` Implementation:** Provides a concrete implementation of `ModelSource` for HuggingFace Hub. It interacts with the HuggingFace API to list files and constructs direct download URLs.
3.  **File Discovery:** When a model ID is provided, `HuggingFaceSource` queries the HuggingFace API to get a list of all files (siblings) in the repository.
4.  **Targeted Download:** It then filters these files, identifying the primary ONNX model file (`.onnx` extension) and common tokenizer files (e.g., `tokenizer.json`, `vocab.txt`, `.txt` files that might contain tokenizer data).
5.  **Direct HTTP Download:** Each identified file is downloaded using standard Go `net/http` calls to the HuggingFace CDN.
6.  **Output:** Downloaded files are saved to the specified output directory, maintaining their original filenames.

---

### 6. Validation and Key Learnings

*   **Successful Conversion:** The `zonnx` tool has been successfully used to parse and convert the entire Google Gemma 3 ONNX model into the ZMF format. This serves as the primary validation of the native parsing architecture.
*   **Proto Versioning:** During initial testing with an older `mnist-8.onnx` model, we encountered parsing errors (`Model graph is nil`). This was traced to a **version mismatch** between the `onnx.proto` file we were using (from the `main` branch) and the older opset version of the test model. This highlights the importance of aligning the `.proto` definition with the target model's opset version. Since our primary target is modern, this was not a blocker, but it is a critical learning for future compatibility.
*   **Robust Download Mechanism:** The `pkg/downloader` design allows for easy integration of new model sources without modifying core CLI logic. The use of direct HTTP downloads ensures minimal external dependencies and maximum portability.

---

### 7. Project Structure

```
/zonnx
├── cmd/
│   └── zonnx/
│       └── main.go         # CLI entry point
│       └── zonnx_test.go   # Integration tests for the CLI
├── docs/
│   └── zonnx.md        # This design document
├── internal/
│   └── onnx/
│       ├── onnx.pb.go      # Generated Go structs from onnx.proto
│       └── onnx.proto      # The ONNX protobuf definition file
├── pkg/
│   ├── importer/
│   │   └── importer.go     # Core ONNX file loading logic
│   ├── converter/
│   │   └── converter.go    # ONNX -> ZMF conversion logic
│   └── downloader/
│       └── downloader.go   # Core model downloading logic and HuggingFace implementation
│       └── downloader_test.go # Unit tests for downloader package
├── go.mod
└── ...
```
