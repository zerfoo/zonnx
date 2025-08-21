# Design Specification: `zerfoo/zonnx`

This document provides a comprehensive, prescriptive, and granular design for the `zerfoo/zonnx` tool. This tool is a standalone command-line utility responsible for converting machine learning models between the ONNX format and the Zerfoo Model Format (ZMF).

---

### 1. Philosophy and Guiding Principles

1.  **Standalone and Decoupled:** `zonnx` is a separate project from the core `zerfoo` framework. It will have its own repository and release cycle. This ensures that the core framework remains lightweight and free of ONNX-specific dependencies.
2.  **Strict and Explicit:** The tool's primary responsibility is translation, not interpretation. It will perform a strict conversion of supported operators and data types. If an ONNX model contains features that are not supported by the `zerfoo` framework, `zonnx` will fail with a clear, informative error message. It will not attempt to approximate or modify the model's architecture.
3.  **Stateless CLI:** `zonnx` is a stateless command-line tool. It reads one or more input files, produces one or more output files, and then exits. It does not manage a database of models or maintain any persistent configuration.
4.  **Unidirectional Logic:** The conversion logic for ONNX-to-ZMF and ZMF-to-ONNX will be implemented as two distinct, unidirectional pathways. This promotes clarity and avoids complex, bidirectional mapping logic.

---

### 2. Core Components

The `zonnx` tool will be composed of the following high-level components:

*   **CLI Interface:** The user-facing command-line interface.
*   **ONNX Frontend:** Responsible for parsing and validating `.onnx` files.
*   **ZMF Frontend:** Responsible for parsing and validating `.zmf` files.
*   **Conversion Engine:** The core component containing the operator mapping and graph translation logic.
*   **ONNX Backend:** Responsible for building and serializing `.onnx` files.
*   **ZMF Backend:** Responsible for building and serializing `.zmf` files.

---

### 3. Command-Line Interface (CLI) Design

The CLI will be implemented using the `github.com/spf13/cobra` library. It will feature two main commands: `import` and `export`.

#### 3.1. `import` Command

The `import` command converts an ONNX model into the ZMF format.

```shell
zonnx import <input-file.onnx> [flags]
```

**Arguments:**
*   `<input-file.onnx>` (string, required): Path to the input ONNX model file.

**Flags:**
*   `--output` (string, optional): Path for the converted output ZMF file. If not provided, it will be derived from the input filename (e.g., `model.onnx` becomes `model.zmf`).
*   `--log-level` (string, optional, default="info"): Sets the logging verbosity. Options: `debug`, `info`, `warn`, `error`.

#### 3.2. `export` Command

The `export` command converts a ZMF model into the ONNX format.

```shell
zonnx export <input-file.zmf> [flags]
```

**Arguments:**
*   `<input-file.zmf>` (string, required): Path to the input ZMF model file.

**Flags:**
*   `--output` (string, optional): Path for the converted output ONNX file. If not provided, it will be derived from the input filename (e.g., `model.zmf` becomes `model.onnx`).
*   `--log-level` (string, optional, default="info"): Sets the logging verbosity. Options: `debug`, `info`, `warn`, `error`.

#### 3.3. Usage Examples

```shell
# Import an ONNX model to ZMF (output name is inferred)
zonnx import /path/to/model.onnx

# Import an ONNX model to ZMF with a specific output path
zonnx import /path/to/model.onnx --output /path/to/custom.zmf

# Export a ZMF model to ONNX (output name is inferred)
zonnx export /path/to/model.zmf

# Export a ZMF model to ONNX with a specific output path
zonnx export /path/to/model.zmf --output /path/to/custom.onnx
```

---

### 4. Conversion Logic: ONNX-to-ZMF (`import`)

This is the more complex pathway, as ONNX is a superset of ZMF's capabilities.

#### 4.1. Loading and Validation

1.  The `.onnx` file will be parsed using the `github.com/yalue/onnxruntime_go` library (or a similar pure Go ONNX parser).
2.  **Initial Validation Pass:** Before conversion, the tool will perform a validation pass on the loaded ONNX graph to ensure it is compatible with `zerfoo`. The process will fail immediately if:
    *   The ONNX opset version is outside a supported range (e.g., < 13 or > 18).
    *   The graph contains any non-tensor data types (`Sequence`, `Map`).
    *   The graph contains operators with graph attributes (`If`, `Loop`, `Scan`).
    *   Any tensor in the graph uses a data type not supported by `zerfoo/numeric` (e.g., `UINT8`, `BOOL`).

#### 4.2. Graph Traversal and Operator Mapping

1.  The conversion engine will maintain a registry of operator converters:
    ```go
    type ONNXConverterFunc func(node *onnx.NodeProto, params map[string]*format.Tensor) (*format.Node, error)

    var onnxConverters = map[string]ONNXConverterFunc{
        "MatMul":      convertMatMul,
        "Add":         convertAdd,
        "Relu":        convertRelu,
        "Reshape":     convertReshape,
        "RMSNorm":     convertRMSNorm, // Assuming a custom op or a recognized pattern
        // ... and so on for all supported operators
    }
    ```
2.  **Parameter Extraction:** All ONNX `initializers` (weights, biases) will be extracted first and converted into the ZMF `format.Tensor` protobuf format. They will be stored in a map keyed by their name.
3.  **Node Conversion:** The tool will iterate through the nodes of the ONNX graph in topological order. For each node:
    *   It will look up the node's `op_type` in the `onnxConverters` map. If not found, the conversion fails with an `ErrUnsupportedOperator` error.
    *   The corresponding `ONNXConverterFunc` is called. This function is responsible for:
        *   Creating a `format.Node` protobuf.
        *   Mapping the ONNX inputs and outputs to the ZMF node's inputs and outputs.
        *   Converting ONNX attributes to ZMF attributes. For example, the `epsilon` attribute of an `RMSNorm` operator.
        *   Handling architectural differences. For example, a `Gemm` operator in ONNX might be decomposed into a `MatMul` and an `Add` node in ZMF, or mapped to a single `Dense` ZMF node.

#### 4.3. I/O and Serialization

1.  The ONNX graph's `input` and `output` `ValueInfoProto` definitions will be converted to ZMF `ValueInfo` messages, preserving the name, data type, and shape.
2.  The complete `format.Model` protobuf will be assembled and serialized to the file specified by the `--output` flag.

---

### 5. Conversion Logic: ZMF-to-ONNX (`export`)

This pathway is simpler as ZMF is more constrained.

1.  **Loading:** The `.zmf` file will be parsed using `google.golang.org/protobuf`.
2.  **Operator Mapping:** A reverse mapping will be used:
    ```go
    type ZMFConverterFunc func(node *format.Node, builder *ONNXGraphBuilder) error

    var zmfConverters = map[string]ZMFConverterFunc{
        "Dense":       convertDense,
        "RMSNorm":     convertRMSNormToONNX,
        // ...
    }
    ```
3.  **Graph Construction:** An `ONNXGraphBuilder` utility will be created to simplify the construction of the ONNX graph.
4.  **Node Conversion:** The tool will iterate through the ZMF nodes. For each node, the corresponding `ZMFConverterFunc` will be called to create one or more ONNX nodes and add them to the graph builder.
5.  **Parameter Conversion:** The ZMF `parameters` map will be converted into ONNX `initializers`.
6.  **Serialization:** The final ONNX model will be serialized to the output file.

---

### 6. Error Handling

Error handling will be robust and user-friendly.

*   Custom error types will be defined for common failure modes:
    *   `ErrUnsupportedOperator{OpName string}`
    *   `ErrUnsupportedDataType{TypeName string}`
    *   `ErrUnsupportedOpset{Version int}`
    *   `ErrInvalidGraphStructure{Details string}`
*   Error messages printed to the console will be clear and actionable, e.g., `"Error: ONNX operator 'Loop' is not supported by the zerfoo framework."`

---

### 7. Testing Strategy

1.  **Unit Tests:** Each individual operator converter function (e.g., `convertMatMul`) will have its own unit tests.
2.  **Integration Tests:** The CLI will be tested end-to-end by executing the `import` and `export` commands as subprocesses and verifying the output files.
3.  **Round-Trip Tests:** A suite of tests will perform a full round-trip conversion (`ZMF -> ONNX -> ZMF`) and verify that the final ZMF file is byte-for-byte identical to the original.
4.  **Reference Model Validation:** The ONNX-to-ZMF converter will be tested against a curated set of simple models from the ONNX Model Zoo (e.g., a small MLP or CNN) to ensure correctness.

---

### 8. Project Structure

```
/zerfoo/zonnx
├── cmd/
│   └── zonnx/
│       └── main.go         // CLI entry point
├── docs/
│   └── zonnx.md        // This design document
├── internal/
│   ├── convert/
│   │   ├── onnx_to_zmf.go  // Main ONNX -> ZMF conversion logic
│   │   ├── zmf_to_onnx.go  // Main ZMF -> ONNX conversion logic
│   │   └── registry.go     // Operator converter registry
│   ├── onnx/
│   │   ├── frontend.go     // ONNX file parsing and validation
│   │   └── backend.go      // ONNX file construction and writing
│   └── zmf/
│       ├── frontend.go     // ZMF file parsing
│       └── backend.go      // ZMF file construction and writing
├── go.mod
├── go.sum
└── README.md
```
