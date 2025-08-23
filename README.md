# zonnx

This tool is a standalone command-line utility responsible for converting machine learning models between the ONNX format and the Zerfoo Model Format (ZMF). It also provides functionality to download ONNX models directly from HuggingFace Hub.

## Features

- **ONNX to ZMF Conversion**: Convert ONNX models to the Zerfoo Model Format.
- **ZMF to ONNX Export**: Export ZMF models back to ONNX format.
- **Model Inspection**: Inspect details of ONNX and ZMF models.
- **HuggingFace Model Download**: Download ONNX models and their associated tokenizer files directly from HuggingFace Hub.

## Usage

### Building the Tool

To build the `zonnx` executable, navigate to the project root and run:

```bash
go build -o zonnx ./cmd/zonnx
```

This will create an executable named `zonnx` in your current directory.

### Commands

#### `download`

Downloads an ONNX model and its associated tokenizer files from HuggingFace Hub.

**Syntax:**

```bash
./zonnx download --model <huggingface-model-id> [--output <output-directory>]
```

**Arguments:**

- `--model <huggingface-model-id>`: (Required) The ID of the HuggingFace model to download (e.g., `openai/whisper-tiny.en`).
- `--output <output-directory>`: (Optional) The directory where the model and tokenizer files will be saved. Defaults to the current directory (`.`).

**Examples:**

Download a model to the current directory:

```bash
./zonnx download --model openai/whisper-tiny.en
```

Download a model to a specific directory:

```bash
./zonnx download --model facebook/bart-large-cnn --output ./models/bart
```

When a model is downloaded, `zonnx` will automatically attempt to identify and download common tokenizer-related files (like `tokenizer.json`, `vocab.txt`, etc.) found in the same HuggingFace repository. These files will be saved alongside the ONNX model in the specified output directory.

#### `import`

(Existing documentation for import command)

#### `export`

(Existing documentation for export command)

#### `inspect`

(Existing documentation for inspect command)

#### `inspect-zmf`

(Existing documentation for inspect-zmf command)

#### `convert`

(Existing documentation for convert command)

## Development

(Existing development section, if any)