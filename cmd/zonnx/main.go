package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/zerfoo/zonnx/pkg/converter"
	"github.com/zerfoo/zonnx/pkg/importer"
	"github.com/zerfoo/zonnx/pkg/zmf_inspector"
	"google.golang.org/protobuf/proto"

	// Import the downloader package
	"github.com/zerfoo/zonnx/pkg/downloader"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "import":
		handleImport()
	case "export":
		handleExport()
	case "inspect":
		handleInspect()
	case "inspect-zmf":
		handleInspectZMF()
	case "convert":
		handleConvert()
	case "download": // Add new case for download command
		handleDownload()
	default:
		printUsage()
		os.Exit(1)
	}
}

func handleImport() {
	importCmd := flag.NewFlagSet("import", flag.ExitOnError)
	outputFile := importCmd.String("output", "", "Path for the converted ZMF file. (optional)")

	if err := importCmd.Parse(os.Args[2:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing flags for import command: %v\n", err)
		os.Exit(1)
	}
	inputFile := importCmd.Arg(0)

	if inputFile == "" {
		fmt.Println("Error: Input file is required for 'import' command.")
		importCmd.Usage()
		os.Exit(1)
	}

	if *outputFile == "" {
		*outputFile = filepath.Base(inputFile[:len(inputFile)-len(filepath.Ext(inputFile))]) + ".zmf"
	}

	fmt.Printf("Importing ONNX model from: %s\n", inputFile)

	// Placeholder for conversion logic
	// err := convert.ONNXToZMF(inputFile, *outputFile)
	// handleErr(err)

	fmt.Printf("Output will be saved to: %s\n", *outputFile)
}

func handleExport() {
	exportCmd := flag.NewFlagSet("export", flag.ExitOnError)
	outputFile := exportCmd.String("output", "", "Path for the converted ONNX file. (optional)")

	if err := exportCmd.Parse(os.Args[2:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing flags for export command: %v\n", err)
		os.Exit(1)
	}
	inputFile := exportCmd.Arg(0)

	if inputFile == "" {
		fmt.Println("Error: Input file is required for 'export' command.")
		exportCmd.Usage()
		os.Exit(1)
	}

	if *outputFile == "" {
		*outputFile = filepath.Base(inputFile[:len(inputFile)-len(filepath.Ext(inputFile))]) + ".onnx"
	}

	fmt.Printf("Exporting ZMF model from: %s\n", inputFile)
	fmt.Printf("Output will be saved to: %s\n", *outputFile)

	// Placeholder for conversion logic
	// err := convert.ZMFToONNX(inputFile, *outputFile)
	// handleErr(err)
}

func handleInspect() {
	inspectCmd := flag.NewFlagSet("inspect", flag.ExitOnError)
	if err := inspectCmd.Parse(os.Args[2:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing flags for inspect command: %v\n", err)
		os.Exit(1)
	}
	inputFile := inspectCmd.Arg(0)

	if inputFile == "" {
		fmt.Println("Error: Input file is required for 'inspect' command.")
		inspectCmd.Usage()
		os.Exit(1)
	}

	fmt.Printf("Inspecting ONNX model from: %s\n", inputFile)

	model, err := importer.LoadOnnxModel(inputFile)
	handleErr(err)

	fmt.Printf("Successfully loaded model with IR version: %d\n", model.GetIrVersion())
	if len(model.GetOpsetImport()) > 0 {
		fmt.Printf("Opset version: %d\n", model.GetOpsetImport()[0].GetVersion())
	}
	fmt.Printf("Graph has %d nodes.\n", len(model.GetGraph().GetNode()))
}

func handleInspectZMF() {
	inspectCmd := flag.NewFlagSet("inspect-zmf", flag.ExitOnError)
	if err := inspectCmd.Parse(os.Args[2:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing flags for inspect-zmf command: %v\n", err)
		os.Exit(1)
	}
	inputFile := inspectCmd.Arg(0)

	if inputFile == "" {
		fmt.Println("Error: Input file is required for 'inspect-zmf' command.")
		inspectCmd.Usage()
		os.Exit(1)
	}

	fmt.Printf("Inspecting ZMF model from: %s\n", inputFile)

	model, err := zmf_inspector.Load(inputFile)
	handleErr(err)

	zmf_inspector.Inspect(model)
}

func handleConvert() {
	convertCmd := flag.NewFlagSet("convert", flag.ExitOnError)
	outputFile := convertCmd.String("output", "", "Path for the converted ZMF file. (optional)")

	if err := convertCmd.Parse(os.Args[2:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing flags for convert command: %v\n", err)
		os.Exit(1)
	}
	inputFile := convertCmd.Arg(0)

	if inputFile == "" {
		fmt.Println("Error: Input file is required for 'convert' command.")
		convertCmd.Usage()
		os.Exit(1)
	}

	if *outputFile == "" {
		*outputFile = filepath.Base(inputFile[:len(inputFile)-len(filepath.Ext(inputFile))]) + ".zmf"
	}

	model, err := importer.LoadOnnxModel(inputFile)
	handleErr(err)

	zmfModel, err := converter.ONNXToZMFWithPath(model, inputFile)
	handleErr(err)

	// Serialize the ZMF model to a file
	outBytes, err := proto.Marshal(zmfModel)
	handleErr(err)

	err = os.WriteFile(*outputFile, outBytes, 0644)
	handleErr(err)

	fmt.Printf("Successfully converted and saved model to: %s\n", *outputFile)
}

func handleDownload() {
	downloadCmd := flag.NewFlagSet("download", flag.ExitOnError)
	modelID := downloadCmd.String("model", "", "HuggingFace model ID (e.g., 'openai/whisper-tiny.en')")
	outputPath := downloadCmd.String("output", ".", "Output directory for downloaded files")
	cliApiKey := downloadCmd.String("api-key", "", "Optional HuggingFace API key for authenticated downloads") // Renamed to avoid conflict

	if err := downloadCmd.Parse(os.Args[2:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing flags for download command: %v\n", err)
		os.Exit(1)
	}

	if *modelID == "" {
		fmt.Println("Error: --model flag is required for 'download' command.")
		downloadCmd.Usage()
		os.Exit(1)
	}

	// Determine the API key to use
	apiKey := *cliApiKey
	if apiKey == "" { // If flag is not provided, check environment variable
		apiKey = os.Getenv("HF_API_KEY")
	}

	// Create the downloader with HuggingFaceSource, passing the API key
	hfSource := downloader.NewHuggingFaceSource(apiKey) // Use the determined apiKey
	d := downloader.NewDownloader(hfSource)

	fmt.Printf("Downloading model '%s' to '%s'...\n", *modelID, *outputPath)

	result, err := d.Download(*modelID, *outputPath)
	handleErr(err)

	fmt.Printf("Successfully downloaded model to: %s\n", result.ModelPath)
	if len(result.TokenizerPaths) > 0 {
		fmt.Println("Downloaded tokenizer files:")
		for _, p := range result.TokenizerPaths {
			fmt.Printf("  - %s\n", p)
		}
	}
}

func printUsage() {
	fmt.Println("Usage: zonnx <command> [arguments]")
	fmt.Println("\nCommands:")
	fmt.Println("  import <input-file.onnx> [-output <output-file.zmf>]")
	fmt.Println("  export <input-file.zmf> [-output <output-file.onnx>]")
	fmt.Println("  inspect <input-file.onnx>")
	fmt.Println("  inspect-zmf <input-file.zmf>")
	fmt.Println("  convert <input-file.onnx> [-output <output-file.zmf>]")
	fmt.Println("  download --model <huggingface-model-id> [--output <output-directory>] [--api-key <your-api-key> | HF_API_KEY=<your-api-key>]") // Update usage
}

func handleErr(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
