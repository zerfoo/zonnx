package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings" // Added for strings.ToLower

	"github.com/zerfoo/zonnx/pkg/importer"
	// "github.com/zerfoo/zonnx/pkg/zmf_inspector" // Removed unused import
	"google.golang.org/protobuf/proto"

	// Import the downloader package
	"github.com/zerfoo/zonnx/pkg/downloader"
	// Import the new inspector package
	"github.com/zerfoo/zonnx/pkg/inspector"
)

func main() {
	logFile, err := os.OpenFile("zonnx-converter.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open log file: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if cerr := logFile.Close(); cerr != nil {
			fmt.Fprintf(os.Stderr, "Error closing log file: %v\n", cerr)
		}
	}()
	log.SetOutput(logFile)

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
	// case "inspect-zmf": // Remove inspect-zmf
	// 	handleInspectZMF()
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
	fileType := inspectCmd.String("type", "", "Type of model to inspect: 'onnx' or 'zmf'")
	prettyPrint := inspectCmd.Bool("pretty", false, "Pretty print the output (human-friendly)") // Add --pretty flag

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

	// Determine file type
	detectedType := strings.ToLower(*fileType)
	if detectedType == "" {
		ext := strings.ToLower(filepath.Ext(inputFile))
		switch ext {
		case ".onnx":
			detectedType = "onnx"
		case ".zmf":
			detectedType = "zmf"
		default:
			fmt.Printf("Error: Could not infer file type from extension '%s'. Please specify --type flag.\n", ext)
			inspectCmd.Usage()
			os.Exit(1)
		}
	}

	var err error
	switch detectedType {
	case "onnx":
		err = inspector.InspectONNX(inputFile)
	case "zmf":
		err = inspector.InspectZMF(inputFile)
	default:
		fmt.Printf("Error: Unsupported model type '%s'. Must be 'onnx' or 'zmf'.\n", detectedType)
		inspectCmd.Usage()
		os.Exit(1)
	}
	handleErr(err)

	// TODO: Implement --pretty printing based on standardized JSON schema
	if *prettyPrint {
		fmt.Println("Pretty printing is not yet implemented.")
	}
}

// Removed handleInspectZMF function
// func handleInspectZMF() { ... }

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

	// Use the refactored importer.ConvertOnnxToZmf directly
	zmfModel, err := importer.ConvertOnnxToZmf(inputFile)
	handleErr(err)

	// Serialize the ZMF model to a file
	outBytes, err := proto.Marshal(zmfModel)
	handleErr(err)

	err = os.WriteFile(*outputFile, outBytes, 0o644)
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
	fmt.Println("  inspect <input-file> [--type <onnx|zmf>] [--pretty]")
	// fmt.Println("  inspect-zmf <input-file.zmf>") // Removed inspect-zmf usage
	fmt.Println("  convert <input-file.onnx> [-output <output-file.zmf>]")
	fmt.Println("  download --model <huggingface-model-id> [--output <output-directory>] [--api-key <your-api-key> | HF_API_KEY=<your-api-key>]")
}

func handleErr(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
