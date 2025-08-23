package main

import (
	"fmt"
	"log"
	"os"

	"github.com/zerfoo/zonnx/pkg/converter"
	"github.com/zerfoo/zonnx/pkg/importer"
	"google.golang.org/protobuf/proto"
)

func main() {
	if len(os.Args) != 3 {
		log.Fatal("Usage: go run test_convert.go <input.onnx> <output.zmf>")
	}

	inputFile := os.Args[1]
	outputFile := os.Args[2]

	fmt.Printf("Loading ONNX model from: %s\n", inputFile)
	model, err := importer.LoadOnnxModel(inputFile)
	if err != nil {
		log.Fatalf("Failed to load ONNX model: %v", err)
	}

	fmt.Printf("Converting to ZMF with external data support...\n")
	zmfModel, err := converter.ONNXToZMFWithPath(model, inputFile)
	if err != nil {
		log.Fatalf("Failed to convert to ZMF: %v", err)
	}

	fmt.Printf("Serializing ZMF model...\n")
	outBytes, err := proto.Marshal(zmfModel)
	if err != nil {
		log.Fatalf("Failed to marshal ZMF: %v", err)
	}

	fmt.Printf("Writing to: %s\n", outputFile)
	err = os.WriteFile(outputFile, outBytes, 0o644)
	if err != nil {
		log.Fatalf("Failed to write file: %v", err)
	}

	fmt.Printf("Successfully converted! Output size: %d bytes\n", len(outBytes))
	fmt.Printf("Parameters in model: %d\n", len(zmfModel.Graph.Parameters))
}
