package inspector

import (
	"fmt"

	"github.com/zerfoo/zonnx/pkg/importer"
	"github.com/zerfoo/zonnx/pkg/zmf_inspector" // Re-using existing zmf_inspector for now
)

// InspectONNX inspects an ONNX model and prints its summary.
func InspectONNX(inputFile string) error {
	fmt.Printf("Inspecting ONNX model from: %s\n", inputFile)

	model, err := importer.LoadOnnxModel(inputFile)
	if err != nil {
		return fmt.Errorf("failed to load ONNX model: %w", err)
	}

	fmt.Printf("Successfully loaded model with IR version: %d\n", model.GetIrVersion())
	if len(model.GetOpsetImport()) > 0 {
		fmt.Printf("Opset version: %d\n", model.GetOpsetImport()[0].GetVersion())
	}
	fmt.Printf("Graph has %d nodes.\n", len(model.GetGraph().GetNode()))

	return nil
}

// InspectZMF inspects a ZMF model and prints its summary.
func InspectZMF(inputFile string) error {
	fmt.Printf("Inspecting ZMF model from: %s\n", inputFile)

	model, err := zmf_inspector.Load(inputFile)
	if err != nil {
		return fmt.Errorf("failed to load ZMF model: %w", err)
	}

	zmf_inspector.Inspect(model)

	return nil
}
