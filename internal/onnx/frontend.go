package onnx

import (
	"fmt"
	"github.com/yalue/onnxruntime_go"
)

// InspectModel loads an ONNX model and prints basic information about it.
func InspectModel(modelPath string) error {
	// Note: onnxruntime_go requires the full path to the shared library.
	// We will need to make this configurable, but for now, we'll assume a
	// standard Homebrew installation path.
	onnxruntime_go.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.dylib")

	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		return fmt.Errorf("failed to initialize ONNX Runtime environment: %w", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	inputs, outputs, err := onnxruntime_go.GetInputOutputInfo(modelPath)
	if err != nil {
		return fmt.Errorf("failed to get input/output info: %w", err)
	}

	fmt.Println("Successfully loaded ONNX model.")
	fmt.Println("Inputs:")
	for _, input := range inputs {
		fmt.Printf("  - Name: %s, Type: %s, Shape: %v\n", input.Name, input.DataType, input.Dimensions)
	}

	fmt.Println("Outputs:")
	for _, output := range outputs {
		fmt.Printf("  - Name: %s, Type: %s, Shape: %v\n", output.Name, output.DataType, output.Dimensions)
	}

	return nil
}