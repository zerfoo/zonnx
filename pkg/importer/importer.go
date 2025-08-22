package importer

import (
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zonnx/internal/onnx"
	"google.golang.org/protobuf/proto"
)

// ConvertOnnxToZmf loads an ONNX model from a file and will (eventually) convert it to a ZMF model.
func ConvertOnnxToZmf(path string) (*model.Model, error) {
	onnxModel, err := loadOnnxModel(path)
	if err != nil {
		return nil, err
	}

	// TODO: Add the logic to convert the onnxModel to a zerfoo/model.Model
	fmt.Printf("Successfully loaded ONNX model: %s\n", onnxModel.GetGraph().GetName())

	return nil, fmt.Errorf("conversion logic not yet implemented")
}

// loadOnnxModel reads an ONNX model file and returns the parsed ModelProto.
func loadOnnxModel(path string) (*onnx.ModelProto, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read ONNX file: %w", err)
	}

	model := &onnx.ModelProto{}
	if err := proto.Unmarshal(data, model); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ONNX protobuf: %w", err)
	}

	return model, nil
}