package importer

import (
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zonnx/internal/onnx"
	_ "github.com/zerfoo/zonnx/pkg/importer/layers" // Blank import to trigger layer registration
	"github.com/zerfoo/zonnx/pkg/registry"
	"google.golang.org/protobuf/proto"
)

// ConvertOnnxToZmf loads an ONNX model and converts it to a zerfoo model.
func ConvertOnnxToZmf[T tensor.Numeric](
	path string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
) (*model.Model[T], error) {
	onnxModel, err := LoadOnnxModel(path)
	if err != nil {
		return nil, err
	}

	// Prepare the conversion context
	ctx := &registry.ConversionContext{
		Initializers: make(map[string]*onnx.TensorProto),
		ValueInfo:    make(map[string]*onnx.ValueInfoProto),
	}
	for _, init := range onnxModel.GetGraph().GetInitializer() {
		ctx.Initializers[init.GetName()] = init
	}
	for _, vi := range onnxModel.GetGraph().GetValueInfo() {
		ctx.ValueInfo[vi.GetName()] = vi
	}
	for _, vi := range onnxModel.GetGraph().GetInput() {
		ctx.ValueInfo[vi.GetName()] = vi
	}
	for _, vi := range onnxModel.GetGraph().GetOutput() {
		ctx.ValueInfo[vi.GetName()] = vi
	}

	// Build the zerfoo graph
	builder := graph.NewBuilder[T](engine)
	nodeOutputMap := make(map[string]graph.Node[T])

	// Add graph inputs
	for _, input := range onnxModel.GetGraph().GetInput() {
		// Skip initializers, which are also listed as inputs
		if _, isInitializer := ctx.Initializers[input.GetName()]; isInitializer {
			continue
		}
		inputNode := builder.Input([]int{}) // TODO: Handle shapes
		nodeOutputMap[input.GetName()] = inputNode
	}

	for _, nodeDef := range onnxModel.GetGraph().GetNode() {
		constructorAny, ok := registry.Get(nodeDef.GetOpType())
		if !ok {
			return nil, fmt.Errorf("unsupported op_type: %s", nodeDef.GetOpType())
		}

		constructor, ok := constructorAny.(registry.LayerConstructor[T])
		if !ok {
			return nil, fmt.Errorf("invalid constructor type for op_type: %s", nodeDef.GetOpType())
		}

		var inputs []graph.Node[T]
		for _, inputName := range nodeDef.GetInput() {
			if _, isInitializer := ctx.Initializers[inputName]; isInitializer {
				// This input is a constant weight/parameter, not a node output
				continue
			}
			if inputNode, exists := nodeOutputMap[inputName]; exists {
				inputs = append(inputs, inputNode)
			} else {
				return nil, fmt.Errorf("could not find input %s for node %s", inputName, nodeDef.GetName())
			}
		}

		zerfooNode, err := constructor(engine, ops, nodeDef, ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to construct node %s (type %s): %w", nodeDef.GetName(), nodeDef.GetOpType(), err)
		}

		builder.AddNode(zerfooNode, inputs...)

		for _, outputName := range nodeDef.GetOutput() {
			nodeOutputMap[outputName] = zerfooNode
		}
	}

	finalOutputName := onnxModel.GetGraph().GetOutput()[0].GetName()
	outputNode, ok := nodeOutputMap[finalOutputName]
	if !ok {
			return nil, fmt.Errorf("could not find final output node: %s", finalOutputName)
	}

	zerfooGraph, err := builder.Build(outputNode)
	if err != nil {
		return nil, fmt.Errorf("failed to build graph: %w", err)
	}

	return model.NewModel(nil, zerfooGraph), nil
}

// LoadOnnxModel reads an ONNX model file and returns the parsed ModelProto.
func LoadOnnxModel(path string) (*onnx.ModelProto, error) {
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
