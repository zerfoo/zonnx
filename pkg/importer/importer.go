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
	"google.golang.org/protobuf/proto"
)

// LayerConstructor defines a function that creates a zerfoo graph.Node from an ONNX NodeProto.
type LayerConstructor[T tensor.Numeric] func(
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	node *onnx.NodeProto,
	params map[string]*graph.Parameter[T],
) (graph.Node[T], error)

// registry holds the mapping from ONNX op_types to our layer constructors.
var registry = make(map[string]any)

// Register adds a new layer constructor to the registry.
func Register[T tensor.Numeric](opType string, constructor LayerConstructor[T]) {
	registry[opType] = constructor
}

// ConvertOnnxToZmf loads an ONNX model and converts it to a zerfoo model.
func ConvertOnnxToZmf[T tensor.Numeric](
	path string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
) (*model.Model[T], error) {
	onnxModel, err := loadOnnxModel(path)
	if err != nil {
		return nil, err
	}

	// TODO: Load parameters
	loadedParams := make(map[string]*graph.Parameter[T])

	// Build the zerfoo graph
	builder := graph.NewBuilder[T](engine)
	// Map ONNX tensor names to the zerfoo nodes that produce them
	nodeOutputMap := make(map[string]graph.Node[T])

	// Add graph inputs
	for _, input := range onnxModel.GetGraph().GetInput() {
		// TODO: Handle shapes properly
		inputNode := builder.Input([]int{})
		nodeOutputMap[input.GetName()] = inputNode
	}

	for _, nodeDef := range onnxModel.GetGraph().GetNode() {
		constructorAny, ok := registry[nodeDef.GetOpType()]
		if !ok {
			return nil, fmt.Errorf("unsupported op_type: %s", nodeDef.GetOpType())
		}

		constructor, ok := constructorAny.(LayerConstructor[T])
		if !ok {
			return nil, fmt.Errorf("invalid constructor type for op_type: %s", nodeDef.GetOpType())
		}

		// Gather inputs for the current node
		var inputs []graph.Node[T]
		for _, inputName := range nodeDef.GetInput() {
			if inputNode, exists := nodeOutputMap[inputName]; exists {
				inputs = append(inputs, inputNode)
			} else {
				return nil, fmt.Errorf("could not find input %s for node %s", inputName, nodeDef.GetName())
			}
		}

		zerfooNode, err := constructor(engine, ops, nodeDef, loadedParams)
		if err != nil {
			return nil, fmt.Errorf("failed to construct node %s (type %s): %w", nodeDef.GetName(), nodeDef.GetOpType(), err)
		}

		// Add the new node to the graph
		builder.AddNode(zerfooNode, inputs...)

		// Map the output tensors of this new node
		for _, outputName := range nodeDef.GetOutput() {
			nodeOutputMap[outputName] = zerfooNode
		}
	}

	// This is a simplification. We need to identify the correct final output node.
	finalOutputName := onnxModel.GetGraph().GetOutput()[0].GetName()
	outputNode, ok := nodeOutputMap[finalOutputName]
	if !ok {
		return nil, fmt.Errorf("could not find final output node: %s", finalOutputName)
	}

	zerfooGraph, err := builder.Build(outputNode)
	if err != nil {
		return nil, fmt.Errorf("failed to build graph: %w", err)
	}

	// TODO: Handle embedding layer properly
	return model.NewModel(nil, zerfooGraph), nil
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
