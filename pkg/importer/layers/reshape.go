package layers

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/importer"
)

func init() {
	importer.Register("Reshape", BuildReshape[float32])
}

// BuildReshape creates a new Reshape layer from an ONNX node.
func BuildReshape[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	node *onnx.NodeProto,
	_ map[string]*graph.Parameter[T],
) (graph.Node[T], error) {

	// In ONNX, the target shape is the second input to the Reshape node.
	// This input must be a constant initializer tensor for this importer to work.
	if len(node.GetInput()) != 2 {
		return nil, fmt.Errorf("ONNX Reshape node %s must have 2 inputs (data, shape)", node.GetName())
	}
	shapeTensorName := node.GetInput()[1]

	// TODO: Find the shape tensor in the graph's initializers
	// and parse its int64 data into a []int slice.
	// This involves looking through the onnx.GraphProto.Initializer list.
	// For now, we will use a placeholder.
	targetShape := []int{1, 768} // Placeholder

	fmt.Printf("Warning: Reshape operator for node %s is using a placeholder shape %v for tensor %s\n", node.GetName(), targetShape, shapeTensorName)

	return core.NewReshape[T](engine, targetShape), nil
}
