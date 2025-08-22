package layers

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/transpose"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/importer"
)

func init() {
	importer.Register("Transpose", BuildTranspose[float32])
}

// BuildTranspose creates a new Transpose layer from an ONNX node.
func BuildTranspose[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	node *onnx.NodeProto,
	_ map[string]*graph.Parameter[T],
) (graph.Node[T], error) {

	var perm []int
	found := false

	for _, attr := range node.GetAttribute() {
		if attr.GetName() == "perm" {
			perm = make([]int, len(attr.GetInts()))
			for i, v := range attr.GetInts() {
				perm[i] = int(v)
			}
		found = true
		break
		}
	}

	if !found {
		// If perm is not present, ONNX specifies reversing the dimensions.
		// TODO: We need the rank (number of dimensions) of the input tensor here.
		// For now, we will assume a rank of 3 for demonstration.
		inputRank := 3
		fmt.Printf("Warning: Transpose operator for node %s is assuming a placeholder rank of %d\n", node.GetName(), inputRank)
		perm = make([]int, inputRank)
		for i := 0; i < inputRank; i++ {
			perm[i] = inputRank - 1 - i
		}
	}

	return transpose.New(engine, perm), nil
}
