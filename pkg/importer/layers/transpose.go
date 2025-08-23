package layers

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/transpose"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("Transpose", BuildTranspose[float32])
}

// BuildTranspose creates a new Transpose layer from an ONNX node.
func BuildTranspose[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	node *onnx.NodeProto,
	ctx *registry.ConversionContext,
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
		// We need to get the rank of the input tensor to do this.
		if len(node.GetInput()) == 0 {
			return nil, fmt.Errorf("transpose node %s has no inputs", node.GetName()) // Fixed capitalization
		}
		inputName := node.GetInput()[0]
		valueInfo, ok := ctx.ValueInfo[inputName]
		if !ok {
			return nil, fmt.Errorf("could not find value info for input tensor %s in Transpose node %s", inputName, node.GetName())
		}

		inputRank := len(valueInfo.GetType().GetTensorType().GetShape().GetDim())
		if inputRank == 0 {
			// A scalar has a rank of 0, its transpose is itself.
			perm = []int{}
		} else {
			perm = make([]int, inputRank)
			for i := 0; i < inputRank; i++ {
				perm[i] = inputRank - 1 - i
			}
		}
	}

	return transpose.New(engine, perm), nil
}
