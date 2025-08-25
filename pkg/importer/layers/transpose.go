package layers

import (
	"fmt"

	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("Transpose", BuildTranspose)
}

// BuildTranspose creates a new Transpose layer from an ONNX node.
func BuildTranspose(
	node *onnx.NodeProto,
	ctx *registry.ConversionContext,
) (interface{}, error) {
	var perm []int64 // Changed to int64 for ZMF compatibility
	found := false

	for _, attr := range node.GetAttribute() {
		if attr.GetName() == "perm" {
			perm = make([]int64, len(attr.GetInts()))
			for i, v := range attr.GetInts() {
				perm[i] = v
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
			perm = []int64{}
		} else {
			perm = make([]int64, inputRank)
			for i := 0; i < inputRank; i++ {
				perm[i] = int64(inputRank - 1 - i)
			}
		}
	}

	// TODO: Return a ZMF representation of Transpose, including perm
	_ = perm
	return nil, nil
}
