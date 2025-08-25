package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("Relu", BuildReLU)
}

// BuildReLU creates a new ReLU layer from an ONNX node.
func BuildReLU(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	// TODO: Return a ZMF representation of ReLU
	return nil, nil
}
