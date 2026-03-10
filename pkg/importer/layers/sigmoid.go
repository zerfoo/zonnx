package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("Sigmoid", BuildSigmoid)
}

// BuildSigmoid creates a Sigmoid layer from an ONNX node.
// Sigmoid has no configurable attributes.
func BuildSigmoid(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
