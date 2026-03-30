package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("Sigmoid", buildSigmoid)
}

// buildSigmoid creates a Sigmoid layer from an ONNX node.
// Sigmoid has no configurable attributes.
func buildSigmoid(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
