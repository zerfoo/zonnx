package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("Erf", buildErf)
}

// buildErf creates an Erf layer from an ONNX node.
// Erf has no configurable attributes.
func buildErf(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
