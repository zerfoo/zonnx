package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("Softmax", BuildSoftmax)
}

// BuildSoftmax creates a Softmax layer from an ONNX node.
// The "axis" attribute (INT, default -1) is preserved by the generic
// convertNode path in the converter; no special handling is needed here.
func BuildSoftmax(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
