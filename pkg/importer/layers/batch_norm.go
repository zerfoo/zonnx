package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("BatchNormalization", buildBatchNormalization)
}

// buildBatchNormalization creates a BatchNormalization layer from an ONNX node.
// The "epsilon" (FLOAT) attribute is preserved by the generic convertNode path.
// scale, B, mean, and var are float initializers treated as regular graph inputs.
func buildBatchNormalization(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
