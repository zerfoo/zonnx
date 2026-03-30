package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("LayerNormalization", buildLayerNormalization)
}

// buildLayerNormalization creates a LayerNormalization layer from an ONNX node.
// The "epsilon" (FLOAT) and "axis" (INT) attributes are preserved by the
// generic convertNode path in the converter; Scale and Bias parameters are
// treated as regular graph inputs.
func buildLayerNormalization(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
