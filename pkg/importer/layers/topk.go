package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("TopK", buildTopK)
}

// buildTopK creates a TopK layer from an ONNX node.
// The ONNX K input tensor is promoted to a ZMF "k" attribute by the
// converter. The "axis", "largest", and "sorted" ONNX attributes are
// preserved by the generic convertNode path.
func buildTopK(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
