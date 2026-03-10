package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("Resize", BuildResize)
}

// BuildResize creates a Resize layer from an ONNX node.
// The converter promotes input[2] (scales, FLOAT tensor) to a "scales" FLOATS
// attribute and input[3] (sizes, INT64 tensor) to a "sizes" INTS attribute.
// The "mode" ONNX attribute is preserved by the generic convertNode path.
func BuildResize(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
