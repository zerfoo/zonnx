package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("Conv", buildConv)
}

// buildConv creates a Conv (2D convolution) layer from an ONNX node.
// The ONNX "Conv" op carries strides, pads, dilations, group, and
// kernel_shape as regular ONNX attributes, which are preserved by the
// generic convertNode path.  X is data input[0]; W (kernel) and optional B
// (bias) are float initializers treated as regular graph inputs.
func buildConv(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
