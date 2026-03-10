package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("Pad", BuildPad)
}

// BuildPad creates a Pad layer from an ONNX node.
// In ONNX opset 11+, pads and constant_value are encoded as input tensors.
// The converter promotes them to ZMF attributes named "pads" and
// "constant_value" before this builder is invoked.
func BuildPad(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
