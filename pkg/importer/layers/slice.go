package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("Slice", buildSlice)
}

// buildSlice creates a Slice layer from an ONNX node.
// In ONNX opset 10+, starts/ends/axes/steps are encoded as input tensors.
// The converter promotes them to ZMF attributes named "starts", "ends",
// "axes", and "steps" before this builder is invoked.
func buildSlice(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
