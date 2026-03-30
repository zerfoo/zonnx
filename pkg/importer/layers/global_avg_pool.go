package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("GlobalAveragePool", buildGlobalAveragePool)
}

// buildGlobalAveragePool creates a GlobalAveragePool layer from an ONNX node.
// The op has no ONNX attributes and a single input X [N, C, H, W].
func buildGlobalAveragePool(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
