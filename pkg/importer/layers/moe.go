package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("MoEGate", buildMoEGate)
	registry.Register("MixtureOfExperts", buildMixtureOfExperts)
}

// buildMoEGate creates a MoEGate layer from an ONNX node.
// The "top_k" INT attribute is preserved by the generic convertNode path.
func buildMoEGate(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}

// buildMixtureOfExperts creates a MixtureOfExperts layer from an ONNX node.
// The "num_experts" and "top_k" INT attributes are preserved by the generic
// convertNode path. Expert sub-graph loading is not yet supported (tech debt).
func buildMixtureOfExperts(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
