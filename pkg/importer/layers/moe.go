package layers

import (
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("MoEGate", BuildMoEGate)
	registry.Register("MixtureOfExperts", BuildMixtureOfExperts)
}

// BuildMoEGate creates a MoEGate layer from an ONNX node.
// The "top_k" INT attribute is preserved by the generic convertNode path.
func BuildMoEGate(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}

// BuildMixtureOfExperts creates a MixtureOfExperts layer from an ONNX node.
// The "num_experts" and "top_k" INT attributes are preserved by the generic
// convertNode path. Expert sub-graph loading is not yet supported (tech debt).
func BuildMixtureOfExperts(
	_ *onnx.NodeProto,
	_ *registry.ConversionContext,
) (interface{}, error) {
	return nil, nil
}
