package layers

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/importer"
)

func init() {
	importer.Register("Relu", BuildReLU[float32])
}

// BuildReLU creates a new ReLU layer from an ONNX node.
func BuildReLU[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	_ *onnx.NodeProto,
	_ *importer.ConversionContext,
) (graph.Node[T], error) {
	return activations.NewReLU[T](engine), nil
}