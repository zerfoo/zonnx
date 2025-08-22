package registry

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zonnx/internal/onnx"
)

// ConversionContext holds all the graph-level information needed during conversion.
type ConversionContext struct {
	Initializers map[string]*onnx.TensorProto
	ValueInfo    map[string]*onnx.ValueInfoProto
}

// LayerConstructor defines a function that creates a zerfoo graph.Node from an ONNX NodeProto.
type LayerConstructor[T tensor.Numeric] func(
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	node *onnx.NodeProto,
	ctx *ConversionContext,
) (graph.Node[T], error)

// registry holds the mapping from ONNX op_types to our layer constructors.
var registry = make(map[string]any)

// Register adds a new layer constructor to the registry.
func Register[T tensor.Numeric](opType string, constructor LayerConstructor[T]) {
	registry[opType] = constructor
}

// Get returns the constructor for a given op type.
func Get(opType string) (any, bool) {
	constructor, ok := registry[opType]
	return constructor, ok
}
