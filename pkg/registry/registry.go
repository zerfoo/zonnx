package registry

import (
	"github.com/zerfoo/zmf"
	"github.com/zerfoo/zonnx/internal/onnx"
)

// ConversionContext holds all the graph-level information needed during conversion.
type ConversionContext struct {
	Initializers map[string]*onnx.TensorProto
	ValueInfo    map[string]*onnx.ValueInfoProto
	QuantizationInfo map[string]*zmf.Quantization // Map from tensor name to its quantization parameters
}

// LayerConstructor is a function that constructs a zerfoo layer from an ONNX node.
// It takes the ONNX node definition and a conversion context.
type LayerConstructor func(
	nodeDef *onnx.NodeProto,
	ctx *ConversionContext,
) (interface{}, error)


// constructors holds the mapping from ONNX op_types to our layer constructors.
var constructors = make(map[string]LayerConstructor)

// Register adds a new layer constructor to the registry.
func Register(opType string, constructor LayerConstructor) {
	constructors[opType] = constructor
}

// Get returns the constructor for a given op type.
func Get(opType string) (LayerConstructor, bool) {
	constructor, ok := constructors[opType]
	return constructor, ok
}