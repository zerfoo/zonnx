package layers

import (
	"encoding/binary"
	"fmt"

	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/registry"
)

func init() {
	registry.Register("Reshape", BuildReshape)
}

// BuildReshape creates a new Reshape layer from an ONNX node.
func BuildReshape(
	node *onnx.NodeProto,
	ctx *registry.ConversionContext,
) (interface{}, error) {
	if len(node.GetInput()) != 2 {
		return nil, fmt.Errorf("ONNX Reshape node %s must have 2 inputs (data, shape)", node.GetName())
	}
	shapeTensorName := node.GetInput()[1]

	shapeTensor, ok := ctx.Initializers[shapeTensorName]
	if !ok {
		return nil, fmt.Errorf("could not find shape initializer tensor '%s' for Reshape node %s", shapeTensorName, node.GetName())
	}

	// Parse the shape tensor data
	if onnx.TensorProto_DataType(shapeTensor.GetDataType()) != onnx.TensorProto_INT64 {
		return nil, fmt.Errorf("shape tensor %s must be of type INT64", shapeTensorName)
	}

	rawData := shapeTensor.GetRawData()
	if len(rawData)%8 != 0 {
		return nil, fmt.Errorf("invalid raw data length for INT64 tensor %s", shapeTensorName)
	}

	numElements := len(rawData) / 8
	targetShape := make([]int64, numElements) // Changed to int64 for ZMF compatibility
	for i := 0; i < numElements; i++ {
		val := binary.LittleEndian.Uint64(rawData[i*8 : (i+1)*8])
		targetShape[i] = int64(val)
	}

	// TODO: Return a ZMF representation of Reshape, including targetShape
	return nil, nil
}
