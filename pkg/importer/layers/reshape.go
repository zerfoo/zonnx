package layers

import (
	"encoding/binary"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zonnx/pkg/importer"
)

func init() {
	importer.Register("Reshape", BuildReshape[float32])
}

// BuildReshape creates a new Reshape layer from an ONNX node.
func BuildReshape[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	node *onnx.NodeProto,
	ctx *importer.ConversionContext,
) (graph.Node[T], error) {

	if len(node.GetInput()) != 2 {
		return nil, fmt.Errorf("ONNX Reshape node %s must have 2 inputs (data, shape)", node.GetName())
	}
	shapeTensorName := node.GetInput()[1]

	shapeTensor, ok := ctx.Initializers[shapeTensorName]
	if !ok {
		return nil, fmt.Errorf("could not find shape initializer tensor '%s' for Reshape node %s", shapeTensorName, node.GetName())
	}

	// Parse the shape tensor data
	if shapeTensor.GetDataType() != onnx.TensorProto_INT64 {
		return nil, fmt.Errorf("shape tensor %s must be of type INT64", shapeTensorName)
	}

	rawData := shapeTensor.GetRawData()
	if len(rawData)%8 != 0 {
		return nil, fmt.Errorf("invalid raw data length for INT64 tensor %s", shapeTensorName)
	}

	numElements := len(rawData) / 8
	targetShape := make([]int, numElements)
	for i := 0; i < numElements; i++ {
		val := binary.LittleEndian.Uint64(rawData[i*8 : (i+1)*8])
		targetShape[i] = int(val)
	}

	// Note: The logic for resolving 0 and -1 in the shape is NOT needed here.
	// That logic was for cleaning the zerfoo layer itself. The zonnx converter's
	// primary job is to read the data as-is from the ONNX constant tensor.
	// If resolution logic were needed, it would be applied here before creating the node.

	return core.NewReshape[T](engine, targetShape), nil
}
