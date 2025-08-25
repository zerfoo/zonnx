package importer

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/zerfoo/zmf"
	"github.com/zerfoo/zonnx/internal/onnx"
	_ "github.com/zerfoo/zonnx/pkg/importer/layers" // Blank import to trigger layer registration
	"github.com/zerfoo/zonnx/pkg/registry"
	"google.golang.org/protobuf/proto"
)

// ConvertOnnxToZmf loads an ONNX model and converts it to a ZMF model.
func ConvertOnnxToZmf(
	path string,
) (*zmf.Model, error) {
	onnxModel, err := LoadOnnxModel(path)
	if err != nil {
		return nil, err
	}

	// Prepare the conversion context
	ctx := &registry.ConversionContext{
		Initializers: make(map[string]*onnx.TensorProto),
		ValueInfo:    make(map[string]*onnx.ValueInfoProto),
	}
	for _, init := range onnxModel.GetGraph().GetInitializer() {
		ctx.Initializers[init.GetName()] = init
	}
	for _, vi := range onnxModel.GetGraph().GetValueInfo() {
		ctx.ValueInfo[vi.GetName()] = vi
	}
	for _, vi := range onnxModel.GetGraph().GetInput() {
		ctx.ValueInfo[vi.GetName()] = vi
	}
	for _, vi := range onnxModel.GetGraph().GetOutput() {
		ctx.ValueInfo[vi.GetName()] = vi
	}

	// Process quantization annotations
	ctx.QuantizationInfo = make(map[string]*zmf.Quantization)
	for _, qa := range onnxModel.GetGraph().GetQuantizationAnnotation() {
		tensorName := qa.GetTensorName()
		var scale float32
		var zeroPoint int64

		for _, param := range qa.GetQuantParameterTensorNames() {
			paramName := param.GetKey()
			paramTensorName := param.GetValue()

			paramTensor, ok := ctx.Initializers[paramTensorName]
			if !ok {
				return nil, fmt.Errorf("quantization parameter tensor %s not found for %s", paramTensorName, tensorName)
			}

			switch paramName {
			case "SCALE_TENSOR":
				val, err := extractScalarFloat(paramTensor)
				if err != nil {
					return nil, fmt.Errorf("failed to extract scale for %s: %w", tensorName, err)
				}
				scale = val
			case "ZERO_POINT_TENSOR":
				val, err := extractScalarInt64(paramTensor)
				if err != nil {
					return nil, fmt.Errorf("failed to extract zero point for %s: %w", tensorName, err)
				}
				zeroPoint = val
			}
		}

		if scale != 0 || zeroPoint != 0 { // Only add if quantization info is present
			ctx.QuantizationInfo[tensorName] = &zmf.Quantization{
				Scale:     scale,
				ZeroPoint: zeroPoint,
			}
		}
	}

	zmfGraph := &zmf.Graph{
		Parameters: make(map[string]*zmf.Tensor),
		Nodes:      make([]*zmf.Node, 0),
		Inputs:     make([]*zmf.ValueInfo, 0),
		Outputs:    make([]*zmf.ValueInfo, 0),
	}

	// Populate parameters from initializers
	for _, init := range onnxModel.GetGraph().GetInitializer() {
		zmfGraph.Parameters[init.GetName()] = onnxTensorToZmfTensor(init, ctx)
	}

	// Populate nodes
	for _, nodeDef := range onnxModel.GetGraph().GetNode() {
		zmfNode := &zmf.Node{
			Name:       nodeDef.GetName(),
			OpType:     nodeDef.GetOpType(),
			Inputs:     nodeDef.GetInput(),
			Outputs:    nodeDef.GetOutput(),
			Attributes: make(map[string]*zmf.Attribute),
		}
		for _, attr := range nodeDef.GetAttribute() {
			switch attr.GetName() {
			case "epsilon":
				if attr.GetType() == onnx.AttributeProto_FLOAT {
					fVal := attr.GetF()
					zmfNode.Epsilon = &fVal
				}
			case "perm":
				if attr.GetType() == onnx.AttributeProto_INTS {
					zmfNode.Perm = attr.GetInts()
				}
			case "axis":
				if attr.GetType() == onnx.AttributeProto_INT {
					iVal := attr.GetI()
					zmfNode.Axis = &iVal
				}
			default:
				zmfAttr, err := onnxAttributeToZmfAttribute(attr)
				if err != nil {
					return nil, fmt.Errorf("failed to convert ONNX attribute %s: %w", attr.GetName(), err)
				}
				zmfNode.Attributes[attr.GetName()] = zmfAttr
			}
		}
		zmfGraph.Nodes = append(zmfGraph.Nodes, zmfNode)
	}

	// Populate inputs and outputs
	for _, input := range onnxModel.GetGraph().GetInput() {
		// Skip initializers, which are also listed as inputs
		if _, isInitializer := ctx.Initializers[input.GetName()]; isInitializer {
			continue
		}
		zmfGraph.Inputs = append(zmfGraph.Inputs, onnxValueInfoToZmfValueInfo(input))
	}
	for _, output := range onnxModel.GetGraph().GetOutput() {
		zmfGraph.Outputs = append(zmfGraph.Outputs, onnxValueInfoToZmfValueInfo(output))
	}

	zmfModel := &zmf.Model{
		Graph: zmfGraph,
		Metadata: &zmf.Metadata{
			ProducerName:    "zonnx",
			ProducerVersion: "0.1.0",                                    // TODO: Get actual version
			OpsetVersion:    onnxModel.GetOpsetImport()[0].GetVersion(), // Assuming single opset for now
		},
	}

	return zmfModel, nil
}

// Helper function to convert ONNX TensorProto to ZMF Tensor
func onnxTensorToZmfTensor(ot *onnx.TensorProto, ctx *registry.ConversionContext) *zmf.Tensor {
	zmfDtype := onnxDataTypeToZmfDataType(onnx.TensorProto_DataType(ot.GetDataType()))
	zmfTensor := &zmf.Tensor{
		Dtype: zmfDtype,
		Shape: ot.GetDims(),
		Data:  ot.GetRawData(),
	}

	// Check for quantization info
	if quantInfo, ok := ctx.QuantizationInfo[ot.GetName()]; ok {
		zmfTensor.Quant = quantInfo
	}

	return zmfTensor
}

// Helper function to convert ONNX ValueInfoProto to ZMF ValueInfo
func onnxValueInfoToZmfValueInfo(ovi *onnx.ValueInfoProto) *zmf.ValueInfo {
	zmfDtype := onnxDataTypeToZmfDataType(onnx.TensorProto_DataType(ovi.GetType().GetTensorType().GetElemType()))
	return &zmf.ValueInfo{
		Name:  ovi.GetName(),
		Dtype: zmfDtype,
		Shape: func() []int64 {
			dims := ovi.GetType().GetTensorType().GetShape().GetDim()
			shape := make([]int64, len(dims))
			for i, d := range dims {
				// Assuming concrete dimensions for now. Symbolic dimensions (DimParam) would need further handling.
				shape[i] = d.GetDimValue()
			}
			return shape
		}(),
	}
}

// onnxDataTypeToZmfDataType converts an ONNX TensorProto_DataType to a ZMF Tensor_DataType.
func onnxDataTypeToZmfDataType(onnxType onnx.TensorProto_DataType) zmf.Tensor_DataType {
	switch onnxType {
	case onnx.TensorProto_FLOAT:
		return zmf.Tensor_FLOAT32
	case onnx.TensorProto_FLOAT16:
		return zmf.Tensor_FLOAT16
	case onnx.TensorProto_BFLOAT16:
		return zmf.Tensor_BFLOAT16
	case onnx.TensorProto_FLOAT8E4M3FN, onnx.TensorProto_FLOAT8E4M3FNUZ, onnx.TensorProto_FLOAT8E5M2, onnx.TensorProto_FLOAT8E5M2FNUZ:
		return zmf.Tensor_FLOAT8
	case onnx.TensorProto_INT32:
		return zmf.Tensor_INT32
	case onnx.TensorProto_INT64:
		return zmf.Tensor_INT64
	case onnx.TensorProto_DOUBLE:
		return zmf.Tensor_FLOAT64
	case onnx.TensorProto_BOOL:
		return zmf.Tensor_BOOL
	case onnx.TensorProto_STRING:
		return zmf.Tensor_STRING
	case onnx.TensorProto_UINT8:
		return zmf.Tensor_UINT8
	case onnx.TensorProto_INT8:
		return zmf.Tensor_INT8
	// Handle cases where there's no direct mapping or we choose a compatible type
	case onnx.TensorProto_UINT16, onnx.TensorProto_INT16, onnx.TensorProto_UINT32, onnx.TensorProto_UINT64:
		return zmf.Tensor_INT64 // Default to INT64 for other integer types
	default:
		panic(fmt.Sprintf("unsupported ONNX data type: %s", onnxType.String()))
	}
}

// onnxAttributeToZmfAttribute converts an ONNX AttributeProto to a ZMF Attribute.
func onnxAttributeToZmfAttribute(oa *onnx.AttributeProto) (*zmf.Attribute, error) {
	zmfAttr := &zmf.Attribute{}

	switch oa.GetType() {
	case onnx.AttributeProto_FLOAT:
		zmfAttr.Value = &zmf.Attribute_F{F: oa.GetF()}
	case onnx.AttributeProto_INT:
		// ONNX represents booleans as INT attributes (0 or 1)
		if oa.GetI() == 0 || oa.GetI() == 1 {
			zmfAttr.Value = &zmf.Attribute_B{B: oa.GetI() == 1}
		} else {
			zmfAttr.Value = &zmf.Attribute_I{I: oa.GetI()}
		}
	case onnx.AttributeProto_STRING:
		zmfAttr.Value = &zmf.Attribute_S{S: string(oa.GetS())}
	case onnx.AttributeProto_FLOATS:
		zmfAttr.Value = &zmf.Attribute_Floats{Floats: &zmf.Floats{Val: oa.GetFloats()}}
	case onnx.AttributeProto_INTS:
		zmfAttr.Value = &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: oa.GetInts()}}
	case onnx.AttributeProto_STRINGS:
		strVals := make([]string, len(oa.GetStrings()))
		for i, s := range oa.GetStrings() {
			strVals[i] = string(s)
		}
		zmfAttr.Value = &zmf.Attribute_Strings{Strings: &zmf.Strings{Val: strVals}}
	case onnx.AttributeProto_TENSOR, onnx.AttributeProto_GRAPH, onnx.AttributeProto_SPARSE_TENSOR, onnx.AttributeProto_TYPE_PROTO,
		onnx.AttributeProto_TENSORS, onnx.AttributeProto_GRAPHS, onnx.AttributeProto_SPARSE_TENSORS, onnx.AttributeProto_TYPE_PROTOS:
		return nil, fmt.Errorf("unsupported ONNX attribute type for ZMF: %s", oa.GetType().String())
	default:
		return nil, fmt.Errorf("unknown ONNX attribute type: %s", oa.GetType().String())
	}

	return zmfAttr, nil
}

// extractScalarFloat extracts a single float32 value from an ONNX TensorProto.
func extractScalarFloat(t *onnx.TensorProto) (float32, error) {
	if len(t.GetDims()) != 0 {
		return 0, fmt.Errorf("expected scalar tensor, got shape %v", t.GetDims())
	}
	if onnx.TensorProto_DataType(t.GetDataType()) != onnx.TensorProto_FLOAT {
		return 0, fmt.Errorf("expected float32 tensor, got type %s", onnx.TensorProto_DataType(t.GetDataType()).String())
	}
	if len(t.GetRawData()) != 4 {
		return 0, fmt.Errorf("expected 4 bytes for float32, got %d bytes", len(t.GetRawData()))
	}
	return math.Float32frombits(binary.LittleEndian.Uint32(t.GetRawData())), nil
}

// extractScalarInt64 extracts a single int64 value from an ONNX TensorProto.
func extractScalarInt64(t *onnx.TensorProto) (int64, error) {
	if len(t.GetDims()) != 0 {
		return 0, fmt.Errorf("expected scalar tensor, got shape %v", t.GetDims())
	}
	if onnx.TensorProto_DataType(t.GetDataType()) != onnx.TensorProto_INT64 {
		return 0, fmt.Errorf("expected int64 tensor, got type %s", onnx.TensorProto_DataType(t.GetDataType()).String())
	}
	if len(t.GetRawData()) != 8 {
		return 0, fmt.Errorf("expected 8 bytes for int64, got %d bytes", len(t.GetRawData()))
	}
	return int64(binary.LittleEndian.Uint64(t.GetRawData())), nil
}

// LoadOnnxModel reads an ONNX model file and returns the parsed ModelProto.
func LoadOnnxModel(path string) (*onnx.ModelProto, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read ONNX file: %w", err)
	}

	model := &onnx.ModelProto{}
	if err := proto.Unmarshal(data, model); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ONNX protobuf: %w", err)
	}

	return model, nil
}
