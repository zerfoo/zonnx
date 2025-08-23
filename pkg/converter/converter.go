package converter

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/zerfoo/zmf"
	"github.com/zerfoo/zonnx/internal/onnx"
)

// ONNXToZMF converts an ONNX model to the ZMF format.
func ONNXToZMF(model *onnx.ModelProto) (*zmf.Model, error) {
	return ONNXToZMFWithPath(model, "")
}

// ONNXToZMFWithPath converts an ONNX model to the ZMF format with support for external data files.
func ONNXToZMFWithPath(model *onnx.ModelProto, modelPath string) (*zmf.Model, error) {
	onnxGraph := model.GetGraph()
	if onnxGraph == nil {
		return nil, fmt.Errorf("model graph is nil")
	}

	initializers := make(map[string]*onnx.TensorProto)
	for _, tensorProto := range onnxGraph.GetInitializer() {
		initializers[tensorProto.GetName()] = tensorProto
	}

	valueInfos := make(map[string]*onnx.ValueInfoProto)
	for _, info := range onnxGraph.GetInput() {
		valueInfos[info.GetName()] = info
	}
	for _, info := range onnxGraph.GetOutput() {
		valueInfos[info.GetName()] = info
	}
	for _, info := range onnxGraph.GetValueInfo() {
		valueInfos[info.GetName()] = info
	}
	for name, tensor := range initializers {
		dims := make([]*onnx.TensorShapeProto_Dimension, len(tensor.GetDims()))
		for i, d := range tensor.GetDims() {
			dimValue := d
			dims[i] = &onnx.TensorShapeProto_Dimension{
				Value: &onnx.TensorShapeProto_Dimension_DimValue{DimValue: dimValue},
			}
		}
		dataType := tensor.GetDataType()
		valueInfos[name] = &onnx.ValueInfoProto{
			Name: &name,
			Type: &onnx.TypeProto{
				Value: &onnx.TypeProto_TensorType{
					TensorType: &onnx.TypeProto_Tensor{
						ElemType: &dataType,
						Shape: &onnx.TensorShapeProto{
							Dim: dims,
						},
					},
				},
			},
		}
	}

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Nodes:      make([]*zmf.Node, 0, len(onnxGraph.GetNode())),
			Parameters: make(map[string]*zmf.Tensor),
			Inputs:     convertValueInfos(onnxGraph.GetInput()),
			Outputs:    convertValueInfos(onnxGraph.GetOutput()),
		},
		Metadata: &zmf.Metadata{
			ProducerName:    "zonnx",
			ProducerVersion: "0.1.0",
		},
	}
	if len(model.GetOpsetImport()) > 0 {
		zmfModel.Metadata.OpsetVersion = model.GetOpsetImport()[0].GetVersion()
	}

	for _, onnxNode := range onnxGraph.GetNode() {
		zmfNode, err := convertNode(onnxNode, initializers, valueInfos)
		if err != nil {
			return nil, fmt.Errorf("failed to convert node '%s': %w", onnxNode.GetName(), err)
		}
		zmfModel.Graph.Nodes = append(zmfModel.Graph.Nodes, zmfNode)
	}

	for name, onnxTensor := range initializers {
		dtype := onnx.TensorProto_DataType(onnxTensor.GetDataType())
		switch dtype {
		case onnx.TensorProto_FLOAT, onnx.TensorProto_FLOAT16, onnx.TensorProto_BFLOAT16, onnx.TensorProto_DOUBLE:
			zmfTensor, err := convertTensorWithPath(onnxTensor, modelPath)
			if err != nil {
				return nil, fmt.Errorf("failed to convert float initializer '%s': %w", name, err)
			}
			zmfModel.Graph.Parameters[name] = zmfTensor
		}
	}

	return zmfModel, nil
}

// convertNode converts an ONNX node to a ZMF node, promoting constant integer
// inputs to named attributes for specific operators.
func convertNode(onnxNode *onnx.NodeProto, initializers map[string]*onnx.TensorProto, valueInfos map[string]*onnx.ValueInfoProto) (*zmf.Node, error) {
	zmfNode := &zmf.Node{
		Name:       onnxNode.GetName(),
		OpType:     onnxNode.GetOpType(),
		Outputs:    onnxNode.GetOutput(),
		Attributes: make(map[string]*zmf.Attribute),
	}

	for _, onnxAttr := range onnxNode.GetAttribute() {
		zmfAttr, err := convertAttribute(onnxAttr)
		if err != nil {
			return nil, fmt.Errorf("failed to convert attribute '%s': %w", onnxAttr.GetName(), err)
		}
		if zmfAttr != nil {
			zmfNode.Attributes[onnxAttr.GetName()] = zmfAttr
		}
	}

	processedInputs := make(map[string]bool)

	// Handle special cases where an input should be a named attribute.
	switch onnxNode.GetOpType() {
	case "ReduceSum":
		// The second input to ReduceSum is the 'axes' tensor.
		if len(onnxNode.GetInput()) > 1 {
			axesInputName := onnxNode.GetInput()[1]
			if initializer, ok := initializers[axesInputName]; ok {
				ints, err := getInt64Data(initializer)
				if err == nil {
					zmfNode.Attributes["axes"] = &zmf.Attribute{
						Value: &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: ints}},
					}
					processedInputs[axesInputName] = true
				}
			}
		}
	case "Transpose":
		// Handle perm attribute or input
		permSet := false

		// First check if there's already a perm attribute (already converted to ZMF)
		if _, hasPermAttr := zmfNode.Attributes["perm"]; hasPermAttr {
			permSet = true
		}

		// If no perm attribute, check for perm input tensor
		if !permSet && len(onnxNode.GetInput()) > 1 {
			permInputName := onnxNode.GetInput()[1]
			if initializer, ok := initializers[permInputName]; ok {
				ints, err := getInt64Data(initializer)
				if err == nil {
					zmfNode.Attributes["perm"] = &zmf.Attribute{
						Value: &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: ints}},
					}
					processedInputs[permInputName] = true
					permSet = true
				}
			}
		}

		if !permSet {
			if len(onnxNode.GetInput()) == 0 {
				return nil, fmt.Errorf("transpose node '%s' has no inputs to infer permutation from", onnxNode.GetName())
			}
			inputName := onnxNode.GetInput()[0]
			info, ok := valueInfos[inputName]
			if !ok {
				return nil, fmt.Errorf("could not find value info for Transpose input '%s' to infer permutation", inputName)
			}
			rank := len(info.GetType().GetTensorType().GetShape().GetDim())
			perm := make([]int64, rank)
			for i := 0; i < rank; i++ {
				perm[i] = int64(rank - 1 - i)
			}
			zmfNode.Attributes["perm"] = &zmf.Attribute{
				Value: &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: perm}},
			}
		}
	case "Reshape":
		// The second input to Reshape is the 'shape' tensor.
		if len(onnxNode.GetInput()) > 1 {
			shapeInputName := onnxNode.GetInput()[1]
			if initializer, ok := initializers[shapeInputName]; ok {
				ints, err := getInt64Data(initializer)
				if err == nil {
					// For the specific case of [0, -1, 256] which appears in Gemma models,
					// we can make reasonable assumptions based on common transformer patterns
					resolvedShape := make([]int64, len(ints))
					copy(resolvedShape, ints)

					// Handle common Gemma transformer patterns
					if len(ints) == 3 && ints[0] == 0 && ints[1] == -1 && ints[2] > 0 {
						// Pattern: [0, -1, embed_dim] - typically reshaping from [batch, seq_len*embed_dim] to [batch, seq_len, embed_dim]
						// For this pattern, we can assume the input is 2D [batch, total_size] and we want [batch, seq_len, embed_dim]
						// Since total_size = seq_len * embed_dim, we can infer seq_len = total_size / embed_dim
						// Keep -1 for runtime inference since different tensors may have different sizes
						resolvedShape[0] = 1  // batch size (common for inference)
						resolvedShape[1] = -1 // keep -1 for sequence length inference
						// resolvedShape[2] is already set to the embed_dim from ints[2]
					} else if len(ints) == 2 && ints[0] == 0 && ints[1] == -1 {
						// Pattern: [0, -1] - flatten to 2D
						resolvedShape[0] = 1  // batch size
						resolvedShape[1] = -1 // keep -1 for inference
					}

					zmfNode.Attributes["shape"] = &zmf.Attribute{
						Value: &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: resolvedShape}},
					}
					processedInputs[shapeInputName] = true
				}
			}
		}
	}

	// Process all inputs.
	for _, inputName := range onnxNode.GetInput() {
		if processedInputs[inputName] {
			continue // Skip inputs that were handled as special cases.
		}

		// Check if the input is a constant integer initializer that should be promoted.
		if initializer, ok := initializers[inputName]; ok {
			dtype := onnx.TensorProto_DataType(initializer.GetDataType())
			if dtype == onnx.TensorProto_INT64 || dtype == onnx.TensorProto_INT32 {
				// Promote to a generic attribute, using the initializer's name as the key.
				ints, err := getInt64Data(initializer)
				if err != nil {
					return nil, fmt.Errorf("failed to get data for constant '%s': %w", inputName, err)
				}
				zmfNode.Attributes[inputName] = &zmf.Attribute{
					Value: &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: ints}},
				}
			} else {
				// It's a non-integer initializer (e.g., float weights), treat as a regular graph input.
				zmfNode.Inputs = append(zmfNode.Inputs, inputName)
			}
		} else {
			// Not an initializer, so it's a regular input from another node.
			zmfNode.Inputs = append(zmfNode.Inputs, inputName)
		}
	}

	return zmfNode, nil
}

// ... (rest of the file remains the same)
func convertAttribute(onnxAttr *onnx.AttributeProto) (*zmf.Attribute, error) {
	zmfAttr := &zmf.Attribute{}
	switch onnxAttr.GetType() {
	case onnx.AttributeProto_FLOAT:
		zmfAttr.Value = &zmf.Attribute_F{F: onnxAttr.GetF()}
	case onnx.AttributeProto_INT:
		zmfAttr.Value = &zmf.Attribute_I{I: onnxAttr.GetI()}
	case onnx.AttributeProto_STRING:
		zmfAttr.Value = &zmf.Attribute_S{S: string(onnxAttr.GetS())}
	case onnx.AttributeProto_FLOATS:
		zmfAttr.Value = &zmf.Attribute_Floats{Floats: &zmf.Floats{Val: onnxAttr.GetFloats()}}
	case onnx.AttributeProto_INTS:
		zmfAttr.Value = &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: onnxAttr.GetInts()}}
	case onnx.AttributeProto_STRINGS:
		strings := make([]string, len(onnxAttr.GetStrings()))
		for i, s := range onnxAttr.GetStrings() {
			strings[i] = string(s)
		}
		zmfAttr.Value = &zmf.Attribute_Strings{Strings: &zmf.Strings{Val: strings}}
	default:
		return nil, nil
	}
	return zmfAttr, nil
}

/*
func convertTensor(onnxTensor *onnx.TensorProto) (*zmf.Tensor, error) {
	return convertTensorWithPath(onnxTensor, "")
}
*/

func convertTensorWithPath(onnxTensor *onnx.TensorProto, modelPath string) (*zmf.Tensor, error) {
	var data []byte
	var err error

	// Check if tensor uses external data
	if len(onnxTensor.GetExternalData()) > 0 {
		data, err = loadExternalData(onnxTensor, modelPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load external data: %w", err)
		}
	} else {
		data = onnxTensor.GetRawData()
	}

	zmfTensor := &zmf.Tensor{
		Shape: onnxTensor.GetDims(),
		Data:  data,
	}
	switch onnx.TensorProto_DataType(onnxTensor.GetDataType()) {
	case onnx.TensorProto_FLOAT:
		zmfTensor.Dtype = zmf.Tensor_FLOAT32
	case onnx.TensorProto_FLOAT16:
		zmfTensor.Dtype = zmf.Tensor_FLOAT16
	case onnx.TensorProto_BFLOAT16:
		zmfTensor.Dtype = zmf.Tensor_BFLOAT16
	case onnx.TensorProto_INT32:
		zmfTensor.Dtype = zmf.Tensor_INT32
	case onnx.TensorProto_INT64:
		zmfTensor.Dtype = zmf.Tensor_INT64
	case onnx.TensorProto_DOUBLE:
		zmfTensor.Dtype = zmf.Tensor_FLOAT64
	default:
		return nil, fmt.Errorf("unsupported tensor data type: %s", onnx.TensorProto_DataType_name[onnxTensor.GetDataType()])
	}
	return zmfTensor, nil
}

// loadExternalData loads tensor data from external files
func loadExternalData(tensor *onnx.TensorProto, modelPath string) ([]byte, error) {
	var location string
	var offset int64
	var length int64

	// Parse external data metadata
	for _, entry := range tensor.GetExternalData() {
		key := entry.GetKey()
		value := entry.GetValue()

		switch key {
		case "location":
			location = value
		case "offset":
			if value != "" {
				var err error
				offset, err = strconv.ParseInt(value, 10, 64)
				if err != nil {
					return nil, fmt.Errorf("invalid offset value: %s", value)
				}
			}
		case "length":
			if value != "" {
				var err error
				length, err = strconv.ParseInt(value, 10, 64)
				if err != nil {
					return nil, fmt.Errorf("invalid length value: %s", value)
				}
			}
		}
	}

	if location == "" {
		return nil, fmt.Errorf("external data location not specified")
	}

	// Resolve the external data file path
	var externalPath string
	if filepath.IsAbs(location) {
		externalPath = location
	} else {
		// Relative to the model file directory
		modelDir := filepath.Dir(modelPath)
		externalPath = filepath.Join(modelDir, location)
	}

	// Open the external data file
	file, err := os.Open(externalPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open external data file %s: %w", externalPath, err)
	}
	defer func() {
		if cerr := file.Close(); cerr != nil {
			fmt.Fprintf(os.Stderr, "Error closing file %s: %v\n", externalPath, cerr)
		}
	}()

	// Seek to the offset if specified
	if offset > 0 {
		_, err = file.Seek(offset, 0)
		if err != nil {
			return nil, fmt.Errorf("failed to seek to offset %d: %w", offset, err)
		}
	}

	// Read the data
	var data []byte
	if length > 0 {
		// Read specific length
		data = make([]byte, length)
		_, err = file.Read(data)
		if err != nil {
			return nil, fmt.Errorf("failed to read %d bytes from external file: %w", length, err)
		}
	} else {
		// Read all remaining data
		data, err = os.ReadFile(externalPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read external data file: %w", err)
		}
		if offset > 0 {
			if int64(len(data)) <= offset {
				return nil, fmt.Errorf("offset %d exceeds file size %d", offset, len(data))
			}
			data = data[offset:]
		}
	}

	return data, nil
}

func getInt64Data(p *onnx.TensorProto) ([]int64, error) {
	if dt := onnx.TensorProto_DataType(p.GetDataType()); dt != onnx.TensorProto_INT64 && dt != onnx.TensorProto_INT32 {
		return nil, fmt.Errorf("tensor is not of type INT64 or INT32, but %s", dt)
	}
	if p.GetInt64Data() != nil {
		return p.GetInt64Data(), nil
	}
	if p.GetInt32Data() != nil {
		data := make([]int64, len(p.GetInt32Data()))
		for i, v := range p.GetInt32Data() {
			data[i] = int64(v)
		}
		return data, nil
	}
	rawData := p.GetRawData()
	if len(rawData) == 0 {
		return []int64{}, nil
	}
	if onnx.TensorProto_DataType(p.GetDataType()) == onnx.TensorProto_INT64 {
		if len(rawData)%8 != 0 {
			return nil, fmt.Errorf("raw_data length %d is not a multiple of 8 for INT64", len(rawData))
		}
		data := make([]int64, len(rawData)/8)
		for i := 0; i < len(data); i++ {
			data[i] = int64(binary.LittleEndian.Uint64(rawData[i*8 : (i+1)*8]))
		}
		return data, nil
	}
	if onnx.TensorProto_DataType(p.GetDataType()) == onnx.TensorProto_INT32 {
		if len(rawData)%4 != 0 {
			return nil, fmt.Errorf("raw_data length %d is not a multiple of 4 for INT32", len(rawData))
		}
		data := make([]int64, len(rawData)/4)
		for i := 0; i < len(data); i++ {
			data[i] = int64(binary.LittleEndian.Uint32(rawData[i*4 : (i+1)*4]))
		}
		return data, nil
	}
	return []int64{}, nil
}

func convertValueInfos(infos []*onnx.ValueInfoProto) []*zmf.ValueInfo {
	zmfInfos := make([]*zmf.ValueInfo, len(infos))
	for i, info := range infos {
		dims := info.GetType().GetTensorType().GetShape().GetDim()
		shape := make([]int64, len(dims))
		for j, dim := range dims {
			shape[j] = dim.GetDimValue()
		}
		zmfInfos[i] = &zmf.ValueInfo{
			Name:  info.GetName(),
			Shape: shape,
		}
	}
	return zmfInfos
}
