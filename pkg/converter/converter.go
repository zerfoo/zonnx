package converter

import (
	"encoding/binary"
	"fmt"
	"math"
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
		switch onnxNode.GetOpType() {
		case "Constant":
			// Constant nodes embed their value as a tensor attribute.
			// Store the value as a ZMF parameter keyed by each output name and skip
			// adding a node to the graph so downstream nodes see it as a regular input.
			zmfTensor, err := extractConstantTensor(onnxNode, modelPath)
			if err != nil {
				return nil, fmt.Errorf("failed to extract Constant node '%s': %w", onnxNode.GetName(), err)
			}
			for _, outName := range onnxNode.GetOutput() {
				if outName != "" {
					zmfModel.Graph.Parameters[outName] = zmfTensor
				}
			}
			// Also register under the node name itself.
			if onnxNode.GetName() != "" {
				zmfModel.Graph.Parameters[onnxNode.GetName()] = zmfTensor
			}

		case "MatMulNBits":
			// Dequantize 4-bit quantized weights to float32 at import time and emit a
			// standard MatMul node.  This avoids needing a specialised runtime kernel.
			zmfNode, err := convertMatMulNBits(onnxNode, initializers, zmfModel.Graph.Parameters, modelPath)
			if err != nil {
				return nil, fmt.Errorf("failed to convert MatMulNBits node '%s': %w", onnxNode.GetName(), err)
			}
			zmfModel.Graph.Nodes = append(zmfModel.Graph.Nodes, zmfNode)

		default:
			zmfNode, err := convertNode(onnxNode, initializers, valueInfos)
			if err != nil {
				return nil, fmt.Errorf("failed to convert node '%s': %w", onnxNode.GetName(), err)
			}
			zmfModel.Graph.Nodes = append(zmfModel.Graph.Nodes, zmfNode)
		}
	}

	for name, onnxTensor := range initializers {
		dtype := onnx.TensorProto_DataType(onnxTensor.GetDataType())
		switch dtype {
		case onnx.TensorProto_FLOAT, onnx.TensorProto_FLOAT16, onnx.TensorProto_BFLOAT16, onnx.TensorProto_DOUBLE,
			onnx.TensorProto_UINT8, onnx.TensorProto_INT8, onnx.TensorProto_INT32, onnx.TensorProto_INT64:
			zmfTensor, err := convertTensorWithPath(onnxTensor, modelPath)
			if err != nil {
				return nil, fmt.Errorf("failed to convert initializer '%s': %w", name, err)
			}
			zmfModel.Graph.Parameters[name] = zmfTensor
		}
	}

	return zmfModel, nil
}

// extractConstantTensor extracts the tensor value from an ONNX Constant op node.
func extractConstantTensor(node *onnx.NodeProto, modelPath string) (*zmf.Tensor, error) {
	for _, attr := range node.GetAttribute() {
		if attr.GetName() == "value" && attr.GetType() == onnx.AttributeProto_TENSOR {
			return convertTensorWithPath(attr.GetT(), modelPath)
		}
	}
	return nil, fmt.Errorf("constant node '%s' has no 'value' TENSOR attribute", node.GetName())
}

// convertMatMulNBits dequantizes a MatMulNBits node's weights to float32 and returns a
// regular MatMul ZMF node whose weight input is the dequantized parameter.
//
// ONNX MatMulNBits computes:  Y = A @ dequantize(B).T
//   - A: float [batch, M, K]
//   - B: uint8 [N, ceil(K/block_size), block_size*bits/8]
//   - scales: float32 [N * ceil(K/block_size)]  (one scale per block)
//   - zero_points (optional): uint8, 4-bit packed
//
// We dequantize B to a float32 [K, N] matrix so that a standard
// MatMul(A, B_dequant) produces the correct [batch, M, N] output.
func convertMatMulNBits(
	node *onnx.NodeProto,
	initializers map[string]*onnx.TensorProto,
	params map[string]*zmf.Tensor,
	_ string, // modelPath reserved for future external-data support
) (*zmf.Node, error) {
	// Read operator attributes.
	var K, N, bits, blockSize int
	for _, attr := range node.GetAttribute() {
		switch attr.GetName() {
		case "K":
			K = int(attr.GetI())
		case "N":
			N = int(attr.GetI())
		case "bits":
			bits = int(attr.GetI())
		case "block_size":
			blockSize = int(attr.GetI())
		}
	}
	if K == 0 || N == 0 || bits == 0 || blockSize == 0 {
		return nil, fmt.Errorf("MatMulNBits node '%s' missing required attributes (K=%d N=%d bits=%d block_size=%d)",
			node.GetName(), K, N, bits, blockSize)
	}

	inputs := node.GetInput()
	if len(inputs) < 3 {
		return nil, fmt.Errorf("MatMulNBits node '%s' requires at least 3 inputs, got %d", node.GetName(), len(inputs))
	}
	activationName := inputs[0]
	weightName := inputs[1]
	scaleName := inputs[2]

	weightTensor, ok := initializers[weightName]
	if !ok {
		return nil, fmt.Errorf("MatMulNBits weight initializer '%s' not found", weightName)
	}
	scaleTensor, ok := initializers[scaleName]
	if !ok {
		return nil, fmt.Errorf("MatMulNBits scale initializer '%s' not found", scaleName)
	}

	var zpTensor *onnx.TensorProto
	if len(inputs) >= 4 && inputs[3] != "" {
		zpTensor = initializers[inputs[3]]
	}

	dequantTensor, err := dequantizeNBits(weightTensor, scaleTensor, zpTensor, N, K, bits, blockSize)
	if err != nil {
		return nil, fmt.Errorf("failed to dequantize MatMulNBits '%s': %w", node.GetName(), err)
	}

	dequantName := weightName + "_dequant"
	params[dequantName] = dequantTensor

	// Build a standard MatMul ZMF node.
	zmfNode := &zmf.Node{
		Name:       node.GetName(),
		OpType:     "MatMul",
		Outputs:    node.GetOutput(),
		Inputs:     []string{activationName, dequantName},
		Attributes: make(map[string]*zmf.Attribute),
	}
	return zmfNode, nil
}

// dequantizeNBits dequantizes packed N-bit quantized weights to float32 [K, N].
// The result is stored transposed ([K, N]) so it can be used directly in A @ B.
func dequantizeNBits(
	weightTensor *onnx.TensorProto,
	scaleTensor *onnx.TensorProto,
	zpTensor *onnx.TensorProto,
	N, K, bits, blockSize int,
) (*zmf.Tensor, error) {
	if bits != 4 {
		return nil, fmt.Errorf("only 4-bit quantization is currently supported, got %d bits", bits)
	}

	kBlocks := (K + blockSize - 1) / blockSize
	bytesPerBlock := blockSize * bits / 8 // = blockSize/2 for 4-bit

	weightData := weightTensor.GetRawData()
	if len(weightData) != N*kBlocks*bytesPerBlock {
		return nil, fmt.Errorf("weight data length mismatch: expected %d, got %d",
			N*kBlocks*bytesPerBlock, len(weightData))
	}

	scaleData := scaleTensor.GetRawData()
	if len(scaleData) != N*kBlocks*4 {
		// scales may also be stored as FloatData
		if len(scaleTensor.GetFloatData()) == N*kBlocks {
			scaleData = make([]byte, N*kBlocks*4)
			for i, v := range scaleTensor.GetFloatData() {
				binary.LittleEndian.PutUint32(scaleData[i*4:], math.Float32bits(v))
			}
		} else {
			return nil, fmt.Errorf("scale data length mismatch: expected %d bytes, got %d",
				N*kBlocks*4, len(scaleData))
		}
	}

	// Dequantize: output [N, K] then transpose to [K, N].
	dequant := make([]float32, N*K)
	for n := 0; n < N; n++ {
		for kblk := 0; kblk < kBlocks; kblk++ {
			scaleOffset := (n*kBlocks + kblk) * 4
			scale := math.Float32frombits(binary.LittleEndian.Uint32(scaleData[scaleOffset : scaleOffset+4]))

			// Default zero point for symmetric 4-bit is 8.
			var zp float32 = 8.0
			if zpTensor != nil {
				zpData := zpTensor.GetRawData()
				zpIdx := n*kBlocks + kblk
				if len(zpData) > zpIdx/2 {
					if zpIdx%2 == 0 {
						zp = float32(zpData[zpIdx/2] & 0xF)
					} else {
						zp = float32((zpData[zpIdx/2] >> 4) & 0xF)
					}
				}
			}

			for byteIdx := 0; byteIdx < bytesPerBlock; byteIdx++ {
				wByte := weightData[n*kBlocks*bytesPerBlock+kblk*bytesPerBlock+byteIdx]
				kLo := kblk*blockSize + byteIdx*2
				kHi := kLo + 1
				if kLo < K {
					dequant[n*K+kLo] = scale * (float32(wByte&0xF) - zp)
				}
				if kHi < K {
					dequant[n*K+kHi] = scale * (float32((wByte>>4)&0xF) - zp)
				}
			}
		}
	}

	// Transpose from [N, K] to [K, N] so MatMul(A[..., K], W[K, N]) = [..., N].
	transposed := make([]float32, K*N)
	for n := 0; n < N; n++ {
		for k := 0; k < K; k++ {
			transposed[k*N+n] = dequant[n*K+k]
		}
	}

	rawData := make([]byte, K*N*4)
	for i, v := range transposed {
		binary.LittleEndian.PutUint32(rawData[i*4:], math.Float32bits(v))
	}

	return &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT32,
		Shape: []int64{int64(K), int64(N)},
		Data:  rawData,
	}, nil
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
	case "GroupQueryAttention":
		// Infer model_dim from the input tensor shape.
		if len(onnxNode.GetInput()) > 0 {
			inputName := onnxNode.GetInput()[0]
			if info, ok := valueInfos[inputName]; ok {
				shape := info.GetType().GetTensorType().GetShape().GetDim()
				if len(shape) > 0 {
					modelDim := shape[len(shape)-1].GetDimValue()
					zmfNode.Attributes["model_dim"] = &zmf.Attribute{
						Value: &zmf.Attribute_I{I: modelDim},
					}
				}
			}
		}
	case "Slice":
		// ONNX Slice (opset 10+): inputs are [data, starts, ends, axes (opt), steps (opt)].
		// Promote starts/ends/axes/steps from initializer tensors to named ZMF attributes
		// so downstream zerfoo Slice layer receives them as attribute values.
		sliceAttrNames := []string{"starts", "ends", "axes", "steps"}
		sliceInputs := onnxNode.GetInput()
		for i, attrName := range sliceAttrNames {
			idx := i + 1 // input[0] is data; starts=1, ends=2, axes=3, steps=4
			if idx >= len(sliceInputs) || sliceInputs[idx] == "" {
				continue
			}
			inputName := sliceInputs[idx]
			if init, ok := initializers[inputName]; ok {
				ints, err := getInt64Data(init)
				if err == nil {
					zmfNode.Attributes[attrName] = &zmf.Attribute{
						Value: &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: ints}},
					}
					processedInputs[inputName] = true
				}
			}
		}

	case "Pad":
		// ONNX Pad (opset 11+): inputs are [data, pads, constant_value (opt)].
		// Promote pads (INT64 tensor) and constant_value (float32 scalar) to ZMF attributes.
		padInputs := onnxNode.GetInput()
		if len(padInputs) > 1 && padInputs[1] != "" {
			if init, ok := initializers[padInputs[1]]; ok {
				if ints, err := getInt64Data(init); err == nil {
					zmfNode.Attributes["pads"] = &zmf.Attribute{
						Value: &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: ints}},
					}
					processedInputs[padInputs[1]] = true
				}
			}
		}
		if len(padInputs) > 2 && padInputs[2] != "" {
			if init, ok := initializers[padInputs[2]]; ok {
				if onnx.TensorProto_DataType(init.GetDataType()) == onnx.TensorProto_FLOAT {
					var f float32
					var ok2 bool
					if rawData := init.GetRawData(); len(rawData) == 4 {
						f = math.Float32frombits(binary.LittleEndian.Uint32(rawData))
						ok2 = true
					} else if fd := init.GetFloatData(); len(fd) == 1 {
						f = fd[0]
						ok2 = true
					}
					if ok2 {
						zmfNode.Attributes["constant_value"] = &zmf.Attribute{
							Value: &zmf.Attribute_F{F: f},
						}
						processedInputs[padInputs[2]] = true
					}
				}
			}
		}

	case "TopK":
		// ONNX TopK: inputs are [X, K] where K is a scalar INT64 initializer tensor.
		// Promote K to a ZMF "k" attribute so the zerfoo TopK layer receives it.
		topkInputs := onnxNode.GetInput()
		if len(topkInputs) > 1 && topkInputs[1] != "" {
			if init, ok := initializers[topkInputs[1]]; ok {
				if ints, err := getInt64Data(init); err == nil && len(ints) == 1 {
					zmfNode.Attributes["k"] = &zmf.Attribute{
						Value: &zmf.Attribute_I{I: ints[0]},
					}
					processedInputs[topkInputs[1]] = true
				}
			}
		}

	case "Resize":
		// ONNX Resize: inputs are [X, roi (opt), scales (opt), sizes (opt)].
		// Promote scales (float32 tensor, input[2]) to a "scales" FLOATS attribute and
		// sizes (INT64 tensor, input[3]) to a "sizes" INTS attribute so the zerfoo
		// Resize layer can read them at build time.
		resizeInputs := onnxNode.GetInput()
		if len(resizeInputs) > 1 && resizeInputs[1] != "" {
			// roi input - mark as processed (not needed for inference)
			processedInputs[resizeInputs[1]] = true
		}
		if len(resizeInputs) > 2 && resizeInputs[2] != "" {
			scaleName := resizeInputs[2]
			if init, ok := initializers[scaleName]; ok {
				if onnx.TensorProto_DataType(init.GetDataType()) == onnx.TensorProto_FLOAT {
					var floats []float32
					if rawData := init.GetRawData(); len(rawData) > 0 && len(rawData)%4 == 0 {
						floats = make([]float32, len(rawData)/4)
						for i := range floats {
							floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(rawData[i*4:]))
						}
					} else {
						floats = init.GetFloatData()
					}
					if len(floats) > 0 {
						zmfNode.Attributes["scales"] = &zmf.Attribute{
							Value: &zmf.Attribute_Floats{Floats: &zmf.Floats{Val: floats}},
						}
					}
				}
				processedInputs[scaleName] = true
			}
		}
		if len(resizeInputs) > 3 && resizeInputs[3] != "" {
			sizeName := resizeInputs[3]
			if init, ok := initializers[sizeName]; ok {
				if ints, err := getInt64Data(init); err == nil {
					zmfNode.Attributes["sizes"] = &zmf.Attribute{
						Value: &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: ints}},
					}
				}
				processedInputs[sizeName] = true
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
		if inputName == "" {
			continue // Empty string means optional input not provided in ONNX.
		}
		if processedInputs[inputName] {
			continue // Skip inputs that were handled as special cases.
		}

		// Keep the input as a regular graph reference. The initializer (if
		// any) is already converted to a ZMF parameter and will be resolved
		// as a parameterNode during graph construction.
		zmfNode.Inputs = append(zmfNode.Inputs, inputName)
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
	case onnx.AttributeProto_TENSOR:
		// Convert the embedded ONNX tensor to ZMF format.
		// Constant node attributes use this to carry their value tensor.
		zmfTensor, err := convertTensorWithPath(onnxAttr.GetT(), "")
		if err != nil {
			return nil, fmt.Errorf("failed to convert TENSOR attribute: %w", err)
		}
		zmfAttr.Value = &zmf.Attribute_Tensor{Tensor: zmfTensor}
	default:
		return nil, fmt.Errorf("unsupported attribute type: %v", onnxAttr.GetType())
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
	case onnx.TensorProto_UINT8:
		zmfTensor.Dtype = zmf.Tensor_UINT8
	case onnx.TensorProto_INT8:
		zmfTensor.Dtype = zmf.Tensor_INT8
	case onnx.TensorProto_BOOL:
		zmfTensor.Dtype = zmf.Tensor_BOOL
	default:
		return nil, fmt.Errorf("unsupported tensor data type: %s", onnx.TensorProto_DataType_name[onnxTensor.GetDataType()])
	}
	return zmfTensor, nil
}

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
