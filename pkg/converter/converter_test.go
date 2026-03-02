package converter

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/zmf"
	"github.com/zerfoo/zonnx/internal/onnx"
)

// --- convertAttribute tests ---

func TestConvertAttribute_Float(t *testing.T) {
	attr := &onnx.AttributeProto{
		Type: onnx.AttributeProto_FLOAT.Enum(),
		F:    proto32(3.14),
	}
	got, err := convertAttribute(attr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	v, ok := got.Value.(*zmf.Attribute_F)
	if !ok {
		t.Fatalf("expected Attribute_F, got %T", got.Value)
	}
	if v.F != 3.14 {
		t.Errorf("expected 3.14, got %v", v.F)
	}
}

func TestConvertAttribute_Int(t *testing.T) {
	attr := &onnx.AttributeProto{
		Type: onnx.AttributeProto_INT.Enum(),
		I:    proto64(42),
	}
	got, err := convertAttribute(attr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	v, ok := got.Value.(*zmf.Attribute_I)
	if !ok {
		t.Fatalf("expected Attribute_I, got %T", got.Value)
	}
	if v.I != 42 {
		t.Errorf("expected 42, got %v", v.I)
	}
}

func TestConvertAttribute_String(t *testing.T) {
	attr := &onnx.AttributeProto{
		Type: onnx.AttributeProto_STRING.Enum(),
		S:    []byte("hello"),
	}
	got, err := convertAttribute(attr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	v, ok := got.Value.(*zmf.Attribute_S)
	if !ok {
		t.Fatalf("expected Attribute_S, got %T", got.Value)
	}
	if v.S != "hello" {
		t.Errorf("expected 'hello', got %v", v.S)
	}
}

func TestConvertAttribute_Tensor(t *testing.T) {
	rawData := make([]byte, 4)
	binary.LittleEndian.PutUint32(rawData, math.Float32bits(1.0))

	tensorProto := &onnx.TensorProto{
		DataType: protoInt32(int32(onnx.TensorProto_FLOAT)),
		Dims:     []int64{1},
		RawData:  rawData,
	}

	attr := &onnx.AttributeProto{
		Type: onnx.AttributeProto_TENSOR.Enum(),
		T:    tensorProto,
	}

	got, err := convertAttribute(attr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	v, ok := got.Value.(*zmf.Attribute_Tensor)
	if !ok {
		t.Fatalf("expected Attribute_Tensor, got %T", got.Value)
	}
	if v.Tensor.Dtype != zmf.Tensor_FLOAT32 {
		t.Errorf("expected FLOAT32, got %v", v.Tensor.Dtype)
	}
}

func TestConvertAttribute_Tensor_UINT8(t *testing.T) {
	tensorProto := &onnx.TensorProto{
		DataType: protoInt32(int32(onnx.TensorProto_UINT8)),
		Dims:     []int64{3},
		RawData:  []byte{0, 127, 255},
	}

	attr := &onnx.AttributeProto{
		Type: onnx.AttributeProto_TENSOR.Enum(),
		T:    tensorProto,
	}

	got, err := convertAttribute(attr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	v, ok := got.Value.(*zmf.Attribute_Tensor)
	if !ok {
		t.Fatalf("expected Attribute_Tensor, got %T", got.Value)
	}
	if v.Tensor.Dtype != zmf.Tensor_UINT8 {
		t.Errorf("expected UINT8, got %v", v.Tensor.Dtype)
	}
}

// --- convertTensorWithPath UINT8/INT8 tests ---

func TestConvertTensorWithPath_UINT8(t *testing.T) {
	rawData := []byte{0, 128, 255}
	onnxTensor := &onnx.TensorProto{
		DataType: protoInt32(int32(onnx.TensorProto_UINT8)),
		Dims:     []int64{3},
		RawData:  rawData,
	}

	got, err := convertTensorWithPath(onnxTensor, "")
	if err != nil {
		t.Fatalf("convertTensorWithPath failed: %v", err)
	}
	if got.Dtype != zmf.Tensor_UINT8 {
		t.Errorf("expected UINT8, got %v", got.Dtype)
	}
	if len(got.Data) != 3 {
		t.Errorf("expected 3 bytes, got %d", len(got.Data))
	}
}

func TestConvertTensorWithPath_INT8(t *testing.T) {
	rawData := []byte{0x80, 0x00, 0x7F} // -128, 0, 127
	onnxTensor := &onnx.TensorProto{
		DataType: protoInt32(int32(onnx.TensorProto_INT8)),
		Dims:     []int64{3},
		RawData:  rawData,
	}

	got, err := convertTensorWithPath(onnxTensor, "")
	if err != nil {
		t.Fatalf("convertTensorWithPath failed: %v", err)
	}
	if got.Dtype != zmf.Tensor_INT8 {
		t.Errorf("expected INT8, got %v", got.Dtype)
	}
}

// --- Constant node handling in ONNXToZMF ---

func TestONNXToZMF_ConstantNode_StoredAsParameter(t *testing.T) {
	rawData := make([]byte, 4)
	binary.LittleEndian.PutUint32(rawData, math.Float32bits(42.0))

	tensorProto := &onnx.TensorProto{
		DataType: protoInt32(int32(onnx.TensorProto_FLOAT)),
		Dims:     []int64{1},
		RawData:  rawData,
	}

	attrName := "value"
	attrType := onnx.AttributeProto_TENSOR
	opType := "Constant"
	constName := "const_42"
	constOut := "const_42_out"

	model := &onnx.ModelProto{
		Graph: &onnx.GraphProto{
			Node: []*onnx.NodeProto{
				{
					Name:   &constName,
					OpType: &opType,
					Output: []string{constOut},
					Attribute: []*onnx.AttributeProto{
						{
							Name: &attrName,
							Type: &attrType,
							T:    tensorProto,
						},
					},
				},
			},
			Output: []*onnx.ValueInfoProto{
				valueInfo(constOut, onnx.TensorProto_FLOAT, []int64{1}),
			},
		},
	}

	zmfModel, err := ONNXToZMF(model)
	if err != nil {
		t.Fatalf("ONNXToZMF failed: %v", err)
	}

	// Constant node should NOT appear in the nodes list
	for _, node := range zmfModel.Graph.Nodes {
		if node.OpType == "Constant" {
			t.Error("Constant node should not appear in ZMF graph nodes")
		}
	}

	// The constant value should be stored as a parameter
	if zmfModel.Graph.Parameters[constOut] == nil {
		t.Errorf("expected parameter '%s' to be stored, got nil", constOut)
	}
}

// --- MatMulNBits dequantization test ---

func TestONNXToZMF_MatMulNBits_DequantizedToMatMul(t *testing.T) {
	// Simple 4-bit quantized weight: N=2, K=4, block_size=4, bits=4
	// 2 blocks total (1 block per row, K=4=block_size)
	// weights[N=2, K_blocks=1, bytes_per_block=2]: 2 rows, 1 block each, 2 bytes per block
	N, K, blockSize, bits := 2, 4, 4, 4
	K_blocks := 1 // K/block_size = 4/4 = 1
	bytesPerBlock := blockSize * bits / 8 // = 2

	// Pack 4-bit values: row0=[1,2,3,4], row1=[5,6,7,8]
	// byte0: lo=1, hi=2 -> 0x21; byte1: lo=3, hi=4 -> 0x43
	// row1: byte0: lo=5, hi=6 -> 0x65; byte1: lo=7, hi=8 -> 0x87
	weightBytes := make([]byte, N*K_blocks*bytesPerBlock)
	weightBytes[0] = 0x21 // row0, block0, byte0
	weightBytes[1] = 0x43 // row0, block0, byte1
	weightBytes[2] = 0x65 // row1, block0, byte0
	weightBytes[3] = 0x87 // row1, block0, byte1

	// Scales: one per block per row, [N*K_blocks] float32
	scaleBytes := make([]byte, N*K_blocks*4)
	binary.LittleEndian.PutUint32(scaleBytes[0:4], math.Float32bits(1.0)) // row0
	binary.LittleEndian.PutUint32(scaleBytes[4:8], math.Float32bits(1.0)) // row1

	weightName := "weight_q"
	scaleName := "scale"
	activationName := "activation"
	nodeName := "matmul_nbits"
	opType := "MatMulNBits"

	model := &onnx.ModelProto{
		Graph: &onnx.GraphProto{
			Initializer: []*onnx.TensorProto{
				{
					Name:     &weightName,
					DataType: protoInt32(int32(onnx.TensorProto_UINT8)),
					Dims:     []int64{int64(N), int64(K_blocks), int64(bytesPerBlock)},
					RawData:  weightBytes,
				},
				{
					Name:     &scaleName,
					DataType: protoInt32(int32(onnx.TensorProto_FLOAT)),
					Dims:     []int64{int64(N * K_blocks)},
					RawData:  scaleBytes,
				},
			},
			Node: []*onnx.NodeProto{
				{
					Name:   &nodeName,
					OpType: &opType,
					Input:  []string{activationName, weightName, scaleName},
					Output: []string{"matmul_out"},
					Attribute: []*onnx.AttributeProto{
						intAttr("K", int64(K)),
						intAttr("N", int64(N)),
						intAttr("bits", int64(bits)),
						intAttr("block_size", int64(blockSize)),
					},
				},
			},
			Input: []*onnx.ValueInfoProto{
				valueInfo(activationName, onnx.TensorProto_FLOAT, []int64{1, int64(K)}),
			},
			Output: []*onnx.ValueInfoProto{
				valueInfo("matmul_out", onnx.TensorProto_FLOAT, []int64{1, int64(N)}),
			},
		},
	}

	zmfModel, err := ONNXToZMF(model)
	if err != nil {
		t.Fatalf("ONNXToZMF failed: %v", err)
	}

	// MatMulNBits node should be converted to MatMul
	foundMatMul := false
	for _, node := range zmfModel.Graph.Nodes {
		if node.OpType == "MatMulNBits" {
			t.Error("MatMulNBits node should not appear in ZMF graph (should be converted)")
		}
		if node.OpType == "MatMul" {
			foundMatMul = true
		}
	}
	if !foundMatMul {
		t.Error("expected a MatMul node in ZMF graph after dequantization")
	}

	// Dequantized weight parameter should exist
	dequantKey := weightName + "_dequant"
	if zmfModel.Graph.Parameters[dequantKey] == nil {
		t.Errorf("expected dequantized parameter '%s'", dequantKey)
	}
}

// --- E38 operator conversion tests ---

func TestONNXToZMF_Softmax_AxisAttribute(t *testing.T) {
	opType := "Softmax"
	nodeName := "softmax_1"
	model := &onnx.ModelProto{
		Graph: &onnx.GraphProto{
			Node: []*onnx.NodeProto{
				{
					Name:      &nodeName,
					OpType:    &opType,
					Input:     []string{"data"},
					Output:    []string{"output"},
					Attribute: []*onnx.AttributeProto{intAttr("axis", -1)},
				},
			},
			Input:  []*onnx.ValueInfoProto{valueInfo("data", onnx.TensorProto_FLOAT, []int64{1, 4})},
			Output: []*onnx.ValueInfoProto{valueInfo("output", onnx.TensorProto_FLOAT, []int64{1, 4})},
		},
	}
	zmfModel, err := ONNXToZMF(model)
	if err != nil {
		t.Fatalf("ONNXToZMF failed: %v", err)
	}
	if len(zmfModel.Graph.Nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(zmfModel.Graph.Nodes))
	}
	node := zmfModel.Graph.Nodes[0]
	if node.OpType != "Softmax" {
		t.Errorf("expected Softmax, got %s", node.OpType)
	}
	attr, ok := node.Attributes["axis"]
	if !ok {
		t.Fatal("expected 'axis' attribute on Softmax node")
	}
	if v, ok2 := attr.Value.(*zmf.Attribute_I); !ok2 || v.I != -1 {
		t.Errorf("expected axis=-1, got %v", attr.Value)
	}
}

func TestONNXToZMF_Sigmoid_NoAttributes(t *testing.T) {
	opType := "Sigmoid"
	nodeName := "sigmoid_1"
	model := &onnx.ModelProto{
		Graph: &onnx.GraphProto{
			Node: []*onnx.NodeProto{
				{Name: &nodeName, OpType: &opType, Input: []string{"data"}, Output: []string{"output"}},
			},
			Input:  []*onnx.ValueInfoProto{valueInfo("data", onnx.TensorProto_FLOAT, []int64{4})},
			Output: []*onnx.ValueInfoProto{valueInfo("output", onnx.TensorProto_FLOAT, []int64{4})},
		},
	}
	zmfModel, err := ONNXToZMF(model)
	if err != nil {
		t.Fatalf("ONNXToZMF failed: %v", err)
	}
	if len(zmfModel.Graph.Nodes) != 1 || zmfModel.Graph.Nodes[0].OpType != "Sigmoid" {
		t.Errorf("expected one Sigmoid node, got %v", zmfModel.Graph.Nodes)
	}
}

func TestONNXToZMF_Erf_NoAttributes(t *testing.T) {
	opType := "Erf"
	nodeName := "erf_1"
	model := &onnx.ModelProto{
		Graph: &onnx.GraphProto{
			Node: []*onnx.NodeProto{
				{Name: &nodeName, OpType: &opType, Input: []string{"data"}, Output: []string{"output"}},
			},
			Input:  []*onnx.ValueInfoProto{valueInfo("data", onnx.TensorProto_FLOAT, []int64{3})},
			Output: []*onnx.ValueInfoProto{valueInfo("output", onnx.TensorProto_FLOAT, []int64{3})},
		},
	}
	zmfModel, err := ONNXToZMF(model)
	if err != nil {
		t.Fatalf("ONNXToZMF failed: %v", err)
	}
	if len(zmfModel.Graph.Nodes) != 1 || zmfModel.Graph.Nodes[0].OpType != "Erf" {
		t.Errorf("expected one Erf node, got %v", zmfModel.Graph.Nodes)
	}
}

func TestONNXToZMF_LayerNormalization_EpsilonAttribute(t *testing.T) {
	opType := "LayerNormalization"
	nodeName := "ln_1"
	attrAxisName := "axis"
	attrEpsilonName := "epsilon"
	axisVal := int64(-1)
	axisAttrType := onnx.AttributeProto_INT
	epsilonAttrType := onnx.AttributeProto_FLOAT
	epsilonVal := float32(1e-5)
	scaleName := "scale"
	model := &onnx.ModelProto{
		Graph: &onnx.GraphProto{
			Initializer: []*onnx.TensorProto{
				{
					Name:     &scaleName,
					DataType: protoInt32(int32(onnx.TensorProto_FLOAT)),
					Dims:     []int64{4},
					RawData:  make([]byte, 16),
				},
			},
			Node: []*onnx.NodeProto{
				{
					Name:   &nodeName,
					OpType: &opType,
					Input:  []string{"data", scaleName},
					Output: []string{"output"},
					Attribute: []*onnx.AttributeProto{
						{Name: &attrAxisName, Type: &axisAttrType, I: &axisVal},
						{Name: &attrEpsilonName, Type: &epsilonAttrType, F: &epsilonVal},
					},
				},
			},
			Input:  []*onnx.ValueInfoProto{valueInfo("data", onnx.TensorProto_FLOAT, []int64{1, 4})},
			Output: []*onnx.ValueInfoProto{valueInfo("output", onnx.TensorProto_FLOAT, []int64{1, 4})},
		},
	}
	zmfModel, err := ONNXToZMF(model)
	if err != nil {
		t.Fatalf("ONNXToZMF failed: %v", err)
	}
	if len(zmfModel.Graph.Nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(zmfModel.Graph.Nodes))
	}
	node := zmfModel.Graph.Nodes[0]
	if node.OpType != "LayerNormalization" {
		t.Errorf("expected LayerNormalization, got %s", node.OpType)
	}
	epAttr, ok := node.Attributes["epsilon"]
	if !ok {
		t.Fatal("expected 'epsilon' attribute on LayerNormalization node")
	}
	if v, ok2 := epAttr.Value.(*zmf.Attribute_F); !ok2 || v.F != epsilonVal {
		t.Errorf("expected epsilon=%v, got %v", epsilonVal, epAttr.Value)
	}
}

// int64Initializer builds an INT64 1-D TensorProto from a []int64 slice.
func int64Initializer(name string, vals []int64) *onnx.TensorProto {
	rawData := make([]byte, len(vals)*8)
	for i, v := range vals {
		binary.LittleEndian.PutUint64(rawData[i*8:], uint64(v))
	}
	n := name
	return &onnx.TensorProto{
		Name:     &n,
		DataType: protoInt32(int32(onnx.TensorProto_INT64)),
		Dims:     []int64{int64(len(vals))},
		RawData:  rawData,
	}
}

// float32Scalar builds a scalar FLOAT32 TensorProto.
func float32Scalar(name string, val float32) *onnx.TensorProto {
	rawData := make([]byte, 4)
	binary.LittleEndian.PutUint32(rawData, math.Float32bits(val))
	n := name
	return &onnx.TensorProto{
		Name:     &n,
		DataType: protoInt32(int32(onnx.TensorProto_FLOAT)),
		RawData:  rawData,
	}
}

func TestONNXToZMF_Slice_PromotesInputsToAttributes(t *testing.T) {
	opType := "Slice"
	nodeName := "slice_1"
	model := &onnx.ModelProto{
		Graph: &onnx.GraphProto{
			Initializer: []*onnx.TensorProto{
				int64Initializer("starts", []int64{0, 1}),
				int64Initializer("ends", []int64{2, 3}),
				int64Initializer("axes", []int64{0, 1}),
				int64Initializer("steps", []int64{1, 1}),
			},
			Node: []*onnx.NodeProto{
				{
					Name:   &nodeName,
					OpType: &opType,
					Input:  []string{"data", "starts", "ends", "axes", "steps"},
					Output: []string{"sliced"},
				},
			},
			Input:  []*onnx.ValueInfoProto{valueInfo("data", onnx.TensorProto_FLOAT, []int64{3, 4})},
			Output: []*onnx.ValueInfoProto{valueInfo("sliced", onnx.TensorProto_FLOAT, []int64{2, 2})},
		},
	}
	zmfModel, err := ONNXToZMF(model)
	if err != nil {
		t.Fatalf("ONNXToZMF failed: %v", err)
	}
	if len(zmfModel.Graph.Nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(zmfModel.Graph.Nodes))
	}
	node := zmfModel.Graph.Nodes[0]
	if node.OpType != "Slice" {
		t.Errorf("expected Slice, got %s", node.OpType)
	}
	for _, key := range []string{"starts", "ends", "axes", "steps"} {
		attr, ok := node.Attributes[key]
		if !ok {
			t.Errorf("expected attribute %q promoted from Slice input", key)
			continue
		}
		if _, ok2 := attr.Value.(*zmf.Attribute_Ints); !ok2 {
			t.Errorf("expected Ints attribute for %q, got %T", key, attr.Value)
		}
	}
	// data must remain as the sole graph input.
	if len(node.Inputs) != 1 || node.Inputs[0] != "data" {
		t.Errorf("expected inputs=[data], got %v", node.Inputs)
	}
}

func TestONNXToZMF_Slice_WithoutAxesSteps(t *testing.T) {
	opType := "Slice"
	nodeName := "slice_2"
	model := &onnx.ModelProto{
		Graph: &onnx.GraphProto{
			Initializer: []*onnx.TensorProto{
				int64Initializer("starts2", []int64{1}),
				int64Initializer("ends2", []int64{3}),
			},
			Node: []*onnx.NodeProto{
				{
					Name:   &nodeName,
					OpType: &opType,
					Input:  []string{"data", "starts2", "ends2"},
					Output: []string{"sliced"},
				},
			},
			Input:  []*onnx.ValueInfoProto{valueInfo("data", onnx.TensorProto_FLOAT, []int64{5})},
			Output: []*onnx.ValueInfoProto{valueInfo("sliced", onnx.TensorProto_FLOAT, []int64{2})},
		},
	}
	zmfModel, err := ONNXToZMF(model)
	if err != nil {
		t.Fatalf("ONNXToZMF failed: %v", err)
	}
	node := zmfModel.Graph.Nodes[0]
	if _, ok := node.Attributes["starts"]; !ok {
		t.Error("expected 'starts' attribute")
	}
	if _, ok := node.Attributes["ends"]; !ok {
		t.Error("expected 'ends' attribute")
	}
	// axes and steps not provided: they must not appear as attributes.
	if _, ok := node.Attributes["axes"]; ok {
		t.Error("unexpected 'axes' attribute")
	}
	if _, ok := node.Attributes["steps"]; ok {
		t.Error("unexpected 'steps' attribute")
	}
}

func TestONNXToZMF_Pad_PromotesPadsToAttribute(t *testing.T) {
	opType := "Pad"
	nodeName := "pad_1"
	model := &onnx.ModelProto{
		Graph: &onnx.GraphProto{
			Initializer: []*onnx.TensorProto{
				int64Initializer("pads", []int64{0, 0, 1, 1}),
			},
			Node: []*onnx.NodeProto{
				{
					Name:   &nodeName,
					OpType: &opType,
					Input:  []string{"data", "pads"},
					Output: []string{"padded"},
				},
			},
			Input:  []*onnx.ValueInfoProto{valueInfo("data", onnx.TensorProto_FLOAT, []int64{2, 2})},
			Output: []*onnx.ValueInfoProto{valueInfo("padded", onnx.TensorProto_FLOAT, []int64{3, 3})},
		},
	}
	zmfModel, err := ONNXToZMF(model)
	if err != nil {
		t.Fatalf("ONNXToZMF failed: %v", err)
	}
	node := zmfModel.Graph.Nodes[0]
	if node.OpType != "Pad" {
		t.Errorf("expected Pad, got %s", node.OpType)
	}
	padsAttr, ok := node.Attributes["pads"]
	if !ok {
		t.Fatal("expected 'pads' attribute")
	}
	ints, ok := padsAttr.Value.(*zmf.Attribute_Ints)
	if !ok {
		t.Fatalf("expected Ints, got %T", padsAttr.Value)
	}
	if len(ints.Ints.Val) != 4 {
		t.Errorf("expected 4 pads values, got %d", len(ints.Ints.Val))
	}
	// data remains as graph input.
	if len(node.Inputs) != 1 || node.Inputs[0] != "data" {
		t.Errorf("expected inputs=[data], got %v", node.Inputs)
	}
}

func TestONNXToZMF_Pad_WithConstantValue(t *testing.T) {
	opType := "Pad"
	nodeName := "pad_cv"
	model := &onnx.ModelProto{
		Graph: &onnx.GraphProto{
			Initializer: []*onnx.TensorProto{
				int64Initializer("pads_cv", []int64{1, 1, 1, 1}),
				float32Scalar("cv", -1.0),
			},
			Node: []*onnx.NodeProto{
				{
					Name:   &nodeName,
					OpType: &opType,
					Input:  []string{"data", "pads_cv", "cv"},
					Output: []string{"padded"},
				},
			},
			Input:  []*onnx.ValueInfoProto{valueInfo("data", onnx.TensorProto_FLOAT, []int64{2, 2})},
			Output: []*onnx.ValueInfoProto{valueInfo("padded", onnx.TensorProto_FLOAT, []int64{4, 4})},
		},
	}
	zmfModel, err := ONNXToZMF(model)
	if err != nil {
		t.Fatalf("ONNXToZMF failed: %v", err)
	}
	node := zmfModel.Graph.Nodes[0]
	cvAttr, ok := node.Attributes["constant_value"]
	if !ok {
		t.Fatal("expected 'constant_value' attribute")
	}
	if v, ok2 := cvAttr.Value.(*zmf.Attribute_F); !ok2 || v.F != float32(-1.0) {
		t.Errorf("expected constant_value=-1.0, got %v", cvAttr.Value)
	}
}

func TestONNXToZMF_TopK_PromotesKToAttribute(t *testing.T) {
	opType := "TopK"
	nodeName := "topk_1"
	model := &onnx.ModelProto{
		Graph: &onnx.GraphProto{
			Initializer: []*onnx.TensorProto{
				int64Initializer("k_val", []int64{3}),
			},
			Node: []*onnx.NodeProto{
				{
					Name:   &nodeName,
					OpType: &opType,
					Input:  []string{"data", "k_val"},
					Output: []string{"values", "indices"},
					Attribute: []*onnx.AttributeProto{
						intAttr("axis", -1),
						intAttr("largest", 1),
						intAttr("sorted", 1),
					},
				},
			},
			Input:  []*onnx.ValueInfoProto{valueInfo("data", onnx.TensorProto_FLOAT, []int64{6})},
			Output: []*onnx.ValueInfoProto{valueInfo("values", onnx.TensorProto_FLOAT, []int64{3})},
		},
	}
	zmfModel, err := ONNXToZMF(model)
	if err != nil {
		t.Fatalf("ONNXToZMF failed: %v", err)
	}
	node := zmfModel.Graph.Nodes[0]
	if node.OpType != "TopK" {
		t.Errorf("expected TopK, got %s", node.OpType)
	}
	kAttr, ok := node.Attributes["k"]
	if !ok {
		t.Fatal("expected 'k' attribute promoted from TopK K input")
	}
	if v, ok2 := kAttr.Value.(*zmf.Attribute_I); !ok2 || v.I != 3 {
		t.Errorf("expected k=3, got %v", kAttr.Value)
	}
	// data must remain as the sole graph input.
	if len(node.Inputs) != 1 || node.Inputs[0] != "data" {
		t.Errorf("expected inputs=[data], got %v", node.Inputs)
	}
}

// --- helpers ---

func proto32(v float32) *float32 { return &v }
func proto64(v int64) *int64     { return &v }
func protoInt32(v int32) *int32  { return &v }

func valueInfo(name string, dtype onnx.TensorProto_DataType, shape []int64) *onnx.ValueInfoProto {
	dt := int32(dtype)
	dims := make([]*onnx.TensorShapeProto_Dimension, len(shape))
	for i, d := range shape {
		dv := d
		dims[i] = &onnx.TensorShapeProto_Dimension{
			Value: &onnx.TensorShapeProto_Dimension_DimValue{DimValue: dv},
		}
	}
	n := name
	return &onnx.ValueInfoProto{
		Name: &n,
		Type: &onnx.TypeProto{
			Value: &onnx.TypeProto_TensorType{
				TensorType: &onnx.TypeProto_Tensor{
					ElemType: &dt,
					Shape: &onnx.TensorShapeProto{
						Dim: dims,
					},
				},
			},
		},
	}
}

func intAttr(name string, val int64) *onnx.AttributeProto {
	n := name
	attrType := onnx.AttributeProto_INT
	return &onnx.AttributeProto{
		Name: &n,
		Type: &attrType,
		I:    &val,
	}
}
