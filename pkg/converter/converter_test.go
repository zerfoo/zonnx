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
