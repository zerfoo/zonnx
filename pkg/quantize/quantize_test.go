package quantize

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/zmf"
)

func makeFloat32Bytes(vals []float32) []byte {
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

func TestQuantizeModel_Q4_0(t *testing.T) {
	vals := make([]float32, 32)
	for i := range vals {
		vals[i] = float32(i) / 31.0
	}

	model := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"weight": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{32},
					Data:  makeFloat32Bytes(vals),
				},
				"bias_int64": {
					Dtype: zmf.Tensor_INT64,
					Shape: []int64{4},
					Data:  make([]byte, 32),
				},
			},
		},
	}

	err := Model(model, Q4_0)
	if err != nil {
		t.Fatalf("QuantizeModel failed: %v", err)
	}

	// Weight should now be Q4_0.
	w := model.Graph.Parameters["weight"]
	if w.Dtype != zmf.Tensor_Q4_0 {
		t.Errorf("weight dtype = %v, want Q4_0", w.Dtype)
	}
	// 1 block of 18 bytes.
	if len(w.Data) != 18 {
		t.Errorf("weight data len = %d, want 18", len(w.Data))
	}

	// Non-float32 tensor should be unchanged.
	b := model.Graph.Parameters["bias_int64"]
	if b.Dtype != zmf.Tensor_INT64 {
		t.Errorf("bias dtype = %v, want INT64", b.Dtype)
	}
}

func TestQuantizeModel_Q8_0(t *testing.T) {
	vals := make([]float32, 64)
	for i := range vals {
		vals[i] = float32(i-32) / 32.0
	}

	model := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"weight": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{64},
					Data:  makeFloat32Bytes(vals),
				},
			},
		},
	}

	err := Model(model, Q8_0)
	if err != nil {
		t.Fatalf("QuantizeModel failed: %v", err)
	}

	w := model.Graph.Parameters["weight"]
	if w.Dtype != zmf.Tensor_Q8_0 {
		t.Errorf("weight dtype = %v, want Q8_0", w.Dtype)
	}
	// 2 blocks of 36 bytes = 72.
	if len(w.Data) != 72 {
		t.Errorf("weight data len = %d, want 72", len(w.Data))
	}
}

func TestQuantizeModel_InvalidType(t *testing.T) {
	model := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{},
		},
	}
	err := Model(model, QuantType("bogus"))
	if err == nil {
		t.Error("expected error for invalid quant type")
	}
}

func TestQuantizeModel_NilGraph(t *testing.T) {
	model := &zmf.Model{}
	err := Model(model, Q4_0)
	if err == nil {
		t.Error("expected error for nil graph")
	}
}

func TestQuantizeModel_CompressionRatio(t *testing.T) {
	n := 1024
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(i) / float32(n)
	}

	model := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"w": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{int64(n)},
					Data:  makeFloat32Bytes(vals),
				},
			},
		},
	}

	origSize := len(model.Graph.Parameters["w"].Data)
	if err := Model(model, Q4_0); err != nil {
		t.Fatal(err)
	}
	newSize := len(model.Graph.Parameters["w"].Data)

	ratio := float64(origSize) / float64(newSize)
	if ratio < 6.0 {
		t.Errorf("compression ratio = %.1fx, want >= 6x", ratio)
	}
}
