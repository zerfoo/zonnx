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
	n := 1024
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(i) / float32(n)
	}

	model := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"model.layers.0.self_attn.q_proj.weight": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{int64(n)},
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

	w := model.Graph.Parameters["model.layers.0.self_attn.q_proj.weight"]
	if w.Dtype != zmf.Tensor_Q4_0 {
		t.Errorf("weight dtype = %v, want Q4_0", w.Dtype)
	}
	// 1024/32 = 32 blocks, 32 * 18 = 576 bytes.
	if len(w.Data) != 576 {
		t.Errorf("weight data len = %d, want 576", len(w.Data))
	}

	// Non-float32 tensor should be unchanged.
	b := model.Graph.Parameters["bias_int64"]
	if b.Dtype != zmf.Tensor_INT64 {
		t.Errorf("bias dtype = %v, want INT64", b.Dtype)
	}
}

func TestQuantizeModel_Q8_0(t *testing.T) {
	n := 1024
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(i-n/2) / float32(n)
	}

	model := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"model.layers.0.mlp.gate_proj.weight": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{int64(n)},
					Data:  makeFloat32Bytes(vals),
				},
			},
		},
	}

	err := Model(model, Q8_0)
	if err != nil {
		t.Fatalf("QuantizeModel failed: %v", err)
	}

	w := model.Graph.Parameters["model.layers.0.mlp.gate_proj.weight"]
	if w.Dtype != zmf.Tensor_Q8_0 {
		t.Errorf("weight dtype = %v, want Q8_0", w.Dtype)
	}
	// 1024/32 = 32 blocks, 32 * 36 = 1152 bytes.
	if len(w.Data) != 1152 {
		t.Errorf("weight data len = %d, want 1152", len(w.Data))
	}
}

func TestQuantizeModel_SkipNormAndSmall(t *testing.T) {
	n := 1024
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(i) / float32(n)
	}
	smallVals := make([]float32, 32)
	for i := range smallVals {
		smallVals[i] = float32(i)
	}

	model := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"model.layers.0.input_layernorm.weight": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{int64(n)},
					Data:  makeFloat32Bytes(vals),
				},
				"model.embed_tokens.weight": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{int64(n)},
					Data:  makeFloat32Bytes(vals),
				},
				"small_tensor": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{32},
					Data:  makeFloat32Bytes(smallVals),
				},
				"model.layers.0.self_attn.q_proj.weight": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{int64(n)},
					Data:  makeFloat32Bytes(vals),
				},
			},
		},
	}

	err := Model(model, Q4_0)
	if err != nil {
		t.Fatalf("QuantizeModel failed: %v", err)
	}

	// Norm weight should stay FLOAT32.
	if model.Graph.Parameters["model.layers.0.input_layernorm.weight"].Dtype != zmf.Tensor_FLOAT32 {
		t.Error("norm weight should not be quantized")
	}
	// Embedding weight should stay FLOAT32.
	if model.Graph.Parameters["model.embed_tokens.weight"].Dtype != zmf.Tensor_FLOAT32 {
		t.Error("embedding weight should not be quantized")
	}
	// Small tensor should stay FLOAT32.
	if model.Graph.Parameters["small_tensor"].Dtype != zmf.Tensor_FLOAT32 {
		t.Error("small tensor should not be quantized")
	}
	// Large attention weight should be Q4_0.
	if model.Graph.Parameters["model.layers.0.self_attn.q_proj.weight"].Dtype != zmf.Tensor_Q4_0 {
		t.Error("attention weight should be quantized to Q4_0")
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
				"model.layers.0.mlp.up_proj.weight": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{int64(n)},
					Data:  makeFloat32Bytes(vals),
				},
			},
		},
	}

	origSize := len(model.Graph.Parameters["model.layers.0.mlp.up_proj.weight"].Data)
	if err := Model(model, Q4_0); err != nil {
		t.Fatal(err)
	}
	newSize := len(model.Graph.Parameters["model.layers.0.mlp.up_proj.weight"].Data)

	ratio := float64(origSize) / float64(newSize)
	if ratio < 6.0 {
		t.Errorf("compression ratio = %.1fx, want >= 6x", ratio)
	}
}
