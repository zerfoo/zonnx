package converter

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// buildSafetensors creates a minimal safetensors file in memory.
// It writes a header with the given tensors and appends raw float32 data.
func buildSafetensors(t *testing.T, tensors map[string][]float32, shapes map[string][]uint64) []byte {
	t.Helper()

	// Build raw data and header entries.
	var dataBuf bytes.Buffer
	header := make(map[string]safetensorsTensorInfo, len(tensors))

	// Sort keys for deterministic layout.
	keys := make([]string, 0, len(tensors))
	for k := range tensors {
		keys = append(keys, k)
	}

	for _, name := range keys {
		values := tensors[name]
		start := uint64(dataBuf.Len())
		for _, v := range values {
			if err := binary.Write(&dataBuf, binary.LittleEndian, v); err != nil {
				t.Fatalf("write tensor data: %v", err)
			}
		}
		end := uint64(dataBuf.Len())
		header[name] = safetensorsTensorInfo{
			Dtype:       dtypeF32,
			Shape:       shapes[name],
			DataOffsets: [2]uint64{start, end},
		}
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}

	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON))); err != nil {
		t.Fatalf("write header length: %v", err)
	}
	buf.Write(headerJSON)
	buf.Write(dataBuf.Bytes())

	return buf.Bytes()
}

func TestParseSafetensorsHeader(t *testing.T) {
	tensors := map[string][]float32{
		"weight": {1.0, 2.0, 3.0, 4.0},
		"bias":   {0.5, 0.5},
	}
	shapes := map[string][]uint64{
		"weight": {2, 2},
		"bias":   {2},
	}

	data := buildSafetensors(t, tensors, shapes)
	r := bytes.NewReader(data)
	sf, err := parseSafetensorsHeader(r)
	if err != nil {
		t.Fatalf("parse header: %v", err)
	}

	if len(sf.Tensors) != 2 {
		t.Errorf("expected 2 tensors, got %d", len(sf.Tensors))
	}

	w, ok := sf.Tensors["weight"]
	if !ok {
		t.Fatal("missing tensor 'weight'")
	}
	if w.Dtype != dtypeF32 {
		t.Errorf("expected dtype F32, got %s", w.Dtype)
	}
	if len(w.Shape) != 2 || w.Shape[0] != 2 || w.Shape[1] != 2 {
		t.Errorf("expected shape [2,2], got %v", w.Shape)
	}

	b, ok := sf.Tensors["bias"]
	if !ok {
		t.Fatal("missing tensor 'bias'")
	}
	if len(b.Shape) != 1 || b.Shape[0] != 2 {
		t.Errorf("expected shape [2], got %v", b.Shape)
	}
}

func TestOpenSafetensorsAndReadData(t *testing.T) {
	tensors := map[string][]float32{
		"layer.weight": {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
	}
	shapes := map[string][]uint64{
		"layer.weight": {2, 3},
	}

	data := buildSafetensors(t, tensors, shapes)

	// Write to temp file.
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}

	sf, err := openSafetensors(path)
	if err != nil {
		t.Fatalf("open safetensors: %v", err)
	}
	defer sf.Close()

	raw, err := sf.ReadTensorData("layer.weight")
	if err != nil {
		t.Fatalf("read tensor data: %v", err)
	}

	// 6 float32 values = 24 bytes.
	if len(raw) != 24 {
		t.Errorf("expected 24 bytes, got %d", len(raw))
	}
}

func TestSafetensorsDtypeToGGUF(t *testing.T) {
	tests := []struct {
		dtype safetensorsDtype
		want  int
		err   bool
	}{
		{dtypeF32, 0, false},  // DTypeF32
		{dtypeF16, 1, false},  // DTypeF16
		{dtypeBF16, 30, false}, // DTypeBF16
		{"INT8", 0, true},
	}
	for _, tc := range tests {
		got, err := safetensorsDtypeToGGUF(tc.dtype)
		if tc.err {
			if err == nil {
				t.Errorf("dtype %s: expected error", tc.dtype)
			}
			continue
		}
		if err != nil {
			t.Errorf("dtype %s: unexpected error: %v", tc.dtype, err)
			continue
		}
		if got != tc.want {
			t.Errorf("dtype %s: got %d, want %d", tc.dtype, got, tc.want)
		}
	}
}

func TestConvertSafetensorsToGGUF(t *testing.T) {
	dir := t.TempDir()

	// Write a minimal config.json for BERT.
	config := map[string]interface{}{
		"hidden_size":          768,
		"num_hidden_layers":    2,
		"num_attention_heads":  12,
		"intermediate_size":    3072,
		"vocab_size":           100,
		"max_position_embeddings": 512,
		"layer_norm_eps":       1e-12,
		"num_labels":           3,
	}
	configJSON, _ := json.Marshal(config)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), configJSON, 0o644); err != nil {
		t.Fatal(err)
	}

	// Build a small safetensors file with two tensors.
	tensors := map[string][]float32{
		"bert.embeddings.word_embeddings.weight": make([]float32, 100*768),
		"classifier.weight":                     make([]float32, 3*768),
	}
	shapes := map[string][]uint64{
		"bert.embeddings.word_embeddings.weight": {100, 768},
		"classifier.weight":                     {3, 768},
	}
	stData := buildSafetensors(t, tensors, shapes)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), stData, 0o644); err != nil {
		t.Fatal(err)
	}

	outputPath := filepath.Join(dir, "output.gguf")
	if err := ConvertSafetensorsToGGUF(dir, outputPath, "bert"); err != nil {
		t.Fatalf("convert: %v", err)
	}

	// Verify output file exists and is non-empty.
	info, err := os.Stat(outputPath)
	if err != nil {
		t.Fatalf("stat output: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output GGUF is empty")
	}
}

func TestParseSafetensorsHeader_SkipsMetadata(t *testing.T) {
	// Build header with __metadata__ key.
	header := map[string]json.RawMessage{
		"__metadata__": json.RawMessage(`{"format":"pt"}`),
		"weight": json.RawMessage(`{"dtype":"F32","shape":[2],"data_offsets":[0,8]}`),
	}
	headerJSON, _ := json.Marshal(header)

	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON)))
	buf.Write(headerJSON)
	// Append 8 bytes of data for the tensor.
	buf.Write(make([]byte, 8))

	r := bytes.NewReader(buf.Bytes())
	sf, err := parseSafetensorsHeader(r)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(sf.Tensors) != 1 {
		t.Errorf("expected 1 tensor, got %d", len(sf.Tensors))
	}
	if _, ok := sf.Tensors["__metadata__"]; ok {
		t.Error("__metadata__ should be skipped")
	}
}
