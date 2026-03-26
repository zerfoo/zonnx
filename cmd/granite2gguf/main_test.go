package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeSafetensors creates a minimal SafeTensors file with the given tensors.
// Each tensor is stored as F32. The file format is:
//
//	[8 bytes: header_len LE] [header JSON] [tensor data...]
func writeSafetensors(t *testing.T, path string, tensors map[string][]float32, shapes map[string][]int) {
	t.Helper()

	type stEntry struct {
		Dtype       string   `json:"dtype"`
		Shape       []int    `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}

	// Build data and header.
	header := make(map[string]stEntry)
	var allData []byte
	// Sort keys for determinism.
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	// Simple sort for small test sets.
	for i := 0; i < len(names); i++ {
		for j := i + 1; j < len(names); j++ {
			if names[i] > names[j] {
				names[i], names[j] = names[j], names[i]
			}
		}
	}

	for _, name := range names {
		floats := tensors[name]
		start := int64(len(allData))
		data := make([]byte, len(floats)*4)
		for i, f := range floats {
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(f))
		}
		allData = append(allData, data...)
		end := int64(len(allData))

		header[name] = stEntry{
			Dtype:       "F32",
			Shape:       shapes[name],
			DataOffsets: [2]int64{start, end},
		}
	}

	headerBytes, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal safetensors header: %v", err)
	}

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create safetensors file: %v", err)
	}
	defer f.Close()

	// Write header length.
	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerBytes))); err != nil {
		t.Fatalf("write header length: %v", err)
	}
	// Write header.
	if _, err := f.Write(headerBytes); err != nil {
		t.Fatalf("write header: %v", err)
	}
	// Write tensor data.
	if _, err := f.Write(allData); err != nil {
		t.Fatalf("write tensor data: %v", err)
	}
}

// ggufTestReader provides minimal GGUF v3 reading for verification.
type ggufTestReader struct {
	data []byte
	pos  int
}

func (r *ggufTestReader) u32() uint32 {
	v := binary.LittleEndian.Uint32(r.data[r.pos:])
	r.pos += 4
	return v
}

func (r *ggufTestReader) u64() uint64 {
	v := binary.LittleEndian.Uint64(r.data[r.pos:])
	r.pos += 8
	return v
}

func (r *ggufTestReader) str() string {
	n := int(r.u64())
	s := string(r.data[r.pos : r.pos+n])
	r.pos += n
	return s
}

func (r *ggufTestReader) f32() float32 {
	return math.Float32frombits(r.u32())
}

func (r *ggufTestReader) alignTo(a int) {
	if rem := r.pos % a; rem != 0 {
		r.pos += a - rem
	}
}

func TestConvertTTM(t *testing.T) {
	dir := t.TempDir()

	// Write config.json for a TTM model.
	config := map[string]interface{}{
		"model_type":        "ttm",
		"context_length":    512,
		"prediction_length": 96,
		"num_mixer_layers":  8,
		"d_model":           64,
		"patch_length":      32,
		"num_patches":       16,
	}
	configData, _ := json.Marshal(config)
	os.WriteFile(filepath.Join(dir, "config.json"), configData, 0o644)

	// Write model.safetensors with two tensors.
	tensors := map[string][]float32{
		"backbone.mixer_layers.0.mlp.fc1.weight": {1.0, 2.0, 3.0, 4.0},
		"backbone.embedding.weight":              {0.5, 1.5},
	}
	shapes := map[string][]int{
		"backbone.mixer_layers.0.mlp.fc1.weight": {2, 2},
		"backbone.embedding.weight":              {2},
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), tensors, shapes)

	// Convert.
	outputPath := filepath.Join(dir, "model.gguf")
	if err := convert(dir, outputPath); err != nil {
		t.Fatalf("convert: %v", err)
	}

	// Verify GGUF output.
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}

	r := &ggufTestReader{data: data}

	// Header.
	magic := r.u32()
	if magic != 0x46554747 {
		t.Fatalf("magic = %#x, want GGUF", magic)
	}
	version := r.u32()
	if version != 3 {
		t.Fatalf("version = %d, want 3", version)
	}
	tensorCount := r.u64()
	if tensorCount != 2 {
		t.Fatalf("tensor_count = %d, want 2", tensorCount)
	}
	kvCount := r.u64()
	if kvCount == 0 {
		t.Fatal("metadata count is 0, expected metadata")
	}

	// Read all metadata KV pairs.
	metadata := make(map[string]interface{})
	for range kvCount {
		key := r.str()
		vt := r.u32()
		switch vt {
		case 8: // string
			metadata[key] = r.str()
		case 4: // uint32
			metadata[key] = r.u32()
		case 6: // float32
			metadata[key] = r.f32()
		default:
			t.Fatalf("unexpected metadata type %d for key %q", vt, key)
		}
	}

	// Verify metadata.
	if v, _ := metadata["general.architecture"].(string); v != "granite_ts" {
		t.Errorf("general.architecture = %q, want granite_ts", v)
	}
	if v, _ := metadata["ts.signal.model_type"].(string); v != "ttm" {
		t.Errorf("ts.signal.model_type = %q, want ttm", v)
	}
	if v, _ := metadata["ts.signal.context_len"].(uint32); v != 512 {
		t.Errorf("ts.signal.context_len = %d, want 512", v)
	}
	if v, _ := metadata["ts.signal.forecast_len"].(uint32); v != 96 {
		t.Errorf("ts.signal.forecast_len = %d, want 96", v)
	}
	if v, _ := metadata["ts.signal.num_mixer_layers"].(uint32); v != 8 {
		t.Errorf("ts.signal.num_mixer_layers = %d, want 8", v)
	}
	if v, _ := metadata["ts.signal.d_model"].(uint32); v != 64 {
		t.Errorf("ts.signal.d_model = %d, want 64", v)
	}
	if v, _ := metadata["ts.signal.patch_len"].(uint32); v != 32 {
		t.Errorf("ts.signal.patch_len = %d, want 32", v)
	}
	if v, _ := metadata["ts.signal.num_patches"].(uint32); v != 16 {
		t.Errorf("ts.signal.num_patches = %d, want 16", v)
	}

	// Read tensor info entries.
	type tensorEntry struct {
		name   string
		nDims  uint32
		dims   []uint64
		dtype  uint32
		offset uint64
	}
	tensorInfos := make([]tensorEntry, tensorCount)
	for i := range tensorCount {
		te := tensorEntry{}
		te.name = r.str()
		te.nDims = r.u32()
		te.dims = make([]uint64, te.nDims)
		for j := range te.nDims {
			te.dims[j] = r.u64()
		}
		te.dtype = r.u32()
		te.offset = r.u64()
		tensorInfos[i] = te
	}

	// Verify tensor names were mapped correctly.
	foundNames := make(map[string]bool)
	for _, te := range tensorInfos {
		foundNames[te.name] = true
		if te.dtype != 0 { // DTypeF32
			t.Errorf("tensor %q dtype = %d, want 0 (F32)", te.name, te.dtype)
		}
	}

	if !foundNames["blk.0.mlp.fc1.weight"] {
		t.Error("expected tensor blk.0.mlp.fc1.weight not found")
	}
	if !foundNames["embedding.weight"] {
		t.Error("expected tensor embedding.weight not found")
	}

	// Align to data section and verify we can read tensor data.
	r.alignTo(64)
	if r.pos >= len(data) {
		t.Fatal("tensor data section beyond file bounds")
	}
}

func TestConvertFlowState(t *testing.T) {
	dir := t.TempDir()

	config := map[string]interface{}{
		"model_type":   "flowstate",
		"d_model":      128,
		"num_layers":   4,
		"scale_factor":  2.5,
	}
	configData, _ := json.Marshal(config)
	os.WriteFile(filepath.Join(dir, "config.json"), configData, 0o644)

	tensors := map[string][]float32{
		"head.linear.weight": {1.0, 2.0},
	}
	shapes := map[string][]int{
		"head.linear.weight": {2},
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), tensors, shapes)

	outputPath := filepath.Join(dir, "model.gguf")
	if err := convert(dir, outputPath); err != nil {
		t.Fatalf("convert: %v", err)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}

	r := &ggufTestReader{data: data}
	r.u32() // magic
	r.u32() // version
	r.u64() // tensor count
	kvCount := r.u64()

	metadata := make(map[string]interface{})
	for range kvCount {
		key := r.str()
		vt := r.u32()
		switch vt {
		case 8:
			metadata[key] = r.str()
		case 4:
			metadata[key] = r.u32()
		case 6:
			metadata[key] = r.f32()
		default:
			t.Fatalf("unexpected type %d for %q", vt, key)
		}
	}

	if v, _ := metadata["ts.signal.model_type"].(string); v != "flowstate" {
		t.Errorf("model_type = %q, want flowstate", v)
	}
	if v, _ := metadata["ts.signal.num_ssm_layers"].(uint32); v != 4 {
		t.Errorf("num_ssm_layers = %d, want 4", v)
	}
	if v, _ := metadata["ts.signal.scale_factor"].(float32); v != 2.5 {
		t.Errorf("scale_factor = %f, want 2.5", v)
	}

	// Verify head tensor name kept as-is.
	name := r.str()
	if name != "head.linear.weight" {
		t.Errorf("tensor name = %q, want head.linear.weight", name)
	}
}

func TestConvertTSPulse(t *testing.T) {
	dir := t.TempDir()

	config := map[string]interface{}{
		"model_type": "tspulse",
		"d_model":    256,
		"mask_type":  "hybrid",
		"head_type":  "dualhead",
	}
	configData, _ := json.Marshal(config)
	os.WriteFile(filepath.Join(dir, "config.json"), configData, 0o644)

	tensors := map[string][]float32{
		"backbone.embedding.weight": {1.0},
	}
	shapes := map[string][]int{
		"backbone.embedding.weight": {1},
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), tensors, shapes)

	outputPath := filepath.Join(dir, "model.gguf")
	if err := convert(dir, outputPath); err != nil {
		t.Fatalf("convert: %v", err)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}

	r := &ggufTestReader{data: data}
	r.u32() // magic
	r.u32() // version
	r.u64() // tensor count
	kvCount := r.u64()

	metadata := make(map[string]interface{})
	for range kvCount {
		key := r.str()
		vt := r.u32()
		switch vt {
		case 8:
			metadata[key] = r.str()
		case 4:
			metadata[key] = r.u32()
		case 6:
			metadata[key] = r.f32()
		default:
			t.Fatalf("unexpected type %d for %q", vt, key)
		}
	}

	if v, _ := metadata["ts.signal.mask_type"].(string); v != "hybrid" {
		t.Errorf("mask_type = %q, want hybrid", v)
	}
	if v, _ := metadata["ts.signal.head_type"].(string); v != "dualhead" {
		t.Errorf("head_type = %q, want dualhead", v)
	}
}

func TestMapGraniteTensorName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"backbone.mixer_layers.0.mlp.fc1.weight", "blk.0.mlp.fc1.weight"},
		{"backbone.mixer_layers.12.norm.weight", "blk.12.norm.weight"},
		{"backbone.embedding.weight", "embedding.weight"},
		{"head.linear.weight", "head.linear.weight"},
		{"some.other.tensor", "some.other.tensor"},
	}

	for _, tt := range tests {
		got := mapGraniteTensorName(tt.input)
		if got != tt.want {
			t.Errorf("mapGraniteTensorName(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}
