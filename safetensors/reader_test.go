package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/float16"
)

// writeSafeTensors builds a minimal SafeTensors file from named raw buffers.
func writeSafeTensors(t *testing.T, path string, tensors []struct {
	Name  string
	Dtype string
	Shape []int
	Data  []byte
}) {
	t.Helper()

	header := make(map[string]headerEntry, len(tensors))
	var offset int64
	for _, ts := range tensors {
		header[ts.Name] = headerEntry{
			Dtype:       ts.Dtype,
			Shape:       ts.Shape,
			DataOffsets: [2]int64{offset, offset + int64(len(ts.Data))},
		}
		offset += int64(len(ts.Data))
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create file: %v", err)
	}
	defer f.Close()

	// Write 8-byte header length.
	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerJSON))); err != nil {
		t.Fatalf("write header length: %v", err)
	}
	// Write header JSON.
	if _, err := f.Write(headerJSON); err != nil {
		t.Fatalf("write header: %v", err)
	}
	// Write tensor data.
	for _, ts := range tensors {
		if _, err := f.Write(ts.Data); err != nil {
			t.Fatalf("write tensor data: %v", err)
		}
	}
}

func float32Bytes(vals ...float32) []byte {
	buf := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func float16Bytes(vals ...float32) []byte {
	buf := make([]byte, len(vals)*2)
	for i, v := range vals {
		h := float16.FromFloat32(v)
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(h))
	}
	return buf
}

func bfloat16Bytes(vals ...float32) []byte {
	buf := make([]byte, len(vals)*2)
	for i, v := range vals {
		b := float16.BFloat16FromFloat32(v)
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(b))
	}
	return buf
}

func TestRoundTripF32(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	want := []float32{1.0, 2.0, 3.0, -4.5}
	writeSafeTensors(t, path, []struct {
		Name  string
		Dtype string
		Shape []int
		Data  []byte
	}{
		{Name: "weight", Dtype: "F32", Shape: []int{2, 2}, Data: float32Bytes(want...)},
	})

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sf.Close()

	names := sf.TensorNames()
	if len(names) != 1 || names[0] != "weight" {
		t.Fatalf("TensorNames = %v, want [weight]", names)
	}

	info, ok := sf.TensorInfo("weight")
	if !ok {
		t.Fatal("TensorInfo: not found")
	}
	if info.Dtype != "F32" {
		t.Errorf("Dtype = %q, want F32", info.Dtype)
	}
	if len(info.Shape) != 2 || info.Shape[0] != 2 || info.Shape[1] != 2 {
		t.Errorf("Shape = %v, want [2 2]", info.Shape)
	}

	got, err := sf.ReadFloat32("weight")
	if err != nil {
		t.Fatalf("ReadFloat32: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestRoundTripF16(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	input := []float32{1.0, 0.5, -2.0, 3.25}
	writeSafeTensors(t, path, []struct {
		Name  string
		Dtype string
		Shape []int
		Data  []byte
	}{
		{Name: "bias", Dtype: "F16", Shape: []int{4}, Data: float16Bytes(input...)},
	})

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sf.Close()

	got, err := sf.ReadFloat32("bias")
	if err != nil {
		t.Fatalf("ReadFloat32: %v", err)
	}
	for i, v := range input {
		if math.Abs(float64(got[i]-v)) > 0.01 {
			t.Errorf("[%d] = %v, want ~%v", i, got[i], v)
		}
	}
}

func TestRoundTripBF16(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	input := []float32{1.0, -0.5, 2.0, 0.125}
	writeSafeTensors(t, path, []struct {
		Name  string
		Dtype string
		Shape []int
		Data  []byte
	}{
		{Name: "scale", Dtype: "BF16", Shape: []int{4}, Data: bfloat16Bytes(input...)},
	})

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sf.Close()

	got, err := sf.ReadFloat32("scale")
	if err != nil {
		t.Fatalf("ReadFloat32: %v", err)
	}
	for i, v := range input {
		if math.Abs(float64(got[i]-v)) > 0.01 {
			t.Errorf("[%d] = %v, want ~%v", i, got[i], v)
		}
	}
}

func TestMultipleTensors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "multi.safetensors")

	writeSafeTensors(t, path, []struct {
		Name  string
		Dtype string
		Shape []int
		Data  []byte
	}{
		{Name: "alpha", Dtype: "F32", Shape: []int{2}, Data: float32Bytes(1.0, 2.0)},
		{Name: "beta", Dtype: "F32", Shape: []int{3}, Data: float32Bytes(3.0, 4.0, 5.0)},
	})

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sf.Close()

	names := sf.TensorNames()
	if len(names) != 2 || names[0] != "alpha" || names[1] != "beta" {
		t.Fatalf("TensorNames = %v, want [alpha beta]", names)
	}

	a, err := sf.ReadFloat32("alpha")
	if err != nil {
		t.Fatalf("ReadFloat32 alpha: %v", err)
	}
	if a[0] != 1.0 || a[1] != 2.0 {
		t.Errorf("alpha = %v", a)
	}

	b, err := sf.ReadFloat32("beta")
	if err != nil {
		t.Fatalf("ReadFloat32 beta: %v", err)
	}
	if b[0] != 3.0 || b[1] != 4.0 || b[2] != 5.0 {
		t.Errorf("beta = %v", b)
	}
}

func TestReadTensorRaw(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "raw.safetensors")

	data := float32Bytes(42.0)
	writeSafeTensors(t, path, []struct {
		Name  string
		Dtype string
		Shape []int
		Data  []byte
	}{
		{Name: "x", Dtype: "F32", Shape: []int{1}, Data: data},
	})

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sf.Close()

	raw, err := sf.ReadTensor("x")
	if err != nil {
		t.Fatalf("ReadTensor: %v", err)
	}
	if len(raw) != 4 {
		t.Fatalf("len = %d, want 4", len(raw))
	}
	bits := binary.LittleEndian.Uint32(raw)
	if math.Float32frombits(bits) != 42.0 {
		t.Errorf("got %v, want 42.0", math.Float32frombits(bits))
	}
}

func TestErrorEmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.safetensors")
	if err := os.WriteFile(path, nil, 0644); err != nil {
		t.Fatal(err)
	}

	_, err := Open(path)
	if err == nil {
		t.Fatal("expected error for empty file")
	}
}

func TestErrorInvalidHeader(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.safetensors")

	// Write header length pointing to garbage.
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	binary.Write(f, binary.LittleEndian, uint64(5))
	f.Write([]byte("hello")) // not valid JSON
	f.Close()

	_, err = Open(path)
	if err == nil {
		t.Fatal("expected error for invalid JSON header")
	}
}

func TestErrorMissingTensor(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	writeSafeTensors(t, path, []struct {
		Name  string
		Dtype string
		Shape []int
		Data  []byte
	}{
		{Name: "exists", Dtype: "F32", Shape: []int{1}, Data: float32Bytes(1.0)},
	})

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sf.Close()

	_, ok := sf.TensorInfo("missing")
	if ok {
		t.Error("TensorInfo should return false for missing tensor")
	}

	_, err = sf.ReadTensor("missing")
	if err == nil {
		t.Error("ReadTensor should return error for missing tensor")
	}

	_, err = sf.ReadFloat32("missing")
	if err == nil {
		t.Error("ReadFloat32 should return error for missing tensor")
	}
}

func TestErrorUnsupportedDtype(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	writeSafeTensors(t, path, []struct {
		Name  string
		Dtype string
		Shape []int
		Data  []byte
	}{
		{Name: "i32", Dtype: "I32", Shape: []int{1}, Data: []byte{0, 0, 0, 0}},
	})

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sf.Close()

	_, err = sf.ReadFloat32("i32")
	if err == nil {
		t.Error("ReadFloat32 should return error for unsupported dtype")
	}
}

func TestErrorHeaderTooShort(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "short.safetensors")

	// Write header length of 100 but only 5 bytes of actual data.
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	binary.Write(f, binary.LittleEndian, uint64(100))
	f.Write([]byte("short"))
	f.Close()

	_, err = Open(path)
	if err == nil {
		t.Fatal("expected error for truncated header")
	}
}
