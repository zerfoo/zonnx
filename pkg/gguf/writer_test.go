package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// testReader provides minimal GGUF v3 reading for round-trip verification.
type testReader struct {
	data []byte
	pos  int
}

func newTestReader(data []byte) *testReader { return &testReader{data: data} }

func (r *testReader) u32() uint32 {
	v := binary.LittleEndian.Uint32(r.data[r.pos:])
	r.pos += 4
	return v
}

func (r *testReader) u64() uint64 {
	v := binary.LittleEndian.Uint64(r.data[r.pos:])
	r.pos += 8
	return v
}

func (r *testReader) str() string {
	n := int(r.u64())
	s := string(r.data[r.pos : r.pos+n])
	r.pos += n
	return s
}

func (r *testReader) bytes(n int) []byte {
	b := make([]byte, n)
	copy(b, r.data[r.pos:r.pos+n])
	r.pos += n
	return b
}

func (r *testReader) skip(n int) { r.pos += n }

func (r *testReader) alignTo(a int) {
	if rem := r.pos % a; rem != 0 {
		r.pos += a - rem
	}
}

func TestEmptyFile(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	if err := w.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	r := newTestReader(buf.Bytes())
	if got := r.u32(); got != Magic {
		t.Fatalf("magic = %#x, want %#x", got, Magic)
	}
	if got := r.u32(); got != Version3 {
		t.Fatalf("version = %d, want %d", got, Version3)
	}
	if got := r.u64(); got != 0 {
		t.Fatalf("tensor_count = %d, want 0", got)
	}
	if got := r.u64(); got != 0 {
		t.Fatalf("metadata_kv_count = %d, want 0", got)
	}
}

func TestMetadataRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.AddMetadataString("general.name", "test-model")
	w.AddMetadataUint32("general.file_type", 7)
	w.AddMetadataFloat32("score", 0.95)
	w.AddMetadataBool("quantized", true)
	w.AddMetadataStringArray("general.tags", []string{"llm", "test"})
	if err := w.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	r := newTestReader(buf.Bytes())
	r.u32() // magic
	r.u32() // version
	r.u64() // tensor count
	kvCount := r.u64()
	if kvCount != 5 {
		t.Fatalf("metadata count = %d, want 5", kvCount)
	}

	// 1: string
	if key := r.str(); key != "general.name" {
		t.Fatalf("key = %q, want general.name", key)
	}
	if vt := r.u32(); vt != TypeString {
		t.Fatalf("type = %d, want %d", vt, TypeString)
	}
	if val := r.str(); val != "test-model" {
		t.Fatalf("value = %q, want test-model", val)
	}

	// 2: uint32
	if key := r.str(); key != "general.file_type" {
		t.Fatalf("key = %q", key)
	}
	if vt := r.u32(); vt != TypeUint32 {
		t.Fatalf("type = %d, want %d", vt, TypeUint32)
	}
	if val := r.u32(); val != 7 {
		t.Fatalf("value = %d, want 7", val)
	}

	// 3: float32
	if key := r.str(); key != "score" {
		t.Fatalf("key = %q", key)
	}
	if vt := r.u32(); vt != TypeFloat32 {
		t.Fatalf("type = %d, want %d", vt, TypeFloat32)
	}
	if val := math.Float32frombits(r.u32()); val != 0.95 {
		t.Fatalf("value = %f, want 0.95", val)
	}

	// 4: bool
	if key := r.str(); key != "quantized" {
		t.Fatalf("key = %q", key)
	}
	if vt := r.u32(); vt != TypeBool {
		t.Fatalf("type = %d, want %d", vt, TypeBool)
	}
	if val := r.bytes(1)[0]; val != 1 {
		t.Fatalf("value = %d, want 1", val)
	}

	// 5: string array
	if key := r.str(); key != "general.tags" {
		t.Fatalf("key = %q", key)
	}
	if vt := r.u32(); vt != TypeArray {
		t.Fatalf("type = %d, want %d", vt, TypeArray)
	}
	if elemType := r.u32(); elemType != TypeString {
		t.Fatalf("array elem type = %d, want %d", elemType, TypeString)
	}
	if count := r.u64(); count != 2 {
		t.Fatalf("array count = %d, want 2", count)
	}
	if s := r.str(); s != "llm" {
		t.Fatalf("array[0] = %q, want llm", s)
	}
	if s := r.str(); s != "test" {
		t.Fatalf("array[1] = %q, want test", s)
	}
}

func TestSingleTensor(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)

	// 2x3 float32 tensor.
	shape := []uint64{2, 3}
	data := make([]byte, 2*3*4)
	for i := range 6 {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i)))
	}

	w.AddTensor("weight", DTypeF32, shape, data)
	if err := w.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	raw := buf.Bytes()
	r := newTestReader(raw)

	// Header.
	r.u32() // magic
	r.u32() // version
	tensorCount := r.u64()
	if tensorCount != 1 {
		t.Fatalf("tensor count = %d, want 1", tensorCount)
	}
	r.u64() // kv count = 0

	// Tensor info.
	name := r.str()
	if name != "weight" {
		t.Fatalf("name = %q, want weight", name)
	}
	nDims := r.u32()
	if nDims != 2 {
		t.Fatalf("n_dims = %d, want 2", nDims)
	}
	// Dimensions are reversed (innermost first): expect 3, 2.
	dim0 := r.u64()
	dim1 := r.u64()
	if dim0 != 3 || dim1 != 2 {
		t.Fatalf("dims = [%d, %d], want [3, 2]", dim0, dim1)
	}
	dtype := r.u32()
	if dtype != DTypeF32 {
		t.Fatalf("dtype = %d, want %d", dtype, DTypeF32)
	}
	offset := r.u64()
	if offset != 0 {
		t.Fatalf("offset = %d, want 0", offset)
	}

	// Alignment padding.
	r.alignTo(Alignment)

	// Verify tensor data starts at a 64-byte aligned position.
	if r.pos%Alignment != 0 {
		t.Fatalf("tensor data starts at %d, not aligned to %d", r.pos, Alignment)
	}

	got := r.bytes(len(data))
	if !bytes.Equal(got, data) {
		t.Fatalf("tensor data mismatch")
	}
}

func TestMultipleTensorsAlignment(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)

	// First tensor: 3 floats (12 bytes -- not aligned).
	data1 := make([]byte, 3*4)
	for i := range 3 {
		binary.LittleEndian.PutUint32(data1[i*4:], math.Float32bits(float32(i)))
	}
	w.AddTensor("a", DTypeF32, []uint64{3}, data1)

	// Second tensor: 5 floats (20 bytes).
	data2 := make([]byte, 5*4)
	for i := range 5 {
		binary.LittleEndian.PutUint32(data2[i*4:], math.Float32bits(float32(i+10)))
	}
	w.AddTensor("b", DTypeF32, []uint64{5}, data2)

	if err := w.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	raw := buf.Bytes()
	r := newTestReader(raw)

	// Skip header.
	r.u32() // magic
	r.u32() // version
	r.u64() // tensor count
	r.u64() // kv count

	// Read tensor info for "a".
	r.str()  // name
	nDimsA := r.u32()
	for range nDimsA {
		r.u64()
	}
	r.u32() // dtype
	offsetA := r.u64()

	// Read tensor info for "b".
	r.str()  // name
	nDimsB := r.u32()
	for range nDimsB {
		r.u64()
	}
	r.u32() // dtype
	offsetB := r.u64()

	// offsetA should be 0.
	if offsetA != 0 {
		t.Fatalf("tensor a offset = %d, want 0", offsetA)
	}

	// offsetB should be aligned: 12 bytes data -> next 64-byte boundary = 64.
	if offsetB%Alignment != 0 {
		t.Fatalf("tensor b offset = %d, not aligned to %d", offsetB, Alignment)
	}
	if offsetB != Alignment {
		t.Fatalf("tensor b offset = %d, want %d", offsetB, Alignment)
	}

	// Verify actual data at the correct positions.
	r.alignTo(Alignment)
	dataStart := r.pos

	gotA := raw[dataStart+int(offsetA) : dataStart+int(offsetA)+len(data1)]
	if !bytes.Equal(gotA, data1) {
		t.Fatalf("tensor a data mismatch")
	}

	gotB := raw[dataStart+int(offsetB) : dataStart+int(offsetB)+len(data2)]
	if !bytes.Equal(gotB, data2) {
		t.Fatalf("tensor b data mismatch")
	}
}

func TestByteForByteRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.AddMetadataString("model", "round-trip-test")
	w.AddMetadataUint32("version", 1)

	data := make([]byte, 4*4)
	for i := range 4 {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i)*1.5))
	}
	w.AddTensor("embed", DTypeF32, []uint64{4}, data)

	if err := w.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	first := buf.Bytes()

	// Write again with same parameters.
	var buf2 bytes.Buffer
	w2 := NewWriter(&buf2)
	w2.AddMetadataString("model", "round-trip-test")
	w2.AddMetadataUint32("version", 1)
	w2.AddTensor("embed", DTypeF32, []uint64{4}, data)
	if err := w2.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	second := buf2.Bytes()

	if !bytes.Equal(first, second) {
		t.Fatalf("two identical writes produced different output (%d vs %d bytes)", len(first), len(second))
	}
}

func TestInt32Metadata(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.AddMetadataInt32("offset", -42)
	if err := w.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	r := newTestReader(buf.Bytes())
	r.u32() // magic
	r.u32() // version
	r.u64() // tensor count
	r.u64() // kv count

	r.str() // key
	r.u32() // type
	raw := r.u32()
	val := int32(raw)
	if val != -42 {
		t.Fatalf("int32 value = %d, want -42", val)
	}
}

func TestBoolFalse(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.AddMetadataBool("flag", false)
	if err := w.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	r := newTestReader(buf.Bytes())
	r.u32() // magic
	r.u32() // version
	r.u64() // tensor count
	r.u64() // kv count

	r.str() // key
	r.u32() // type
	if val := r.bytes(1)[0]; val != 0 {
		t.Fatalf("bool value = %d, want 0", val)
	}
}
