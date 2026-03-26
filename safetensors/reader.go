// Package safetensors implements a reader for HuggingFace's SafeTensors
// binary format, which stores named tensors with a JSON header followed
// by contiguous raw data.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/zerfoo/float16"
)

// TensorInfo describes a single tensor stored in a SafeTensors file.
type TensorInfo struct {
	Name        string
	Dtype       string
	Shape       []int
	DataOffsets [2]int64
}

// headerEntry is the JSON representation of a single tensor in the header.
type headerEntry struct {
	Dtype       string  `json:"dtype"`
	Shape       []int   `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// File provides read access to tensors stored in a SafeTensors file.
type File struct {
	f          *os.File
	tensors    map[string]TensorInfo
	names      []string
	dataOffset int64 // byte offset where tensor data begins (8 + headerLen)
}

// Open opens a SafeTensors file and parses its header.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: open: %w", err)
	}

	sf, err := parse(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	return sf, nil
}

// parse reads the header from r and builds the File struct.
func parse(f *os.File) (*File, error) {
	// Read 8-byte header length.
	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		return nil, fmt.Errorf("safetensors: read header length: %w", err)
	}

	if headerLen == 0 {
		return nil, fmt.Errorf("safetensors: header length is zero")
	}
	if headerLen > 100*1024*1024 { // 100 MiB sanity limit
		return nil, fmt.Errorf("safetensors: header length %d exceeds 100 MiB limit", headerLen)
	}

	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, fmt.Errorf("safetensors: read header: %w", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, fmt.Errorf("safetensors: parse header JSON: %w", err)
	}

	tensors := make(map[string]TensorInfo, len(raw))
	names := make([]string, 0, len(raw))

	for name, data := range raw {
		// The "__metadata__" key is reserved for file-level metadata; skip it.
		if name == "__metadata__" {
			continue
		}

		var entry headerEntry
		if err := json.Unmarshal(data, &entry); err != nil {
			return nil, fmt.Errorf("safetensors: parse tensor %q: %w", name, err)
		}
		tensors[name] = TensorInfo{
			Name:        name,
			Dtype:       entry.Dtype,
			Shape:       entry.Shape,
			DataOffsets: entry.DataOffsets,
		}
		names = append(names, name)
	}
	sort.Strings(names)

	return &File{
		f:          f,
		tensors:    tensors,
		names:      names,
		dataOffset: int64(8 + headerLen),
	}, nil
}

// TensorNames returns the sorted names of all tensors in the file.
func (sf *File) TensorNames() []string {
	out := make([]string, len(sf.names))
	copy(out, sf.names)
	return out
}

// TensorInfo returns the metadata for the named tensor.
func (sf *File) TensorInfo(name string) (TensorInfo, bool) {
	info, ok := sf.tensors[name]
	return info, ok
}

// ReadTensor reads the raw bytes for the named tensor.
func (sf *File) ReadTensor(name string) ([]byte, error) {
	info, ok := sf.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}

	start := sf.dataOffset + info.DataOffsets[0]
	size := info.DataOffsets[1] - info.DataOffsets[0]
	if size < 0 {
		return nil, fmt.Errorf("safetensors: tensor %q has negative size", name)
	}

	buf := make([]byte, size)
	if _, err := sf.f.ReadAt(buf, start); err != nil {
		return nil, fmt.Errorf("safetensors: read tensor %q: %w", name, err)
	}
	return buf, nil
}

// ReadFloat32 reads a tensor and returns its data as a []float32 slice.
// Supported dtypes: F32, F16, BF16.
func (sf *File) ReadFloat32(name string) ([]float32, error) {
	info, ok := sf.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}

	raw, err := sf.ReadTensor(name)
	if err != nil {
		return nil, err
	}

	switch info.Dtype {
	case "F32":
		return decodeF32(raw)
	case "F16":
		return decodeF16(raw)
	case "BF16":
		return decodeBF16(raw)
	default:
		return nil, fmt.Errorf("safetensors: unsupported dtype %q for ReadFloat32", info.Dtype)
	}
}

// Close closes the underlying file.
func (sf *File) Close() error {
	return sf.f.Close()
}

func decodeF32(data []byte) ([]float32, error) {
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("safetensors: F32 data length %d not a multiple of 4", len(data))
	}
	n := len(data) / 4
	out := make([]float32, n)
	for i := range n {
		bits := binary.LittleEndian.Uint32(data[i*4 : i*4+4])
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

func decodeF16(data []byte) ([]float32, error) {
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("safetensors: F16 data length %d not a multiple of 2", len(data))
	}
	n := len(data) / 2
	out := make([]float32, n)
	for i := range n {
		bits := binary.LittleEndian.Uint16(data[i*2 : i*2+2])
		out[i] = float16.Float16(bits).ToFloat32()
	}
	return out, nil
}

func decodeBF16(data []byte) ([]float32, error) {
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("safetensors: BF16 data length %d not a multiple of 2", len(data))
	}
	n := len(data) / 2
	out := make([]float32, n)
	for i := range n {
		bits := binary.LittleEndian.Uint16(data[i*2 : i*2+2])
		out[i] = float16.BFloat16(bits).ToFloat32()
	}
	return out, nil
}
