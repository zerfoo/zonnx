package converter

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	sharedgguf "github.com/zerfoo/ztensor/gguf"
	"github.com/zerfoo/zonnx/pkg/gguf"
)

// safetensorsDtype represents a data type in the safetensors format.
type safetensorsDtype string

const (
	dtypeF32  safetensorsDtype = "F32"
	dtypeF16  safetensorsDtype = "F16"
	dtypeBF16 safetensorsDtype = "BF16"
)

// safetensorsTensorInfo describes a single tensor in the safetensors header.
type safetensorsTensorInfo struct {
	Dtype       safetensorsDtype `json:"dtype"`
	Shape       []uint64         `json:"shape"`
	DataOffsets [2]uint64        `json:"data_offsets"`
}

// safetensorsFile holds parsed safetensors header and a reader for tensor data.
type safetensorsFile struct {
	Tensors    map[string]safetensorsTensorInfo
	DataOffset int64 // byte offset where tensor data begins in the file
	file       *os.File
}

// openSafetensors parses a safetensors file and returns its metadata.
// The caller must call Close() when done.
func openSafetensors(path string) (*safetensorsFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}

	sf, err := parseSafetensorsHeader(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	sf.file = f
	return sf, nil
}

// parseSafetensorsHeader reads the safetensors header from r.
func parseSafetensorsHeader(r io.ReadSeeker) (*safetensorsFile, error) {
	// First 8 bytes: uint64 header length (little-endian).
	var headerLen uint64
	if err := binary.Read(r, binary.LittleEndian, &headerLen); err != nil {
		return nil, fmt.Errorf("read header length: %w", err)
	}
	if headerLen > 100*1024*1024 { // sanity check: 100MB max header
		return nil, fmt.Errorf("header length %d exceeds 100MB limit", headerLen)
	}

	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	// Header is JSON: map of tensor name -> {dtype, shape, data_offsets}.
	// There may also be a "__metadata__" key which we skip.
	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, fmt.Errorf("parse header JSON: %w", err)
	}

	tensors := make(map[string]safetensorsTensorInfo, len(rawHeader))
	for name, raw := range rawHeader {
		if name == "__metadata__" {
			continue
		}
		var info safetensorsTensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, fmt.Errorf("parse tensor %q: %w", name, err)
		}
		tensors[name] = info
	}

	return &safetensorsFile{
		Tensors:    tensors,
		DataOffset: int64(8 + headerLen),
	}, nil
}

// ReadTensorData reads the raw bytes for the named tensor.
func (sf *safetensorsFile) ReadTensorData(name string) ([]byte, error) {
	info, ok := sf.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found", name)
	}
	size := info.DataOffsets[1] - info.DataOffsets[0]
	data := make([]byte, size)
	offset := sf.DataOffset + int64(info.DataOffsets[0])
	if _, err := sf.file.ReadAt(data, offset); err != nil {
		return nil, fmt.Errorf("read tensor %q data: %w", name, err)
	}
	return data, nil
}

// Close releases the underlying file.
func (sf *safetensorsFile) Close() error {
	if sf.file != nil {
		return sf.file.Close()
	}
	return nil
}

// safetensorsDtypeToGGUF maps safetensors dtype strings to GGUF dtype constants.
func safetensorsDtypeToGGUF(dtype safetensorsDtype) (int, error) {
	switch dtype {
	case dtypeF32:
		return sharedgguf.TypeF32, nil
	case dtypeF16:
		return sharedgguf.TypeF16, nil
	case dtypeBF16:
		return sharedgguf.TypeBF16, nil
	default:
		return 0, fmt.Errorf("unsupported safetensors dtype: %s", dtype)
	}
}

// ConvertSafetensorsToGGUF reads a HuggingFace model directory containing
// config.json and model.safetensors, maps tensor names and metadata using
// the existing GGUF mapping functions, and writes a GGUF file.
func ConvertSafetensorsToGGUF(inputDir, outputPath, arch string) error {
	// Read config.json.
	configPath := filepath.Join(inputDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("read config.json: %w", err)
	}
	var config map[string]interface{}
	if err := json.Unmarshal(configData, &config); err != nil {
		return fmt.Errorf("parse config.json: %w", err)
	}

	// Find safetensors file.
	stPath := filepath.Join(inputDir, "model.safetensors")
	if _, err := os.Stat(stPath); err != nil {
		return fmt.Errorf("model.safetensors not found in %s: %w", inputDir, err)
	}

	sf, err := openSafetensors(stPath)
	if err != nil {
		return err
	}
	defer sf.Close()

	// Create output file.
	if dir := filepath.Dir(outputPath); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create output directory: %w", err)
		}
	}
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create output file: %w", err)
	}
	defer outFile.Close()

	w := sharedgguf.NewWriter()

	// Write metadata.
	for _, entry := range gguf.MapMetadata(arch, config) {
		switch entry.Type {
		case sharedgguf.MetaTypeString:
			w.AddMetadataString(entry.Key, entry.Value.(string))
		case sharedgguf.MetaTypeUint32:
			w.AddMetadataUint32(entry.Key, entry.Value.(uint32))
		case sharedgguf.MetaTypeFloat32:
			w.AddMetadataFloat32(entry.Key, entry.Value.(float32))
		}
	}

	// Sort tensor names for deterministic output.
	names := make([]string, 0, len(sf.Tensors))
	for name := range sf.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	// Write tensors.
	for _, name := range names {
		info := sf.Tensors[name]
		ggufDtype, err := safetensorsDtypeToGGUF(info.Dtype)
		if err != nil {
			return fmt.Errorf("tensor %q: %w", name, err)
		}

		data, err := sf.ReadTensorData(name)
		if err != nil {
			return err
		}

		ggufName := gguf.MapTensorName(name)
		shape := make([]int, len(info.Shape))
		for i, d := range info.Shape {
			shape[i] = int(d)
		}
		w.AddTensor(ggufName, ggufDtype, shape, data)
	}

	if err := w.Write(outFile); err != nil {
		return fmt.Errorf("write GGUF: %w", err)
	}

	return nil
}
