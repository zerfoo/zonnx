package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

const (
	Magic     = 0x46554747
	Version3  = 3
	Alignment = 64

	// Metadata value types.
	TypeUint8   uint32 = 0
	TypeInt8    uint32 = 1
	TypeUint16  uint32 = 2
	TypeInt16   uint32 = 3
	TypeUint32  uint32 = 4
	TypeInt32   uint32 = 5
	TypeFloat32 uint32 = 6
	TypeBool    uint32 = 7
	TypeString  uint32 = 8
	TypeArray   uint32 = 9
	TypeUint64  uint32 = 10
	TypeInt64   uint32 = 11
	TypeFloat64 uint32 = 12

	// Tensor data types.
	DTypeF32  uint32 = 0
	DTypeF16  uint32 = 1
	DTypeQ4_0 uint32 = 2
	DTypeQ8_0 uint32 = 8
	DTypeBF16 uint32 = 30
)

type metadataKV struct {
	key       string
	valueType uint32
	value     any
}

type tensorInfo struct {
	name  string
	dtype uint32
	shape []uint64
	data  []byte
}

// Writer buffers GGUF v3 metadata and tensor data, then writes the complete
// file in a single Flush call.
type Writer struct {
	w        io.Writer
	metadata []metadataKV
	tensors  []tensorInfo
}

// NewWriter returns a Writer that writes GGUF v3 data to w.
func NewWriter(w io.Writer) *Writer {
	return &Writer{w: w}
}

func (w *Writer) AddMetadataString(key, value string) {
	w.metadata = append(w.metadata, metadataKV{key: key, valueType: TypeString, value: value})
}

func (w *Writer) AddMetadataUint32(key string, value uint32) {
	w.metadata = append(w.metadata, metadataKV{key: key, valueType: TypeUint32, value: value})
}

func (w *Writer) AddMetadataInt32(key string, value int32) {
	w.metadata = append(w.metadata, metadataKV{key: key, valueType: TypeInt32, value: value})
}

func (w *Writer) AddMetadataFloat32(key string, value float32) {
	w.metadata = append(w.metadata, metadataKV{key: key, valueType: TypeFloat32, value: value})
}

func (w *Writer) AddMetadataBool(key string, value bool) {
	w.metadata = append(w.metadata, metadataKV{key: key, valueType: TypeBool, value: value})
}

func (w *Writer) AddMetadataStringArray(key string, values []string) {
	w.metadata = append(w.metadata, metadataKV{key: key, valueType: TypeArray, value: values})
}

// AddTensor registers a tensor. shape uses the caller's convention (outermost
// first); the writer reverses dimensions when serializing per GGUF spec.
func (w *Writer) AddTensor(name string, dtype uint32, shape []uint64, data []byte) {
	w.tensors = append(w.tensors, tensorInfo{name: name, dtype: dtype, shape: shape, data: data})
}

// Flush writes the complete GGUF v3 file to the underlying writer.
func (w *Writer) Flush() error {
	var written int64
	write := func(v any) error {
		if err := binary.Write(w.w, binary.LittleEndian, v); err != nil {
			return err
		}
		written += int64(binary.Size(v))
		return nil
	}
	writeBytes := func(b []byte) error {
		n, err := w.w.Write(b)
		written += int64(n)
		return err
	}
	writeString := func(s string) error {
		if err := write(uint64(len(s))); err != nil {
			return err
		}
		return writeBytes([]byte(s))
	}
	writePad := func() error {
		rem := written % Alignment
		if rem == 0 {
			return nil
		}
		pad := make([]byte, Alignment-rem)
		return writeBytes(pad)
	}

	// Header.
	if err := write(uint32(Magic)); err != nil {
		return fmt.Errorf("write magic: %w", err)
	}
	if err := write(uint32(Version3)); err != nil {
		return fmt.Errorf("write version: %w", err)
	}
	if err := write(uint64(len(w.tensors))); err != nil {
		return fmt.Errorf("write tensor count: %w", err)
	}
	if err := write(uint64(len(w.metadata))); err != nil {
		return fmt.Errorf("write metadata count: %w", err)
	}

	// Metadata KV pairs.
	for _, kv := range w.metadata {
		if err := writeString(kv.key); err != nil {
			return fmt.Errorf("write metadata key %q: %w", kv.key, err)
		}
		if err := write(kv.valueType); err != nil {
			return fmt.Errorf("write metadata type %q: %w", kv.key, err)
		}
		if err := writeMetadataValue(write, writeBytes, writeString, kv); err != nil {
			return fmt.Errorf("write metadata value %q: %w", kv.key, err)
		}
	}

	// Tensor info entries. Compute each tensor's offset from the start of
	// the tensor data section (after header+metadata+tensorinfo+padding).
	var dataOffset uint64
	for _, t := range w.tensors {
		if err := writeString(t.name); err != nil {
			return fmt.Errorf("write tensor name %q: %w", t.name, err)
		}
		// Reverse dimensions for GGUF (innermost first).
		nDims := uint32(len(t.shape))
		if err := write(nDims); err != nil {
			return err
		}
		for i := len(t.shape) - 1; i >= 0; i-- {
			if err := write(t.shape[i]); err != nil {
				return err
			}
		}
		if err := write(t.dtype); err != nil {
			return err
		}
		if err := write(dataOffset); err != nil {
			return err
		}
		dataOffset += uint64(len(t.data))
		// Align next tensor's offset.
		if rem := dataOffset % Alignment; rem != 0 {
			dataOffset += Alignment - rem
		}
	}

	// Pad to alignment boundary before tensor data.
	if err := writePad(); err != nil {
		return fmt.Errorf("write header padding: %w", err)
	}

	// Tensor data, each aligned.
	for _, t := range w.tensors {
		if err := writeBytes(t.data); err != nil {
			return fmt.Errorf("write tensor data %q: %w", t.name, err)
		}
		if err := writePad(); err != nil {
			return fmt.Errorf("write tensor padding %q: %w", t.name, err)
		}
	}

	return nil
}

func writeMetadataValue(
	write func(any) error,
	writeBytes func([]byte) error,
	writeString func(string) error,
	kv metadataKV,
) error {
	switch kv.valueType {
	case TypeUint32:
		return write(kv.value.(uint32))
	case TypeInt32:
		return write(kv.value.(int32))
	case TypeFloat32:
		return write(math.Float32bits(kv.value.(float32)))
	case TypeBool:
		var b uint8
		if kv.value.(bool) {
			b = 1
		}
		return write(b)
	case TypeString:
		return writeString(kv.value.(string))
	case TypeArray:
		arr := kv.value.([]string)
		if err := write(uint32(TypeString)); err != nil {
			return err
		}
		if err := write(uint64(len(arr))); err != nil {
			return err
		}
		for _, s := range arr {
			if err := writeString(s); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unsupported metadata type %d", kv.valueType)
	}
}
