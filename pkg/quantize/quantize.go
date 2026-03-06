package quantize

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/zerfoo/zmf"
)

// QuantType selects the block quantization format.
type QuantType string

const (
	Q4_0 QuantType = "q4_0"
	Q8_0 QuantType = "q8_0"
)

const blockSize = 32

// Model quantizes all FLOAT32 parameter tensors in the given ZMF model in-place.
func Model(m *zmf.Model, qt QuantType) error {
	if m.Graph == nil {
		return fmt.Errorf("model has no graph")
	}

	var encode func([]float32) ([]byte, zmf.Tensor_DataType)

	switch qt {
	case Q4_0:
		encode = func(f32 []float32) ([]byte, zmf.Tensor_DataType) {
			return encodeQ4Blocks(f32), zmf.Tensor_Q4_0
		}
	case Q8_0:
		encode = func(f32 []float32) ([]byte, zmf.Tensor_DataType) {
			return encodeQ8Blocks(f32), zmf.Tensor_Q8_0
		}
	default:
		return fmt.Errorf("unsupported quantization type: %q", qt)
	}

	for name, t := range m.Graph.Parameters {
		if t.Dtype != zmf.Tensor_FLOAT32 {
			continue
		}
		if skipQuantize(name, t) {
			continue
		}

		f32 := decodeFloat32(t.Data)
		data, dtype := encode(f32)
		m.Graph.Parameters[name] = &zmf.Tensor{
			Dtype: dtype,
			Shape: t.Shape,
			Data:  data,
		}
	}

	return nil
}

// minQuantElements is the minimum number of elements for a tensor to be quantized.
// Tensors with fewer elements (e.g., norm weights, bias vectors) lose too much
// precision from block quantization and are kept as float32.
const minQuantElements = 1024

// skipQuantize returns true for tensors that should NOT be quantized.
// Norm weights, bias vectors, and small tensors are kept in float32.
func skipQuantize(name string, t *zmf.Tensor) bool {
	lower := strings.ToLower(name)

	// Skip normalization weights (RMSNorm, LayerNorm).
	if strings.Contains(lower, "norm") {
		return true
	}
	// Skip bias tensors.
	if strings.HasSuffix(lower, ".bias") || strings.HasSuffix(lower, "_bias") {
		return true
	}
	// Skip embedding tables (used in Gather, dequantization happens per-lookup).
	if strings.Contains(lower, "embed") {
		return true
	}

	// Skip small tensors.
	numElements := 1
	for _, d := range t.Shape {
		numElements *= int(d)
	}
	return numElements < minQuantElements
}

func decodeFloat32(b []byte) []float32 {
	n := len(b) / 4
	f32 := make([]float32, n)
	for i := range n {
		f32[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4 : i*4+4]))
	}
	return f32
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// encodeQ4Blocks converts float32 values to Q4_0 block bytes.
// Q4_0: 2-byte float16 scale + 16-byte packed 4-bit data = 18 bytes per 32 values.
func encodeQ4Blocks(src []float32) []byte {
	n := len(src)
	nBlocks := (n + blockSize - 1) / blockSize
	out := make([]byte, nBlocks*18)

	for bi := range nBlocks {
		offset := bi * blockSize
		var absMax float32
		for j := range blockSize {
			idx := offset + j
			var v float32
			if idx < n {
				v = src[idx]
			}
			if av := float32(math.Abs(float64(v))); av > absMax {
				absMax = av
			}
		}

		var scale float32
		if absMax > 0 {
			scale = absMax / 7.0
		}

		// Encode scale as float16 (IEEE 754 half-precision).
		binary.LittleEndian.PutUint16(out[bi*18:], float32ToFloat16Bits(scale))

		var invScale float32
		if scale > 0 {
			invScale = 1.0 / scale
		}
		for j := 0; j < blockSize; j += 2 {
			var v0, v1 float32
			if offset+j < n {
				v0 = src[offset+j]
			}
			if offset+j+1 < n {
				v1 = src[offset+j+1]
			}
			q0 := clampInt(int(math.Round(float64(v0*invScale))), -8, 7)
			q1 := clampInt(int(math.Round(float64(v1*invScale))), -8, 7)
			out[bi*18+2+j/2] = byte(q0+8) | (byte(q1+8) << 4)
		}
	}
	return out
}

// encodeQ8Blocks converts float32 values to Q8_0 block bytes.
// Q8_0: 4-byte float32 scale + 32-byte int8 data = 36 bytes per 32 values.
func encodeQ8Blocks(src []float32) []byte {
	n := len(src)
	nBlocks := (n + blockSize - 1) / blockSize
	out := make([]byte, nBlocks*36)

	for bi := range nBlocks {
		offset := bi * blockSize
		var absMax float32
		for j := range blockSize {
			idx := offset + j
			var v float32
			if idx < n {
				v = src[idx]
			}
			if av := float32(math.Abs(float64(v))); av > absMax {
				absMax = av
			}
		}

		var scale float32
		if absMax > 0 {
			scale = absMax / 127.0
		}
		binary.LittleEndian.PutUint32(out[bi*36:], math.Float32bits(scale))

		var invScale float32
		if scale > 0 {
			invScale = 1.0 / scale
		}
		for j := range blockSize {
			var v float32
			if offset+j < n {
				v = src[offset+j]
			}
			out[bi*36+4+j] = byte(int8(clampInt(int(math.Round(float64(v*invScale))), -128, 127)))
		}
	}
	return out
}

// float32ToFloat16Bits converts a float32 to IEEE 754 half-precision bits.
func float32ToFloat16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := (b >> 16) & 0x8000
	exp := int((b>>23)&0xFF) - 127 + 15
	frac := b & 0x7FFFFF

	if exp <= 0 {
		return uint16(sign) // flush to zero
	}
	if exp >= 31 {
		return uint16(sign | 0x7C00) // infinity
	}
	return uint16(sign | uint32(exp)<<10 | (frac >> 13))
}
