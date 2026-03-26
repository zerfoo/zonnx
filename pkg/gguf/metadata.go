package gguf

import (
	"fmt"
	"math"

	sharedgguf "github.com/zerfoo/ztensor/gguf"
)

// MetadataEntry represents a single GGUF metadata key-value pair.
type MetadataEntry struct {
	Key   string
	Type  uint32
	Value any
}

// configMapping defines how HuggingFace config keys map to GGUF metadata keys.
// The placeholder {arch} is replaced with the architecture name at runtime.
var configMapping = []struct {
	hfKey    string
	ggufKey  string
	ggufType uint32
}{
	{"hidden_size", "{arch}.embedding_length", sharedgguf.MetaTypeUint32},
	{"num_hidden_layers", "{arch}.block_count", sharedgguf.MetaTypeUint32},
	{"num_attention_heads", "{arch}.attention.head_count", sharedgguf.MetaTypeUint32},
	{"num_key_value_heads", "{arch}.attention.head_count_kv", sharedgguf.MetaTypeUint32},
	{"intermediate_size", "{arch}.feed_forward_length", sharedgguf.MetaTypeUint32},
	{"vocab_size", "{arch}.vocab_size", sharedgguf.MetaTypeUint32},
	{"max_position_embeddings", "{arch}.context_length", sharedgguf.MetaTypeUint32},
	{"rms_norm_eps", "{arch}.attention.layer_norm_rms_epsilon", sharedgguf.MetaTypeFloat32},
	{"rope_theta", "{arch}.rope.freq_base", sharedgguf.MetaTypeFloat32},
}

// bertExtraMapping defines BERT-specific config keys not covered by configMapping.
var bertExtraMapping = []struct {
	hfKey    string
	ggufKey  string
	ggufType uint32
}{
	{"layer_norm_eps", "{arch}.attention.layer_norm_epsilon", sharedgguf.MetaTypeFloat32},
	{"num_labels", "{arch}.num_labels", sharedgguf.MetaTypeUint32},
}

// bertStaticMetadata defines BERT-specific metadata with fixed values.
var bertStaticMetadata = []MetadataEntry{
	{Key: "{arch}.pooler_type", Type: sharedgguf.MetaTypeString, Value: "cls"},
}

// MapMetadata converts HuggingFace/ONNX config fields to GGUF metadata entries.
func MapMetadata(arch string, config map[string]interface{}) []MetadataEntry {
	entries := []MetadataEntry{
		{Key: "general.architecture", Type: sharedgguf.MetaTypeString, Value: arch},
		{Key: "general.file_type", Type: sharedgguf.MetaTypeUint32, Value: uint32(0)}, // F32
	}

	for _, m := range configMapping {
		val, ok := config[m.hfKey]
		if !ok {
			continue
		}

		key := replaceArch(m.ggufKey, arch)

		switch m.ggufType {
		case sharedgguf.MetaTypeUint32:
			if u, err := toUint32(val); err == nil {
				entries = append(entries, MetadataEntry{Key: key, Type: sharedgguf.MetaTypeUint32, Value: u})
			}
		case sharedgguf.MetaTypeFloat32:
			if f, err := toFloat32(val); err == nil {
				entries = append(entries, MetadataEntry{Key: key, Type: sharedgguf.MetaTypeFloat32, Value: f})
			}
		}
	}

	if arch == "bert" {
		for _, m := range bertExtraMapping {
			val, ok := config[m.hfKey]
			if !ok {
				continue
			}
			key := replaceArch(m.ggufKey, arch)
			switch m.ggufType {
			case sharedgguf.MetaTypeUint32:
				if u, err := toUint32(val); err == nil {
					entries = append(entries, MetadataEntry{Key: key, Type: sharedgguf.MetaTypeUint32, Value: u})
				}
			case sharedgguf.MetaTypeFloat32:
				if f, err := toFloat32(val); err == nil {
					entries = append(entries, MetadataEntry{Key: key, Type: sharedgguf.MetaTypeFloat32, Value: f})
				}
			}
		}
		for _, m := range bertStaticMetadata {
			entries = append(entries, MetadataEntry{
				Key:   replaceArch(m.Key, arch),
				Type:  m.Type,
				Value: m.Value,
			})
		}
	}

	return entries
}

func replaceArch(pattern, arch string) string {
	result := make([]byte, 0, len(pattern))
	for i := 0; i < len(pattern); i++ {
		if i+5 < len(pattern) && pattern[i:i+6] == "{arch}" {
			result = append(result, arch...)
			i += 5
		} else {
			result = append(result, pattern[i])
		}
	}
	return string(result)
}

func toUint32(v interface{}) (uint32, error) {
	switch n := v.(type) {
	case float64:
		if n < 0 || n > math.MaxUint32 || n != math.Trunc(n) {
			return 0, fmt.Errorf("float64 %f out of uint32 range", n)
		}
		return uint32(n), nil
	case int:
		if n < 0 {
			return 0, fmt.Errorf("negative int %d", n)
		}
		return uint32(n), nil
	case int64:
		if n < 0 {
			return 0, fmt.Errorf("negative int64 %d", n)
		}
		return uint32(n), nil
	case uint32:
		return n, nil
	default:
		return 0, fmt.Errorf("unsupported type %T", v)
	}
}

func toFloat32(v interface{}) (float32, error) {
	switch n := v.(type) {
	case float64:
		return float32(n), nil
	case float32:
		return n, nil
	case int:
		return float32(n), nil
	case int64:
		return float32(n), nil
	default:
		return 0, fmt.Errorf("unsupported type %T", v)
	}
}
