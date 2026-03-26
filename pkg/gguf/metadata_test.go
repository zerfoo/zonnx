package gguf

import (
	"testing"

	sharedgguf "github.com/zerfoo/ztensor/gguf"
)

func TestMapMetadata_AllFields(t *testing.T) {
	config := map[string]interface{}{
		"hidden_size":             float64(4096),
		"num_hidden_layers":       float64(32),
		"num_attention_heads":     float64(32),
		"num_key_value_heads":     float64(8),
		"intermediate_size":       float64(11008),
		"vocab_size":              float64(32000),
		"max_position_embeddings": float64(2048),
		"rms_norm_eps":            float64(1e-5),
		"rope_theta":              float64(10000.0),
	}

	entries := MapMetadata("llama", config)

	expected := map[string]MetadataEntry{
		"general.architecture":                   {Type: sharedgguf.MetaTypeString, Value: "llama"},
		"general.file_type":                      {Type: sharedgguf.MetaTypeUint32, Value: uint32(0)},
		"llama.embedding_length":                 {Type: sharedgguf.MetaTypeUint32, Value: uint32(4096)},
		"llama.block_count":                      {Type: sharedgguf.MetaTypeUint32, Value: uint32(32)},
		"llama.attention.head_count":              {Type: sharedgguf.MetaTypeUint32, Value: uint32(32)},
		"llama.attention.head_count_kv":           {Type: sharedgguf.MetaTypeUint32, Value: uint32(8)},
		"llama.feed_forward_length":              {Type: sharedgguf.MetaTypeUint32, Value: uint32(11008)},
		"llama.vocab_size":                       {Type: sharedgguf.MetaTypeUint32, Value: uint32(32000)},
		"llama.context_length":                   {Type: sharedgguf.MetaTypeUint32, Value: uint32(2048)},
		"llama.attention.layer_norm_rms_epsilon":  {Type: sharedgguf.MetaTypeFloat32, Value: float32(1e-5)},
		"llama.rope.freq_base":                   {Type: sharedgguf.MetaTypeFloat32, Value: float32(10000.0)},
	}

	if len(entries) != len(expected) {
		t.Fatalf("expected %d entries, got %d", len(expected), len(entries))
	}

	entryMap := make(map[string]MetadataEntry)
	for _, e := range entries {
		entryMap[e.Key] = e
	}

	for key, want := range expected {
		got, ok := entryMap[key]
		if !ok {
			t.Errorf("missing entry for key %q", key)
			continue
		}
		if got.Type != want.Type {
			t.Errorf("key %q: type = %d, want %d", key, got.Type, want.Type)
		}
		switch wv := want.Value.(type) {
		case uint32:
			if gv, ok := got.Value.(uint32); !ok || gv != wv {
				t.Errorf("key %q: value = %v, want %v", key, got.Value, wv)
			}
		case float32:
			if gv, ok := got.Value.(float32); !ok || gv != wv {
				t.Errorf("key %q: value = %v, want %v", key, got.Value, wv)
			}
		case string:
			if gv, ok := got.Value.(string); !ok || gv != wv {
				t.Errorf("key %q: value = %v, want %v", key, got.Value, wv)
			}
		}
	}
}

func TestMapMetadata_EmptyConfig(t *testing.T) {
	entries := MapMetadata("llama", map[string]interface{}{})

	if len(entries) != 2 {
		t.Fatalf("expected 2 entries (general.architecture + general.file_type), got %d", len(entries))
	}
	if entries[0].Key != "general.architecture" || entries[0].Value != "llama" {
		t.Errorf("first entry = %+v, want general.architecture=llama", entries[0])
	}
	if entries[1].Key != "general.file_type" || entries[1].Value != uint32(0) {
		t.Errorf("second entry = %+v, want general.file_type=0", entries[1])
	}
}

func TestMapMetadata_DifferentArch(t *testing.T) {
	config := map[string]interface{}{
		"hidden_size": float64(768),
	}

	entries := MapMetadata("gemma", config)

	entryMap := make(map[string]MetadataEntry)
	for _, e := range entries {
		entryMap[e.Key] = e
	}

	if e, ok := entryMap["gemma.embedding_length"]; !ok {
		t.Error("missing gemma.embedding_length")
	} else if e.Value != uint32(768) {
		t.Errorf("gemma.embedding_length = %v, want 768", e.Value)
	}

	if e, ok := entryMap["general.architecture"]; !ok || e.Value != "gemma" {
		t.Errorf("general.architecture = %v, want gemma", e.Value)
	}
}

func TestMapMetadata_BERT(t *testing.T) {
	config := map[string]interface{}{
		"hidden_size":             float64(768),
		"num_hidden_layers":       float64(12),
		"num_attention_heads":     float64(12),
		"intermediate_size":       float64(3072),
		"vocab_size":              float64(30522),
		"max_position_embeddings": float64(512),
		"layer_norm_eps":          float64(1e-12),
		"num_labels":              float64(3),
	}

	entries := MapMetadata("bert", config)

	entryMap := make(map[string]MetadataEntry)
	for _, e := range entries {
		entryMap[e.Key] = e
	}

	expected := map[string]MetadataEntry{
		"general.architecture":              {Type: sharedgguf.MetaTypeString, Value: "bert"},
		"general.file_type":                 {Type: sharedgguf.MetaTypeUint32, Value: uint32(0)},
		"bert.embedding_length":             {Type: sharedgguf.MetaTypeUint32, Value: uint32(768)},
		"bert.block_count":                  {Type: sharedgguf.MetaTypeUint32, Value: uint32(12)},
		"bert.attention.head_count":          {Type: sharedgguf.MetaTypeUint32, Value: uint32(12)},
		"bert.feed_forward_length":          {Type: sharedgguf.MetaTypeUint32, Value: uint32(3072)},
		"bert.vocab_size":                   {Type: sharedgguf.MetaTypeUint32, Value: uint32(30522)},
		"bert.context_length":               {Type: sharedgguf.MetaTypeUint32, Value: uint32(512)},
		"bert.attention.layer_norm_epsilon":  {Type: sharedgguf.MetaTypeFloat32, Value: float32(1e-12)},
		"bert.num_labels":                   {Type: sharedgguf.MetaTypeUint32, Value: uint32(3)},
		"bert.pooler_type":                  {Type: sharedgguf.MetaTypeString, Value: "cls"},
	}

	if len(entries) != len(expected) {
		t.Fatalf("expected %d entries, got %d", len(expected), len(entries))
	}

	for key, want := range expected {
		got, ok := entryMap[key]
		if !ok {
			t.Errorf("missing entry for key %q", key)
			continue
		}
		if got.Type != want.Type {
			t.Errorf("key %q: type = %d, want %d", key, got.Type, want.Type)
		}
		switch wv := want.Value.(type) {
		case uint32:
			if gv, ok := got.Value.(uint32); !ok || gv != wv {
				t.Errorf("key %q: value = %v, want %v", key, got.Value, wv)
			}
		case float32:
			if gv, ok := got.Value.(float32); !ok || gv != wv {
				t.Errorf("key %q: value = %v, want %v", key, got.Value, wv)
			}
		case string:
			if gv, ok := got.Value.(string); !ok || gv != wv {
				t.Errorf("key %q: value = %v, want %v", key, got.Value, wv)
			}
		}
	}
}

func TestMapMetadata_PartialConfig(t *testing.T) {
	config := map[string]interface{}{
		"hidden_size":       float64(2048),
		"num_hidden_layers": float64(16),
	}

	entries := MapMetadata("llama", config)

	// 2 general + 2 from config
	if len(entries) != 4 {
		t.Fatalf("expected 4 entries, got %d", len(entries))
	}
}
