package gguf

import (
	"regexp"
)

// layerPattern matches "model.layers.N." and captures the layer number.
var layerPattern = regexp.MustCompile(`^model\.layers\.(\d+)\.(.+)$`)

// staticMappings maps non-layer HuggingFace tensor names to GGUF names.
var staticMappings = map[string]string{
	"model.embed_tokens.weight": "token_embd.weight",
	"model.norm.weight":         "output_norm.weight",
	"lm_head.weight":            "output.weight",
}

// layerSuffixMappings maps per-layer HuggingFace suffixes to GGUF suffixes.
var layerSuffixMappings = map[string]string{
	"self_attn.q_proj.weight":            "attn_q.weight",
	"self_attn.k_proj.weight":            "attn_k.weight",
	"self_attn.v_proj.weight":            "attn_v.weight",
	"self_attn.o_proj.weight":            "attn_output.weight",
	"mlp.gate_proj.weight":               "ffn_gate.weight",
	"mlp.up_proj.weight":                 "ffn_up.weight",
	"mlp.down_proj.weight":               "ffn_down.weight",
	"input_layernorm.weight":             "attn_norm.weight",
	"post_attention_layernorm.weight":     "ffn_norm.weight",
}

// MapTensorName converts a HuggingFace tensor name to its GGUF equivalent.
// If no mapping matches, the name is returned unchanged.
func MapTensorName(name string) string {
	if gguf, ok := staticMappings[name]; ok {
		return gguf
	}

	m := layerPattern.FindStringSubmatch(name)
	if m == nil {
		return name
	}

	layerNum := m[1]
	suffix := m[2]

	if ggufSuffix, ok := layerSuffixMappings[suffix]; ok {
		return "blk." + layerNum + "." + ggufSuffix
	}

	return name
}
