package gguf

import (
	"regexp"
)

// layerPattern matches "model.layers.N." and captures the layer number.
var layerPattern = regexp.MustCompile(`^model\.layers\.(\d+)\.(.+)$`)

// bertLayerPattern matches "bert.encoder.layer.N." and "roberta.encoder.layer.N."
// and captures the layer number and suffix.
var bertLayerPattern = regexp.MustCompile(`^(?:bert|roberta)\.encoder\.layer\.(\d+)\.(.+)$`)

// staticMappings maps non-layer HuggingFace tensor names to GGUF names.
var staticMappings = map[string]string{
	// Llama-style
	"model.embed_tokens.weight": "token_embd.weight",
	"model.norm.weight":         "output_norm.weight",
	"lm_head.weight":            "output.weight",

	// BERT
	"bert.embeddings.word_embeddings.weight":     "token_embd.weight",
	"bert.embeddings.position_embeddings.weight": "position_embd.weight",
	"bert.embeddings.token_type_embeddings.weight": "token_type_embd.weight",
	"bert.embeddings.LayerNorm.weight":           "token_embd_norm.weight",
	"bert.embeddings.LayerNorm.bias":             "token_embd_norm.bias",
	"bert.pooler.dense.weight":                   "cls_pooler.weight",
	"bert.pooler.dense.bias":                     "cls_pooler.bias",
	"classifier.weight":                          "cls.weight",
	"classifier.bias":                            "cls.bias",

	// RoBERTa (same structure, different prefix)
	"roberta.embeddings.word_embeddings.weight":     "token_embd.weight",
	"roberta.embeddings.position_embeddings.weight": "position_embd.weight",
	"roberta.embeddings.token_type_embeddings.weight": "token_type_embd.weight",
	"roberta.embeddings.LayerNorm.weight":           "token_embd_norm.weight",
	"roberta.embeddings.LayerNorm.bias":             "token_embd_norm.bias",
	"roberta.pooler.dense.weight":                   "cls_pooler.weight",
	"roberta.pooler.dense.bias":                     "cls_pooler.bias",
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

// bertLayerSuffixMappings maps BERT per-layer suffixes to GGUF suffixes.
var bertLayerSuffixMappings = map[string]string{
	"attention.self.query.weight":        "attn_q.weight",
	"attention.self.query.bias":          "attn_q.bias",
	"attention.self.key.weight":          "attn_k.weight",
	"attention.self.key.bias":            "attn_k.bias",
	"attention.self.value.weight":        "attn_v.weight",
	"attention.self.value.bias":          "attn_v.bias",
	"attention.output.dense.weight":      "attn_output.weight",
	"attention.output.dense.bias":        "attn_output.bias",
	"attention.output.LayerNorm.weight":  "attn_norm.weight",
	"attention.output.LayerNorm.bias":    "attn_norm.bias",
	"intermediate.dense.weight":          "ffn_up.weight",
	"intermediate.dense.bias":            "ffn_up.bias",
	"output.dense.weight":               "ffn_down.weight",
	"output.dense.bias":                  "ffn_down.bias",
	"output.LayerNorm.weight":            "ffn_norm.weight",
	"output.LayerNorm.bias":              "ffn_norm.bias",
}

// MapTensorName converts a HuggingFace tensor name to its GGUF equivalent.
// If no mapping matches, the name is returned unchanged.
func MapTensorName(name string) string {
	if gguf, ok := staticMappings[name]; ok {
		return gguf
	}

	// Try Llama-style layer pattern.
	if m := layerPattern.FindStringSubmatch(name); m != nil {
		if ggufSuffix, ok := layerSuffixMappings[m[2]]; ok {
			return "blk." + m[1] + "." + ggufSuffix
		}
	}

	// Try BERT/RoBERTa layer pattern.
	if m := bertLayerPattern.FindStringSubmatch(name); m != nil {
		if ggufSuffix, ok := bertLayerSuffixMappings[m[2]]; ok {
			return "blk." + m[1] + "." + ggufSuffix
		}
	}

	return name
}
