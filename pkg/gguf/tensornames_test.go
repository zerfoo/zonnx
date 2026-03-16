package gguf

import "testing"

func TestMapTensorName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		// Static mappings
		{"model.embed_tokens.weight", "token_embd.weight"},
		{"model.norm.weight", "output_norm.weight"},
		{"lm_head.weight", "output.weight"},

		// Layer 0
		{"model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"},
		{"model.layers.0.self_attn.k_proj.weight", "blk.0.attn_k.weight"},
		{"model.layers.0.self_attn.v_proj.weight", "blk.0.attn_v.weight"},
		{"model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight"},
		{"model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"},
		{"model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"},
		{"model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"},
		{"model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight"},
		{"model.layers.0.post_attention_layernorm.weight", "blk.0.ffn_norm.weight"},

		// Higher layer numbers
		{"model.layers.31.self_attn.q_proj.weight", "blk.31.attn_q.weight"},
		{"model.layers.127.mlp.down_proj.weight", "blk.127.ffn_down.weight"},

		// Unknown names pass through unchanged
		{"some.unknown.tensor", "some.unknown.tensor"},
		{"model.layers.0.unknown_suffix.weight", "model.layers.0.unknown_suffix.weight"},
	}

	for _, tt := range tests {
		got := MapTensorName(tt.input)
		if got != tt.want {
			t.Errorf("MapTensorName(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}
