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

		// BERT static mappings
		{"bert.embeddings.word_embeddings.weight", "token_embd.weight"},
		{"bert.embeddings.position_embeddings.weight", "position_embd.weight"},
		{"bert.embeddings.token_type_embeddings.weight", "token_type_embd.weight"},
		{"bert.embeddings.LayerNorm.weight", "token_embd_norm.weight"},
		{"bert.embeddings.LayerNorm.bias", "token_embd_norm.bias"},
		{"bert.pooler.dense.weight", "cls_pooler.weight"},
		{"bert.pooler.dense.bias", "cls_pooler.bias"},
		{"classifier.weight", "cls.weight"},
		{"classifier.bias", "cls.bias"},

		// BERT layer mappings
		{"bert.encoder.layer.0.attention.self.query.weight", "blk.0.attn_q.weight"},
		{"bert.encoder.layer.0.attention.self.query.bias", "blk.0.attn_q.bias"},
		{"bert.encoder.layer.0.attention.self.key.weight", "blk.0.attn_k.weight"},
		{"bert.encoder.layer.0.attention.self.key.bias", "blk.0.attn_k.bias"},
		{"bert.encoder.layer.0.attention.self.value.weight", "blk.0.attn_v.weight"},
		{"bert.encoder.layer.0.attention.self.value.bias", "blk.0.attn_v.bias"},
		{"bert.encoder.layer.0.attention.output.dense.weight", "blk.0.attn_output.weight"},
		{"bert.encoder.layer.0.attention.output.dense.bias", "blk.0.attn_output.bias"},
		{"bert.encoder.layer.0.attention.output.LayerNorm.weight", "blk.0.attn_norm.weight"},
		{"bert.encoder.layer.0.attention.output.LayerNorm.bias", "blk.0.attn_norm.bias"},
		{"bert.encoder.layer.0.intermediate.dense.weight", "blk.0.ffn_up.weight"},
		{"bert.encoder.layer.0.intermediate.dense.bias", "blk.0.ffn_up.bias"},
		{"bert.encoder.layer.0.output.dense.weight", "blk.0.ffn_down.weight"},
		{"bert.encoder.layer.0.output.dense.bias", "blk.0.ffn_down.bias"},
		{"bert.encoder.layer.0.output.LayerNorm.weight", "blk.0.ffn_norm.weight"},
		{"bert.encoder.layer.0.output.LayerNorm.bias", "blk.0.ffn_norm.bias"},
		{"bert.encoder.layer.11.attention.self.query.weight", "blk.11.attn_q.weight"},
		{"bert.encoder.layer.23.output.dense.weight", "blk.23.ffn_down.weight"},

		// RoBERTa static mappings
		{"roberta.embeddings.word_embeddings.weight", "token_embd.weight"},
		{"roberta.embeddings.position_embeddings.weight", "position_embd.weight"},
		{"roberta.embeddings.LayerNorm.weight", "token_embd_norm.weight"},
		{"roberta.pooler.dense.weight", "cls_pooler.weight"},

		// RoBERTa layer mappings
		{"roberta.encoder.layer.0.attention.self.query.weight", "blk.0.attn_q.weight"},
		{"roberta.encoder.layer.5.intermediate.dense.weight", "blk.5.ffn_up.weight"},

		// Unknown names pass through unchanged
		{"some.unknown.tensor", "some.unknown.tensor"},
		{"model.layers.0.unknown_suffix.weight", "model.layers.0.unknown_suffix.weight"},
		{"bert.encoder.layer.0.unknown_suffix.weight", "bert.encoder.layer.0.unknown_suffix.weight"},
	}

	for _, tt := range tests {
		got := MapTensorName(tt.input)
		if got != tt.want {
			t.Errorf("MapTensorName(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}
