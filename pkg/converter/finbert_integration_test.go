package converter

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestConvertFinBERTSafetensorsToGGUF creates a synthetic FinBERT model
// with the correct architecture (12 layers, 768 hidden, 3 classes) and
// verifies end-to-end safetensors-to-GGUF conversion.
func TestConvertFinBERTSafetensorsToGGUF(t *testing.T) {
	dir := t.TempDir()

	// FinBERT config.json matching ProsusAI/finbert.
	config := map[string]interface{}{
		"architectures":          []string{"BertForSequenceClassification"},
		"attention_probs_dropout_prob": 0.1,
		"hidden_act":             "gelu",
		"hidden_dropout_prob":    0.1,
		"hidden_size":            768,
		"initializer_range":      0.02,
		"intermediate_size":      3072,
		"layer_norm_eps":         1e-12,
		"max_position_embeddings": 512,
		"model_type":             "bert",
		"num_attention_heads":    12,
		"num_hidden_layers":      12,
		"num_labels":             3,
		"type_vocab_size":        2,
		"vocab_size":             30522,
	}
	configJSON, _ := json.MarshalIndent(config, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, "config.json"), configJSON, 0o644); err != nil {
		t.Fatal(err)
	}

	// Build synthetic safetensors with all expected FinBERT tensors.
	// We use tiny (1-element) data to keep the test fast; shapes are correct.
	type tensorDef struct {
		name  string
		shape []uint64
	}

	defs := []tensorDef{
		// Embeddings
		{"bert.embeddings.word_embeddings.weight", []uint64{30522, 768}},
		{"bert.embeddings.position_embeddings.weight", []uint64{512, 768}},
		{"bert.embeddings.token_type_embeddings.weight", []uint64{2, 768}},
		{"bert.embeddings.LayerNorm.weight", []uint64{768}},
		{"bert.embeddings.LayerNorm.bias", []uint64{768}},
		// Pooler
		{"bert.pooler.dense.weight", []uint64{768, 768}},
		{"bert.pooler.dense.bias", []uint64{768}},
		// Classifier
		{"classifier.weight", []uint64{3, 768}},
		{"classifier.bias", []uint64{3}},
	}

	// Add 12 encoder layers.
	layerSuffixes := []struct {
		suffix string
		shape  []uint64
	}{
		{"attention.self.query.weight", []uint64{768, 768}},
		{"attention.self.query.bias", []uint64{768}},
		{"attention.self.key.weight", []uint64{768, 768}},
		{"attention.self.key.bias", []uint64{768}},
		{"attention.self.value.weight", []uint64{768, 768}},
		{"attention.self.value.bias", []uint64{768}},
		{"attention.output.dense.weight", []uint64{768, 768}},
		{"attention.output.dense.bias", []uint64{768}},
		{"attention.output.LayerNorm.weight", []uint64{768}},
		{"attention.output.LayerNorm.bias", []uint64{768}},
		{"intermediate.dense.weight", []uint64{3072, 768}},
		{"intermediate.dense.bias", []uint64{3072}},
		{"output.dense.weight", []uint64{768, 3072}},
		{"output.dense.bias", []uint64{768}},
		{"output.LayerNorm.weight", []uint64{768}},
		{"output.LayerNorm.bias", []uint64{768}},
	}

	for layer := range 12 {
		for _, s := range layerSuffixes {
			name := "bert.encoder.layer." + itoa(layer) + "." + s.suffix
			defs = append(defs, tensorDef{name, s.shape})
		}
	}

	// Build safetensors binary. Use minimal data (all zeros) with correct byte sizes.
	var dataBuf bytes.Buffer
	header := make(map[string]safetensorsTensorInfo, len(defs))

	for _, d := range defs {
		numElements := uint64(1)
		for _, dim := range d.shape {
			numElements *= dim
		}
		start := uint64(dataBuf.Len())
		// Write numElements float32 zeros.
		zeros := make([]byte, numElements*4) // 4 bytes per float32
		dataBuf.Write(zeros)
		end := uint64(dataBuf.Len())
		header[d.name] = safetensorsTensorInfo{
			Dtype:       dtypeF32,
			Shape:       d.shape,
			DataOffsets: [2]uint64{start, end},
		}
	}

	headerJSON, _ := json.Marshal(header)
	var stBuf bytes.Buffer
	binary.Write(&stBuf, binary.LittleEndian, uint64(len(headerJSON)))
	stBuf.Write(headerJSON)
	stBuf.Write(dataBuf.Bytes())

	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), stBuf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	// Convert.
	outputPath := filepath.Join(dir, "finbert.gguf")
	if err := ConvertSafetensorsToGGUF(dir, outputPath, "bert"); err != nil {
		t.Fatalf("convert: %v", err)
	}

	info, err := os.Stat(outputPath)
	if err != nil {
		t.Fatalf("stat output: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output GGUF is empty")
	}

	// Verify the GGUF was written correctly by parsing the header.
	verifyGGUFHeader(t, outputPath, len(defs))
}

// verifyGGUFHeader does a minimal GGUF header check: magic, version, tensor count.
func verifyGGUFHeader(t *testing.T, path string, expectedTensors int) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()

	var magic, version uint32
	var tensorCount, metadataCount uint64

	binary.Read(f, binary.LittleEndian, &magic)
	binary.Read(f, binary.LittleEndian, &version)
	binary.Read(f, binary.LittleEndian, &tensorCount)
	binary.Read(f, binary.LittleEndian, &metadataCount)

	if magic != 0x46554747 {
		t.Errorf("bad magic: 0x%08X", magic)
	}
	if version != 3 {
		t.Errorf("expected version 3, got %d", version)
	}
	if int(tensorCount) != expectedTensors {
		t.Errorf("expected %d tensors, got %d", expectedTensors, tensorCount)
	}
	// FinBERT BERT metadata should have at least:
	// general.architecture, general.file_type, embedding_length, block_count,
	// head_count, feed_forward_length, vocab_size, context_length,
	// layer_norm_epsilon, num_labels, pooler_type
	if metadataCount < 10 {
		t.Errorf("expected at least 10 metadata entries, got %d", metadataCount)
	}
}

func itoa(i int) string {
	buf := make([]byte, 0, 3)
	if i == 0 {
		return "0"
	}
	for i > 0 {
		buf = append([]byte{byte('0' + i%10)}, buf...)
		i /= 10
	}
	return string(buf)
}
