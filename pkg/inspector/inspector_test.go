package inspector

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zonnx/internal/onnx"
	"github.com/zerfoo/zmf"
	"google.golang.org/protobuf/proto"
)

// Helper function to create a dummy ONNX model file
func createDummyOnnxModel(t *testing.T, dir, filename string) string {
	model := &onnx.ModelProto{
		IrVersion: proto.Int64(4),
		OpsetImport: []*onnx.OperatorSetIdProto{
			{Version: proto.Int64(9)},
		},
		Graph: &onnx.GraphProto{
			Node: []*onnx.NodeProto{
				{Name: proto.String("node1"), OpType: proto.String("Add")},
				{Name: proto.String("node2"), OpType: proto.String("Mul")},
			},
		},
	}
	data, err := proto.Marshal(model)
	if err != nil {
		t.Fatalf("Failed to marshal dummy ONNX model: %v", err)
	}
	filePath := filepath.Join(dir, filename)
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		t.Fatalf("Failed to write dummy ONNX model: %v", err)
	}
	return filePath
}

// Helper function to create a dummy ZMF model file
func createDummyZmfModel(t *testing.T, dir, filename string) string {
	zmfModel := &zmf.Model{
		Metadata: &zmf.Metadata{
			ProducerName:    "test-producer",
			ProducerVersion: "1.0",
			OpsetVersion:    1,
		},
		Graph: &zmf.Graph{
			Nodes: []*zmf.Node{
				{Name: "zmf_node1", OpType: "Add"},
			},
			Parameters: make(map[string]*zmf.Tensor),
		},
	}
	data, err := proto.Marshal(zmfModel)
	if err != nil {
		t.Fatalf("Failed to marshal dummy ZMF model: %v", err)
	}
	filePath := filepath.Join(dir, filename)
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		t.Fatalf("Failed to write dummy ZMF model: %v", err)
	}
	return filePath
}

func TestInspectONNX(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "inspect_onnx_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if cerr := os.RemoveAll(tempDir); cerr != nil {
			t.Errorf("Error removing temp dir %s: %v", tempDir, cerr)
		}
	}()

	onnxFile := createDummyOnnxModel(t, tempDir, "test.onnx")

	// Capture stdout
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	inspectErr := InspectONNX(onnxFile)

	if cerr := w.Close(); cerr != nil {
		t.Errorf("Error closing writer: %v", cerr)
	}
	os.Stdout = oldStdout
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("Failed to read stdout: %v", err)
	}

	if inspectErr != nil {
		t.Errorf("InspectONNX returned an error: %v", inspectErr)
	}

	output := string(out)
	if !strings.Contains(output, "Inspecting ONNX model from:") {
		t.Errorf("Output missing expected string: %s", output)
	}
	if !strings.Contains(output, "Successfully loaded model with IR version: 4") {
		t.Errorf("Output missing expected IR version: %s", output)
	}
	if !strings.Contains(output, "Opset version: 9") {
		t.Errorf("Output missing expected Opset version: %s", output)
	}
	if !strings.Contains(output, "Graph has 2 nodes.") {
		t.Errorf("Output missing expected node count: %s", output)
	}
}

func TestInspectZMF(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "inspect_zmf_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if cerr := os.RemoveAll(tempDir); cerr != nil {
			t.Errorf("Error removing temp dir %s: %v", tempDir, cerr)
		}
	}()

	zmfFile := createDummyZmfModel(t, tempDir, "test.zmf")

	// Capture stdout
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	inspectErr := InspectZMF(zmfFile)

	if cerr := w.Close(); cerr != nil {
		t.Errorf("Error closing writer: %v", cerr)
	}
	os.Stdout = oldStdout
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("Failed to read stdout: %v", err)
	}

	if inspectErr != nil {
		t.Errorf("InspectZMF returned an error: %v", inspectErr)
	}

	output := string(out)
	if !strings.Contains(output, "Inspecting ZMF model from:") {
		t.Errorf("Output missing expected string: %s", output)
	}
	if !strings.Contains(output, "Producer: test-producer 1.0") {
		t.Errorf("Output missing expected producer info: %s", output)
	}
	if !strings.Contains(output, "Opset version: 1") {
		t.Errorf("Output missing expected opset version: %s", output)
	}
	if !strings.Contains(output, "Graph has 1 nodes.") {
		t.Errorf("Output missing expected node count: %s", output)
	}
	if !strings.Contains(output, "Graph has 0 parameters.") {
		t.Errorf("Output missing expected parameter count: %s", output)
	}
	if !strings.Contains(output, "- Node: zmf_node1, OpType: Add") {
		t.Errorf("Output missing expected node details: %s", output)
	}
}