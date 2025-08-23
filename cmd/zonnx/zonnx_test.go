package main_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestDownloadCommand(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "zonnx_download_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if cerr := os.RemoveAll(tempDir); cerr != nil {
			t.Errorf("Error removing temp dir %s: %v", tempDir, cerr)
		}
	}()

	// Create mock HuggingFace API server
	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := fmt.Fprint(w, `{
			"modelId": "test-org/test-model",
			"siblings": [
				{"rfilename": "model.onnx"},
				{"rfilename": "tokenizer.json"}
			]
		}`); err != nil {
			t.Errorf("Error writing to response writer: %v", err)
		}
	}))
	defer apiServer.Close()

	// Create mock HuggingFace CDN server
	cdnServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "model.onnx") {
			if _, err := fmt.Fprint(w, "mock onnx content"); err != nil {
				t.Errorf("Error writing to response writer: %v", err)
			}
		} else if strings.HasSuffix(r.URL.Path, "tokenizer.json") {
			if _, err := fmt.Fprint(w, "mock tokenizer content"); err != nil {
				t.Errorf("Error writing to response writer: %v", err)
			}
		} else {
			http.Error(w, "Not Found", http.StatusNotFound)
		}
	}))
	defer cdnServer.Close()

	// Build the zonnx executable
	cmd := exec.Command("go", "build", "-o", filepath.Join(tempDir, "zonnx"), "./cmd/zonnx")
	cmd.Dir = "/Users/dndungu/Code/dndungu/zerfoo/zonnx" // Explicitly set working directory
	cmd.Stderr = os.Stderr                               // Print build errors to stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed to build zonnx executable: %v", err)
	}

	zonnxPath := filepath.Join(tempDir, "zonnx")

	// Set environment variables to point to mock servers
	if err := os.Setenv("HUGGINGFACE_API_URL", apiServer.URL+"/"); err != nil {
		t.Fatalf("Failed to set HUGGINGFACE_API_URL: %v", err)
	}
	if err := os.Setenv("HUGGINGFACE_CDN_URL", cdnServer.URL+"/"); err != nil {
		t.Fatalf("Failed to set HUGGINGFACE_CDN_URL: %v", err)
	}
	defer func() {
		if err := os.Unsetenv("HUGGINGFACE_API_URL"); err != nil {
			t.Errorf("Failed to unset HUGGINGFACE_API_URL: %v", err)
		}
		if err := os.Unsetenv("HUGGINGFACE_CDN_URL"); err != nil {
			t.Errorf("Failed to unset HUGGINGFACE_CDN_URL: %v", err)
		}
	}()

	// Run the zonnx download command
	downloadCmd := exec.Command(zonnxPath, "download", "--model", "test-org/test-model", "--output", tempDir)
	output, err := downloadCmd.CombinedOutput()
	if err != nil {
		t.Fatalf("zonnx download command failed: %v\nOutput: %s", err, output)
	}

	// Verify downloaded files
	expectedModelPath := filepath.Join(tempDir, "model.onnx")
	expectedTokenizerPath := filepath.Join(tempDir, "tokenizer.json")

	if _, err := os.Stat(expectedModelPath); os.IsNotExist(err) {
		t.Errorf("ONNX model file not found: %s", expectedModelPath)
	}
	if _, err := os.Stat(expectedTokenizerPath); os.IsNotExist(err) {
		t.Errorf("Tokenizer file not found: %s", expectedTokenizerPath)
	}

	// Verify content (optional, but good for robustness)
	modelContent, err := os.ReadFile(expectedModelPath)
	if err != nil {
		t.Fatalf("Failed to read model file: %v", err)
	}
	if string(modelContent) != "mock onnx content" {
		t.Errorf("Model content mismatch: got \"%s\", want \"mock onnx content\"", string(modelContent))
	}

	tokenizerContent, err := os.ReadFile(expectedTokenizerPath)
	if err != nil {
		t.Fatalf("Failed to read tokenizer file: %v", err)
	}
	if string(tokenizerContent) != "mock tokenizer content" {
		t.Errorf("Tokenizer content mismatch: got \"%s\", want \"mock tokenizer content\"", string(tokenizerContent))
	}
}
