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

	// Test cases for download command
	tests := []struct {
		name           string
		modelID        string
		outputPath     string
		apiKey         string // API key passed via flag
		envApiKey      string // API key passed via environment variable
		apiHandler     http.HandlerFunc
		cdnHandler     http.HandlerFunc
		expectedModel  string
		expectedTokens []string
		expectedError  string
	}{
		{
			name:       "Successful public download",
			modelID:    "test-org/public-model",
			outputPath: tempDir,
			apiKey:     "",
			envApiKey:  "",
			apiHandler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"modelId": "test-org/public-model","siblings": [{"rfilename": "model.onnx"},{"rfilename": "tokenizer.json"}]}`)
			},
			cdnHandler: func(w http.ResponseWriter, r *http.Request) {
				if strings.HasSuffix(r.URL.Path, "model.onnx") {
					_, _ = fmt.Fprint(w, "mock onnx content")
				} else if strings.HasSuffix(r.URL.Path, "tokenizer.json") {
					_, _ = fmt.Fprint(w, "mock tokenizer content")
				} else {
					http.Error(w, "Not Found", http.StatusNotFound)
				}
			},
			expectedModel:  "model.onnx",
			expectedTokens: []string{"tokenizer.json"},
			expectedError:  "",
		},
		{
			name:       "Successful authenticated download via flag",
			modelID:    "test-org/private-model-flag",
			outputPath: tempDir,
			apiKey:     "test-api-key-flag",
			envApiKey:  "",
			apiHandler: func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "Bearer test-api-key-flag" {
					http.Error(w, "Unauthorized", http.StatusUnauthorized)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_,_ = fmt.Fprint(w, `{"modelId": "test-org/private-model-flag","siblings": [{"rfilename": "model.onnx"},{"rfilename": "tokenizer.json"}]}`)
			},
			cdnHandler: func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "Bearer test-api-key-flag" {
					http.Error(w, "Unauthorized", http.StatusUnauthorized)
					return
				}
				if strings.HasSuffix(r.URL.Path, "model.onnx") {
					_, _ = fmt.Fprint(w, "authenticated onnx content")
				} else if strings.HasSuffix(r.URL.Path, "tokenizer.json") {
					_, _ = fmt.Fprint(w, "authenticated tokenizer content")
				} else {
					http.Error(w, "Not Found", http.StatusNotFound)
				}
			},
			expectedModel:  "model.onnx",
			expectedTokens: []string{"tokenizer.json"},
			expectedError:  "",
		},
		{
			name:       "Successful authenticated download via env var",
			modelID:    "test-org/private-model-env",
			outputPath: tempDir,
			apiKey:     "", // No flag
			envApiKey:  "test-api-key-env",
			apiHandler: func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "Bearer test-api-key-env" {
					http.Error(w, "Unauthorized", http.StatusUnauthorized)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"modelId": "test-org/private-model-env","siblings": [{"rfilename": "model.onnx"},{"rfilename": "tokenizer.json"}]}`)
			},
			cdnHandler: func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "Bearer test-api-key-env" {
					http.Error(w, "Unauthorized", http.StatusUnauthorized)
					return
				}
				if strings.HasSuffix(r.URL.Path, "model.onnx") {
					_, _ = fmt.Fprint(w, "authenticated onnx content")
				} else if strings.HasSuffix(r.URL.Path, "tokenizer.json") {
					_, _ = fmt.Fprint(w, "authenticated tokenizer content")
				} else {
					http.Error(w, "Not Found", http.StatusNotFound)
				}
			},
			expectedModel:  "model.onnx",
			expectedTokens: []string{"tokenizer.json"},
			expectedError:  "",
		},
		{
			name:       "Authenticated download unauthorized",
			modelID:    "test-org/unauthorized-model",
			outputPath: tempDir,
			apiKey:     "wrong-api-key",
			envApiKey:  "",
			apiHandler: func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
			},
			cdnHandler:     nil, // Not used in this case
			expectedModel:  "",
			expectedTokens: nil,
			expectedError:  "HuggingFace API returned non-OK status: 401 Unauthorized",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock API server
			apiServer := httptest.NewServer(tt.apiHandler)
			defer apiServer.Close()

			// Create mock CDN server
			var cdnServer *httptest.Server
			if tt.cdnHandler != nil {
				cdnServer = httptest.NewServer(tt.cdnHandler)
			} else {
				cdnServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					http.Error(w, "Not Found", http.StatusNotFound)
				}))
			}
			defer cdnServer.Close()

			// Temporarily override constants for testing
			if err := os.Setenv("HUGGINGFACE_API_URL", apiServer.URL+"/"); err != nil {
				t.Fatalf("Failed to set HUGGINGFACE_API_URL: %v", err)
			}
			if err := os.Setenv("HUGGINGFACE_CDN_URL", cdnServer.URL+"/"); err != nil {
				t.Fatalf("Failed to set HUGGINGFACE_CDN_URL: %v", err)
			}
			// Set HF_API_KEY environment variable if provided in test case
			if tt.envApiKey != "" {
				if err := os.Setenv("HF_API_KEY", tt.envApiKey); err != nil {
					t.Fatalf("Failed to set HF_API_KEY: %v", err)
				}
			}

			defer func() {
				if err := os.Unsetenv("HUGGINGFACE_API_URL"); err != nil {
					t.Errorf("Failed to unset HUGGINGFACE_API_URL: %v", err)
				}
				if err := os.Unsetenv("HUGGINGFACE_CDN_URL"); err != nil {
					t.Errorf("Failed to unset HUGGINGFACE_CDN_URL: %v", err)
				}
				if tt.envApiKey != "" {
					if err := os.Unsetenv("HF_API_KEY"); err != nil {
						t.Errorf("Failed to unset HF_API_KEY: %v", err)
					}
				}
			}()

			// Build the zonnx executable
			cmd := exec.Command("go", "build", "-o", filepath.Join(tempDir, "zonnx"), "./cmd/zonnx")
			cmd.Dir = "/Users/dndungu/Code/dndungu/zerfoo/zonnx" // Explicitly set working directory
			cmd.Stderr = os.Stderr
			if err := cmd.Run(); err != nil {
				t.Fatalf("Failed to build zonnx executable: %v", err)
			}

			zonnxPath := filepath.Join(tempDir, "zonnx")

			// Prepare command arguments
			args := []string{"download", "--model", tt.modelID, "--output", tt.outputPath}
			if tt.apiKey != "" {
				args = append(args, "--api-key", tt.apiKey)
			}

			// Run the zonnx download command
			downloadCmd := exec.Command(zonnxPath, args...)
			output, err := downloadCmd.CombinedOutput()

			if tt.expectedError != "" {
				if err == nil || !strings.Contains(string(output), tt.expectedError) {
					t.Errorf("Expected error containing \"%s\", but got: %v\nOutput: %s", tt.expectedError, err, output)
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, but got: %v\nOutput: %s", err, output)
				}

				expectedModelPath := filepath.Join(tt.outputPath, tt.expectedModel)
				if _, err := os.Stat(expectedModelPath); os.IsNotExist(err) {
					t.Errorf("ONNX model file not found: %s", expectedModelPath)
				}

				for _, expectedToken := range tt.expectedTokens {
					expectedTokenPath := filepath.Join(tt.outputPath, expectedToken)
					if _, err := os.Stat(expectedTokenPath); os.IsNotExist(err) {
						t.Errorf("Tokenizer file not found: %s", expectedTokenPath)
					}
				}
			}
		})
	}
}
