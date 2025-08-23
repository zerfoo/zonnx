package downloader

import (
	"bytes"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// MockModelSource is a mock implementation of the ModelSource interface for testing.
type MockModelSource struct {
	mockDownloadModel func(modelID string, destination string) (*DownloadResult, error)
}

func (m *MockModelSource) DownloadModel(modelID string, destination string) (*DownloadResult, error) {
	if m.mockDownloadModel != nil {
		return m.mockDownloadModel(modelID, destination)
	}
	return nil, errors.New("DownloadModel not implemented for mock")
}

func TestNewDownloader(t *testing.T) {
	mockSource := &MockModelSource{}
	d := NewDownloader(mockSource)

	if d == nil {
		t.Fatal("NewDownloader returned nil")
	}
	if d.source != mockSource {
		t.Errorf("NewDownloader did not set the correct ModelSource")
	}
}

func TestDownloader_Download(t *testing.T) {
	tests := []struct {
		name          string
		modelID       string
		destination   string
		mockResult    *DownloadResult
		mockError     error
		expectedError bool
	}{
		{
			name:        "Successful download",
			modelID:     "test-model",
			destination: "/tmp/download",
			mockResult: &
				DownloadResult{
					ModelPath:      "/tmp/download/model.onnx",
					TokenizerPaths: []string{"/tmp/download/tokenizer.json"},
				},
			mockError:     nil,
			expectedError: false,
		},
		{
			name:          "Download with error",
			modelID:       "error-model",
			destination:   "/tmp/download",
			mockResult:    nil,
			mockError:     errors.New("mock download error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockSource := &MockModelSource{
				mockDownloadModel: func(modelID string, destination string) (*DownloadResult, error) {
					return tt.mockResult, tt.mockError
				},
			}
			d := NewDownloader(mockSource)

			result, err := d.Download(tt.modelID, tt.destination)

			if tt.expectedError {
				if err == nil {
					t.Errorf("Expected an error, but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, but got: %v", err)
				}
				if result == nil {
					t.Fatal("Expected a DownloadResult, but got nil")
				}
				if result.ModelPath != tt.mockResult.ModelPath {
					t.Errorf("Expected ModelPath %s, got %s", tt.mockResult.ModelPath, result.ModelPath)
				}
				if len(result.TokenizerPaths) != len(tt.mockResult.TokenizerPaths) {
					t.Errorf("Expected %d tokenizer paths, got %d", len(tt.mockResult.TokenizerPaths), len(result.TokenizerPaths))
				}
			}
		})
	}
}

func Test_downloadFile(t *testing.T) {
	// Create a temporary directory for testing downloads
	tempDir, err := os.MkdirTemp("", "download_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if cerr := os.RemoveAll(tempDir); cerr != nil {
			t.Errorf("Error removing temp dir %s: %v", tempDir, cerr)
		}
	}()


	tests := []struct {
		name           string
		serverHandler  http.HandlerFunc
		fileName       string
		expectedErrMsg string
	}{
		{
			name: "Successful download",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				if _, err := fmt.Fprint(w, "test content"); err != nil {
					t.Errorf("Error writing to response writer: %v", err)
				}
			},
			fileName:       "test.txt",
			expectedErrMsg: "",
		},
		{
			name: "HTTP error status",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "Not Found", http.StatusNotFound);
			},
			fileName:       "error.txt",
			expectedErrMsg: "status code 404",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(tt.serverHandler)
			defer server.Close()

			filePath := filepath.Join(tempDir, tt.fileName)
			err := downloadFile(server.URL, filePath)

			if tt.expectedErrMsg != "" {
				if err == nil || !bytes.Contains([]byte(err.Error()), []byte(tt.expectedErrMsg)) {
					t.Errorf("Expected error containing \"%s\", got \"%v\"", tt.expectedErrMsg, err)
				}
				// Ensure file was not created or is empty on error
				_, fileErr := os.Stat(filePath)
				if fileErr == nil || !os.IsNotExist(fileErr) {
					t.Errorf("File %s should not exist or be empty on error, but it does", filePath)
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, but got: %v", err)
				}
				content, readErr := os.ReadFile(filePath)
				if readErr != nil {
					t.Fatalf("Failed to read downloaded file: %v", readErr)
				}
				if string(content) != "test content" {
					t.Errorf("Downloaded content mismatch: got \"%s\", want \"test content\"", string(content))
				}
			}
		})
	}
}

func Test_copyFile(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		expectedBytes int64
		expectedError bool
	}{
		{
			name:          "Empty content",
			input:         "",
			expectedBytes: 0,
			expectedError: false,
		},
		{
			name:          "Some content",
			input:         "hello world",
			expectedBytes: 11,
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := bytes.NewBufferString(tt.input)
			dst := &bytes.Buffer{}

			n, err := copyFile(src, dst)

			if tt.expectedError {
				if err == nil {
					t.Errorf("Expected an error, but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, but got: %v", err)
				}
				if n != tt.expectedBytes {
					t.Errorf("Expected %d bytes copied, got %d", tt.expectedBytes, n)
				}
				if dst.String() != tt.input {
					t.Errorf("Copied content mismatch: got \"%s\", want \"%s\"", dst.String(), tt.input)
				}
			}
		})
	}
}