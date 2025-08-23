package downloader

import (
	"errors"
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
