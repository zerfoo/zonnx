package downloader

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os" // Added for os.Getenv
	"path/filepath"
	"strings"
)

// Changed to var to allow overriding for testing
var (
	huggingFaceAPI = "https://huggingface.co/api/models/"
	huggingFaceCDN = "https://huggingface.co/" // Base URL for direct file downloads
)

func init() {
	if apiURL := os.Getenv("HUGGINGFACE_API_URL"); apiURL != "" {
		huggingFaceAPI = apiURL
	}
	if cdnURL := os.Getenv("HUGGINGFACE_CDN_URL"); cdnURL != "" {
		huggingFaceCDN = cdnURL
	}
}

// ModelSource defines the interface for a model source, such as HuggingFace.
// It provides methods to download a model and its associated files.
type ModelSource interface {
	// DownloadModel downloads the specified model and its associated files
	// to the given destination. It returns a DownloadResult containing the
	// paths to the downloaded files, or an error if the download fails.
	DownloadModel(modelID string, destination string) (*DownloadResult, error)
}

// DownloadResult contains the paths to the downloaded model and tokenizer files.
type DownloadResult struct {
	ModelPath      string
	TokenizerPaths []string
}

// Downloader handles the overall download process using a ModelSource.
type Downloader struct {
	source ModelSource
}

// NewDownloader creates a new Downloader with the given ModelSource.
func NewDownloader(source ModelSource) *Downloader {
	return &Downloader{source: source}
}

// Download orchestrates the download of a model and its associated files
// using the configured ModelSource.
func (d *Downloader) Download(modelID string, destination string) (*DownloadResult, error) {
	return d.source.DownloadModel(modelID, destination)
}

// downloadFile downloads a single file from a URL to a local path.
func downloadFile(url, filePath string) error {
	// Create the directory if it doesn't exist
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download file from %s: %w", url, err)
	}

	if resp.StatusCode != http.StatusOK {
		// Close the body before returning the error
		if cerr := resp.Body.Close(); cerr != nil {
			fmt.Fprintf(os.Stderr, "Error closing response body after non-OK status for %s: %v\n", url, cerr)
		}
		return fmt.Errorf("failed to download file from %s: status code %s", url, resp.Status) // Fixed govet
	}

	out, err := os.Create(filePath)
	if err != nil {
		// Close the body before returning the error
		if cerr := resp.Body.Close(); cerr != nil {
			fmt.Fprintf(os.Stderr, "Error closing response body after file creation error for %s: %v\n", url, cerr)
		}
		return fmt.Errorf("failed to create file %s: %w", filePath, err)
	}

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		// Close both body and file before returning the error
		if cerr := resp.Body.Close(); cerr != nil {
			fmt.Fprintf(os.Stderr, "Error closing response body after copy error for %s: %v\n", url, cerr)
		}
		if cerr := out.Close(); cerr != nil {
			fmt.Fprintf(os.Stderr, "Error closing file after copy error for %s: %v\n", filePath, cerr)
		}
		return fmt.Errorf("failed to write file %s: %w", filePath, err)
	}

	// Explicitly check errors from Close()
	if err := resp.Body.Close(); err != nil {
		return fmt.Errorf("failed to close response body for %s: %w", url, err)
	}
	if err := out.Close(); err != nil {
		return fmt.Errorf("failed to close file %s: %w", filePath, err)
	}

	return nil
}

// copyFile copies content from a source reader to a destination writer.
func copyFile(src io.Reader, dst io.Writer) (int64, error) {
	return io.Copy(dst, src)
}

// HuggingFaceSource implements the ModelSource interface for HuggingFace Hub.
type HuggingFaceSource struct {
	client *http.Client
}

// NewHuggingFaceSource creates a new HuggingFaceSource.
func NewHuggingFaceSource() *HuggingFaceSource {
	return &HuggingFaceSource{
		client: &http.Client{},
	}
}

// HuggingFaceModelInfo represents the structure of the JSON response from HuggingFace API.
type HuggingFaceModelInfo struct {
	ModelID string `json:"modelId"`
	// Other fields might be present, but we only care about siblings for now
	Siblings []struct {
		RPath string `json:"rfilename"` // Relative path of the file
	} `json:"siblings"`
}

// DownloadModel downloads the specified model and its associated files from HuggingFace Hub.
func (h *HuggingFaceSource) DownloadModel(modelID string, destination string) (result *DownloadResult, err error) { // Added named return values
	apiURL := huggingFaceAPI + modelID

	resp, err := h.client.Get(apiURL)
	if err != nil {
		err = fmt.Errorf("failed to fetch model info from HuggingFace API: %w", err)
		return
	}
	defer func() {
		if cerr := resp.Body.Close(); cerr != nil && err == nil { // Only assign if no other error occurred
			err = fmt.Errorf("failed to close response body for %s: %w", apiURL, cerr)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		err = fmt.Errorf("HuggingFace API returned non-OK status: %s", resp.Status)
		return
	}

	var modelInfo HuggingFaceModelInfo
	if err = json.NewDecoder(resp.Body).Decode(&modelInfo); err != nil {
		err = fmt.Errorf("failed to decode HuggingFace API response: %w", err)
		return
	}

	var modelPath string
	var tokenizerPaths []string

	// Iterate through siblings to find ONNX model and tokenizer files
	for _, sibling := range modelInfo.Siblings {
		rPath := sibling.RPath
		if strings.HasSuffix(rPath, ".onnx") {
			modelPath = filepath.Join(destination, filepath.Base(rPath))
			downloadURL := huggingFaceCDN + modelID + "/resolve/main/" + rPath // Assuming 'main' branch
			// Ensure the URL is correctly formed for HuggingFace CDN
			downloadURL = strings.ReplaceAll(downloadURL, "//resolve/main/", "/resolve/main/")
			if err = downloadFile(downloadURL, modelPath); err != nil {
				err = fmt.Errorf("failed to download ONNX model %s: %w", rPath, err)
				return
			}
		} else if strings.Contains(rPath, "tokenizer") || strings.HasSuffix(rPath, ".json") || strings.HasSuffix(rPath, ".txt") {
			// Heuristic for tokenizer files: contains "tokenizer", or is a .json or .txt file
			// This might need refinement based on actual tokenizer file naming conventions
			tokenizerFilePath := filepath.Join(destination, filepath.Base(rPath))
			downloadURL := huggingFaceCDN + modelID + "/resolve/main/" + rPath // Assuming 'main' branch
			// Ensure the URL is correctly formed for HuggingFace CDN
			downloadURL = strings.ReplaceAll(downloadURL, "//resolve/main/", "/resolve/main/")
			if err = downloadFile(downloadURL, tokenizerFilePath); err != nil {
				err = fmt.Errorf("failed to download tokenizer file %s: %w", rPath, err)
				return
			}
			tokenizerPaths = append(tokenizerPaths, tokenizerFilePath)
		}
	}

	if modelPath == "" {
		err = fmt.Errorf("no ONNX model found for model ID: %s", modelID)
		return
	}

	result = &DownloadResult{
		ModelPath:      modelPath,
		TokenizerPaths: tokenizerPaths,
	}
	return
}
