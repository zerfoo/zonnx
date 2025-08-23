package downloader

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

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
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download file from %s: %w", url, err)
	}
	defer func() {
		if cerr := resp.Body.Close(); cerr != nil {
			// Log the error or handle it as appropriate, but don't return it
			// as the primary error of the function.
			fmt.Fprintf(os.Stderr, "Error closing response body for %s: %v\n", url, cerr)
		}
	}()


	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download file from %s: status code %d", url, resp.StatusCode)
	}

	out, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", filePath, err)
	}
	defer func() {
		if cerr := out.Close(); cerr != nil {
			// Log the error or handle it as appropriate.
			fmt.Fprintf(os.Stderr, "Error closing file %s: %v\n", filePath, cerr)
		}
	}()


	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write file %s: %w", filePath, err)
	}

	return nil
}

// copyFile copies content from a source reader to a destination writer.
func copyFile(src io.Reader, dst io.Writer) (int64, error) {
	return io.Copy(dst, src)
}