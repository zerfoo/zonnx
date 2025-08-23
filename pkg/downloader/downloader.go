package downloader

import "io"

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

// downloadFile is a helper function to download a single file from a URL to a local path.
func downloadFile(url, filepath string) error {
	// TODO: Implement actual file download logic using net/http
	// For now, this is a placeholder.
	return nil
}

// copyFile is a helper function to copy content from a source reader to a destination writer.
func copyFile(src io.Reader, dst io.Writer) (int64, error) {
	// TODO: Implement actual file copy logic
	// For now, this is a placeholder.
	return 0, nil
}
