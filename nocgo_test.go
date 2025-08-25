package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestBuildWithCGODisabled ensures the zonnx module builds with CGO disabled,
// which verifies there is no CGo linkage dependency.
func TestBuildWithCGODisabled(t *testing.T) {
	modRoot, err := os.Getwd()
	if err != nil {
		t.Fatalf("failed to get working directory: %v", err)
	}

	cmd := exec.Command("go", "build", "./...")
	cmd.Dir = modRoot
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		// Provide a helpful error that includes the module root for context.
		t.Fatalf("build failed with CGO disabled in %s: %v", filepath.Base(modRoot), err)
	}
}
