// Command granite2gguf converts IBM Granite Time Series SafeTensors models
// to GGUF format for use with zerfoo inference.
//
// Usage:
//
//	granite2gguf -input /path/to/model/dir -output model.gguf
//
// The input directory must contain config.json and model.safetensors.
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/zerfoo/zonnx/pkg/gguf"
	"github.com/zerfoo/zonnx/safetensors"
)

func main() {
	inputDir := flag.String("input", "", "Path to model directory containing config.json and model.safetensors")
	outputPath := flag.String("output", "", "Path for the output GGUF file (default: <input>/model.gguf)")
	flag.Parse()

	if *inputDir == "" {
		fmt.Fprintln(os.Stderr, "Error: -input flag is required")
		flag.Usage()
		os.Exit(1)
	}

	if *outputPath == "" {
		*outputPath = filepath.Join(*inputDir, "model.gguf")
	}

	if err := convert(*inputDir, *outputPath); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Successfully converted Granite TS model to: %s\n", *outputPath)
}

// graniteConfig holds the parsed config.json fields relevant to Granite TS models.
type graniteConfig struct {
	raw map[string]interface{}

	// Resolved fields.
	modelType string // "ttm", "flowstate", "tspulse"
	modelName string
}

func parseConfig(dir string) (*graniteConfig, error) {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("read config.json: %w", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse config.json: %w", err)
	}

	gc := &graniteConfig{raw: raw}

	// Determine model family from model_type or _name_or_path.
	mt, _ := raw["model_type"].(string)
	mt = strings.ToLower(mt)

	nameOrPath, _ := raw["_name_or_path"].(string)
	nameLower := strings.ToLower(nameOrPath)

	switch {
	case strings.Contains(mt, "ttm") || strings.Contains(nameLower, "ttm"):
		gc.modelType = "ttm"
	case strings.Contains(mt, "flowstate") || strings.Contains(nameLower, "flowstate"):
		gc.modelType = "flowstate"
	case strings.Contains(mt, "tspulse") || strings.Contains(nameLower, "tspulse"):
		gc.modelType = "tspulse"
	default:
		// Fall back to whatever model_type says.
		gc.modelType = mt
	}

	// Derive a human-readable model name.
	if nameOrPath != "" {
		gc.modelName = nameOrPath
	} else {
		gc.modelName = "granite-timeseries-" + gc.modelType
	}

	return gc, nil
}

// writeMetadata adds ts.signal.* metadata keys to the GGUF writer.
func writeMetadata(w *gguf.Writer, gc *graniteConfig) {
	w.AddMetadataString("general.architecture", "granite_ts")
	w.AddMetadataString("general.name", gc.modelName)
	w.AddMetadataString("ts.signal.model_type", gc.modelType)

	// Map numeric fields.
	uint32Fields := []struct {
		configKey string
		ggufKey   string
	}{
		{"context_length", "ts.signal.context_len"},
		{"prediction_length", "ts.signal.forecast_len"},
		{"d_model", "ts.signal.d_model"},
		{"patch_length", "ts.signal.patch_len"},
		{"num_patches", "ts.signal.num_patches"},
	}

	for _, f := range uint32Fields {
		if v, ok := gc.raw[f.configKey]; ok {
			if u, err := toUint32(v); err == nil {
				w.AddMetadataUint32(f.ggufKey, u)
			}
		}
	}

	// num_layers / num_mixer_layers -> architecture-specific key.
	layerCount := lookupNumeric(gc.raw, "num_layers", "num_mixer_layers")
	if layerCount > 0 {
		switch gc.modelType {
		case "ttm":
			w.AddMetadataUint32("ts.signal.num_mixer_layers", uint32(layerCount))
		case "flowstate":
			w.AddMetadataUint32("ts.signal.num_ssm_layers", uint32(layerCount))
		default:
			w.AddMetadataUint32("ts.signal.num_layers", uint32(layerCount))
		}
	}

	// Float fields.
	if v, ok := gc.raw["scale_factor"]; ok {
		if f, err := toFloat32(v); err == nil {
			w.AddMetadataFloat32("ts.signal.scale_factor", f)
		}
	}

	// String fields for TSPulse.
	stringFields := []struct {
		configKey string
		ggufKey   string
	}{
		{"mask_type", "ts.signal.mask_type"},
		{"head_type", "ts.signal.head_type"},
	}

	for _, f := range stringFields {
		if v, ok := gc.raw[f.configKey]; ok {
			if s, ok := v.(string); ok {
				w.AddMetadataString(f.ggufKey, s)
			}
		}
	}
}

// mixerLayerPattern matches "backbone.mixer_layers.N.suffix".
var mixerLayerPattern = regexp.MustCompile(`^backbone\.mixer_layers\.(\d+)\.(.+)$`)

// mapGraniteTensorName converts HuggingFace Granite TS tensor names to
// GGUF-compatible names. The rules are:
//   - "backbone.mixer_layers.N.suffix" -> "blk.N.suffix"
//   - "backbone.X" -> "X" (strip backbone. prefix)
//   - Everything else: keep as-is
func mapGraniteTensorName(name string) string {
	if m := mixerLayerPattern.FindStringSubmatch(name); m != nil {
		return "blk." + m[1] + "." + m[2]
	}
	if strings.HasPrefix(name, "backbone.") {
		return strings.TrimPrefix(name, "backbone.")
	}
	return name
}

func convert(inputDir, outputPath string) error {
	gc, err := parseConfig(inputDir)
	if err != nil {
		return err
	}

	// Open safetensors file.
	stPath := filepath.Join(inputDir, "model.safetensors")
	sf, err := safetensors.Open(stPath)
	if err != nil {
		return fmt.Errorf("open model.safetensors: %w", err)
	}
	defer sf.Close()

	// Create output file.
	if dir := filepath.Dir(outputPath); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create output directory: %w", err)
		}
	}
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create output file: %w", err)
	}
	defer outFile.Close()

	w := gguf.NewWriter(outFile)

	// Write metadata.
	writeMetadata(w, gc)

	// Sort tensor names for deterministic output.
	names := sf.TensorNames()
	sort.Strings(names)

	// Read and write each tensor as F32.
	for _, name := range names {
		info, _ := sf.TensorInfo(name)
		floats, err := sf.ReadFloat32(name)
		if err != nil {
			return fmt.Errorf("read tensor %q: %w", name, err)
		}

		// Encode float32 slice to bytes.
		data := make([]byte, len(floats)*4)
		for i, f := range floats {
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(f))
		}

		shape := make([]uint64, len(info.Shape))
		for i, d := range info.Shape {
			shape[i] = uint64(d)
		}

		ggufName := mapGraniteTensorName(name)
		w.AddTensor(ggufName, gguf.DTypeF32, shape, data)
	}

	if err := w.Flush(); err != nil {
		return fmt.Errorf("write GGUF: %w", err)
	}

	return nil
}

// lookupNumeric tries multiple config keys and returns the first found numeric value.
func lookupNumeric(config map[string]interface{}, keys ...string) int64 {
	for _, k := range keys {
		if v, ok := config[k]; ok {
			switch n := v.(type) {
			case float64:
				return int64(n)
			case int:
				return int64(n)
			case int64:
				return n
			}
		}
	}
	return 0
}

func toUint32(v interface{}) (uint32, error) {
	switch n := v.(type) {
	case float64:
		if n < 0 || n > math.MaxUint32 || n != math.Trunc(n) {
			return 0, fmt.Errorf("float64 %f out of uint32 range", n)
		}
		return uint32(n), nil
	case int:
		if n < 0 {
			return 0, fmt.Errorf("negative int %d", n)
		}
		return uint32(n), nil
	case int64:
		if n < 0 {
			return 0, fmt.Errorf("negative int64 %d", n)
		}
		return uint32(n), nil
	case uint32:
		return n, nil
	default:
		return 0, fmt.Errorf("unsupported type %T", v)
	}
}

func toFloat32(v interface{}) (float32, error) {
	switch n := v.(type) {
	case float64:
		return float32(n), nil
	case float32:
		return n, nil
	case int:
		return float32(n), nil
	default:
		return 0, fmt.Errorf("unsupported type %T", v)
	}
}
