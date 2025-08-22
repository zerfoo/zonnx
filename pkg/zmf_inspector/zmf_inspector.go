package zmf_inspector

import (
	"fmt"
	"io/ioutil"

	"github.com/zerfoo/zmf"
	"google.golang.org/protobuf/proto"
)

// Load reads and deserializes a ZMF model from a file.
func Load(file string) (*zmf.Model, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	model := &zmf.Model{}
	if err := proto.Unmarshal(data, model); err != nil {
		return nil, err
	}

	return model, nil
}

// Inspect prints a human-readable summary of a ZMF model.
func Inspect(model *zmf.Model) {
	fmt.Printf("Producer: %s %s\n", model.GetMetadata().GetProducerName(), model.GetMetadata().GetProducerVersion())
	fmt.Printf("Opset version: %d\n", model.GetMetadata().GetOpsetVersion())
	fmt.Printf("Graph has %d nodes.\n", len(model.GetGraph().GetNodes()))
	fmt.Printf("Graph has %d parameters.\n", len(model.GetGraph().GetParameters()))

	fmt.Println("\nNodes:")
	for _, node := range model.GetGraph().GetNodes() {
		fmt.Printf("- Node: %s, OpType: %s\n", node.GetName(), node.GetOpType())
		fmt.Printf("  Inputs: %v\n", node.GetInputs())
		fmt.Printf("  Outputs: %v\n", node.GetOutputs())
		if len(node.GetAttributes()) > 0 {
			fmt.Println("  Attributes:")
			for name, attr := range node.GetAttributes() {
				fmt.Printf("    - %s: %v\n", name, attr.GetValue())
			}
		}
	}
}