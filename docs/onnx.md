# ONNX Conversion Strategy

This document outlines the strategy for converting ONNX models to the ZMF format, specifically addressing how to handle ONNX's flexibility to keep the `zerfoo` runtime simple and efficient.

## Guiding Principle: The Converter is the Compiler

The core design principle is to treat the `zonnx` converter as a "compiler" for ONNX models. It is the converter's responsibility to understand the complexities and variations of the ONNX specification and "compile them down" into a simpler, stricter, and more canonical representation in the ZMF format.

This approach provides a clean separation of concerns:
- **`zonnx`:** Handles the complex, flexible, and sometimes ambiguous nature of the ONNX standard.
- **`zerfoo`:** Works with a simple, predictable, and strongly-typed graph format (`.zmf`), allowing its core engine to be optimized for execution without needing to handle numerous special cases from the ONNX spec.

## Handling Constant Inputs as Attributes

A key example of this principle is how we handle constant inputs. In ONNX, some operator arguments can be provided either as a node `attribute` or as a constant `initializer` connected to an input.

A prime example is the `Transpose` operator. The permutation order (`perm`) can be an attribute of the `Transpose` node itself. However, it is also common for the permutation to be defined in a separate `Constant` node (an initializer) and fed into the second input of the `Transpose` node.

**The `zonnx` Strategy:**

The converter is responsible for resolving this. It will always transform the constant input pattern into a canonical attribute-based representation in the `.zmf` file.

1.  **Identify Constant Inputs:** The converter iterates through a node's inputs. If an input is linked to an `initializer` containing non-float data (e.g., `INT64` or `INT32`), it is treated as a constant.
2.  **Promote to Attribute:** This constant data is extracted and converted into a `zmf.Attribute` on the node.
3.  **Remove from Inputs:** The original input is removed from the node's input list in the `.zmf` graph.

**Example: `Transpose` Operator**

-   **ONNX Graph:**
    -   `Node A` -> `Transpose` (Input 0)
    -   `Constant [0, 2, 1]` -> `Transpose` (Input 1)

-   **`zonnx` Conversion Process:**
    1.  The converter sees the `Transpose` node.
    2.  It inspects the second input and finds it's an `initializer` with `INT64` data `[0, 2, 1]`.
    3.  It creates a new `zmf.Attribute` on the `Transpose` node called `perm` with the value `[0, 2, 1]`.
    4.  It removes the second input from the `Transpose` node's input list.

-   **Resulting `zmf` Graph:**
    -   `Node A` -> `Transpose` (Input 0)
        -   `Attribute "perm": [0, 2, 1]`

This ensures that the `zerfoo` `Transpose` layer only needs to look for a `perm` attribute and doesn't need complex logic to handle multiple ways of receiving its arguments. This keeps the runtime code clean, simple, and efficient.
