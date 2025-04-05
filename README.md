
# rearrange: Custom Tensor Rearrangement Utility
Inspired by the elegance of einops.rearrange, this rearrange function is a compact, efficient, and readable numpy-native utility to reshape and permute multi-dimensional arrays using a declarative string pattern.

Function Signature
rearrange(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray

Features
Supports axis merging/splitting using (...) notation.

Supports broadcasting dimensions (1 -> b).

Handles ellipsis (...) for unknown axis positions.

Works with batch dimensions and large-scale data.

Numpy-only, no external dependencies.

Automatically infers shape dimensions when needed.

How It Works – Algorithm Breakdown
Pattern Parsing The input pattern 'a b -> b a' is parsed into:
input_tokens: tokens from the left-hand side (e.g., ['a', 'b'])

output_tokens: tokens from the right-hand side (e.g., ['b', 'a'])

Parentheses like (h w) are preserved for now to represent merged axes.

Ellipsis Resolution If ... is used, it is replaced by generated axis names (e.g., _axis0, _axis1) based on tensor shape and remaining known axes.

Shape Inference If merged axes like (h w) appear, the function:

Computes the product of known dimensions.

Infers missing ones (e.g., infer w from h=3, shape=12 → w=4).

Ensures dimensionality matches the tensor's actual shape.

Input Expansion Expands composite patterns like (h w) into flat tokens (['h', 'w']) to align with the raw tensor shape.

Transpose Mapping Determines how axes should be reordered:

Builds a mapping of input axes → indices.

Constructs a transpose_order for reordering only axes that exist in both input and output.

Final Shape Calculation Uses the tokenized output pattern to:
Multiply dimensions if necessary (e.g., (h w) becomes h*w)

Broadcast or reshape to desired layout.

Performance & Efficiency
Compact & Readable: ~100 lines of clear Python, yet expressive and extensible.

Efficient: Avoids unnecessary reshapes/transposes. Only modifies axes as needed.

Broadcast-Friendly: Smart handling of broadcast scenarios like 1 -> b.

Large Dataset Ready: Designed with batching in mind. Easily integrates with tensors of shape (B, C, H, W) or larger.

Zero Dependencies: Pure NumPy implementation – fast, portable, and minimal.

Example Usages
x = np.random.rand(2, 3, 4)

rearranged = rearrange(x, 'b c h -> b h c') # simple transpose

x = np.random.rand(12, 10)

rearranged = rearrange(x, '(h w) c -> h w c', h=3) # split flat spatial dim

x = np.random.rand(2, 1, 5)

rearranged = rearrange(x, 'a 1 c -> a b c', b=4) # broadcast singleton dim

x = np.random.rand(2, 3, 4, 5)

rearranged = rearrange(x, 'b c h w -> b (c h w)') # flatten all but batch

Error Handling
Throws descriptive ValueError when:

Axis lengths are inconsistent or ambiguous.

Cannot reshape due to size mismatch.

Multiple unknown dimensions in merged patterns.

Limitations
Does not yet support advanced slicing or stride tricks (like ::2 in einops).

All axes must be explicitly handled in pattern.

Ideal Use Cases
Preprocessing images/videos for ML pipelines.

Reshaping batch data in deep learning models.

Compact manipulation of large tensors.

Readable replacements for chains of .reshape(), .transpose(), .expand_dims(), etc.

Tests You Should Try
Simple spatial rearrange
x = np.random.rand(12, 10)

assert np.allclose(rearrange(x, '(h w) c -> h w c', h=3), einops.rearrange(x, '(h w) c -> h w c', h=3))

Broadcasting
x = np.random.rand(2, 1, 5)

assert np.allclose(rearrange(x, 'a 1 c -> a b c', b=4), einops.rearrange(x, 'a 1 c -> a b c', b=4))
