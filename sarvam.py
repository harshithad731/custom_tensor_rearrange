
import numpy as np
import re
from typing import List, Tuple, Dict


def parse_pattern(pattern: str) -> Tuple[str, str]:
    if "->" not in pattern:
        raise ValueError(f"Invalid pattern '{pattern}'. Expected '->' to separate input and output.")
    return tuple(map(str.strip, pattern.split("->")))


def tokenize_pattern(pattern: str) -> List[str]:
    tokens = []
    buffer = ""
    in_paren = 0
    for ch in pattern:
        if ch == "(":
            in_paren += 1
            buffer += ch
        elif ch == ")":
            in_paren -= 1
            buffer += ch
        elif ch == " " and in_paren == 0:
            if buffer:
                tokens.append(buffer)
                buffer = ""
        else:
            buffer += ch
    if buffer:
        tokens.append(buffer)
    return tokens


def expand_pattern(tokens: List[str]) -> List[str]:
    expanded = []
    for token in tokens:
        if token == "...":
            expanded.append(token)
        elif token.startswith("("):
            sub_axes = re.findall(r'\w+', token)
            expanded.extend(sub_axes)
        else:
            expanded.append(token)
    return expanded


def resolve_ellipsis(pattern_tokens: List[str], tensor_shape: Tuple[int], other_pattern_tokens: List[str]) -> List[str]:
    # Replace ... with appropriate number of unnamed axes
    if "..." not in pattern_tokens:
        return pattern_tokens

    ellipsis_index = pattern_tokens.index("...")
    known = [t for t in pattern_tokens if t != "..."]
    num_missing = len(tensor_shape) - len(expand_pattern(known))

    if num_missing < 0:
        raise ValueError("Pattern has more named dimensions than tensor has axes.")

    unnamed_axes = [f"_axis{i}" for i in range(num_missing)]
    return pattern_tokens[:ellipsis_index] + unnamed_axes + pattern_tokens[ellipsis_index + 1:]


def extract_shape_mapping(input_tokens: List[str], shape: Tuple[int], axes_lengths: Dict[str, int]) -> Dict[str, int]:
    if len(input_tokens) != len(shape):
        raise ValueError(f"Input pattern tokens {input_tokens} don't match tensor shape {shape}")

    mapping = dict(axes_lengths)
    for token, dim in zip(input_tokens, shape):
        if token.startswith("("):  # Merged axes like (h w)
            sub_axes = re.findall(r'\w+', token)
            known = [a for a in sub_axes if a in mapping]
            unknown = [a for a in sub_axes if a not in mapping]

            if not unknown:
                # All dimensions are known, verify match
                total = np.prod([mapping[a] for a in sub_axes])
                if total != dim:
                    raise ValueError(f"Merged dimension {token} expected {total}, got {dim}")
            elif len(unknown) == 1:
                # One unknown, infer it
                known_product = np.prod([mapping[a] for a in known])
                if dim % known_product != 0:
                    raise ValueError(f"Cannot infer dimension for {unknown[0]}: {dim} not divisible by {known_product}")
                mapping[unknown[0]] = dim // known_product
            else:
                raise ValueError(f"Cannot infer more than one axis in merged dimension: {token}")
        else:
            if token in mapping and mapping[token] != dim:
                print(f"[Note] Overriding axes_lengths[{token}] = {mapping[token]} with actual shape {dim}")
            mapping[token] = dim
    return mapping


def compute_shape(tokens: List[str], axes_lengths: Dict[str, int]) -> List[int]:
    shape = []
    for token in tokens:
        if token.startswith("("):  # Merged axes
            sub_axes = re.findall(r'\w+', token)
            size = np.prod([axes_lengths[a] for a in sub_axes])
            shape.append(size)
        else:
          if token.isdigit():  # literal dimension
              shape.append(int(token))
          else:
              shape.append(axes_lengths[token])
    return shape

def compute_shape(tokens: List[str], axes_lengths: Dict[str, int]) -> List[int]:
    shape = []
    for token in tokens:
        if token.startswith("("):  # Merged axes
            sub_axes = re.findall(r'\w+', token)
            size = np.prod([axes_lengths[a] for a in sub_axes])
            shape.append(size)
        elif token.isdigit():
            shape.append(int(token))
        elif token in axes_lengths:
            shape.append(axes_lengths[token])
        else:
            raise ValueError(f"Cannot determine shape for axis '{token}'")
    return shape

def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    input_pat, output_pat = parse_pattern(pattern)
    input_tokens = tokenize_pattern(input_pat)
    output_tokens = tokenize_pattern(output_pat)

    input_tokens = resolve_ellipsis(input_tokens, tensor.shape, output_tokens)
    output_tokens = resolve_ellipsis(output_tokens, tensor.shape, input_tokens)

    axes_lengths = extract_shape_mapping(input_tokens, tensor.shape, axes_lengths)

    input_expanded = expand_pattern(input_tokens)
    output_expanded = expand_pattern(output_tokens)

    # Reshape to match flat input pattern
    input_shape = [axes_lengths[t] if not t.isdigit() else int(t) for t in input_expanded]
    tensor = tensor.reshape(input_shape)

    # Transpose only input axes that appear in output, in output order
    index_map = {name: i for i, name in enumerate(input_expanded)}
    transpose_order = [index_map[name] for name in output_expanded if name in index_map]
    if transpose_order != list(range(len(transpose_order))):
        tensor = tensor.transpose(transpose_order)

    # Compute final shape and reshape or broadcast
    final_shape = compute_shape(output_tokens, axes_lengths)
    if tensor.size == np.prod(final_shape):
        tensor = tensor.reshape(final_shape)
    else:
        try:
            tensor = np.broadcast_to(tensor, final_shape)
        except Exception as e:
            raise ValueError(f"Cannot reshape or broadcast to shape {final_shape}: {e}")

    return tensor

def parse_slice_str(s: str) -> Tuple[int, int, int]:
    """Parse a slice string like '1:5:2' or '::2'."""
    parts = s.split(":")
    parts = [int(p) if p else None for p in parts]
    return slice(*parts)


def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    input_pat, output_pat = parse_pattern(pattern)
    input_tokens = tokenize_pattern(input_pat)
    output_tokens = tokenize_pattern(output_pat)

    # Ellipsis
    input_tokens = resolve_ellipsis(input_tokens, tensor.shape, output_tokens)
    output_tokens = resolve_ellipsis(output_tokens, tensor.shape, input_tokens)

    # Mapping
    axes_lengths = extract_shape_mapping(input_tokens, tensor.shape, axes_lengths)

    # Expand
    input_expanded = expand_pattern(input_tokens)
    output_expanded = expand_pattern(output_tokens)

    # Reshape to flat input
    input_shape = [int(t) if t.isdigit() else axes_lengths[t] for t in input_expanded]
    tensor = tensor.reshape(input_shape)

    # Transpose if needed
    index_map = {name: i for i, name in enumerate(input_expanded)}
    transpose_order = []
    for name in output_expanded:
        if name in index_map:
            transpose_order.append(index_map[name])

    if len(transpose_order) != tensor.ndim:
        # Don't transpose if dims don't align; wait for reshape
        pass
    elif transpose_order != list(range(len(transpose_order))):
        tensor = tensor.transpose(transpose_order)


    # Final reshape
    final_shape = compute_shape(output_tokens, axes_lengths)
    if tensor.size != np.prod(final_shape):
        try:
            tensor = np.broadcast_to(tensor, final_shape)
        except Exception as e:
            raise ValueError(f"Cannot reshape or broadcast to {final_shape}: {e}")
    else:
        tensor = tensor.reshape(final_shape)

    return tensor

