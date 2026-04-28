import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from utils import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from matmul_kernels import nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

@nki.jit
def nki_transpose(in_tensor):
    """NKI kernel to transpose a 2D tensor.

    Args:
        in_tensor: an input tensor of shape [#rows, #cols]

    Returns:
        out_tensor: an output (transposed) tensor of shape [#cols, #rows]
    """
    i_rows, i_cols = in_tensor.shape
    o_rows, o_cols = i_cols, i_rows

    out_tensor = nl.ndarray((o_rows, o_cols), dtype=in_tensor.dtype, buffer=nl.hbm)

    # YOUR CODE HERE
    for i in nl.affine_range(i_rows // nl.tile_size.pmax):
      i_p = i * nl.tile_size.pmax
      for j in nl.affine_range(i_cols // nl.tile_size.pmax):
        j_p = j * nl.tile_size.pmax
        tile = nl.load_transpose2d(in_tensor[i_p:i_p+nl.tile_size.pmax, j_p:j_p+nl.tile_size.pmax])
        nl.store(out_tensor[j_p:j_p+nl.tile_size.pmax, i_p:i_p+nl.tile_size.pmax], tile)
    return out_tensor
    

@nki.jit
def nki_bias_add_act(A, b, act='relu'):
    """NKI kernel to add a bias vector to each row of a 2D tensor, and apply activation.

    Args:
        A: an input tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
        b: a bias vector of shape [1, HIDDEN_SIZE]
        act: an activation function to apply (e.g., 'relu', 'softmax')
    Returns:
        result: the resulting output tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
    """
    # Gather input shapes
    BATCH_SIZE, HIDDEN_SIZE = A.shape
    _, HIDDEN_SIZE_ = b.shape
    assert HIDDEN_SIZE == HIDDEN_SIZE_, "A and b must have the same HIDDEN_SIZE"

    # Create an output tensor
    result = nl.ndarray((BATCH_SIZE, HIDDEN_SIZE), dtype=A.dtype, buffer=nl.hbm)

    # YOUR CODE HERE
    if act == 'relu':
      for i in nl.affine_range(BATCH_SIZE):
        row_idx = nl.mgrid[0:1, 0:nl.tile_size.pmax]
        for j in nl.affine_range(HIDDEN_SIZE // nl.tile_size.pmax):
          j_p = j * nl.tile_size.pmax
          tile = nl.ndarray((1, nl.tile_size.pmax), dtype=A.dtype, buffer=nl.sbuf)
          nisa.dma_copy(dst=tile, src=A[i+row_idx.p, j_p+row_idx.x])
          bias = nl.load(b[row_idx.p, j_p+row_idx.x])
          sum_tile = nl.add(tile, bias)
          nisa.dma_copy(dst=result[i+row_idx.p, j_p+row_idx.x], src=nl.relu(sum_tile))

    if act == 'softmax':
      for i in nl.affine_range(BATCH_SIZE):
        row_idx = nl.mgrid[0:1, 0:nl.tile_size.pmax]
        row_max = nl.ndarray((1, 1), dtype=A.dtype, buffer=nl.sbuf)
        row_sum = nl.ndarray((1, 1), dtype=A.dtype, buffer=nl.sbuf)
        tile = nl.load(A[i + row_idx.p, row_idx.x])
        bias = nl.load(b[row_idx.p, row_idx.x])
        sum_tile = nl.add(tile, bias)
        row_max[...] = nl.max(sum_tile, axis=1)

        for j in nl.sequential_range(1, HIDDEN_SIZE // nl.tile_size.pmax):
          j_p = j * nl.tile_size.pmax
          tile = nl.load(A[i+row_idx.p, j_p+row_idx.x])
          bias = nl.load(b[row_idx.p, j_p+row_idx.x])
          sum_tile = nl.add(tile, bias)
          row_max[...] = nl.maximum(row_max, nl.max(sum_tile, axis=1))
  
        tile = nl.load(A[i+row_idx.p, row_idx.x])
        bias = nl.load(b[row_idx.p, row_idx.x])
        sum_tile = nl.add(tile, bias)
        exp_tile = nl.exp(nl.subtract(sum_tile, row_max))
        row_sum[...] = nl.sum(exp_tile, axis=1)

        for j in nl.sequential_range(1, HIDDEN_SIZE // nl.tile_size.pmax):
          j_p = j * nl.tile_size.pmax
          tile = nl.load(A[i + row_idx.p, j_p+row_idx.x])
          bias = nl.load(b[row_idx.p, j_p+row_idx.x])
          sum_tile = nl.add(tile, bias)
          exp_tile = nl.exp(nl.subtract(sum_tile, row_max))
          row_sum[...] = nl.add(row_sum, nl.sum(exp_tile, axis=1))
        
        for j in nl.affine_range(0, HIDDEN_SIZE // nl.tile_size.pmax):
          j_p = j * nl.tile_size.pmax
          tile = nl.load(A[i+row_idx.p, j_p+row_idx.x])
          bias = nl.load(b[row_idx.p, j_p+row_idx.x])
          sum_tile = nl.add(tile, bias)
          exp_tile = nl.exp(nl.subtract(sum_tile, row_max))
          out_tile = nl.divide(exp_tile, row_sum)
          nl.store(result[i+row_idx.p, j_p+row_idx.x], out_tile)

    return result

@nki.jit
def nki_forward(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel to compute the forward pass of the feedforward neural network with 1 hidden layer.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      probs: the resulting probability output tensor of shape [BATCH_SIZE, OUTPUT_SIZE]
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'
  """
  if matmul_kernel == 'tiled':
    nki_matmul = nki_matmul_tiled_
  elif matmul_kernel == 'hoist_load':
    nki_matmul = nki_matmul_hoist_load_
  elif matmul_kernel == 'block_free_dimension':
    nki_matmul = nki_matmul_block_free_dimension_
  elif matmul_kernel == 'fully_optimized':
    nki_matmul = nki_matmul_fully_optimized_
  else:
    raise ValueError(f"Unsupported matmul kernel: {matmul_kernel}")

  # Layer 1
  probs = nl.ndarray((BATCH_SIZE, HIDDEN_SIZE), dtype=X.dtype, buffer=nl.hbm)
  probs = nki_bias_add_act(nki_matmul(nki_transpose(X), W1), b1)
  probs = nki_bias_add_act(nki_matmul(nki_transpose(probs), W2), b2, act='softmax')
  return probs
  

@nki.jit
def nki_predict(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel run forward pass and predict the classes of the input tensor.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'

  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  """
  probs = nki_forward(X, W1, b1, W2, b2, matmul_kernel)
  BATCH_SIZE, OUTPUT_SIZE = probs.shape
  predictions = nl.ndarray((BATCH_SIZE,), dtype=np.int32, buffer=nl.hbm)

  # YOUR CODE HERE
  for i in nl.affine_range(BATCH_SIZE):
    tile = nl.load(probs[i:i+1, :])
    max8_values = nisa.max8(src=tile)
    max8_indices = nisa.nc_find_index8(vals=max8_values, data=tile)
    nl.store(predictions[i], max8_indices[0, 0])
  return predictions