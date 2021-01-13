from deepNets import layers

def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = layers.affine_forward(x, w, b)
    out, relu_cache = layers.relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = layers.relu_backward(dout, relu_cache)
    dx, dw, db = layers.affine_backward(da, fc_cache)
    return dx, dw, db

def affine_relu_norm_forward(x,w,b,gamma,beta,bn_param,normalization):

    """
    Convenience layer that performs an affine transform followed by batch norm 
    followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma,beta : for scaling batch/layer norm
    - bn_param : Parameters for batch/layer norm
    - normalization: batch/layer normalization

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass 
    """
    a,fc_cache = layers.affine_forward(x,w,b)
    if normalization == "batchnorm":
      batch,bc_cache = layers.batchnorm_forward(a, gamma, beta, bn_param)
    elif normalization == "layernorm":
      batch,bc_cache = layers.layernorm_forward(a, gamma, beta, bn_param)
    out,relu_cache = layers.relu_forward(batch)
    return out , (fc_cache,bc_cache,relu_cache)

def affine_relu_norm_backward(dout,caches,normalization):
    """
    Backward pass for the affine-batch/layer-relu convenience layer

    Inputs:
    - dout: input gradient of loss function
    - caches : forward pass caches of forward,batch/layer,relu
    - normalization: type of normalization
    Returns a set of:
    - dx: gradient of x with respect to output
    - dw: gradient of w with respect to output
    - db: gradient of b with respect to output
    - dgamma: gradient of gamma with respect to output
    - dbeta: gradient of beta with respect to output
    """
    fc_cache,bc_cache,relu_cache = caches
    drelu_out = layers.relu_backward(dout,relu_cache)
    if normalization == "batchnorm":
      dbatch,dgamma,dbeta = layers.batchnorm_backward(drelu_out,bc_cache)
    elif normalization == "layernorm":
      dbatch,dgamma,dbeta = layers.layernorm_backward(drelu_out,bc_cache)
    dx,dw,db = layers.affine_backward(dbatch,fc_cache)
    return dx,dw,db,dgamma,dbeta

def norm_affine_forward(x,w,b,gamma,beta,bn_param,normalization):

  """
    Convenience layer that performs batch norm followed by affine transform

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma,beta : for scaling batch/layer norm
    - bn_param : Parameters for batch/layer norm
    - normalization: batch/layer normalization

    Returns a tuple of:
    - out: Output from the affine
    - cache: Object to give to the backward pass 
  """
  if normalization == "batchnorm":
    batch,bc_cache = layers.batchnorm_forward(x, gamma, beta, bn_param)
  elif normalization == "layernorm":
    batch,bc_cache = layers.layernorm_forward(x, gamma, beta, bn_param)
  out,fc_cache = layers.affine_forward(batch,w,b)
  return out,(fc_cache,bc_cache)

def norm_affine_backward(dout,caches,normalization):
  """
    Backward pass for the batch/layer-affine convenience layer

    Inputs:
    - dout: input gradient of loss function
    - caches : forward pass caches of forward,batch/layer,relu
    - normalization: type of normalization
    Returns a set of:
    - dx: gradient of x with respect to output
    - dw: gradient of w with respect to output
    - db: gradient of b with respect to output
    - dgamma: gradient of gamma with respect to output
    - dbeta: gradient of beta with respect to output
  """
  fc_cache,bc_cache = caches
  dx,dw,db = layers.affine_backward(dout,fc_cache)
  if normalization == "batchnorm":
    dx,dgamma,dbeta = layers.batchnorm_backward(dx,bc_cache)
  elif normalization == "layernorm":
    dx,dgamma,dbeta = layers.layernorm_backward(dx,bc_cache)
  return dx,dw,db,dgamma,dbeta