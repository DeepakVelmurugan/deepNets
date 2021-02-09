from deepNets import layers
#from layers import *

#affine -> relu
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
    # a, fc_cache = affine_forward(x, w, b)
    # out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = layers.relu_backward(dout, relu_cache)
    dx, dw, db = layers.affine_backward(da, fc_cache)
    # da = relu_backward(dout, relu_cache)
    # dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db
#affine -> batch -> relu
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
    # a,fc_cache = affine_forward(x,w,b)
    if normalization == "batchnorm":
      batch,bc_cache = layers.batchnorm_forward(a, gamma, beta, bn_param)
      #batch,bc_cache = batchnorm_forward(a, gamma, beta, bn_param)
    elif normalization == "layernorm":
      batch,bc_cache = layers.layernorm_forward(a, gamma, beta, bn_param)
      #batch,bc_cache = layernorm_forward(a, gamma, beta, bn_param)
    out,relu_cache = layers.relu_forward(batch)
    #out,relu_cache = relu_forward(batch)
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
    #drelu_out = relu_backward(dout,relu_cache)
    if normalization == "batchnorm":
      dbatch,dgamma,dbeta = layers.batchnorm_backward(drelu_out,bc_cache)
      #dbatch,dgamma,dbeta = batchnorm_backward(drelu_out,bc_cache)
    elif normalization == "layernorm":
      dbatch,dgamma,dbeta = layers.layernorm_backward(drelu_out,bc_cache)
      #dbatch,dgamma,dbeta = layernorm_backward(drelu_out,bc_cache)
    dx,dw,db = layers.affine_backward(dbatch,fc_cache)
    #dx,dw,db = affine_backward(dbatch,fc_cache)
    return dx,dw,db,dgamma,dbeta
#batch -> affine
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
    #batch,bc_cache = batchnorm_forward(x, gamma, beta, bn_param)
  elif normalization == "layernorm":
    batch,bc_cache = layers.layernorm_forward(x, gamma, beta, bn_param)
    #batch,bc_cache = layernorm_forward(x, gamma, beta, bn_param)
  out,fc_cache = layers.affine_forward(batch,w,b)
  #out,fc_cache = affine_forward(batch,w,b)
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
  #dx,dw,db = affine_backward(dout,fc_cache)
  if normalization == "batchnorm":
    dx,dgamma,dbeta = layers.batchnorm_backward(dx,bc_cache)
    #dx,dgamma,dbeta = batchnorm_backward(dx,bc_cache)
  elif normalization == "layernorm":
    dx,dgamma,dbeta = layers.layernorm_backward(dx,bc_cache)
    #dx,dgamma,dbeta = layernorm_backward(dx,bc_cache)
  return dx,dw,db,dgamma,dbeta
#conv -> relu
def conv_relu_forward(x, w, b, conv_param):
  """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
  """
  a, conv_cache = layers.conv_forward(x, w, b, conv_param)
  out, relu_cache = layers.relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache

def conv_relu_backward(dout, cache):
  """
    Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = layers.relu_backward(dout, relu_cache)
  dx, dw, db = layers.conv_backward(da, conv_cache)
  return dx, dw, db

def conv_batch_relu_pool_forward(x,w,b,conv_param,gamma,beta,bn_param,pool_param):
  """
    A convenience layer that performs conv-batch-relu-pool forward pass

    Inputs : 
    - x : Input to convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - gamma, beta, bn_param : batch norm parameters
    - pool_param: pooling layer parameters

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass 
  """
  convOut,conv_cache = layers.conv_forward(x,w,b,conv_param)
  normOut,norm_cache = layers.spatial_batchnorm_forward(convOut, gamma, beta, bn_param)
  reluOut,relu_cache = layers.relu_forward(normOut)
  out,pool_cache = layers.max_pool_forward(reluOut, pool_param)
  cache = (conv_cache,norm_cache,relu_cache,pool_cache)
  return out,cache

def conv_batch_relu_pool_backward(dout, cache):
  """
    Backward pass for the conv-batch-relu-pool convenience layer.
  """
  conv_cache,norm_cache,relu_cache,pool_cache = cache
  dpool_out = layers.max_pool_backward(dout,pool_cache)
  drelu_out = layers.relu_backward(dpool_out, relu_cache)
  dnorm_out,dgamma,dbeta = layers.spatial_batchnorm_backward(drelu_out, norm_cache)
  dx,dw,db = layers.conv_backward(dnorm_out, conv_cache)
  return dx,dw,db,dgamma,dbeta

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    convOut, conv_cache = layers.conv_forward(x, w, b, conv_param)
    reluOut, relu_cache = layers.relu_forward(convOut)
    out, pool_cache = layers.max_pool_forward(reluOut, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    dpool_out = layers.max_pool_backward(dout, pool_cache)
    drelu_out = layers.relu_backward(dpool_out, relu_cache)
    dx, dw, db = layers.conv_backward(drelu_out, conv_cache)
    return dx, dw, db

def conv_batch_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """
    Convenience layer that performs a convolution, a batch, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - gamma, beta, bn_param : batch norm parameters

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    convOut, conv_cache = layers.conv_forward(x, w, b, conv_param)
    normOut, norm_cache = layers.spatial_batchnorm_forward(convOut, gamma, beta, bn_param)
    out, relu_cache = layers.relu_forward(normOut)
    cache = (conv_cache, norm_cache, relu_cache)
    return out, cache

def conv_batch_relu_backward(dout, cache):
    """
    Backward pass for the conv-batch-relu convenience layer
    """
    conv_cache, norm_cache, relu_cache = cache
    drelu_out = layers.relu_backward(dout, relu_cache)
    dnorm_out, dgamma, dbeta = layers.spatial_batchnorm_backward(drelu_out, norm_cache)
    dx, dw, db = layers.conv_backward(dnorm_out, conv_cache)
    return dx, dw, db, dgamma, dbeta
