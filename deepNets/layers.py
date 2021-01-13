import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    xDim_each = x[0].shape
    out = np.dot(x.reshape(x.shape[0],np.prod(xDim_each)),w) + b.reshape(1,-1)
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #out = S(x*w) + b
    dS = dout #NxM
    db = dout #NxM->(M,)
    db = np.sum(db,axis=0)
    dx = np.dot(dS,w.T) #NxM X MxD -> NxD
    dx = dx.reshape(x.shape)
    X = x.reshape(x.shape[0],np.prod(x[0].shape)) #-> NxD
    dw = np.dot(X.T,dS) #DxN X NxM -> DxM
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0,x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout
    dx[x<0] = 0
    return dx


""" Normalisation layers """

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    #for layernorm
    layernorm = bn_param.get("layernorm",0)
    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
      MEAN = x.mean(axis=0)
      var_part = x - MEAN
      var_part_sqrt = var_part**2
      var_sum = np.sum(var_part_sqrt,axis=0)
      var = (N**(-1)) * var_sum    #mini_batch
      norm_deno_part = var + eps
      norm_deno_sqrt =  norm_deno_part**0.5
      norm_deno = norm_deno_sqrt**(-1)  #inversing denominator
      norm = var_part * norm_deno  
      scaled_norm = gamma * norm
      out = scaled_norm + beta
      cache = {'norm':norm,'gamma':gamma,'norm_deno':norm_deno,'var_part':var_part,'norm_deno_sqrt':norm_deno_sqrt,'norm_deno_part':norm_deno_part}
      if layernorm:
        cache["axis"] = 1
      else:
        cache["axis"] = 0
        running_mean = momentum * running_mean + (1 - momentum) * MEAN
        running_var = momentum * running_var + (1 - momentum) * var
    elif mode == "test":
      z = (x - running_mean)/np.sqrt(running_var + eps)
      out = gamma*z + beta
    else:
      raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    N,D = dout.shape
    gamma = cache['gamma']
    norm,gamma,norm_deno,var_part = cache['norm'],cache['gamma'],cache['norm_deno'],cache['var_part']
    norm_deno_sqrt,norm_deno_part,axis = cache['norm_deno_sqrt'],cache['norm_deno_part'],cache["axis"]
    dscaled_norm = dout
    dbeta = np.sum(dout,axis=axis) #since beta is (D,)
    dgamma = np.sum(norm*dscaled_norm,axis=axis) #since gamma is (D,)
    dnorm = gamma*dscaled_norm
    dvar_part = norm_deno * dnorm
    dnorm_deno = np.sum(var_part*dnorm,axis=0) #since denominator is (N,)
    dnorm_deno_sqrt = -1 * norm_deno_sqrt**(-1-1) * dnorm_deno
    dnorm_deno_part = 0.5 * 1/np.sqrt(norm_deno_part) * dnorm_deno_sqrt #pow -> 0.5-1 = -0.5(inverse root)
    dvar = dnorm_deno_part
    dvar_sum = 1./N * dvar
    dvar_part_sqrt = np.ones((N,D)) * dvar_sum
    dvar_part += 2 * var_part * dvar_part_sqrt
    dx = 1*dvar_part
    dmean = -1 * np.sum(dvar_part,axis=0)
    dx += 1./N * np.ones((N,D)) * dmean  #summing of x gradient
    return dx, dgamma, dbeta

def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ln_param["mode"] = "train"
    ln_param["layernorm"] = 1
    out,cache = batchnorm_forward(x.T, gamma.reshape(-1,1), beta.reshape(-1,1), ln_param)
    out = out.T
    return out, cache

def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.
    
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    dx,dgamma,dbeta = batchnorm_backward(dout.T,cache)
    dx = dx.T
    return dx, dgamma, dbeta

"""Loss layers"""

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx

