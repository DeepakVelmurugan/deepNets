from deepNets import layers
from deepNets import layer_utils
import numpy as np
# from layers import *
# from layer_utils import *

class Net(object):
    def __init__(self):
        self.params = {}
        self.reg = 0.0
        self.dtype = np.float64
        self.seed = None
        self.layer_defs = []
        self.layers_length = 0
        self.layerDims = []
        self.bn_params = []

    def check_syntax(self,layer,inp=False,loss=False):
        layer_check = layer.get("layer_type",None)
        if inp:
            assert layer_check != None and layer_check == "input", "Input layer required"
        elif loss:
            assert layer_check != None and layer_check == "loss", "loss layer required"
            layer_check = layer.get("num_classes",None)
            assert layer_check != None and layer_check > 0 , "Number of classes required"
        else:
            assert layer_check != None and (layer_check == "relu"  or layer_check == "batchnorm" or layer_check == "layernorm" or layer_check == "conv" or layer_check == "pool") , "Incorrect layer found"
            if layer_check == "relu":
                hidden = layer.get("hidden_layers",0)
                assert hidden > 0 ,"Hidden layers not specified"
            elif layer_check == "conv":
                filters = layer.get("filters",0)
                filter_size = layer.get("filter_size",0)
                padding = layer.get("padding",0)
                stride = layer.get("stride",1)
                assert filters > 0 and filter_size > 0 and padding >= 0 and stride >= 1 , "Wrong parameters for conv layer"
            elif layer_check == "pool":
                filter_size = layer.get("filter_size",0)
                stride = layer.get("stride",1)
                assert filter_size > 0 and stride >= 1 , "Wrong parameters for conv layer"
            else:
                self.layers_length -= 1

    def makeLayerDims(self,layer_defs):
        inp_dim = layer_defs[0].get("inp",None)
        if(len(inp_dim[0].shape)==3):
            c,h,w = inp_dim[0].shape
        conv_flag = False
        for i in range(len(layer_defs)-1):
            if i == 0:
                checkConv = layer_defs[i+1].get("layer_type",None)
                if checkConv == "conv":
                    conv_flag = True
                    continue
                get_inp = layer_defs[i].get("inp",None)
                xdim = get_inp[0].shape
                self.layerDims.append(np.prod(xdim))
                continue
            self.check_syntax(layer_defs[i],False,False)
            get_layer = layer_defs[i].get("layer_type",None)
            if get_layer == "relu":
                if conv_flag:
                    conv_flag = False
                    self.layerDims.append(c*h*w)                    
                get_hidden = layer_defs[i].get("hidden_layers",None)
                self.layerDims.append(get_hidden)
            elif get_layer == "conv":                
                dims = [layer_defs[i].get("filters",0),layer_defs[i].get("filter_size",0)]
                assert conv_flag , "Cannot unpack " + str(self.layerDims[-1]) + " into " + str(dims[0]) + " x " + str(dims[1])  + " x " + str(dims[1])
                padding = layer_defs[i].get("padding",0)
                stride = layer_defs[i].get("stride",1)
                #updating the parameters channel,height and width for flattening
                c = dims[0]
                assert (h - dims[1] + 2*padding)%stride == 0, "Conv filter height not proper"
                assert (w - dims[1] + 2*padding)%stride == 0, "Conv filter width not proper"
                h = (h - dims[1] + 2*padding)//stride + 1
                w = (w - dims[1] + 2*padding)//stride + 1
                self.layerDims.append(dims)
            elif get_layer == "pool":
                assert conv_flag , "Input layer cannot be pooled without conv layer on top"
                filter_size = layer_defs[i].get("filter_size",0)
                stride = layer_defs[i].get("stride",1) #Default 1
                assert (h - filter_size)%stride == 0, "Pool filter height not proper"
                assert (w - filter_size)%stride == 0, "Pool filter width not proper"
                h = (h - filter_size)//stride + 1
                w = (w - filter_size)//stride + 1
                self.layers_length -= 1
        #For loss layer
        if conv_flag:
            conv_flag = False
            self.layerDims.append(c*h*w)
        self.layerDims.append(layer_defs[-1].get("num_classes",None))
        return

    def makeLayers(self,layer_defs,initialization="None",weight_scale=1e-3):
        self.layer_defs = layer_defs
        self.layers_length = len(layer_defs) - 1
        input_layer = layer_defs[0]
        self.check_syntax(input_layer,True,False)
        loss_layer = layer_defs[-1]
        self.check_syntax(loss_layer,False,True)
        self.makeLayerDims(layer_defs)
        channel = 0
        for i in range(self.layers_length):   
            if type(self.layerDims[i]) is list:
                if channel == 0:
                    channel = input_layer.get("inp").shape[1] # 3 channel image is preferred
                self.params["W"+str(i+1)] = weight_scale * np.random.randn(self.layerDims[i][0],channel,self.layerDims[i][1],self.layerDims[i][1])
                self.params["b"+str(i+1)] = np.zeros(self.layerDims[i][0])
                self.params["gamma"+str(i+1)] = np.ones(self.layerDims[i][0])
                self.params["beta"+str(i+1)] = np.zeros(self.layerDims[i][0])
                channel = self.layerDims[i][0]
                continue
            #Kaiming He initialization not supported for conv layers
            elif initialization == "kaimenghe":
                self.params["W"+str(i+1)] = np.random.randn(self.layerDims[i],self.layerDims[i+1]) / np.sqrt((self.layerDims[i])/2)
            else:
                self.params["W"+str(i+1)] = weight_scale * np.random.randn(self.layerDims[i], self.layerDims[i+1]) #
            self.params["b"+str(i+1)] = np.zeros(self.layerDims[i+1])
            self.params["gamma"+str(i+1)] = np.ones(self.layerDims[i+1])
            self.params["beta"+str(i+1)] = np.zeros(self.layerDims[i+1])
        self.bn_params = [{"mode":"train"} for i in range(self.layers_length)]
        for k,v in self.params.items():
            self.params[k] = v.astype(self.dtype)
        
    def loss(self,X,y=None):
        X = X.astype(self.dtype)       
        if y is None:
            mode = "test"
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        else:
            mode = "train"
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores,caches = None,{}
        params_idx = 1
        for i in range(1,len(self.layer_defs) - 1):
            layer_type = self.layer_defs[i].get("layer_type")
            if layer_type == "relu":
                W = self.params["W"+str(params_idx)]
                b = self.params["b"+str(params_idx)]
                normalization = self.layer_defs[i-1].get("layer_type",None)
                if normalization == "batchnorm" or normalization == "layernorm":
                    gamma = self.params["gamma"+str(params_idx)]
                    beta = self.params["beta"+str(params_idx)]
                    X,cache = layer_utils.affine_relu_norm_forward(X,W,b,gamma,beta,self.bn_params[params_idx-1],normalization)
                    #X,cache = affine_relu_norm_forward(X,W,b,gamma,beta,self.bn_params[params_idx-1],normalization)
                else:
                    X,cache = layer_utils.affine_relu_forward(X,W,b)
                    #X,cache = affine_relu_forward(X,W,b)               
                caches[params_idx] = cache
                params_idx += 1
            elif layer_type == "conv":
                W = self.params["W"+str(params_idx)]
                b = self.params["b"+str(params_idx)]
                layerBefore = self.layer_defs[i-1].get("layer_type",None)
                layerAfter = self.layer_defs[i+1].get("layer_type",None)
                conv_param = {"padding":self.layer_defs[i].get("padding",0) , "stride" : self.layer_defs[i].get("stride",1)}
                if (layerBefore == "batchnorm" or layerBefore == "layernorm") and layerAfter == "pool":
                    #conv -> batch -> relu -> pool
                    gamma = self.params["gamma"+str(params_idx)]
                    beta = self.params["beta"+str(params_idx)]
                    pool_param = {"filter_size":self.layer_defs[i+1].get("filter_size",None) , "stride" : self.layer_defs[i+1].get("stride",1)}                    
                    X,cache = layer_utils.conv_batch_relu_pool_forward(X,W,b,conv_param,gamma,beta,self.bn_params[params_idx-1],pool_param)
                elif(layerBefore == "batchnorm" or layerBefore == "layernorm"):
                    #conv -> batch -> relu           
                    gamma = self.params["gamma"+str(params_idx)]
                    beta = self.params["beta"+str(params_idx)]
                    X,cache = layer_utils.conv_batch_relu_forward(X, W, b, gamma, beta, conv_param, self.bn_params[params_idx-1])
                elif(layerAfter == "pool" or layerAfter == "batchnorm"):
                    #conv -> relu -> pool
                    if(layerAfter == "pool"):
                        pool_param = {"filter_size":self.layer_defs[i+1].get("filter_size",None) , "stride" : self.layer_defs[i+1].get("stride",1)}                  
                        X,cache = layer_utils.conv_relu_pool_forward(X, W, b, conv_param, pool_param)
                    elif(layerAfter=="batchnorm"):
                        #check if conv -> batch -> pool -> relu
                        if(i+2<len(self.layer_defs)-1):
                            layerAfterBatch = self.layer_defs[i+2].get("layer_type",None)
                            if layerAfterBatch == "pool":
                                gamma = self.params["gamma"+str(params_idx)]
                                beta = self.params["beta"+str(params_idx)]
                                pool_param = {"filter_size":self.layer_defs[i+2].get("filter_size",None) , "stride" : self.layer_defs[i+2].get("stride",1)}                    
                                X,cache = layer_utils.conv_batch_relu_pool_forward(X,W,b,conv_param,gamma,beta,self.bn_params[params_idx-1],pool_param)
                            else:
                                X,cache = layer_utils.conv_relu_forward(X, W, b, conv_param)
                    else:
                        X,cache = layer_utils.conv_relu_forward(X, W, b, conv_param)
                else:
                    #conv-> relu
                    X,cache = layer_utils.conv_relu_forward(X, W, b, conv_param)
                caches[params_idx] = cache
                params_idx += 1
        W = self.params["W"+str(params_idx)]    
        b = self.params["b"+str(params_idx)]
        #Check X shape because shape varies due to conv   
        assert len(X.shape) == 4 or len(X.shape) == 2 , "Shape error"
        conv_flag = False 
        if len(X.shape) == 4:
            X = X.reshape(X.shape[0],np.prod(X[0].shape))
            conv_flag = True
        normalization = self.layer_defs[-1-1].get("layer_type",None)
        if normalization == "batchnorm" or normalization == "layernorm":
            if conv_flag:
                gamma = np.ones(W.shape[0])
                beta = np.zeros(W.shape[0])
            else:
                gamma = self.params["gamma"+str(params_idx-1)]
                beta = self.params["beta"+str(params_idx-1)]
            scores,cache = layer_utils.norm_affine_forward(X,W,b,gamma,beta,self.bn_params[params_idx-1],normalization)
            #scores,cache = norm_affine_forward(X,W,b,gamma,beta,self.bn_params[params_idx-1],normalization)
            caches[params_idx] = cache
        else:
            scores,cache = layers.affine_forward(X,W,b)
            #scores,cache = affine_forward(X,W,b)
            caches[params_idx] = cache
        if y is None:
            return scores
        grads = {}
        loss,dloss = layers.softmax_loss(scores,y)
        #loss,dloss = softmax_loss(scores,y)
        #Regularization
        for i in range(params_idx):
            W = self.params["W"+str(i+1)]
            loss += 0.5 * self.reg * np.sum(W*W)
        normalization = self.layer_defs[-1-1].get("layer_type",None)
        if normalization == "batchnorm" or normalization == "layernorm":
            #affine - batch
            dx,dw,db,dgamma,dbeta = layer_utils.norm_affine_backward(dloss,caches[params_idx],normalization)
            #dx,dw,db,dgamma,dbeta = norm_affine_backward(dloss,caches[params_idx],normalization)
            grads["gamma"+str(params_idx-1)] =  dgamma
            grads["beta"+str(params_idx-1)] = dbeta
        else:
            dx,dw,db = layers.affine_backward(dloss,caches[params_idx])
            #dx,dw,db = affine_backward(dloss,caches[params_idx])
        grads["W"+str(params_idx)] = dw + self.reg * self.params["W"+str(params_idx)]
        grads["b"+str(params_idx)] = db
        params_idx -=1
        for i in range(len(self.layer_defs)-1,0,-1):
            layer_type = self.layer_defs[i].get("layer_type")
            if layer_type == "relu":
                normalization = self.layer_defs[i-1].get("layer_type",None)
                if normalization == "batchnorm" or normalization == "layernorm":
                    #relu - batch - affine
                    dx,dw,db,dgamma,dbeta = layer_utils.affine_relu_norm_backward(dx,caches[params_idx],normalization) 
                    #dx,dw,db,dgamma,dbeta = affine_relu_norm_backward(dx,caches[params_idx],normalization)                   
                    grads["W"+str(params_idx)] = dw + self.reg * self.params["W"+str(params_idx)]
                    grads["b"+str(params_idx)] = db
                    #check if already present
                    if len(grads.get("gamma"+str(params_idx),[]))>0:
                        grads["gamma"+str(params_idx)] +=  dgamma
                        grads["beta"+str(params_idx)] += dbeta
                    else:
                        grads["gamma"+str(params_idx)] =  dgamma
                        grads["beta"+str(params_idx)] = dbeta
                else:
                    #relu - affine
                    dx,dw,db = layer_utils.affine_relu_backward(dx,caches[params_idx])
                    #dx,dw,db = affine_relu_backward(dx,caches[params_idx])
                    grads["W"+str(params_idx)] = dw + self.reg * self.params["W"+str(params_idx)]
                    grads["b"+str(params_idx)] = db
                params_idx -= 1
            if layer_type == "conv":
                #Converting flattened to 4 dimensional
                if len(dx.shape) != 4:
                    if(len(caches[params_idx])==2):
                        _,r = caches[params_idx]
                    elif(len(caches[params_idx])==3):
                        _,_,r = caches[params_idx]
                    elif(len(caches[params_idx])==4):
                        _,_,_,r = caches[params_idx]                  
                    if type(r) is not np.ndarray:
                        r = r[1] #getting the pool dimension
                    N,C,H,W = r.shape
                    dx = np.reshape(dx,(N,C,H,W))
                layerAfter = self.layer_defs[i+1].get("layer_type",None)
                layerBefore = self.layer_defs[i-1].get("layer_type",None)
                if layerBefore == "batchnorm" and layerAfter == "pool":
                    #pool - relu - batch - conv
                    dx,dw,db ,dgamma,dbeta = layer_utils.conv_batch_relu_pool_backward(dx, caches[params_idx])
                    grads["W"+str(params_idx)] = dw + self.reg * self.params["W"+str(params_idx)]
                    grads["b"+str(params_idx)] = db
                    #check if already present
                    if len(grads.get("gamma"+str(params_idx),[]))>0:
                        grads["gamma"+str(params_idx)] +=  dgamma
                        grads["beta"+str(params_idx)] += dbeta
                    else:
                        grads["gamma"+str(params_idx)] =  dgamma
                        grads["beta"+str(params_idx)] = dbeta
                elif layerBefore == "batchnorm":
                    #relu - batch - conv
                    dx,dw,db,dgamma,dbeta = layer_utils.conv_batch_relu_backward(dx,caches[params_idx])
                    grads["W"+str(params_idx)] = dw + self.reg * self.params["W"+str(params_idx)]
                    grads["b"+str(params_idx)] = db
                    #check if already present
                    if len(grads.get("gamma"+str(params_idx),[]))>0:
                        grads["gamma"+str(params_idx)] +=  dgamma
                        grads["beta"+str(params_idx)] += dbeta
                    else:
                        grads["gamma"+str(params_idx)] =  dgamma
                        grads["beta"+str(params_idx)] = dbeta
                elif layerAfter == "batchnorm" or layerAfter == "pool":
                    if layerAfter == "pool":
                        #pool - relu - conv
                        dx,dw,db = layer_utils.conv_relu_pool_backward(dx,caches[params_idx])
                        grads["W"+str(params_idx)] = dw + self.reg * self.params["W"+str(params_idx)]
                        grads["b"+str(params_idx)] = db
                    elif(i+2<len(self.layer_defs)-1):
                        layerAfterBatch = self.layer_defs[i+2].get("layer_type",None)
                        if layerAfterBatch == "pool":
                            #pool - relu - batch - conv
                            dx,dw,db ,dgamma,dbeta = layer_utils.conv_batch_relu_pool_backward(dx, caches[params_idx])
                            grads["W"+str(params_idx)] = dw + self.reg * self.params["W"+str(params_idx)]
                            grads["b"+str(params_idx)] = db
                            #check if already present
                            if len(grads.get("gamma"+str(params_idx),[]))>0:
                                grads["gamma"+str(params_idx)] +=  dgamma
                                grads["beta"+str(params_idx)] += dbeta
                            else:
                                grads["gamma"+str(params_idx)] =  dgamma
                                grads["beta"+str(params_idx)] = dbeta
                        else:
                            #relu - conv
                            dx,dw,db = layer_utils.conv_relu_backward(dx,caches[params_idx])
                            grads["W"+str(params_idx)] = dw + self.reg * self.params["W"+str(params_idx)]
                            grads["b"+str(params_idx)] = db
                    else:
                        #relu - conv
                        dx,dw,db = conv_relu_backward(dx,caches[params_idx])
                        grads["W"+str(params_idx)] = dw + self.reg * self.params["W"+str(params_idx)]
                        grads["b"+str(params_idx)] = db
                else:
                    #relu - conv
                    dx,dw,db = layer_utils.conv_relu_backward(dx,caches[params_idx])
                    grads["W"+str(params_idx)] = dw + self.reg * self.params["W"+str(params_idx)]
                    grads["b"+str(params_idx)] = db
                params_idx -= 1
        return loss,grads 
                

                
                
            
