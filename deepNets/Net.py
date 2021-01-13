from deepNets import layers
from deepNets import layer_utils
import numpy as np

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
            # layer_check  = layer.get("inp",0)
            # layer_length = len(layer_check.shape)
            # assert layer_length != 0 and layer_length == 4, "Expected input size N,C,H,W"
            return
        elif loss:
            assert layer_check != None and layer_check == "loss", "loss layer required"
            layer_check = layer.get("num_classes",None)
            assert layer_check != None and layer_check > 0 , "Number of classes required"
            return
        else:
            assert layer_check != None and (layer_check == "relu"  or layer_check == "batchnorm" or layer_check == "layernorm") , "Incorrect layer found"
            if layer_check == "relu":
                hidden = layer.get("hidden_layers",0)
                assert hidden > 0 ,"Hidden layers not specified"
            else:
                self.layers_length -= 1

    def makeLayerDims(self,layer_defs):
        for i in range(len(layer_defs)-1):
            if i == 0:
                get_inp = layer_defs[i].get("inp",None)
                xdim = get_inp[0].shape
                self.layerDims.append(np.prod(xdim))
                continue
            self.check_syntax(layer_defs[i],False,False)
            get_layer = layer_defs[i].get("layer_type",None)
            if get_layer == "relu":
                get_hidden = layer_defs[i].get("hidden_layers",None)
                self.layerDims.append(get_hidden)
        #For loss layer
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
        for i in range(self.layers_length):
            #Kaiming He initialization
            if initialization == "kaimenghe":
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
                else:
                    X,cache = layer_utils.affine_relu_forward(X,W,b)              
                caches[params_idx] = cache
                params_idx += 1
        W = self.params["W"+str(params_idx)]    
        b = self.params["b"+str(params_idx)]
        normalization = self.layer_defs[-1-1].get("layer_type",None)
        if normalization == "batchnorm" or normalization == "layernorm":
            gamma = self.params["gamma"+str(params_idx-1)]
            beta = self.params["beta"+str(params_idx-1)]     
            scores,cache = layer_utils.norm_affine_forward(X,W,b,gamma,beta,self.bn_params[params_idx-1],normalization)
            caches[params_idx] = cache
        else:
            scores,cache = layers.affine_forward(X,W,b)
            caches[params_idx] = cache
        if y is None:
            return scores
        grads = {}
        loss,dloss = layers.softmax_loss(scores,y)
        #Regularization
        for i in range(params_idx):
            W = self.params["W"+str(i+1)]
            loss += 0.5 * self.reg * np.sum(W*W)
        normalization = self.layer_defs[-1-1].get("layer_type",None)
        if normalization == "batchnorm" or normalization == "layernorm":
            dx,dw,db,dgamma,dbeta = layer_utils.norm_affine_backward(dloss,caches[params_idx],normalization)
            grads["gamma"+str(params_idx-1)] =  dgamma
            grads["beta"+str(params_idx-1)] = dbeta
        else:
            dx,dw,db = layers.affine_backward(dloss,caches[params_idx])
        grads["W"+str(params_idx)] = dw + self.reg * self.params["W"+str(params_idx)]
        grads["b"+str(params_idx)] = db
        params_idx -=1
        for i in range(len(self.layer_defs)-1,0,-1):
            layer_type = self.layer_defs[i].get("layer_type")
            if layer_type == "relu":
                normalization = self.layer_defs[i-1].get("layer_type",None)
                if normalization == "batchnorm" or normalization == "layernorm":
                    dx,dw,db,dgamma,dbeta = layer_utils.affine_relu_norm_backward(dx,caches[params_idx],normalization)                   
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
                    dx,dw,db = layer_utils.affine_relu_backward(dx,caches[params_idx])
                    grads["W"+str(params_idx)] = dw + self.reg * self.params["W"+str(params_idx)]
                    grads["b"+str(params_idx)] = db
                params_idx -= 1
        return loss,grads 
                

                
                
            