![deepNets_Logo](https://github.com/DeepakVelmurugan/deepNets/blob/main/deepNets.png)
-------------------------------------------------------------------------------------
<h1>DeepNets</h1>
<ul>
<li>Deep learning library for constructing neutral networks.</li>
<li>Supports neural network modules(conv layers,fully-connected
layers,non-linearities) and classification(softmax) functions.</li>
<li>Ability to specify and train Convolutional Networks that process images.</li>
</ul>

<p>Latest version : 0.1.7 :white_check_mark:</p>

<b>Install using pip :</b></br>
```pip3 install deepNets```

<b>Example for constructing a simple fully connected network</b></br>
```python
    #Import files you need
    from deepNets import Net as nt
    from deepNets import Trainer as trn
    #Create a list of dictionaries to include your layers
    make_layers = []
    #Remember to specify the first layer as input, also note the layer type syntax
    make_layers.append({"layer_type":"input","inp":x})
    #You can add relu non linearities with hidden units accounting to any valid number
    make_layers.append({"layer_type":"relu","hidden_layers":100})
    #Normalization layers like batchnorm and layernorm are supported
    make_layers.append({"layer_type":"batchnorm"})
    #Always specify the last layer as loss
    make_layers.append({"layer_type":"loss","num_classes":10})
    #Create a object for neural net
    net = nt.Net()
    #This function constructs your neural net
    #You can also specify weight_scale = 1e-2 . Default weight_scale = 1e-3
    net.makeLayers(make_layers)
    #The Trainer function has net object as input
    #Specify training data(x,y) and validation data(x_val,y_val)
    #Set the update rules to sgd,rmsprop,adam etc and learning_rate and batch_size
    trainer = trn.Trainer(net,x,y,(x_val,y_val),update_rule="sgd",optim_config={'learning_rate':0.001},batch_size=100,verbose=True)
    #Setting verbose to true will print values of training accuracy and val accuracy
    trainer.train()
```
</br>

<b>Example for constructing a simple convolutional network on images</b></br>
```python
    #Import files you need
    from deepNets import Net as nt
    from deepNets import Trainer as trn
    make_layers = []
    make_layers.append({"layer_type":"input","inp":x})
    #Add Convolutional layers you can specify padding,stride,filter_size etc
    make_layers.append({"layer_type":"conv","filters":16,"filter_size":5,"padding":2})
    #You can also pooling layers currently maxpooling is done
    make_layers.append({"layer_type":"pool","filter_size":2,"stride":2})
    #No need for flatten layers , the below layer flattens and uses a dense layer
    make_layers.append({"layer_type":"relu","hidden_layers":50})
    #Make sure the last layer is loss
    make_layers.append({"layer_type":"loss","num_classes":10})
    net = nt.Net()
    #You can add regularization as below
    net.reg = 0.5
    net.makeLayers(make_layers)
    trainer = trn.Trainer(net,x,y,(x_val,y_val),update_rule="adam",optim_config={'learning_rate':0.01},batch_size=50)
    #Train your network
    trainer.train()
```
<i>For example and tutorial : [predicting numbers using deepNets in MNIST dataset](https://github.com/DeepakVelmurugan/deepNets/blob/main/MNISTdeepNets.ipynb)</i>
<pre>Currently only batch norm is supported for Conv nets.</pre> 
<i>[For info visit Pypi page](https://pypi.org/project/deepNets/0.1.7/)</i>
</br>
<i>The current version does not support GPU :shit:</i>
