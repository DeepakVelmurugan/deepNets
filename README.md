<h1>DeepNets</h1>
<ul>
<li>Deep learning library for constructing neutral networks.</li>
<li>Currently supports neural network modules(fully-connected
layers,non-linearities) and classification(softmax) functions.</li>
</ul>
</br>
<i>Currently under construction.</i>
</br>

<p>Latest version : 0.1.4</p>

<b>Install using pip :</b></br>
```pip3 install deepNets```

<b>Example Code</b></br>
```python
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
    net = Net()
    #This function constructs your neural net
    #You can also specify weight_scale = 1e-2 . Default weight_scale = 1e-3
    net.makeLayers(make_layers)
    #The Trainer function has net object as input
    #Specify training data(x,y) and validation data(x_val,y_val)
    #Set the update rules to sgd,rmsprop,adam etc and learning_rate and batch_size
    trainer = Trainer(net,x,y,(x_val,y_val),update_rule="sgd",lr_decay=0.95,optim_config={'learning_rate':0.001},batch_size=100,verbose=True)
    #Setting verbose to true will print values of training accuracy and val accuracy
    trainer.train()
```
<i>[For info visit Pypi page](https://pypi.org/project/deepNets/0.1.4/)</i>
</br>
<i>The current version does not support GPU</i>
