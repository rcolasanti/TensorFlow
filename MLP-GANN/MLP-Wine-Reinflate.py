
# Load Libraries


import tensorflow as tf
from sklearn.metrics import confusion_matrix
from pprint import pprint
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import json
sess = tf.InteractiveSession()
    


# TensorFlow MLP Class



class TfAnn(object):
    
    def __init__(self):
        self.hidden=[]
        self.np_hidden=[]
        
        self.n_nodes=[]
        self.n_classes = 0
        self.n_hiden_layers = 0

    # create empty network for training
    def init_empty(self,layers,n_classes,size):
        self.n_classes = n_classes
        self.n_hiden_layers = len(layers)
        for i in range(self.n_hiden_layers):
            self.hidden.append({'weights':[],'biases':[]})
            self.np_hidden.append({'weights':[],'biases':[]})
            self.n_nodes.append(layers[i])
        self.output_layer = {'weights':[],'biases':[]}
        self.np_output_layer={"weights":[],"biases":[]}

        for i in range(self.n_hiden_layers):
            self.hidden[i] = {'weights':tf.Variable(tf.random_normal([size, self.n_nodes[i]])),
                      'biases':tf.Variable(tf.random_normal([self.n_nodes[i]]))}
            
        self.output_layer = {'weights':tf.Variable(tf.random_normal([self.n_nodes[-1], self.n_classes])),
                        'biases':tf.Variable(tf.random_normal([self.n_classes]))}

    
    # Reinflate network from json description
    def init_json(self,jfile):
        with open(jfile) as json_data:
            tf_data = json.load(json_data)
        self.n_classes = tf_data["n_classes"]
        self.n_hiden_layers = tf_data["n_hiden_layers"]
        self.hidden =tf_data["hidden"]
        self.output_layer =tf_data["output"]


        
    def create(self,data):
        # This is the heart of the ann where multiply the data by the wights to the layers 
        for i in range(self.n_hiden_layers):
            layer = tf.add(tf.matmul(data,self.hidden[i]['weights']), self.hidden[i]['biases'])
            layer= tf.nn.relu(layer)
        output =  tf.add(tf.matmul(layer,self.output_layer['weights']) , self.output_layer['biases'])
        return output
    
    
    #Save a trained network as a json file
    def extract(self,jfile):
        for i in range(self.n_hiden_layers):
            self.np_hidden[i]["weights"] = neural_network_model.hidden[i]["weights"].eval().tolist()
            self.np_hidden[i]["biases"] = neural_network_model.hidden[i]["biases"].eval().tolist()
        self.np_output_layer["weights"] = neural_network_model.output_layer["weights"].eval().tolist()
        self.np_output_layer["biases"] = neural_network_model.output_layer["biases"].eval().tolist()
        with open(jfile,"w") as jout:
            json.dump({"n_classes":self.n_classes, # number of input classifier classes
                       "n_hiden_layers":self.n_hiden_layers, # number of 
                       "hidden":self.np_hidden,# weights and biases
                       # each layer is defined by dict {'weights':[],'biases':[]}
                       "output":self.np_output_layer} # as for hidden
                      ,jout)    
        
def test_neural_network(neural_network_model,x_data_test,y_data_test):
    # set up network
    x = tf.placeholder('float')
    prediction = neural_network_model.create(x)    
    
    # ren test data
    y_test_res=(sess.run(prediction,feed_dict={x:x_data_test}))       
    # the correct data
    true_class=np.argmax(y_data_test,1)
    
    # get the index of the outpt array with heighest value
    predicted_class=np.argmax(y_test_res,1)
    
    # calculate confusion matix
    cm = confusion_matrix(predicted_class,true_class)
    cm = cm.astype('float')*10000 / cm.sum(axis=1)[:, np.newaxis]
    cm = np.nan_to_num(cm, copy=True)
    cm = cm.astype('int')
    return cm*0.01

with open("datasets/wine/wine_test.json") as json_data:
    test_dataset = json.load(json_data)

test_x = np.asarray(test_dataset["attribs"])
test_y = np.asarray(test_dataset["target_hot"])

neural_network_model = TfAnn()
neural_network_model.init_json("classifiers/wine-mlp.json")
cf = test_neural_network(neural_network_model,test_x,test_y)
pprint(cf)



