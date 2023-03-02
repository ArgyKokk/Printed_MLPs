import tensorflow as tf

import pandas as pd
import seaborn as sb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from callbacks import all_callbacks
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import model_save_quantized_weights
import tensorflow.compat.v1 as tf1
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning

from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qkeras.utils import model_save_quantized_weights
import hls4ml
import write_mlp_mergemult_ps as wv
import clustering as cl
from importlib import reload
from joblib import dump
import os


def generate(model,identity,relusize,weight_size1,bias_size1,weight_size2,bias_size2,inputsize, relusize_int,layer1,layer2,layer3,X_train,Y_train,X_test,Y_test):

    #1) extract the solution's parameters
    relusize_f = int(relusize)
    weight_size_f1 = int(weight_size1)
    bias_size_f1 = int(bias_size1)
    weight_size_f2 = int(weight_size2)
    bias_size_f2 = int(bias_size2)
    input_s = int(inputsize)
    reluint = int(relusize_int)
    norm = 2**input_s
    layer_1 = layer1
    layer_2 = layer2
    layer_3 = layer3
    
    adam = Adam()
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    accuracy=model.evaluate(X_test,Y_test)

    #2) normalize the input
    sc = MinMaxScaler(feature_range=(0,1))
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    for i in range(0,len(X_train)):
        X_train[i] = [round(x*norm)/norm for x in X_train[i]]
    for i in range(0,len(X_test)):
        X_test[i] = [round(x*norm)/norm for x in X_test[i]]

    weight_bias_size=[ [ (weight_size_f1,1), (bias_size_f1,1) ], [ (weight_size_f2,1), (bias_size_f2,1) ] ]
    relu_size=(relusize_f,reluint)

    #3) get the model's weights
    w1=model.get_weights()

    #4) weight clustering for the 1st layer
    target_layer=0
    iterations=0
    new_accuracy=1
    accuracies=[]
    weights=[]
    clustering_info=0
    while(new_accuracy > 0):
        reload(cl)
        new_weights, new_accuracy = cl.clustering(model,target_layer,layer1,layer2,layer3,X_test,Y_test,X_train,Y_train, weight_bias_size, relu_size);
        iterations = iterations + 1
        print("Accuracy of iteration "+str(iterations)+" for the target layer "+str(target_layer)+" is: "+str(new_accuracy))
        model.set_weights(new_weights)
        accuracies.append(new_accuracy)
        weights.append(new_weights)
        if new_accuracy < accuracy[1]-0.03:
            break

    max_value = max(accuracies)
    if max_value >= accuracy[1]-0.03:
        max_index = accuracies.index(max_value)
        model.set_weights(weights[max_index])
        clustering_info=1
    else:
        model.set_weights(w1)
    previous_weights=model.get_weights()

    #5) weight clustering for the second layer
    target_layer=2
    iterations=0
    new_accuracy=1
    accuracies_1=[]
    weights_1=[]

    while(new_accuracy > 0):
        reload(cl)
        new_weights, new_accuracy = cl.clustering(model,target_layer,layer1,layer2,layer3,X_test,Y_test,X_train,Y_train, weight_bias_size, relu_size);
        iterations = iterations + 1
        print("Accuracy of iteration "+str(iterations)+" for the target layer "+str(target_layer)+" is: "+str(new_accuracy))
        model.set_weights(new_weights)
        accuracies_1.append(new_accuracy)
        weights_1.append(new_weights)
        if new_accuracy < accuracy[1]-0.03:
            break

    max_value = max(accuracies_1)
    if max_value >= accuracy[1]-0.03:
        max_index = accuracies_1.index(max_value)
        model.set_weights(weights_1[max_index])
        clustering_info=1
    else:
        model.set_weights(previous_weights)
    
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    model_save_quantized_weights(model, "test_shared_weights")
    accuracy=model.evaluate(X_test,Y_test)
    #4)Create the verilog
    def aptype_to_size(ap):
      if "ap_int" in ap:
          s=ap.split('<')[1].split('>')[0]
          i=s
      else:
          s,i=ap.split('<')[1].split('>')[0].split(',')
      return (int(s),int(i))

    #scale weights
    allweights=[t*2**(weight_bias_size[j//2][j%2][0]-weight_bias_size[j//2][j%2][1]) for j,t in enumerate(model.get_weights())]
    #cast weights to integer & transpose
    allweightsT=[wl.T.astype(int).tolist() for wl in allweights]
    weight_list=allweightsT[0::2]
    bias_list=allweightsT[1::2]
    #set params
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    last_layer="linear"
    input_size = (input_s,0)
    relu_size= aptype_to_size(config['LayerName']['relu1']['Precision']['result'])

    weight_bias_size= [
      [aptype_to_size(config['LayerName']['fc1']['Precision']['weight']),aptype_to_size(config['LayerName']['fc1']['Precision']['bias'])],
      [aptype_to_size(config['LayerName']['output']['Precision']['weight']),aptype_to_size(config['LayerName']['output']['Precision']['bias'])]
    ]
    sum_relu_size=[
      [(16,6),relu_size],
      [(16,6),(32,6)]
    ]
    filename = "top_pareto"+str(identity)+"_"+str(relusize_f)+str(weight_size_f1)+str(bias_size_f1)+str(weight_size_f2)+str(bias_size_f2)+str(input_s)+str(reluint)+"-"+str(accuracy[1])+"WS"+str(clustering_info)
    x = os.path.join("./paretos_weight_sharing", filename)
    f=open(x,"w")

    wv.write_mlp_verilog(f, input_size, bias_list, weight_list, weight_bias_size, sum_relu_size,last_layer)
    f.close()

    f=open("sim.Xtest","w")
    np.savetxt(f,(X_test*2**input_size[0]).astype(int),fmt='%d',delimiter=' ')
    f.close()

    dump(np.argmax(Y_test,axis=1), "sim.Ytest")

    print("copy top.v, sim.Xtest, and sim.Ytest to the synopsys project ")

