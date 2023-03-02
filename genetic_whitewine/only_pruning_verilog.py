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
from joblib import dump
import os


def generate(identity,layer1,layer2,layer3,lrate,nepochs,X_train,Y_train,X_test,Y_test,pretrained1,pretrained2):

    #1) extract the solution's parameters
    sparsity_vals = [0.8]
    relusize_f = 8
    weight_size_f1 = 8
    bias_size_f1 = 8
    weight_size_f2 = 8
    bias_size_f2 = 8
    input_s = 4
    norm = 2**input_s
    layer_1 = layer1
    layer_2 = layer2
    layer_3 = layer3
    wb1 = pretrained1
    wb2 = pretrained2
    learning_rate = lrate
    num_epochs = nepochs

    #2) normalize the input
    sc = MinMaxScaler(feature_range=(0,0.9))
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    for i in range(0,len(X_train)):
        X_train[i] = [int(x*norm)/norm for x in X_train[i]]
    for i in range(0,len(X_test)):
        X_test[i] = [int(x*norm)/norm for x in X_test[i]]

    weight_bias_size=[ [ (weight_size_f1,2), (bias_size_f1,1) ], [ (weight_size_f2,2), (bias_size_f2,1) ] ]
    relu_size=(relusize_f,1)
    for i in sparsity_vals:
        model = Sequential()
        model.add(QDense(layer_2, input_shape=(layer_1,), name='fc1', kernel_quantizer=quantized_bits(weight_bias_size[0][0][0],1,alpha=1,use_stochastic_rounding=True),bias_quantizer=quantized_bits(weight_bias_size[0][1][0],0,alpha=1),
                        kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)   ))
        model.add(QActivation(activation=quantized_relu(relu_size[0],relu_size[1],use_stochastic_rounding=False), name='relu1'))
        model.add(QDense(layer_3, name='output',
                        kernel_quantizer=quantized_bits(weight_bias_size[1][0][0],1,alpha=1,use_stochastic_rounding=True), bias_quantizer=quantized_bits(weight_bias_size[1][1][0],0,alpha=1),
                        kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001 ) ))
        model.add(Activation(activation='softmax', name='softmax'))


        pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(i, begin_step=300, frequency=100)}
        model = prune.prune_low_magnitude(model, **pruning_params)
        model.layers[0].set_weights(wb1)
        model.layers[2].set_weights(wb2)

        #3)QAT retrainning
        adam = Adam(lr=learning_rate)
        model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
        callbacks= all_callbacks( outputDir = 'model_seeds_classification_Pruning')
        callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())
        model.fit(X_train, Y_train, batch_size=1,
                epochs=num_epochs,validation_split=0.2, verbose=0, shuffle=True,
                callbacks = callbacks.callbacks);
        model = strip_pruning(model)
        model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
        model_save_quantized_weights(model, "test_weights")
  
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
            [(32,6),relu_size],
            [(32,6),(32,6)]
        ]

        filename = "top_Pruning_"+str(i*100)+"-"+str(accuracy[1])
        x = os.path.join("./only_pruning", filename)
        f=open(x,"w")

        wv.write_mlp_verilog(f, input_size, bias_list, weight_list, weight_bias_size, sum_relu_size,last_layer)
        f.close()

        f=open("sim.Xtest4bit","w")
        np.savetxt(f,(X_test*2**input_size[0]).astype(int),fmt='%d',delimiter=' ')
        f.close()

        dump(np.argmax(Y_test,axis=1), "sim.Ytest")

        print("copy top.v, sim.Xtest, and sim.Ytest to the synopsys project ")

