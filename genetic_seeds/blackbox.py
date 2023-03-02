from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

import tensorflow as tf
import os
import pandas as pd
import seaborn as sb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from callbacks import all_callbacks
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
import tensorflow.compat.v1 as tf1
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from qkeras.utils import model_save_quantized_weights
from sklearn.preprocessing import MinMaxScaler


def blackbox(wb1,wb2,relusize,weight_size1, bias_size1, weight_size2, bias_size2, sparsity, inputsize,int_relusize ,layer_1,layer_2,layer_3,X_test,Y_test,X_train,Y_train):

    model = Sequential()
    sparsity_val = float(sparsity / 10)
    relusize_f = int(relusize)
    weight_size_f1 = int(weight_size1)
    bias_size_f1 = int(bias_size1)
    weight_size_f2 = int(weight_size2)
    bias_size_f2 = int(bias_size2)
    input_s = int(inputsize)
    int_relu = int(int_relusize)
    
    norm = 2**input_s
    sc = MinMaxScaler(feature_range=(0,0.9))
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    for i in range(0,len(X_train)):
       X_train[i] = [int(x*norm)/norm for x in X_train[i]]
    for i in range(0,len(X_test)):
       X_test[i] = [int(x*norm)/norm for x in X_test[i]]
    
    
    print("INPUT IS " + str(input_s)+ "NORM IS " + str(norm)+ "RELU IS "+str(relusize_f)+" INT RELU" +str(int_relu)+" WEIGHTS 1 ARE "+str(weight_size_f1)+" BIASES 1 ARE "+str(bias_size_f1)+" WEIGHTS 2 ARE "+str(weight_size_f2)+" BIASES 2 ARE "+str(bias_size_f2)+" SPARSITY : "+str(sparsity_val))

    weight_bias_size=[ [ (weight_size_f1,1), (bias_size_f1,1) ], [ (weight_size_f2,1), (bias_size_f2,1) ] ]
    relu_size=(relusize_f,int_relu)

    model.add(QDense(layer_2, input_shape=(layer_1,), name='fc1', kernel_quantizer=quantized_bits(weight_bias_size[0][0][0],0,alpha=1,use_stochastic_rounding=True),bias_quantizer=quantized_bits(weight_bias_size[0][1][0],0,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)   ))
    model.add(QActivation(activation=quantized_relu(relu_size[0],relu_size[1],use_stochastic_rounding=False), name='relu1'))
    model.add(QDense(layer_3, name='output',
                    kernel_quantizer=quantized_bits(weight_bias_size[1][0][0],0,alpha=1,use_stochastic_rounding=True), bias_quantizer=quantized_bits(weight_bias_size[1][1][0],0,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001 ) ))
    model.add(Activation(activation='softmax', name='softmax'))


    pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(sparsity_val, begin_step=300, frequency=100)}
    model = prune.prune_low_magnitude(model, **pruning_params)

    model.layers[0].set_weights(wb1)
    model.layers[2].set_weights(wb2)

    adam = Adam(lr=0.003)

    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    callbacks= all_callbacks( outputDir = 'seeds_classification_prune')
    callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())
    model.fit(X_train, Y_train, batch_size=1,
              epochs=14,validation_split=0.2, verbose=0, shuffle=True,
              callbacks = callbacks.callbacks);
    model = strip_pruning(model)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    model_save_quantized_weights(model, "test_weights")


    accuracy=model.evaluate(X_test,Y_test)
    #print(model.get_weights())
    print(" ACCURACY IS "+str(accuracy[1]) )
    print("INPUT WAS " + str(input_s)+ "NORM WAS " + str(norm)+ "RELU WAS "+str(relusize_f)+ "INT RELU" +str(int_relu)+" WEIGHTS 1 WERE "+str(weight_size_f1)+" BIASES 1 WERE "+str(bias_size_f1)+" WEIGHTS 2 ARE "+str(weight_size_f2)+" BIASES 2 WERE "+str(bias_size_f2)+" SPARSITY : "+str(sparsity_val))

    return accuracy[1], model.get_weights()

