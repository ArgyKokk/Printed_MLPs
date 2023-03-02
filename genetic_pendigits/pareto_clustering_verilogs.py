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
#import clustering as cl
from importlib import reload
from joblib import dump
import tensorflow_model_optimization as tfmot
import os

def generate(model,identity,relusize,weight_size1,bias_size1,weight_size2,bias_size2,inputsize, int_relu, w1_int, b1_int, w2_int, b2_int, layer1,layer2,layer3,X_train,Y_train,X_test,Y_test):

    #1) extract the solution's parameters
    relusize_f = 8#int(relusize)
    weight_size_f1 = 7#int(weight_size1)
    bias_size_f1 = 7#nt(bias_size1)
    weight_size_f2 = 7#int(weight_size2)
    bias_size_f2 = 7#int(bias_size2)
    input_s = 4#int(inputsize)
    relusize_int = 2#int(int_relu)
    w1=2#int(w1_int)
    b1=1#int(b1_int)
    w2=2#int(w2_int)
    b2=1#int(b2_int)
    norm = 2**input_s
    layer_1 = layer1
    layer_2 = layer2
    layer_3 = layer3

    #2) normalize the input
    sc = MinMaxScaler(feature_range=(0,0.9))
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    for i in range(0,len(X_train)):
        X_train[i] = [int(x*norm)/norm for x in X_train[i]]
    for i in range(0,len(X_test)):
        X_test[i] = [int(x*norm)/norm for x in X_test[i]]

    weight_bias_size=[ [ (weight_size_f1,w1+1), (bias_size_f1,b1+1) ], [ (weight_size_f2,w2+1), (bias_size_f2,b2+1) ] ]
    relu_size=(relusize_f,relusize_int)

    #3) get the model's weights
    w1=model.get_weights()


    #4) define a max number of clusters and start clustering
    max_clusters=9

    for i in range(2,max_clusters):
        #initialize the model
        model_cl = Sequential()
        model_cl.add(Dense(layer2, input_shape=(layer1,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
        model_cl.add(Activation(activation='relu', name='relu1'))
        model_cl.add(Dense(layer3, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
        model_cl.add(Activation(activation='softmax', name='softmax'))

        model_cl.set_weights(w1)
        cluster_weights = tfmot.clustering.keras.cluster_weights
        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

        clustering_params = {
            'number_of_clusters': i,
            'cluster_centroids_init': CentroidInitialization.LINEAR
        }
        # Cluster a whole model
        clustered_model = cluster_weights(model_cl, **clustering_params)

        # Use smaller learning rate for fine-tuning clustered model
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

        clustered_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=opt,
            metrics=['accuracy'])

        clustered_model.summary()

        new_Y_train = np.argmax(Y_train,axis=1)

        clustered_model.fit(
            X_train,
            new_Y_train,
            batch_size=1,
            epochs=1,
            validation_split=0.2)
       
        final_clustered_model = tfmot.clustering.keras.strip_clustering(clustered_model)
        clustered_model_weights=final_clustered_model.get_weights()

        #7) create a QKeras model
        model_clustered_quantized = Sequential()
        model_clustered_quantized.add(QDense(layer_2, input_shape=(layer_1,), name='fc1', kernel_quantizer=quantized_bits(weight_bias_size[0][0][0],weight_bias_size[0][0][1]-1,alpha=1,use_stochastic_rounding=True),bias_quantizer=quantized_bits(weight_bias_size[0][1][0],weight_bias_size[0][1][1]-1,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)   ))
        model_clustered_quantized.add(QActivation(activation=quantized_relu(relu_size[0],relu_size[1],use_stochastic_rounding=False), name='relu1'))
        model_clustered_quantized.add(QDense(layer3, name='output',
                    kernel_quantizer=quantized_bits(weight_bias_size[1][0][0],weight_bias_size[1][0][1]-1,alpha=1,use_stochastic_rounding=True), bias_quantizer=quantized_bits(weight_bias_size[1][1][0],weight_bias_size[1][1][1]-1,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001 ) ))
        model_clustered_quantized.add(Activation(activation='softmax', name='softmax'))

        model_clustered_quantized.set_weights(clustered_model_weights)
        adam = Adam()
        model_clustered_quantized.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
        model_save_quantized_weights(model_clustered_quantized, "test_weights_tf_cluster")
  
        accuracy=model_clustered_quantized.evaluate(X_test,Y_test)
        

        #8)Create the verilog
        def aptype_to_size(ap):
           if "ap_int" in ap:
               s=ap.split('<')[1].split('>')[0]
               i=s
           else:
               s,i=ap.split('<')[1].split('>')[0].split(',')
           return (int(s),int(i))

        #scale weights
        allweights=[t*2**(weight_bias_size[j//2][j%2][0]-weight_bias_size[j//2][j%2][1]) for j,t in enumerate(model_clustered_quantized.get_weights())]
        #cast weights to integer & transpose
        allweightsT=[wl.T.astype(int).tolist() for wl in allweights]
        weight_list=allweightsT[0::2]
        bias_list=allweightsT[1::2]
        #set params
        config = hls4ml.utils.config_from_keras_model(model_clustered_quantized, granularity='name')

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

        filename = "top_pareto"+str(identity)+"_"+str(relusize_f)+str(weight_size_f1)+str(bias_size_f1)+str(weight_size_f2)+str(bias_size_f2)+str(input_s)+str(relusize_int)+str(i)+"-"+str(accuracy[1])+str("C")
        x = os.path.join("./paretos_clustering", filename)        
        f =open(x,"w")

        wv.write_mlp_verilog(f, input_size, bias_list, weight_list, weight_bias_size, sum_relu_size,last_layer)
        f.close()

        f=open("sim.Xtest"+str(identity),"w")
        np.savetxt(f,(X_test*2**input_size[0]).astype(int),fmt='%d',delimiter=' ')
        f.close()

        dump(np.argmax(Y_test,axis=1), "sim.Ytest")


        print("copy top.v, sim.Xtest, and sim.Ytest to the synopsys project ")


