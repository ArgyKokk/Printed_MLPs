import sys
import string

import pandas as pd
import seaborn as sb

import tensorflow.compat.v1 as tf1
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
import numpy as np
from heapq import nsmallest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from callbacks import all_callbacks
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import model_save_quantized_weights
import collections
from importlib import reload

from mult_area import mult_area
import write_mlp_mergemult_ps as wv1
from sklearn.cluster import KMeans
import copy
import hls4ml
import os
from importlib import reload
from joblib import dump



def reload_class(class_obj):
    module_name = class_obj.__module__
    module = sys.modules[module_name]
    pycfile = module.__file__
    modulepath = str.replace(pycfile, ".pyc", ".py")
    code=open(modulepath, 'rU').read()
    compile(code, module_name, "exec")
    module = reload(module)
    return getattr(module,class_obj.__name__)



def top_sharing(weights, bits_l1, bits_int_l1, bits_l2, bits_int_l2, bias_l1, bias_int_l1 , bias_l2 , bias_int_l2, inputsize, relu, relu_int, X_test, Y_test, layer1, layer2, layer3, clusters1, clusters2, window):

    #layer1=16
    #layer2=5
    #layer3=10

    #bits_l1=7
    #bits_int_l1=3
    #bits_l2=7
    #bits_int_l2=3

    #bias_l1=7
    #bias_l2=7
    #bias_int_l1=2
    #bias_int_l2=2
    #relu_int=3
    #relu=8

    relusize_f= relu
    weight_size_f1= bits_l1
    bias_size_f1= bias_l1
    weight_size_f2= bits_l2
    bias_size_f2= bias_l2
    input_s= inputsize
    int_relu = relu_int
    int_w1= bits_int_l1
    int_b1= bias_int_l1
    int_w2= bits_int_l2
    int_b2= bias_int_l2
    half_window = int(window / 2)


    # create a model with the selected bitwidths and find the zero cost weights (or the duplicates)
    model1 = Sequential()
    model1.add(QDense(layer2, input_shape=(layer1,), name='fc1', kernel_quantizer=quantized_bits(bits_l1,bits_int_l1-1,alpha=1,use_stochastic_rounding=True),bias_quantizer=quantized_bits(bias_l1,bias_int_l1-1,0,alpha=1),kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)   ))
    model1.add(QActivation(activation=quantized_relu(relu,relu_int-1,use_stochastic_rounding=False), name='relu1'))
    model1.add(QDense(layer3, name='output',
              kernel_quantizer=quantized_bits(bits_l2,bits_int_l2-1,alpha=1,use_stochastic_rounding=True), bias_quantizer=quantized_bits(bias_l2,bias_int_l2-1,alpha=1),
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model1.add(Activation(activation='softmax', name='softmax'))
    model1.set_weights(weights)

    # for the two layers
    for node in range(0,1):
        GoldenList1=[]
        GoldenList2=[]
        for i in range(0,bits_l1):
            GoldenList1.append( 2**i/(2**(bits_l1-bits_int_l1) ))
            GoldenList1.append(- 2**i/(2**(bits_l1-bits_int_l1) ))
        for i in range(0,bits_l2):
            GoldenList2.append(2**i/(2**(bits_l2-bits_int_l2) ))
            GoldenList2.append(-2**i/(2**(bits_l2-bits_int_l2) ))
        GoldenList1.append(0)
        GoldenList2.append(0)


        # We will create "layer1" datasets for the first set of weights and "layer2" datasets for the second set
        # Each one of the first set datasets will contain "layer2" weights and each one of the second set of datasets will contain "layer3" weights
        for ii in range(0,2):

            target_layer = ii

            # make an intialization depending if we aim the first or the second sets
            if target_layer==0:
                GoldenList=GoldenList1
                inputs=layer1
                tvars=model1.trainable_variables[0]
                bits=bits_l1
                bits_int=bits_int_l1
                i_bits=input_s
                desired_clusters= clusters1
            else:
                GoldenList=GoldenList2
                inputs=layer2
                tvars=model1.trainable_variables[2]
                bits=bits_l2
                target_layer=2
                bits_int=bits_int_l2
                i_bits=relu
                desired_clusters= clusters2

            # extract the datasets in a list format
            l=[]
            datasets=[]
            for i in range(0,inputs):
                l.append(tf.get_static_value(tvars[i]))

            for i in range(0,inputs):
                ls=[]
                for i in l[i]:
                    ls.append(float(i))
                datasets.append(ls)

            #create the datasets for the clustering
            cluster_datasets=[]
            positions=[]
            for i in range(0,inputs):
                ls_new=[]
                pos=[]
                index=0
                for j in datasets[i]:
                    if j not in GoldenList:
                        ls_new.append(j)
                        pos.append(index)
                    index = index + 1
                if len(ls_new):
                    cluster_datasets.append(ls_new)
                    positions.append(pos)


            #do clustering
            for dataset in cluster_datasets:

                #re-define the number of clusters
                if desired_clusters > len(dataset):
                    num_clusters = len(dataset)
                else:
                    num_clusters = desired_clusters
                array = np.array(dataset)
                kmeans = KMeans(n_clusters=num_clusters,random_state=0, n_init=100).fit(array.reshape(-1,1))
                centroids=kmeans.cluster_centers_.reshape(-1).tolist()

                # round the centroids up to 6 decimals accuracy
                for i in range(0,len(centroids)):
                    centroids[i]=round(centroids[i],6)
                    # calculate the estimated area overhead of this centroid
                    #save the sign
                    qcentroid = int(centroids[i]*(2 ** (bits-bits_int)))
                    final_centroid = qcentroid
                    multarea = mult_area[int(i_bits)][abs(int(qcentroid))]
                    #print("here")

                    # now for half window to the right if (< 127) and half window to the left if (> 0 ) evaluate the neighbor points
                    for w in range(1,half_window+1):
                        new_centroid = w + qcentroid
                        if new_centroid < 127:
                            new_multarea = mult_area[int(i_bits)][abs(int(new_centroid))]
                            if new_multarea < multarea:
                                multarea = new_multarea
                                final_centroid = new_centroid

                    for w in range(1,half_window+1):
                        new_centroid = qcentroid - w
                        if new_centroid > 0:
                            new_multarea = mult_area[int(i_bits)][abs(int(new_centroid))]
                            if new_multarea < multarea:
                                multarea = new_multarea
                                final_centroid = new_centroid
                    fixed_point_centroid = final_centroid/(2 ** (bits-bits_int))
                    #print("FOUND ANOTHER CENTROID"+str(final_centroid)+" AKA "+str(fixed_point_centroid)+" INSTEAD OF "+str(qcentroid)+" AKA "+str(centroids[i]))
                    centroids[i] = fixed_point_centroid
                labels=kmeans.labels_.tolist()
                index = 0
                for i in range(0,len(dataset)):
                    dataset[i] = centroids[labels[i]]
                    dataset[i] = int(dataset[i]*(2 ** (bits-bits_int)))
                    dataset[i] = dataset[i]/(2 ** (bits-bits_int))


            #restore weights
            restored_weights= copy.deepcopy(datasets)
            for i in range(0,len(positions)):
                index = 0
                for j in positions[i]:
                    restored_weights[i][j] = cluster_datasets[i][index]
                    index = index + 1

            model1.trainable_variables[target_layer].assign(restored_weights)
            adam = Adam(lr=0.00001)
                #callbacks= all_callbacks( outputDir = 'callbacks')
    model1.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    new_accuracy=model1.evaluate(X_test,Y_test)
    print("accuracy = "+str(new_accuracy[1]))

    #-----------------------------------------------
    # Clustering is completed
    # create the verilog
    weight_bias_size=[ [ (weight_size_f1,int_w1), (bias_size_f1,int_b1) ], [ (weight_size_f2,int_w2), (bias_size_f2,int_b2) ] ]
    relu_size=(relusize_f,int_relu-1)
    #4)Create the verilog
    def aptype_to_size(ap):
        if "ap_int" in ap:
            s=ap.split('<')[1].split('>')[0]
            i=s
        else:
            s,i=ap.split('<')[1].split('>')[0].split(',')
        return (int(s),int(i))

    #scale weights
    allweights=[t*2**(weight_bias_size[j//2][j%2][0]-weight_bias_size[j//2][j%2][1]) for j,t in enumerate(model1.get_weights())]
    #cast weights to integer & transpose
    allweightsT=[wl.T.astype(int).tolist() for wl in allweights]
    weight_list=allweightsT[0::2]
    bias_list=allweightsT[1::2]
    #set params
    config = hls4ml.utils.config_from_keras_model(model1, granularity='name')

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

    filename = "Kmeans"+"-"+str(clusters1 )+"C-"+str(clusters2)+"C-"+str(weight_size_f1)+str(bias_size_f1)+str(weight_size_f2)+str(bias_size_f2)+str(input_s)+ str(int_relu)+"-"+str(new_accuracy[1])
    x = os.path.join("./custom_clustering", filename)
    f=open(x,"w")

    wv1.write_mlp_verilog(f, input_size, bias_list, weight_list, weight_bias_size, sum_relu_size,last_layer)
    f.close()

    f=open("sim.Xtest"+filename,"w")
    np.savetxt(f,(X_test*2**input_size[0]).astype(int),fmt='%d',delimiter=' ')
    f.close()

    dump(np.argmax(Y_test,axis=1), "sim.Ytest")


    print("copy top.v, sim.Xtest, and sim.Ytest to the synopsys project ")
