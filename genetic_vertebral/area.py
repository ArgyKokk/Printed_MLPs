from mult_area import mult_area

def area(weights,inputsize,relusize,weightsize1,weightsize2, w1_int, w2_int,layer1,layer2,layer3):

    input_bits = int(inputsize)
    relusize = int(relusize)
    weights1 = int(weightsize1)
    weights2 = int(weightsize2)
    w1int = int(w1_int)
    w2int = int(w2_int)
    #bias1 = int(biassize1)
    #bias2 = int(biassize2)
    print("weights are")
    print(weights)
    
    l1=[]
    l2=[]
    #append the weights in two lists after each forward pass
    for i in range(0,layer1):
        l1.append(weights[0][i]* (2**(weights1-w1_int)))
    for i in range(0,layer2):
        l2.append(weights[2][i]* (2**(weights2-w2_int))) 

    #layer 1 area
    area=0
    summary=0
    for i in range(0,layer1):
        #for every input create a list of the weights that have been used for the area estimation
        checkedWeights=[]
        for j in range(0,layer2):
            weight_val = abs(int(l1[i][j]))
            #print("weight val  is "+str(weight_val))
            if weight_val not in checkedWeights:
                area += mult_area[input_bits][weight_val]
                #print("enter with area "+str(mult_area[input_bits][weight_val]))
            checkedWeights.append(weight_val)
            summary += weight_val
    #print("summary area is: "+str(summary))
    area1 = area + 0.1 * summary
    #layer 2 area
    area=0
    summary=0
    input_bits=relusize
    for i in range(0,layer2):
        #for every input create a list of the weights that have been used for the area estimation
        checkedWeights=[]
        for j in range(0,layer3):
            weight_val = abs(int(l2[i][j]))
            if weight_val not in checkedWeights:
                area += mult_area[input_bits][weight_val]
            checkedWeights.append(weight_val)
            summary += weight_val
            
    area2 = area + 0.1 * summary
    overall_area = area1 + area2
    print("Overall area is "+str(overall_area))
    return overall_area

