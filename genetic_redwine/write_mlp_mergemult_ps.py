import math
import random
from numpy import binary_repr
import sys

INPNAME="inp"
OUTNAME="out"

def get_width (a):
    return int(a).bit_length()


def writeneuron(nprefix, neuron, activation, bias, nweights, inputs, weight_bias_inp_size, sum_relu_size,merge_list):
        prefix = nprefix+str(neuron)
        sumname = prefix+"_sum"
        sumname_pos =  sumname+"_pos"
        sumname_neg =  sumname+"_neg"

        size_w,size_b,size_i=weight_bias_inp_size
        width_inp=size_i[0]
        decimal_w=size_w[0]-size_w[1]
        decimal_b=size_b[0]-size_b[1]
        decimal_i=size_i[0]-size_i[1]
        fixb=(decimal_w+decimal_i)-decimal_b
        if decimal_b > (decimal_w + decimal_i):
           decimal_b = decimal_w + decimal_i
        fixb=(decimal_w+decimal_i)-decimal_b


        count_neg_w=0
        pos=[]
        neg=[]

        max_pos=0
        max_neg=0

        if bias != 0:
            #print(decimal_w+decimal_i)
            #print(bias)
            b=abs(bias)<<fixb
            #print(abs(bias), fixb, b)
            width_b=get_width(b)
            bin_b=str(width_b)+"'b"+binary_repr(b,width_b)
            if bias > 0:
                pos.append(bin_b)
                max_pos+=b
            else:
                neg.append(bin_b)
                max_neg+=b

        for i in range(len(nweights)):
            w=nweights[i]
            if w == 0:
                print("    //weight %d : skip" % (w))
                continue
            a=inputs[i]
            name=prefix+"_po_"+str(i)
            #bit_h,bit_l=prod_width[i]
            #pwidth=bit_h
            width_w=get_width(abs(w))
            pwidth=width_w+width_inp
            bit_h=pwidth
            bin_w=str(width_w)+"'b"+binary_repr(abs(w),width_w)

            if w > 0:
                print("    //weight %d : %s" % (w, bin_w))
            else:
                print("    //weight abs(%d) : %s" % (w, bin_w))
            print("    wire [%d:0] %s;" % ((pwidth-1),name))
            if merge_list[i] < 0:
                print("    assign %s = $unsigned(%s) * $unsigned(%s);" % (name,a,bin_w))
            else:
                print("    //merging with node %d" %(merge_list[i]))
                mergename=nprefix+str(merge_list[i])+"_po_"+str(i)
                print("    assign %s = %s;" % (name,mergename))

            max_prod=int(2**bit_h-1)
            if w > 0:
                pos.append(name)
                max_pos+=max_prod
            else:
                neg.append(name)
                max_neg+=max_prod
            print()

        spwidth=get_width(max_pos)
        snwidth=get_width(max_neg)
        swidth=max(spwidth,snwidth)+1
        decimal_s=decimal_w+decimal_i
        #swidth=sum_relu_size[0][0]
        #decimal_s=swidth-sum_relu_size[0][1]

        #fix relu dimensions
        rwidth=sum_relu_size[1][0]
        decimal_r=rwidth-sum_relu_size[1][1]
        if rwidth == 32: 
            decimal_r=decimal_s
            if activation=="linear":
                rwidth=swidth
            else:
                rwidth=swidth-1
        fixrwidth=0
        if decimal_r > decimal_s:
            fixrwidth=decimal_r-decimal_s
        ####

        print("    //accumulate positive/negative subproducts")
        if len(pos):
            pos_str=" + ".join(str(x) for x in pos)
            print("    wire [%d:0] %s;" % ((swidth-2),sumname_pos))
            print("    assign %s = %s;" % (sumname_pos,pos_str))

        if len(neg) and ( len(pos) or activation=="linear" ):
            neg_str=" + ".join(str(x) for x in neg)
            print("    wire [%d:0] %s;" % ((swidth-2),sumname_neg))
            print("    assign %s = %s;" % (sumname_neg,neg_str))

        if len(pos) and len(neg):            
            print("    wire signed [%d:0] %s;" % ((swidth-1),sumname))
            print("    assign %s = $signed({1'b0,%s}) - $signed({1'b0,%s});" % (sumname,sumname_pos,sumname_neg))
        elif len(pos) and not len(neg):
            print()
            print("    //WARN: only positive weights. Using identity")
            print("    wire signed [%d:0] %s;" % ((swidth-1),sumname))
            print("    assign %s = $signed({1'b0,%s});" % (sumname, sumname_pos))
        elif not len(pos) and len(neg) and activation=="linear":
            print()
            print("    //WARN: only negative weights with linear. Negate.")
            print("    wire signed [%d:0] %s;" % ((swidth-1),sumname))
            print("    assign %s = -$signed({1'b0,%s});" % (sumname, sumname_neg))
        elif not len(pos) and len(neg) and activation=="relu":
            print()
            print("    //WARN: only negative weights with relu. Using zero")
            print("    wire signed [%d:0] %s;" % ((swidth-1),sumname))
            print("    assign %s = $signed({%d{1'b0}});" % (sumname,swidth))            
        elif not len(pos) and not len(neg):
            print()
            print("    //WARN: no weights. Using zero")
            print("    wire signed [%d:0] %s;" % ((swidth-1),sumname))
            print("    assign %s = $signed({%d{1'b0}});" % (sumname,swidth))
        print()

        #fix relu dimensions
        sumname0=sumname
        if fixrwidth > 0:
            sumname='{'+sumname+','+str(fixrwidth)+"'b"+binary_repr(0,fixrwidth)+'}'
            swidth=fixrwidth+swidth
            decimal_s=decimal_r

        qrelu=prefix+"_qrelu"
        decimal_s=decimal_w+decimal_i
        msb_sat=decimal_s+(rwidth-decimal_r)-1
        lsb_sat=msb_sat-rwidth+1
        if activation == "relu":
            print("    //relu")
            if rwidth >= swidth-1:
                print("    wire [%d:0] %s;" % ((rwidth-1),prefix))
                print("    assign %s = (%s<0) ? $unsigned({%d{1'b0}}) : $unsigned(%s);" % (prefix,sumname,rwidth, sumname ))
            else:
                print("    wire [%d:0] %s, %s;" % ((rwidth-1),prefix,qrelu))
                if swidth-1==msb_sat:
                    print ("    assign %s = %s[%d:%d];" %(qrelu, sumname, msb_sat, lsb_sat))
                elif fixrwidth > 0:
                    print("    wire [%d:0] %s;" % ((swidth-1),sumname0))
                    print("    assign %s = %s;" % (sumname0, sumname))
                    print("    DW01_satrnd #(%d, %d, %d) USR_%s ( .din(%s[%d:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(%s));" % ((swidth-1), msb_sat, lsb_sat, prefix, sumname0, (swidth-2), qrelu))
                else: 
                    print("    DW01_satrnd #(%d, %d, %d) USR_%s ( .din(%s[%d:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(%s));" % ((swidth-1), msb_sat, lsb_sat, prefix, sumname, (swidth-2), qrelu))
                print("    assign %s = (%s<0) ? $unsigned({%d{1'b0}}) : $unsigned(%s);" % (prefix,sumname,rwidth, qrelu ))
        elif activation == "linear":
             print("    //linear")
             print("    wire signed [%d:0] %s;" % ((rwidth-1),prefix))
             if rwidth >= swidth:  
                 print("    assign %s = %s;" % (prefix,sumname))
             else:
                 print("    DW01_satrnd #(%d, %d, %d) USR_%s ( .din(%s), .tc(1'b1), .rnd(1'b0), .ov(), .sat(1'b1), .dout(%s));" % (swidth, msb_sat, lsb_sat, prefix, sumname, prefix))

        #return (rwidth,sum_relu_size[1][1])
        return rwidth



def argmax (prefix,act,vwidth,iwidth,signed):
        lvl=0
        vallist=list(act)
        print("// argmax inp: " + ', '.join(vallist))
        idxlist=[str(iwidth)+"'b"+binary_repr(i,iwidth) for i in range(len(act))]
        while len(vallist) > 1:
            newV=[]
            newI=[]
            comp=0
            print("    //comp level %d" % lvl)
            for i in range(0,len(vallist)-1,2):
                cmpname="cmp_"+str(lvl)+"_"+str(i)
                vname=prefix+"_val_"+str(lvl)+"_"+str(i)
                iname=prefix+"_idx_"+str(lvl)+"_"+str(i)
                vname1=vallist[i]
                vname2=vallist[i+1]
                iname1=idxlist[i]
                iname2=idxlist[i+1]
                print("    wire %s;" % cmpname)
                if signed:
                    print("    wire signed [%d:0] %s;" % ((vwidth-1),vname))
                else:
                    print("    wire [%d:0] %s;" % ((vwidth-1),vname))
                print("    wire [%d:0] %s;" % ((iwidth-1),iname))

                print("    assign {%s} = ( %s >= %s );" % (cmpname,vname1,vname2))
                print("    assign {%s} = ( %s ) ? %s : %s;" % (vname,cmpname,vname1,vname2))
                print("    assign {%s} = ( %s ) ? %s : %s;" % (iname,cmpname,iname1,iname2))
                print()
                newV.append(vname)
                newI.append(iname)
            if len(vallist) % 2 == 1:
                newV.append(vallist[-1])
                newI.append(idxlist[-1])
            lvl+=1
            vallist = list(newV)
            idxlist = list(newI)
        return idxlist[-1]


def write_mlp_verilog (f, input_size, biases, weights, weight_bias_size, sum_relu_size, last_layer):
    stdoutbckp=sys.stdout
    sys.stdout=f
    width_a=input_size[0]

    REGRESSOR=False
    inp_num=len(weights[0][0])


    width_o=get_width(len(weights[-1]))

    print("//weights:",weights)
    print("//intercepts:",biases)

    print("module top ("+INPNAME+", "+OUTNAME+");")
    print("input ["+str(inp_num*width_a-1)+":"+str(0)+"] " + INPNAME +";")
    print("output ["+str(width_o-1)+":"+str(0)+"] " + OUTNAME +";")
    print()

    act_next=[]
    act_next_size=[]
    all_act_size=[]
    for i in range(inp_num):
        a=INPNAME+"["+str((i+1)*width_a-1)+":"+str(i*width_a)+"]"
        act_next.append(a)
    all_act_size.append(input_size)


    ver_relu_size=0
    for j in range(len(weights)):
        act=list(act_next)
        act_next=[]
        act_next_size=[]
        for i in range(len(weights[j])):
            print("// layer: %d - neuron: %d" % (j,i) )
            prefix = "n_"+str(j)+"_"
            nweights=weights[j][i]
            bias=biases[j][i]
            nweight_bias_inp_size=list(weight_bias_size[j])
            nweight_bias_inp_size.append(all_act_size[j])
            nsum_relu_size=sum_relu_size[j]
            merge_list= [-1] * len(weights[j][i])
            for k in range(len(weights[j][i])):
                for ii in range(i):
                    if abs(weights[j][i][k]) == abs(weights[j][ii][k]):
                        merge_list[k]=ii
                        break
            if j == len(weights)-1:
                activation=last_layer
            else:
                activation="relu"
            ver_relu_size=max(ver_relu_size,writeneuron (prefix,i, activation,bias,nweights,act, nweight_bias_inp_size,nsum_relu_size,merge_list))
            prefix=prefix+str(i)
            #act_next_size.append(sum_size)
            act_next.append(prefix)
            print()
        #all_act_size.append(act_next_size)
        all_act_size.append(sum_relu_size[j][1])

    vw=max(all_act_size[-1])
    if vw ==32:
       vw=ver_relu_size
    iw=width_o
    prefix="argmax"
    print("// argmax: %d classes, need %d bits" % (len(weights[-1]),iw) )
    if last_layer == "linear":
        signed=True
    else:
        signed=False
    out=argmax(prefix,act_next,vw,iw,signed)
    print("    assign "+OUTNAME+" = " + out + ";")
    print()
    print("endmodule")
    sys.stdout=stdoutbckp
    return all_act_size, width_o

def main():
    if len(sys.argv)==2:
        f=open(sys.argv[1],'w')
    else:
        f=sys.stdout

    list = []
    listw2 = []
    b2 = []
    b5 = []
    with open('w5_int.txt') as f1:
        lines = f1.read().splitlines()
        for i in lines:
            val = int(i)
            list.append(val)
    f1.close()
#print(list)

    with open('w2_int.txt') as f1:
        lines = f1.read().splitlines()
        for i in lines:
            val = int(i)
            listw2.append(val)
    f1.close()

    with open('b2_int.txt') as f1:
        lines = f1.read().splitlines()
        for i in lines:
            val = int(i)
            b2.append(val)
    f1.close()

    with open('b5_int.txt') as f1:
        lines = f1.read().splitlines()
        for i in lines:
            val = int(i)
            b5.append(val)
    f1.close()
#print(listw2)


    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    list6=[]
    b= []
    b.append(b2)
    b.append(b5)

    list1.append(list[0])
    list1.append(list[6])
    list1.append(list[12])
    list1.append(list[18])
    list1.append(list[24])

    list2.append(list[1])
    list2.append(list[7])
    list2.append(list[13])
    list2.append(list[19])
    list2.append(list[25])

    list3.append(list[2])
    list3.append(list[8])
    list3.append(list[14])
    list3.append(list[20])
    list3.append(list[26])

    list4.append(list[3])
    list4.append(list[9])
    list4.append(list[15])
    list4.append(list[21])
    list4.append(list[27])

    list5.append(list[4])
    list5.append(list[10])
    list5.append(list[16])
    list5.append(list[22])
    list5.append(list[28])

    list6.append(list[5])
    list6.append(list[11])
    list6.append(list[17])
    list6.append(list[23])
    list6.append(list[29])

    list1w2 = []
    list2w2 = []
    list3w2 = []
    list4w2 = []
    list5w2 = []

    for i in range(0,11):
        list1w2.append(listw2[i*5])

    for i in range(0,11):
        list2w2.append(listw2[1 + i*5])

    for i in range(0,11):
        list3w2.append(listw2[2 + i*5])

    for i in range(0,11):
        list4w2.append(listw2[3 + i*5])

    for i in range(0,11):
        list5w2.append(listw2[4 + i*5])

    listw2 = []
    listw2.append(list1w2)
    listw2.append(list2w2)
    listw2.append(list3w2)
    listw2.append(list4w2)
    listw2.append(list5w2)
#print(listw2)

    listw5 = []
    listw5.append(list1)
    listw5.append(list2)
    listw5.append(list3)
    listw5.append(list4)
    listw5.append(list5)
    listw5.append(list6)

    final_list = []
    final_list.append(listw2)
    final_list.append(listw5)

    biases = b
    weights = final_list


    #print(weights)
    #print(biases)

    #last_layer: linear, relu
    last_layer="linear"
    input_size = (4,0)
    weight_bias_size=[ [ (5,2), (5,3) ], [ (5,2), (5,3) ] ]

    sum_relu_size =  [ [ (12,5), (6,0)], [ (16,7), (16,7)] ]

    write_mlp_verilog(f, input_size, biases, weights, weight_bias_size, sum_relu_size,last_layer)


if __name__ == "__main__":
    main()
