#in this code i use some standard terms for defining various terms so i prefer you research on your own for each u know word for better understanding

#importing numpy library of python
import numpy as np
#defining a sigmoid function you can use inbuilt np. Sign 
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x) #this is a derivative of signum function. This need to calculate gradient decent
    return 1/(1+np.exp(-x))# this is a signum function
#input values you can use any input matrix it can be image also
X = np.array([[0,0,1], 
	            [0,1,1],
	            [1,0,1], 
	            [1,1,1]
	            ])
#output of input matrix
y = np.array([[0], 		
	             	[1], 
	           			[1], 
	           			[0]
	           			])
#random seed will ensure that each time we get same random number as weights so that we can easily analyse the output	           			
np.random.seed(1) 
#initialyzing initial random weights of first synaps(snaps are the link joining one layer to another layer of the Nodes which carry weights) 
syn0 = 2*np.random.random((len(X[0]),len(X))) - 1 
#initialyzing second later of synaps
syn1 = 2*np.random.random((len(X),len(X))) - 1
#initialyzing thirdayier of synaps
syn2 = 2*np.random.random((len(X),1)) - 1 
#here comes the actual part training our model and updating weights
for j in xrange(60000):#here i use 60000 loops to train you can change it according to complexity of your model
    l0 = X #input layer 
    l1 = nonlin(np.dot(l0,syn0)) #first layer(hidden layer 1)  output
    l2 = nonlin(np.dot(l1,syn1)) #second layer(hidden layer 2) output
    l3 = nonlin(np.dot(l2,syn2)) #third layer(output layer) or final output
    l3_error = y - l3 #calculating error using final output and given output
    if (j% 10000) == 0: #print error at every 10000 loops
        print "Error:" + str(np.mean(np.abs(l3_error)))
    l3_delta = l3_error*nonlin(l3,deriv=True) #calculating gradient for third layer
    l2_error = l3_delta.dot(syn2.T) #calculating layer 2 error using back propagation
    l2_delta = l2_error*nonlin(l2,deriv=True)  #calculating second layer gradient
    l1_error = l2_delta.dot(syn1.T) #calculating first layer error again by using back propagation
    l1_delta = l1_error * nonlin(l1,deriv=True) #calculating gradient of first layer
    syn1 += l1.T.dot(l2_delta) #updating  first synaps layer
    syn0 += l0.T.dot(l1_delta) #updating second layer synaps
    syn2 += l2.T.dot(l3_delta) #updating third synaps synaps
    
print l3 #predicting output on training set

  
 
 