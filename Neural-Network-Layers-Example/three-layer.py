#!/usr/bin/python
import sys
import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

def predict(W,v):
	for w in W:
		v=nonlin(np.dot(v,w))
	return v
	
def three_layer_nn(X,y,iterations=10000):
	#initialize weights
	syn0=2*np.random.random((3,4))-1 
	syn1=2*np.random.random((4,1))-1

	
	for j in xrange(iterations):
		#forward propagation
		L0=X
		L1=nonlin(np.dot(L0,syn0))
		L2=nonlin(np.dot(L1,syn1))
		
		#Difference bwteen result and target value:
		L2_error=y-L2
		
		#Create back prop amoutns, back one layer:
		L2_delta=L2_error*nonlin(L2,deriv=True)
		
		#How much did each L1 value contribute to the L2 error
		L1_error=L2_delta.dot(syn1.T)
		
		#create back-prop amoutns, back two layers:
		L1_delta=L1_error*nonlin(L1,deriv=True)
		
		#Modify neural net:
		syn1 +=L1.T.dot(L2_delta)
		syn0 +=L0.T.dot(L1_delta)
		
	return[syn0,syn1] #return list of layers (of weights)

def main(arglist):
	np.random.seed(100)
	
	#input
	X=np.array([
		[0,0,1],
		[0,1,1],
		[1,0,1],
		[1,1,1]
	])
	
	#output
	#y=np.array([[0,0,1,1]]).T #easy
	y=np.array([[0,1,1,0]]).T #hard
	
	np.random.seed(100)
	W=three_layer_nn(X,y) #create two layer neural net
	
	print predict (W,np.array([[1,1,1]])) #should be 1
	print predict (W,np.array([[0,1,1]])) #should be 0
	print predict (W,np.array([[1,1,0]])) #should be 1


if __name__ == "__main__":
	main(sys.argv[1:])
