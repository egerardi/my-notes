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
	
def two_layer_nn(X,y,iterations=10000):
	syn0=2*np.random.random((3,1))-1 #initialize weights
	
	for iter in xrange(iterations):
		#forward propagation
		L0=X
		L1=nonlin(np.dot(L0,syn0))
		
		#print L1
		L1_error=y-L1 #how much did we miss
		
		#multiply how much we missed by the slope of the sigmoid at the values in L1
		L1_delta=L1_error*nonlin(L1,True)
		syn0 += np.dot(L0.T,L1_delta) #update weights
	return[syn0] #return list of layers (of weights)

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
	y=np.array([[0,0,1,1]]).T #easy
	#y=np.array([[0,1,1,0]]).T #hard

	W=two_layer_nn(X,y) #create two layer neural net
	
	print predict (W,np.array([[1,1,1]])) #should be 1
	print predict (W,np.array([[0,1,1]])) #should be 0
	print predict (W,np.array([[1,1,0]])) #should be 1


if __name__ == "__main__":
	main(sys.argv[1:])
