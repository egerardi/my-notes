#!/usr/bin/python
import sys
import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

def main(arglist):
	np.random.seed(100)
	print nonlin(-10)
	print nonlin(0)
	print nonlin(10)

if __name__ == "__main__":
	main(sys.argv[1:])
