# Utilities to deal with multioutput capabilities
# Licensed under the BSD 3-clause license (see LICENSE.txt)

###TODO: Change names + comment


import numpy as np

def extend_X(X, nb_index):
    """
    """
    indices = np.arange(nb_index)
    return np.vstack([np.c_[(np.repeat(x[np.newaxis, :], nb_index, 0), indices)] for x in X])

def extend_Y(Y, nb_index):
	assert np.shape(Y)[1] == nb_index, "shape of Y: {}".format(Y.shape)
	return np.reshape(Y, (np.size(Y), 1))


def extend_XY(X, Y, nb_index):
	assert X.shape[0] == Y.shape[0], "shape of X: {}, shape of Y: {}".format(X.shape, Y.shape)
	X_ext = extend_X(X, nb_index)
	Y_ext = extend_Y(Y, nb_index)
	return X_ext, Y_ext

# def gather_Y(Y, nb_index, collapse_fn=None, target_len=None):
# 	""" Average per input or average abs distance to a target if provided"""
# 	assert Y.shape[1] == 1, "shape of Y: {}".format(Y.shape)
# 	if collapse_fn is None:
# 		collapse_fn = lambda x: np.average(x, axis = 1)[:, np.newaxis]
# 	len_Y = int(len(Y)/nb_index)
# 	if target_len is not None:
# 		assert len_Y == target_len

# 	return collapse_fn(np.reshape(Y, (len_Y, nb_index)))


def contract_X(X, nb_index):
    """ Verif the X contained in the gpmodel object has the expected 
    structure i.e. if multiouput [(X0,0),.., (X0,D), ...., (XN,0), ...,]
    else no test is needed"""
    D = nb_index
    if(D>1):
        N = int(len(X)/D)
        X_contracted = X[0::D,0:-1:1]
        X_test = np.c_[(np.repeat(X_contracted, D, 0), np.tile(np.arange(D), N))]
        assert np.allclose(X_test, X)
    else:
        X_contracted = X.copy()
    return X_contracted

def contract_Y(Y, nb_index, contract_fn=None):
	if contract_fn is None:
		contract_fn = lambda x: x
		#contract_fn = lambda x: np.average(x, axis = 1)[:, np.newaxis]
	D = nb_index
	if(D>1):
		N = int(len(Y)/D)
		Y_contracted = contract_fn(Y.reshape(N, D))
	else:
		Y_contracted = Y.copy()
	return Y_contracted


def contract_XY(X, Y, nb_index, contract_fn = None):
	X_contracted = contract_X(X, nb_index)
	Y_contracted = contract_Y(Y, nb_index, contract_fn)
	return X_contracted, Y_contracted


def contract_XYs(X, list_Y, nb_index, contract_fn = None):
	X_contracted = contract_X(X, nb_index)
	Y_contracted = [contract_Y(Y, nb_index, contract_fn) for Y in list_Y]
	return X_contracted, Y_contracted



