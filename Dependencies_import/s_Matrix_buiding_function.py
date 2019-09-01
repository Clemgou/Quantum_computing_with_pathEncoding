#!/usr/bin/env python
# -*- coding: utf-8 -*-


################################################################################################
# IMPORTATIONS
################################################################################################

import numpy as np

################################################################################################
# FUNCTIONS
################################################################################################

def makeMatDiagFromSquareMatrices(L):
    '''
    L must be a list of either scalar or square np.array.
    return the square matrix of diagonal L.
    '''
    n = len(L)
    N = 0
    for i in range(n):
        if type(L[i])==float:
            N+=1
        elif type(L[i])==type(np.array(0)):
            N+=L[i].shape[0]
        else:
            print('Error: one element of the list argument is not in a right type')
            return None
    M = np.zeros([N,N],dtype=complex)
    j = 0
    for i in range(n):
        if type(L[i])==float:
            M[j,j] = L[i]
            j+=1
        elif type(L[i])==type(np.array(0)):
            k = L[i].shape[0]
            M[j:j+k, j:j+k] = L[i][:,:]
            j+=k
    return M

def matrixSwitchingOutput(N, mode_switch_list):
    '''
    Generate a matrix that operate the change of output modes.
    For instance if input 1 is then the output 3, then M_{3,1} = 1 et M_{1,1} = 0.
    The input of the function are:
        - N: size of the square matrix
        - the list of switched modes: mode_switch_list = [[input, output], ...]
    '''
    M = np.eye(N)
    for i in range(len(mode_switch_list)):
        in_i, out_j = mode_switch_list[i]
        M[in_i,in_i]  = 0
        M[out_j,in_i] = 1
    return M

def matrixPhaseShift(N, mode_phase_list):
    '''
    Generate a matrix that operate a phase shift on the specified modes. For instance if
    mode_phase_list=[[2, np.pi/2]], then M_{2,2} = np.exp(1j*(np.pi/2))
    The input of the function are:
        - N: size of the square matrix
        - the list of switched modes: mode_switch_list = [[mode, phase], ...]
    '''
    M = np.eye(N, dtype='complex')
    for i in range(len(mode_phase_list)):
        mode_i, phase = mode_phase_list[i]
        M[mode_i,mode_i]  = np.exp(1j*phase)
    return M

def beamSplitter2Modes():
    '''
    return a 2x2 np.array representing a 50:50 balanced beam splitter
    '''
    BS = 1/np.sqrt(2) * np.array([[1, 1j],[1j, 1]])
    #BS = 1/np.sqrt(2) * np.array([[1, 1],[-1, 1]])
    return BS

def makeMatTransfBSArray(size, couplelist):
    '''
    Generate the transformation matrix of a lign of coupled channels by BS.
    - size        = nbr of channel
    - couplelist  = [[c1,c2],[c3,c4],...] where ci are the channel index.
    Thus [ci,cj] means that the channels ci is coupled to the channel cj by a BS.
    '''
    n = size
    BS = beamSplitter()
    T = np.eye(n).astype('complex')
    for i in range(len(couplelist)):
        l = couplelist[i][0]
        k = couplelist[i][1]
        T[l,l] = BS[0,0]
        T[k,k] = BS[1,1]
        T[l,k] = BS[0,1]
        T[k,l] = BS[1,0]
    return T

def matrixDensityState(state):
    '''
    Return the matrix density of the pure state 'state'.
    The state vector should be a 1D np.array 
    '''
    n  = len(state)
    st = state.reshape([1,n])
    g  = np.dot(np.transpose(st),st)
    return g

def npPermanent(M):
    '''
    Compute the permanent of the matrix M, 2d np.array
    Uses the formula of Ryser.
    '''
    n = M.shape[0]
    d = np.ones(n).astype('complex')
    j =  0
    t = 1
    f = np.arange(n)  # helps compute the next position that needs to be changed when iterating a Gray code.
    # --- initialisation --- #
    v = M.sum(axis=0)
    p = np.prod(v)
    # --- sum over the submatrices of M --- #
    while (j < n-1):
        v -= 2*d[j]*M[j]
        d[j] = -d[j]
        t = -t
        prod = np.prod(v)
        p += t*prod
        f[0] = 0
        f[j] = f[j+1]
        f[j+1] = j+1
        j = f[0]
    return p/2**(n-1)

################################################################################################
# CODE
################################################################################################
if __name__=='__main__':
    print('STARTING')






