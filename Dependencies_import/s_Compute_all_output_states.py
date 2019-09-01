#!/usr/bin/env python
# -*- coding: utf-8 -*-


################################################################################################
# IMPORTATIONS
################################################################################################

import numpy as np

from itertools                  import combinations_with_replacement
from s_Photon_quantum_state     import *
from s_Matrix_buiding_function  import *

################################################################################################
# FUNCTIONS
################################################################################################

def OutputStateFromInputStateAndNetworkMatrix(self, inp_state, T, printable=False):
    '''
    Return the possible output states of the beam splitter array, represented by the transformation matrix T,
    with respect to the input photon. The inp_L lenth should be of the same order than the matrix T.
    inp_state = {key : [coeff, 1d np.array]}
    T         = 2d np.array
    The function returns:
    OutState = [1d np.array(N,1), 2d np.array(N,n)], with n the number of channels, ie n=len(inp_L).
    The first  array = coefficent of the state.
    The second array = states --> 1 line <=> state with the number of photon in each channel.
    '''
    t0 = time.time()
    ##############################################################
    # --- redefining and verifying if the inputs are correct --- #
    ##############################################################
    key = list(inp_state.keys())
    if len(key) > 1:
        print('Error, more than one input state: {}'.format(key))
    else:
        key = key[0]
    #print('ERROR with: ', inp_state)
    inp_L     = np.array(inp_state[key][1]).astype('int')
    inp_coeff = inp_state[key][0]
    n = len(inp_L)       # number of channel
    if n!=T.shape[0]:
        print('Error: not same dimensionality.')
        return None
    # --- we get the input index, i.e the channel of the BS array, and the photon number --- #
    input_index = []
    for i in range(n):
        if inp_L[i] != 0:
            input_index.append([i,inp_L[i]])
    input_index = np.array(input_index)
    if printable: print('Verification: OK')
    ##################################################################################
    # --- computing the global coefficient --- #
    '''
    indeed, when decomposing the input in its creation operators, we should not forget
    the coefficient the number of photon: |2> = (a^\dagger)**2 * 1/np.sqrt(2) |0> !!
    '''
    ##################################################################################
    coeff_glob = inp_coeff
    for ph_nbr in inp_L:
        for i in range(ph_nbr):
            coeff_glob = coeff_glob* 1/np.sqrt(i+1)
    if printable: print('Global coefficient: OK')
    ##################################################################################
    # --- building matrix for computing permanent and get coefficient out output --- #
    ##################################################################################
    l = len(input_index[:,0])
    N = np.sum(input_index[:,1]) # total photon number
    # --- initialising --- #
    P = np.zeros([N, T.shape[1]], dtype='complex')
    ind_extract = []
    for i in range(l):
        for j in range(input_index[i,1]):
            ind_extract.append(input_index[i,0])
    ind_extract = np.array(ind_extract)
    # --- iterating --- #
    for i in range(P.shape[0]):
        P[i,:] = T[ind_extract[i],:]
        # ---  --- #
    if N != P.shape[0]:#should be the same
        print("Error: not the same line in intermediate matrix than the photon number.")
    if printable: print('Matrix for permanent: OK')
    ####################################################
    # --- construction of output with coefficients --- #
    ####################################################
    if printable: print('Starting output coeff calculation...')
    OUTPUT_states = {}
    #permL = permutationWithoutRepetition(np.arange(n), N) #np.array(list(product(np.arange(n), repeat=N))) # old version
    permL = np.array(list(combinations_with_replacement(np.arange(n), N))) # much more efficient, uses itertool !
    s_permL = permL.shape
    if printable: print('Permutation shape: {}'.format(s_permL))
    for i in range(s_permL[0]):  #can be parallelised
        if printable: print(i/s_permL[0])
        # --- reboot state --- #
        outp_L = np.zeros(n)
        outp_L_coeff = 1.
        # --- initialising --- #
        Q = np.zeros([P.shape[0],s_permL[1]],dtype='complex')
        if Q.shape[0]!=Q.shape[1]:
            print('Error: not a square matrix')
            return None
        ind_extract = permL[i,:]
        # --- iterating --- #
        for j in range(len(ind_extract)):
            ind      = ind_extract[j]
            Q[:,j] = P[:,ind]
            outp_L[ind_extract[j]] += 1
            outp_L_coeff = outp_L_coeff * 1/np.sqrt(outp_L[permL[i,j]])
        Qperm = npPermanent(Q)#matPermntRyser(Q)#npPermanent(Q)
        # --- add state in output state dictionary --- #
        coeff_state = Qperm*outp_L_coeff*coeff_glob
        if np.abs(coeff_state) > 1e-16:
            OUTPUT_states[i] = [ coeff_state , outp_L ]
    if printable: print('End calculation output.')
    print('Computation time for output state: {:.4f}s'.format(time.time()-t0))
    return OUTPUT_states

################################################################################################
# CODE
################################################################################################
if __name__=='__main__':
    print('STARTING')

