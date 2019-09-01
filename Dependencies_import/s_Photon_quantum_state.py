#!/usr/bin/env python
# -*- coding: utf-8 -*-


################################################################################################
# IMPORTATIONS
################################################################################################

import time
import numpy as np

################################################################################################
# FUNCTIONS
################################################################################################

def doesExists(outi, state3):
    for n in range(state3.shape[0]):
        val = 1
        for i in range(len(outi)):
            val = val*(state3[n,i]==outi[i])
        if val:
            return [True, n]
    return[False]

def sumState(state1, state2):
    coeff_stt1 = state1[0]
    coeff_stt2 = state2[0]
    coeff_stt3 = coeff_stt1
    state3     = state1[1]
    n1         = state1[1].shape[0]
    n2         = state2[1].shape[0]
    for i in range(n2):
        outi = state2[1][i,:]
        if doesExists(outi, state3)[0]:
            coeff_stt3[i] += coeff_stt2[i]
        else:
            outi = outi.reshape([1,len(outi)])
            #print(state3.shape, outi.shape)
            state3 = np.concatenate((state3, outi), axis=0)
            new_coeff = np.array([coeff_stt2[i]])
            #print( coeff_stt3.shape , new_coeff.shape)
            coeff_stt3 = np.concatenate( (coeff_stt3, new_coeff), axis=0)
    return [coeff_stt3, state3]

def visualisationInOutstate(input_state, T=None):
    '''
    Gives a brief visualisation in the braket notation of the output state from te beam splitter array T.
    '''
    if type(T) == type(None):
        output_state = input_state
    else:
        strin = ''
        for i in range(len(input_state)):
            strin = strin+str(int(input_state[i]))
        print('INPUT:\n(+1.00+0.00i) |'+strin+'>')
        # --- making output state --- #
        output_state = whichState(input_state, T)
    # --- Visualisation of the output state --- #
    n = output_state[1].shape[1]
    l = output_state[1].shape[0]
    print('OUPUT:')
    for i in range(l):
        strout = ''
        for j in range(n):
            strout = strout + str(int(output_state[1][i,j]))
        print('({0:+4.2f}{1:+4.2f}i) |'.format(np.real(output_state[0][i]), np.imag(output_state[0][i])) + strout + '>')
    return None

def postSelect(Output_state, mode_cond_list):
    '''
    Chack the states with the number of photon that satisfy the given condition. For instance if
    mode_cond_list=[[2, '<1']], then we check 
    The input of the function are:
        - N: size of the square matrix
        - the list of switched modes: mode_switch_list = [[mode, string_condition], ...]
    '''
    delete_list = []
    for j in range(Output_state[1].shape[0]):
        out_state = Output_state[1][j]
        for i in range(len(mode_cond_list)):
            mode, cond = mode_cond_list[i]
            bool_ = eval(str(out_state[mode])+cond)
            if not bool_:
                delete_list.append(j)
    delete_list =  list(dict.fromkeys(delete_list)) # to remove duplicates
    return [np.delete(Output_state[0],delete_list), np.delete(Output_state[1], delete_list, axis=0)]

################################################################################################
# CODE
################################################################################################
if __name__=='__main__':
    print('STARTING')
