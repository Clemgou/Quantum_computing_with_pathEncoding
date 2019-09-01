#!/usr/bin/env python
# -*- coding: utf-8 -*-


################################################################################################
# IMPORTATIONS
################################################################################################
import sys
sys.path.insert(0, './Dependencies_import')

import numpy as np

from s_Photon_quantum_state      import *
from s_Matrix_buiding_function   import *
from s_Compute_all_output_states import *

################################################################################################
# FUNCTIONS
################################################################################################

class PhotonicNetwork():
    def __init__(self):
        self.channel_nbr = 0
        self.channels  = {}
        self.input     = {}
        self.output    = {}
        self.networMat = None
    # --- define external method --- #
    allOutputStatefromNetwork = OutputStateFromInputStateAndNetworkMatrix
    makeBeamSplitter          = beamSplitter2Modes
    # ---  --- #

    def setInputSate(self, inputState):
        self.input = inputState

    def setNetworkMatrix(self, T):
        self.networMat = T

    def setChannelNumber(self, N):
        self.channel_nbr = N

    def addElementBetweenChannels(self, ch_list, element):
        return None

    def makeBeamSplitter(self):
        '''
        return a 2x2 np.array representing a 50:50 balanced beam splitter
        '''
        BS = 1/np.sqrt(2) * np.array([[1, 1j],[1j, 1]])
        #BS = 1/np.sqrt(2) * np.array([[1, 1],[-1, 1]])
        return BS

    def buildNetworkMatrix(self):
        N = len(list(self.channels.keys()))
        T = np.eye(N)
        return T

    def visualisationSecondQuantisation(self, state, name=''):
        '''
        Gives a brief visualisation in the braket notation of the state in the second quantisation.
        The param 'state' must be a dictionary such that:
            state[key] = [coeff, list_of_photon_occupation]
        '''
        print('State {}:'.format(name))
        KEYS = list(state.keys())
        for key in KEYS:
            ph_occupation = ''
            for i in state[key][1]:
                ph_occupation += str(int(i))
            print( '({0:+4.2f}{1:+4.2f}i) |{2}>'.format(np.real(state[key][0]), np.imag(state[key][0]), ph_occupation) )
        return None

    def postSelect(Output_state, mode_cond_list):
        '''
        Check the states with the number of photon that satisfy the given condition. For instance if
        mode_cond_list = [[2, '<1']], then we check if the mode 2 has at less than 1 photon,
        otherwise we discard the state.
        The Output_state is a dictionary such that: Output_state[key] = [coeff, list_of_photon_occupation]
        '''
        post_selected_state = Output_state.copy()
        for key in Output_state:
            out_state = Output_state[key][1]
            for i in range(len(mode_cond_list)):
                mode, cond = mode_cond_list[i]
                bool_ = eval(str(out_state[mode])+cond)
                if not bool_:
                    post_selected_state.pop(key)
        return post_selected_state

    def simulatePhotonicNetwork(self, input_state=None, network_mat=None, print_all=False):
        if   type(input_state) == type(None) and len(list(self.input.keys())) != 0:
            input_state = self.input
        elif type(input_state) != type(dict([])):
            print('Error in simulatePhotonicNetwork: input_state not valid.')
            return None
        if   type(network_mat) == type(None) and type(self.networMat) != type(None):
            network_mat = self.networMat
        elif type(network_mat) != type(np.array([])):
            print('Error in simulatePhotonicNetwork: input_state not valid.')
            return None
        # ---  --- #
        self.output = self.allOutputStatefromNetwork( input_state, network_mat , printable=print_all)
        self.visualisationSecondQuantisation( self.output )

################################################################################################
# CODE
################################################################################################
if __name__=='__main__':
    print('STARTING')
    PN = PhotonicNetwork()
    # ---  --- #
    BS = PN.makeBeamSplitter()
    PN.setNetworkMatrix(BS)
    PN.simulatePhotonicNetwork( {1 : [1., [3,1] ]} , BS)

