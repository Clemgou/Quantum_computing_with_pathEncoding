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
        self.networLayers = {}
    # --- define external method --- #
    allOutputStatefromNetwork = OutputStateFromInputStateAndNetworkMatrix
    matrixPhaseShift          = matrixPhaseShift
    matrixSwitchingOutput     = matrixSwitchingOutput
    # ---  --- #

    def setInputSate(self, inputState):
        self.input = inputState

    def setNetworkMatrix(self, T):
        self.networMat = T

    def setChannelNumber(self, N):
        self.channel_nbr = N

    def addElementBetweenChannels(self, ch_list, element):
        return None

    def BS(self):
        '''
        return a 2x2 np.array representing a 50:50 balanced beam splitter
        '''
        return 1/np.sqrt(2) * np.array([[1, 1j],[1j, 1]])

    def PS(self,  mode_phase_dic):
        '''
        return the square matrix for adding phase shifts specified by mode_phase_list:
            -  mode_phase_dic = { mode_nbr : phase_shift , ...}
        '''
        if self.channel_nbr == 0:
            print('ERROR in PS: the network has no channels.')
            return None
        mode_phase_list = []
        for key in mode_phase_dic:
            mode_phase_list.append([key, mode_phase_dic[key]])
        PS_mat = matrixPhaseShift(self.channel_nbr, mode_phase_list)
        return PS_mat

    def switchMat(self, mode_switch_dic):
        '''
        return the square matrix that commute two outputs.
        It re-organises the given input into the desired output,
            - mode_switch_dic = {input : output , ... }
        '''
        if self.channel_nbr == 0:
            print('ERROR in switchMat: the network has no channels.')
            return None
        mode_switch_list = []
        for key in mode_switch_dic:
            mode_switch_list.append([key, mode_switch_dic[key]])
        return matrixSwitchingOutput(self.channel_nbr, mode_switch_list)

    def initNetworkMatrix(self, T=None):
        '''
        Initiat the network matrix with unity matrix by default.
        Also initiat the channel_nbr.
        '''
        if   type(T) == type(None):
            if self.channel_nbr != 0:
                self.networMat   = np.eye(self.channel_nbr)
            else:
                print('ERROR in initNetworkMatrix: no channel to initiat the netwrk.')
                return None
        else:
            self.networMat   = T.copy()
            self.channel_nbr = T.shape[0]

    def addNetworkLayer(self, T):
        s = T.shape
        if s[0] != s[1] or s[0] != self.channel_nbr:
            print('ERROR in addNetworkLayer: new layer matrix is not the same size than the network.')
            return None
        self.networMat = np.dot(self.networMat, T)

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
        '''
        Compute the final state resulting from the input state in the photonic network.
        It returns a sum of all the non-zero states in the second quantisation formalism, ie in terms of modes occupation.
        '''
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
        # --- verification --- #
        key = list(input_state.keys())[0]
        if network_mat.shape[0] != network_mat.shape[1] or len(input_state[key][1]) != network_mat.shape[0]:
            print('ERRROR in simulatePhotonicNetwork: wrong dimension for input or matrix network.')
            return None
        # --- simulation --- #
        self.output = self.allOutputStatefromNetwork( input_state, network_mat , printable=print_all)
        self.visualisationSecondQuantisation( self.output )

################################################################################################
# CODE
################################################################################################
if __name__=='__main__':
    print('STARTING')
    PN = PhotonicNetwork()
    # ---  --- #
    PN.initNetworkMatrix(np.eye(4))
    PN.addNetworkLayer( makeMatDiagFromSquareMatrices([1.,PN.BS(),1.]) )
    PN.addNetworkLayer( makeMatDiagFromSquareMatrices([PN.BS(),PN.BS()]) )
    PN.simulatePhotonicNetwork( {1 : [1., [0,1,1,0] ]})

