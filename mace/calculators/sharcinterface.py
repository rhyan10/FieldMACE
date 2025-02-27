#from typing import Dict, List, Union
#from sharc.pysharc.interface import SHARC_INTERFACE
from mace.calculators.sharc_calculator import SharcCalculator
import sys
import math
import datetime
import struct
from multiprocessing import Pool
from copy import deepcopy
from socket import gethostname
import numpy as np
#from schnarc import calculators
import torch
import time
# internal
from SHARC_FAST import SHARC_FAST
from globals import DEBUG, PRINT
from utils import *
from constants import IToMult, rcm_to_Eh
from io import TextIOWrapper
from utils import timer

class SHARC_NN(SHARC_FAST):
    """
    Class for SHARC NN
    """
    def __init__(self, model_path, n_states, threshold, cutoff, nac_keys, properties):
        self.model_path = model_path
        self.n_states = n_states
        self.threshold = threshold
        self.cutoff = cutoff
        self.nac_keys = nac_keys
        self.properties = properties

    def get_features(self, KEYSTROKES: TextIOWrapper = None) -> set:
        return {
            "h",
            "dm",
            "grad",
            "nacdr",
            "point_charges",
            "grad_pc",
        }

    def read_template(self, template_filename='SCHNARC.template'):
        '''reads the template file
        has to be called after setup_mol!'''
        
        kw_whitelist = {'model_file','qmmm','charge','paddingstates'}
        QMin = self.QMin
        QMin.template.types={
                "model_file":str,
                "qmmm":bool,
                "charge":list,
                "paddingstates":list
                }
        QMin.template.data={
                "model_file":"best_model",
                 "qmmm":False,
                 "charge":[0],
                 'paddingstates': [0,0,0,0,0]
                }

        super().read_template(template_filename, kw_whitelist = kw_whitelist)
        return

    def read_resources(self, resources_filename="SCHNARC.resources"):
        super().read_resources(resources_filename)
        self._read_resources=True
        return

    def setup_interface(self):
        # param are some paramters for the neuralnetwork, which can be changed in the definition below
        # the SchNarculator is initialized with dummy coordinates and fieldschnet is enabled if point charges are found
        param = parameters()
        dummy_crd=torch.zeros(len(self.QMin.molecule["elements"]),3)
        if self.QMin.resources["ngpu"]:
            device=torch.device('cuda')
        else:
            device=torch.device('cpu')
        self.models = SharcCalculator("")

    def run(self):

        sharc_out = self.QMin.coords["coords"]
        print(self.QMin)
        sharc_out={'positions':self.QMin.coords["coords"],'external_charges':self.QMin.coords["pccharge"]}
        NN_out = self.models.calculate(sharc_out)
        keys = NN_out.keys()
        #'dm', 'nacdr', 'h', 'grad', 'dydf', 'pc_grad'

        self.log.debug(NN_out.keys())
        for key in NN_out.keys():
            if key == "h":
                self.log.debug(key)
                self.QMout.h = np.array(NN_out["h"])

            elif key == "grad":
                self.log.debug(key)
                self.QMout.grad = np.array(NN_out["grad"])

            elif key == "dydf":
                if not self.QMin.molecule["point_charges"]:
            # technically you could have an NN, which predicts this gradient even though
            # there are no point charges in SHARC
                    continue
                else:
            # computing the gradient of each point charge with respect to each atom
            # summing over the corresponding axis afterwards to get the correct dimensions
                    pc_grads = self.get_pc_grad(NN_out["dydf"])
                    self.QMout.grad += np.sum(pc_grads, axis=2)*(-1)
                    self.QMout.grad_pc = np.sum(pc_grads, axis=1) #*(-1)
                self.log.debug(key)

            elif key == "dm":
                self.QMout.dm = np.array(NN_out["dm"])

            elif key == "nacdr":
                self.QMout.nacdr = np.array(NN_out["nacdr"])
                self.log.debug(key)

            elif key == "socdr":
                self.QMout.socdr = np.array(NN_out["socdr"])

            else:
                self.log.warning(key, " is not implemented")

        return None

    def getQMout(self):
        # everything is already 
        self.QMout.states = self.QMin.molecule['states']
        self.QMout.nstates = self.QMin.molecule['nstates']
        self.QMout.nmstates = self.QMin.molecule['nmstates']
        self.QMout.natom = self.QMin.molecule['natom']
        self.QMout.npc = self.QMin.molecule['npc']
        self.QMout.point_charges = False
        return self.QMout

    def create_restart_files(self):
        x=open("restart/restart","w")
        x.write(str(self.QMin.coords['coords']))
        x.close()
        return None 



    def get_pc_grad(self,dEdF):
        # computes the gradients of the energy, which include the point charges
        # requires the change of energy with the electric field
        N_MM = self.QMin.coords["pccharge"].shape[0]
        N_QM = self.QMin.coords["coords"].shape[0]
        dF_dx = np.zeros((N_QM, N_MM, 3,3))
        # loop over all qm atoms computing the change of the electric field with respect to the coordinates of atoms and point charges
        for qm_idx, qm_atom in enumerate(self.QMin.coords["coords"]):
            dists = np.subtract(qm_atom,self.QMin.coords["pccoords"])
            norms = np.linalg.norm(dists, axis=1).reshape(N_MM,1,1)
            mat = (3.*np.einsum('ij,il->ijl', dists, dists)/np.power(norms,2)- np.identity(3))
            mat /= np.power(norms,3)
            dF_dx[qm_idx] = mat * (self.QMin.coords["pccharge"].reshape(N_MM, 1,1))
        return np.einsum('ijk,jlkm->ijlm', dEdF, dF_dx)

    def final_print(self):
        self.sharc_writeQMin()

    def readParameter(self, param,  *args, **kwargs):
        pass

