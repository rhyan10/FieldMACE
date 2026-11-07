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
        """Load the FieldMACE model given by the template keyword 'model_file'."""
        device = "cuda" if self.QMin.resources["ngpu"] else "cpu"
        properties = getattr(self, "properties", None) or ["energy", "forces"]
        self.models = SharcCalculator(
            list(self.QMin.molecule["elements"]),
            self.QMin.template["model_file"],
            device=device,
            n_states=getattr(self, "n_states", None),
            properties=properties,
            mm_positions_in_angstrom=False,  # SHARC passes point-charge coordinates in bohr
        )

    def run(self):
        coords = np.asarray(self.QMin.coords["coords"], dtype=float)
        pccoords = self.QMin.coords.get("pccoords") if self.QMin.molecule.get("point_charges") else None
        pccharge = self.QMin.coords.get("pccharge") if pccoords is not None else None
        qm_out = self.models.calculate(coords, pccoords, pccharge)
        self.QMout.h = np.array(qm_out["h"])
        if "grad" in qm_out:
            self.QMout.grad = np.array(qm_out["grad"])
        if "grad_pc" in qm_out and pccoords is not None:
            self.QMout.grad_pc = np.array(qm_out["grad_pc"])
        if "dm" in qm_out:
            self.QMout.dm = np.array(qm_out["dm"])
        if "nacdr" in qm_out:
            self.QMout.nacdr = np.array(qm_out["nacdr"])
        return None

    def getQMout(self):
        # everything is already 
        self.QMout.states = self.QMin.molecule['states']
        self.QMout.nstates = self.QMin.molecule['nstates']
        self.QMout.nmstates = self.QMin.molecule['nmstates']
        self.QMout.natom = self.QMin.molecule['natom']
        self.QMout.npc = self.QMin.molecule['npc']
        self.QMout.point_charges = bool(self.QMin.molecule.get("point_charges", False))
        return self.QMout

    def create_restart_files(self):
        x=open("restart/restart","w")
        x.write(str(self.QMin.coords['coords']))
        x.close()
        return None 



    def final_print(self):
        self.sharc_writeQMin()

    def readParameter(self, param,  *args, **kwargs):
        pass

