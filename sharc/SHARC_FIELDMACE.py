#!/usr/bin/env python3
# ******************************************
#
#    SHARC 4 interface for FieldMACE
#    Author: Rhyan Barrett
#
#    FieldMACE (https://github.com/rhyan10/FieldMACE) predicts energies, gradients,
#    non-adiabatic couplings, dipoles and spin-orbit couplings of a QM region in the
#    electrostatic field of MM point charges.
#
#    Install: copy or link this file into $SHARC (the bin directory of SHARC 4), with
#    FieldMACE importable in the same Python environment:
#        ln -s /path/to/FieldMACE/sharc/SHARC_FIELDMACE.py $SHARC/
#
#    Use it
#      * on its own (gas phase or with "point_charges" in QM.in), interface name FIELDMACE;
#      * as the QM program of SHARC_QMMM (electrostatic embedding): "qm-program FIELDMACE"
#        in QMMM.template, with an MM program such as OPENMM.
#
#    FIELDMACE.template (see FIELDMACE.template next to this file):
#        model_file   /path/to/model.model        (or "modelpath" in FIELDMACE.resources)
#        properties   energy forces nacs dipoles socs
#        device       cpu | cuda
#        energy_unit  eV | Hartree                (units the model was trained in)
#        distance_unit Angstrom | Bohr
#        nac_to_au / dipole_to_au / soc_to_au     (optional unit conversions, see SharcCalculator)
#        nac_type     nac | smooth                (smooth: the model predicts (E_j - E_i) * NAC_ij)
#        soc_selection_rules true | false         (default true: SOCs between singlets, between the M_S
#                                                  components of one multiplet and for |dM_S| > 1 are zero)
#
#    This file is distributed with FieldMACE under the MIT license; SHARC is GPL-3.
#
# ******************************************

import datetime
import os
import sys
import shutil
from io import TextIOWrapper

import numpy as np
from SHARC_FAST import SHARC_FAST
from utils import expand_path, question

from mace.calculators.sharc_calculator import SharcCalculator

__all__ = ["SHARC_FIELDMACE"]

AUTHORS = "Rhyan Barrett"
VERSION = "2.4"
VERSIONDATE = datetime.datetime(2026, 11, 8)
NAME = "FIELDMACE"
DESCRIPTION = "     FAST interface for FieldMACE (ML for QM/MM; energies, gradients, NACs, dipoles, SOCs)"
CHANGELOGSTRING = """
2.1  SHARC 4 interface: point charges (grad_pc), multiplicities, SOCs, phase tracking
2.3  smooth NACs (nac_type), SOC selection rules, double-precision arrays for pysharc
     (float32 models gave zero gradients in pysharc); tutorial: tutorials/04_sharc_dynamics
2.4  diabatic models: diagonalised every step (per multiplicity), NACs on point charges (nacdr_pc),
     wave-function overlaps (coupling overlap); tutorial: tutorials/05_diabatic_states
2.4.1 unknown template keywords reported, units validated
2.4.2 bug-fix release (CHANGELOG.md); the calculator no longer reads labels from the frames
"""
PREV_FILE = "FIELDMACE_previous.npz"


class SHARC_FIELDMACE(SHARC_FAST):
    """SHARC interface for FieldMACE"""

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.QMin.resources.update({"modelpath": None})
        self.QMin.resources.types.update({"modelpath": str})
        self.QMin.template.update(
            {
                "model_file": None,
                "properties": ["energy", "forces"],
                "device": "cpu",
                "energy_unit": "eV",
                "distance_unit": "Angstrom",
                "nac_to_au": None,
                "dipole_to_au": None,
                "soc_to_au": None,
                "soc_selection_rules": True,
                "nac_type": "nac",
            }
        )
        self.QMin.template.types.update(
            {
                "model_file": str,
                "properties": list,
                "device": str,
                "energy_unit": str,
                "distance_unit": str,
                "nac_to_au": float,
                "dipole_to_au": float,
                "soc_to_au": float,
                "soc_selection_rules": bool,
                "nac_type": str,
            }
        )
        self.calculator = None
        self._prediction = None
        self._template_file = None
        self._resources_file = None
        self._warned = set()

    # ----- metadata -----
    @staticmethod
    def version() -> str:
        return SHARC_FIELDMACE._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_FIELDMACE._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_FIELDMACE._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_FIELDMACE._authors

    @staticmethod
    def name() -> str:
        return SHARC_FIELDMACE._name

    @staticmethod
    def description() -> str:
        return SHARC_FIELDMACE._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_FIELDMACE._name}\n{SHARC_FIELDMACE._description}"

    def _properties(self) -> list:
        props = self.QMin.template["properties"] or []
        if props and isinstance(props[0], list):  # the template parser may nest lists
            props = props[0]
        return [str(p).lower() for p in props]

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set[str]:
        """Requests this interface can answer (dm is always answered; zeros without a dipole model)."""
        features = {"h", "grad", "dm", "point_charges", "phases"}
        if not self._read_template:
            return features | {"nacdr", "soc"}
        props = self._properties()
        if any("nac" in p for p in props):
            features.add("nacdr")
        if self.calculator is not None and self.calculator.diabatic:
            features |= {"nacdr", "overlap"}  # from the diabatic Hamiltonian
        if "socs" in props or sum(self.QMin.molecule.get("states", [1])[1:]) == 0:
            features.add("soc")  # singlets only: SOCs are zero by definition
        return features

    # ----- setup (setup_traj.py) -----
    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info(f"{'FieldMACE interface setup':=^80}\n")
        if os.path.isfile("FIELDMACE.template") and question(
            "Use FIELDMACE.template in the current directory?", bool, KEYSTROKES=KEYSTROKES, default=True
        ):
            self._template_file = "FIELDMACE.template"
        else:
            while not os.path.isfile(path := question("Path to FIELDMACE.template:", str, KEYSTROKES=KEYSTROKES)):
                self.log.info(f"File {path} does not exist!")
            self._template_file = path
        if question("Do you have a FIELDMACE.resources file?", bool, KEYSTROKES=KEYSTROKES, default=False):
            while not os.path.isfile(path := question("Path to FIELDMACE.resources:", str, KEYSTROKES=KEYSTROKES)):
                self.log.info(f"File {path} does not exist!")
            self._resources_file = path
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str):
        create = os.symlink if INFOS.get("link_files") else shutil.copy
        create(expand_path(self._template_file), os.path.join(dir_path, "FIELDMACE.template"))
        if self._resources_file:
            create(expand_path(self._resources_file), os.path.join(dir_path, "FIELDMACE.resources"))

    def read_resources(self, resources_file="FIELDMACE.resources", kw_whitelist=None):
        if not os.path.isfile(resources_file):
            self._read_resources = True
            return
        super().read_resources(resources_file, kw_whitelist)

    def read_template(self, template_file="FIELDMACE.template", kw_whitelist=None):
        super().read_template(template_file, kw_whitelist)
        # SHARC ignores unknown keywords silently; a typo (e.g. "energy_units") would quietly use the default
        known = {k.lower() for k in self.QMin.template.keys()}
        if os.path.isfile(template_file):
            with open(template_file) as fh:
                for line in fh:
                    words = line.split("#")[0].split()
                    if words and words[0].lower() not in known:
                        msg = (f"{template_file}: unknown keyword '{words[0]}' is ignored "
                               f"(known: {', '.join(sorted(known))})")
                        self.log.warning(msg)
                        print("WARNING: " + msg, file=sys.stderr)

    # ----- run -----
    def setup_interface(self):
        super().setup_interface()
        model = self.QMin.template["model_file"] or self.QMin.resources["modelpath"]
        if not model:
            self.log.error("No model given: set 'model_file' in FIELDMACE.template (or 'modelpath' in FIELDMACE.resources)")
            raise ValueError("missing model_file")
        t = self.QMin.template
        self.calculator = SharcCalculator(
            atom_types=self.QMin.molecule["elements"],
            modelpath=expand_path(model),
            device=t["device"],
            states=list(self.QMin.molecule["states"]),
            properties=self._properties(),
            energy_unit=t["energy_unit"],
            distance_unit=t["distance_unit"],
            nac_to_au=t["nac_to_au"],
            dipole_to_au=t["dipole_to_au"],
            soc_to_au=t["soc_to_au"],
            soc_selection_rules=t["soc_selection_rules"],
            nac_type=t["nac_type"],
            mm_positions_in_angstrom=False,  # SHARC passes point-charge coordinates in bohr
        )
        self.log.info(
            f"FieldMACE model {model}: {self.calculator.n_model_states} model states, SHARC states "
            f"{self.calculator.states} ({self.calculator.mode} mapping), properties {self.calculator.properties}"
        )

    def _prev_file(self) -> str:
        return os.path.join(self.QMin.save["savedir"], PREV_FILE)

    def run(self):
        # phase reference: the previous step's prediction. Kept in memory (pysharc, persistent) and in
        # SAVE/FIELDMACE_previous.npz, so file-based sharc.x runs and restarts continue with consistent signs.
        if self.calculator.last_prediction is None and not self.QMin.save.get("init", False):
            self.calculator.load_state(self._prev_file())
        pc = self.QMin.molecule["point_charges"] and self.QMin.molecule["npc"] > 0
        self._prediction = self.calculator.predict(
            self.QMin.coords["coords"],
            self.QMin.coords["pccoords"] if pc else None,
            np.asarray(self.QMin.coords["pccharge"], dtype=float) if pc else None,
        )
        os.makedirs(self.QMin.save["savedir"], exist_ok=True)
        self.calculator.save_state(self._prev_file())

    def getQMout(self):
        req = self.QMin.requests
        requests = {k for k, v in req.items() if v and k != "retain"}
        npc = self.QMin.molecule["npc"] if self.QMin.molecule["point_charges"] else 0
        self.QMout.allocate(
            states=self.QMin.molecule["states"], natom=self.QMin.molecule["natom"], npc=npc, requests=requests
        )
        calc = self.calculator
        props = set(calc.properties)
        if req["soc"] and "socs" not in props and sum(calc.states[1:]) > 0:
            raise ValueError("SOCs requested, but FIELDMACE.template does not list 'socs' (or the model has none)")
        if req["nacdr"] and "nacs" not in props and not calc.diabatic:
            raise ValueError("NACs requested, but FIELDMACE.template does not list 'nacs' (or the model has none)")
        qm = calc.get_qm(self._prediction, as_arrays=True)

        if req["h"] or req["soc"]:
            h = np.asarray(qm["h"], dtype=complex)
            if not req["soc"]:
                h = np.diag(np.diag(h))
            self.QMout["h"] = h
        if req["grad"]:
            self.QMout["grad"] = qm["grad"]
            if npc > 0:
                self.QMout["grad_pc"] = qm["grad_pc"]
        if req["nacdr"]:
            self.QMout["nacdr"] = qm["nacdr"]
            if npc > 0 and "nacdr_pc" in qm:  # diabatic models: NACs on the point charges too
                self.QMout["nacdr_pc"] = qm["nacdr_pc"]
        if req.get("overlap"):
            if "overlap" not in qm:
                raise ValueError("overlaps need a diabatic model (train with --diabatic=True)")
            self.QMout["overlap"] = np.asarray(qm["overlap"], dtype=complex)
        if req["dm"]:
            if "dm" in qm:
                self.QMout["dm"] = qm["dm"]
            elif "dm" not in self._warned:
                self.log.warning("Dipoles requested but not predicted (add 'dipoles' to properties): returning zeros")
                self._warned.add("dm")
        if req["phases"]:
            self.QMout["phases"] = np.ones(self.QMin.molecule["nmstates"], dtype=complex)
        self.QMout["runtime"] = self.clock.measuretime(False)
        return self.QMout

    def create_restart_files(self):
        super().create_restart_files()


if __name__ == "__main__":
    from logger import loglevel

    try:
        SHARC_FIELDMACE(loglevel=loglevel).main()
    except KeyboardInterrupt:
        print("\nCTRL+C makes me a sad SHARC ;-(")
