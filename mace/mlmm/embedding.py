# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)

"""
The *embedding frame*: how raw simulation coordinates are turned into the
inputs FieldMACE sees (QM positions, MM point-charge positions and charges).

The same object is used by the OpenMM force **and** by the training-data
exporter, so the model is always evaluated in exactly the frame it was
trained in. Everything is written in torch so that forces through the frame
transformation (centering, periodic imaging, link atoms, charge switching)
come out of autograd exactly.

Conventions (all lengths in Angstrom, charges in e):
  * The QM region is made whole (each QM atom is imaged next to QM atom 0).
  * origin="centroid" (default): the geometric centre of the QM atoms is put at
    (0, 0, 0). "com" uses the mass-weighted centre, "none" keeps raw coordinates.
    For PerAtomFieldEMACE the result is invariant to this choice; for
    FieldEMACE (single multipole expansion about the origin) it is not, and the
    centroid is the natural expansion centre.
  * Every MM charge is placed at its minimum image relative to the QM centre.
  * Optional cutoff: charges are scaled by a C2 switching function of their
    distance to the QM centre (or the nearest QM atom), going from 1 at
    (cutoff - switch_width) to 0 at cutoff. Charges outside are dropped.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .system import QMMMSpec


@dataclass
class EmbeddingSettings:
    origin: str = "centroid"  # "centroid" | "com" | "none"
    cutoff: Optional[float] = None  # Angstrom; None = all MM charges
    switch_width: float = 1.0  # Angstrom
    cutoff_reference: str = "centroid"  # "centroid" | "nearest"

    def __post_init__(self):
        if self.origin not in ("centroid", "com", "none"):
            raise ValueError(f"origin '{self.origin}': use 'centroid', 'com' or 'none'")
        if self.cutoff_reference not in ("centroid", "nearest"):
            raise ValueError(f"cutoff_reference '{self.cutoff_reference}': use 'centroid' or 'nearest'")
        if self.cutoff is not None:
            if self.cutoff <= 0:
                raise ValueError(f"cutoff={self.cutoff}: give a positive distance in Angstrom, or None for no cutoff")
            if not 0 <= self.switch_width < self.cutoff:
                raise ValueError(f"switch_width={self.switch_width} must be >= 0 and smaller than cutoff={self.cutoff} "
                                 "(charges are switched off between cutoff - switch_width and cutoff)")

    def to_dict(self):
        return asdict(self)


def _min_image(d: torch.Tensor, box: Optional[torch.Tensor]) -> torch.Tensor:
    """Minimum-image convention for OpenMM's reduced triclinic boxes (rows a, b, c)."""
    if box is None:
        return d
    d = d - torch.round(d[..., 2:3] / box[2, 2]) * box[2]
    d = d - torch.round(d[..., 1:2] / box[1, 1]) * box[1]
    d = d - torch.round(d[..., 0:1] / box[0, 0]) * box[0]
    return d


def _switch(x: torch.Tensor) -> torch.Tensor:
    """C2 smoothstep, 1 at x<=0, 0 at x>=1."""
    x = torch.clamp(x, 0.0, 1.0)
    return 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


class EmbeddingFrame:
    """Maps the positions of the force's particle subset to FieldMACE inputs.

    The particle subset is ``spec.qm_atoms + extra + spec.mm_atoms`` where
    ``extra`` are MM link hosts that carry no embedding charge.
    """

    def __init__(self, spec: QMMMSpec, settings: EmbeddingSettings = EmbeddingSettings()):
        self.spec = spec
        self.settings = settings
        if settings.origin not in ("centroid", "com", "none"):
            raise ValueError("origin must be 'centroid', 'com' or 'none'")
        if settings.cutoff_reference not in ("centroid", "nearest"):
            raise ValueError("cutoff_reference must be 'centroid' or 'nearest'")
        qm = list(spec.qm_atoms)
        mm_ = list(spec.mm_atoms)
        extra = sorted({lk.mm_atom for lk in spec.link_atoms} - set(mm_))
        self.particles = qm + extra + mm_  # global OpenMM indices, in subset order
        local = {g: k for k, g in enumerate(self.particles)}
        self.n_qm = len(qm)
        self.link_qm = torch.tensor([local[lk.qm_atom] for lk in spec.link_atoms], dtype=torch.long)
        self.link_mm = torch.tensor([local[lk.mm_atom] for lk in spec.link_atoms], dtype=torch.long)
        self.link_ratio = torch.tensor([lk.ratio for lk in spec.link_atoms], dtype=torch.float64)
        self.mm_local = torch.tensor([local[g] for g in mm_], dtype=torch.long)
        self.mm_charges = torch.tensor(spec.mm_charges, dtype=torch.float64)
        self.qm_masses = torch.tensor(spec.qm_masses, dtype=torch.float64)

    # ------------------------------------------------------------------ #
    def __call__(self, pos: torch.Tensor, box: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        pos : (n_subset, 3) Angstrom, may require grad.
        box : (3, 3) Angstrom box vectors (rows) or None.
        returns qm_pos (n_qm + n_link, 3), mm_pos (M, 3), mm_q (M,), mm_sel (M,)
        where mm_sel indexes spec.mm_atoms (the charges that were kept).
        """
        dt, dev = pos.dtype, pos.device
        s = self.settings
        if box is not None:
            box = box.to(dt)
        qm = pos[: self.n_qm]
        qm = qm[0] + _min_image(qm - qm[0], box)  # make QM region whole

        if s.origin == "com":
            w = self.qm_masses.to(dt).to(dev)
            centre = (w[:, None] * qm).sum(0) / w.sum()
        else:
            centre = qm.mean(0)

        # link atoms (H) on the QM-MM bonds
        if len(self.link_qm):
            a = qm[self.link_qm.to(dev)]
            b = a + _min_image(pos[self.link_mm.to(dev)] - a, box)
            r = self.link_ratio.to(dt).to(dev)[:, None]
            links = a + r * (b - a)
            qm_all = torch.cat([qm, links], 0)
        else:
            qm_all = qm

        mm_rel = _min_image(pos[self.mm_local.to(dev)] - centre, box)
        q = self.mm_charges.to(dt).to(dev)
        sel = torch.arange(len(q), device=dev)

        if s.cutoff is not None and box is not None and not getattr(self, "_warned_box", False):
            with torch.no_grad():  # every charge within the cutoff must have a unique image
                reach = s.cutoff + (float((qm_all - centre).norm(dim=-1).max()) if s.cutoff_reference == "nearest" else 0.0)
                half = 0.5 * float(torch.min(torch.diagonal(box)))
            if reach > half:
                import warnings

                self._warned_box = True
                warnings.warn(f"embedding cutoff {s.cutoff} A + QM-region radius gives {reach:.1f} A, more than half "
                              f"the box ({half:.1f} A): some charges are within the cutoff with two periodic images, "
                              "but only one is embedded. Use a smaller cutoff or a larger box")
        if s.cutoff is not None:
            with torch.no_grad():
                if s.cutoff_reference == "centroid":
                    dist = mm_rel.norm(dim=-1)
                else:
                    dist = torch.cdist(mm_rel, qm_all - centre).min(dim=1).values
                keep = dist < s.cutoff
            sel = sel[keep]
            mm_rel = mm_rel[keep]
            if s.cutoff_reference == "centroid":
                d = mm_rel.norm(dim=-1)
            else:
                d = torch.cdist(mm_rel, qm_all - centre).min(dim=1).values
            r_on = s.cutoff - s.switch_width
            q = q[keep] * _switch((d - r_on) / max(s.switch_width, 1e-12))

        if s.origin == "none":
            return qm_all, mm_rel + centre, q, sel
        return qm_all - centre, mm_rel, q, sel

    # ------------------------------------------------------------------ #
    def numpy(self, positions_A: np.ndarray, box_A: Optional[np.ndarray] = None):
        """Same as __call__ for *global* OpenMM positions (N, 3) in Angstrom, numpy in/out."""
        pos = torch.as_tensor(np.asarray(positions_A)[self.particles], dtype=torch.float64)
        box = None
        if box_A is not None and self.spec.periodic:
            box = torch.as_tensor(np.asarray(box_A), dtype=torch.float64)
        with torch.no_grad():
            qm, mmp, q, sel = self(pos, box)
        return qm.numpy(), mmp.numpy(), q.numpy(), sel.numpy()
