"""This file implements utility functions that are useful when interacting with protein data.

TODO: I will fully deprecate this file. Instead, I will be using protein.py
"""

# Dependencies
from Bio.PDB import Model, Atom, Residue, Chain
import numpy as np
import py3Dmol


def parse_pdb_coordinates(parser, directory):
    """This method reads a clean PDB file and returns an array of shape [N, b, 3] where N is the number of residues
    and b is the number of backbone atoms per residue. If there are unclean pdb files with ligands etc.,
    it will produce an error. parser is a PDBParser object.
    (unit-tested)"""
    structure = parser.get_structure("protein", directory)
    model = structure[0]  # access the first model
    coordinates = []
    for chain in structure[0]:
        for residue in chain:
            # I will extract the first 4 backbone atoms
            # N, CA, C, O atoms in backbone
            backbone_atoms = list(residue.get_atoms())[:4]  # Get the first 4 atoms
            atom_coordinates = [atom.coord for atom in backbone_atoms]
            coordinates.append(atom_coordinates)
        break  # I am for now using only the first chain
    return np.stack(coordinates, axis=0)


def model_from_coordinates(model_id, coordinates):
    new_model = Model.Model(id=model_id)
    chain = Chain.Chain(id='A')
    for i in range(coordinates.shape[0]):
        res_id = (' ', i + 1, ' ')
        resname = 'GLY'
        seg_id = '    '
        res_obj = Residue.Residue(res_id, resname, seg_id)

        if coordinates.shape[1] == 1:
            # Only CA atoms
            ca_coord = coordinates[i][0]
            ca_atom = Atom.Atom(name="CA",
                                coord=ca_coord,
                                bfactor=0.0,
                                occupancy=1.0,
                                altloc=' ',
                                fullname="CA",
                                element="C",
                                serial_number=2)
            res_obj.add(ca_atom)
            chain.add(res_obj)

        elif coordinates.shape[1] == 4:
            # Backbone atoms
            # residue.shape == (B, 3)
            n_coord = coordinates[i][0]
            n_atom = Atom.Atom(name="N",
                               coord=n_coord,
                               bfactor=0.0,
                               occupancy=1.0,
                               altloc=' ',
                               fullname="N",
                               element="N",
                               serial_number=1)
            ca_coord = coordinates[i][1]
            ca_atom = Atom.Atom(name="CA",
                                coord=ca_coord,
                                bfactor=0.0,
                                occupancy=1.0,
                                altloc=' ',
                                fullname="CA",
                                element="C",
                                serial_number=2)
            c_coord = coordinates[i][2]
            c_atom = Atom.Atom(name="C",
                               coord=c_coord,
                               bfactor=0.0,
                               occupancy=1.0,
                               altloc=' ',
                               fullname="C",
                               element="C",
                               serial_number=3)
            o_coord = coordinates[i][3]
            o_atom = Atom.Atom(name="O",
                               coord=o_coord,
                               bfactor=0.0,
                               occupancy=1.0,
                               altloc=' ',
                               fullname="O",
                               element="O",
                               serial_number=4)
            res_obj.add(n_atom)
            res_obj.add(ca_atom)
            res_obj.add(c_atom)
            res_obj.add(o_atom)

            chain.add(res_obj)
        else:
            print("Error!")
    new_model.add(chain)
    return new_model
