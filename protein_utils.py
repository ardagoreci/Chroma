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
    """
    Create a Biopython PDB.Model object from a numpy array of shape [n_res, 4, 3].

    Parameters:
    - coords (numpy array): Array of shape [n_res, 4, 3] representing protein coordinates.

    Returns:
    - model (Bio.PDB.Model): A Biopython model of the protein.
    """

    # Ensure the input array has the correct shape
    if len(coordinates.shape) != 3 or coordinates.shape[1] != 4 or coordinates.shape[2] != 3:
        raise ValueError("Input array must have shape [n_res, 4, 3]")

    # Create a new model
    model = Model.Model(id=model_id)

    # For simplicity, let's assume a single chain
    chain = Chain.Chain("A")
    model.add(chain)

    # Define atom names (assuming a simple representation with only backbone atoms)
    atom_names = ["N", "CA", "C", "O"]
    atom_elements = ["N", "C", "C", "O"]

    # Iterate through coordinates to construct the model
    for i, residue_coords in enumerate(coordinates):
        # Create a new residue object
        res_id = (' ', i + 1, ' ')
        resname = 'GLY'
        seg_id = '    '
        res = Residue.Residue(res_id, resname, seg_id)

        # Add atoms to the residue
        for j, atom_coord in enumerate(residue_coords):
            # Create atom object
            atom = Atom.Atom(name=atom_names[j],
                             coord=atom_coord,
                             bfactor=0.0,
                             occupancy=1.0,
                             altloc=' ',
                             fullname=atom_names[j],
                             element=atom_elements[j],
                             serial_number=j + 1)
            res.add(atom)
        chain.add(res)
    return model
