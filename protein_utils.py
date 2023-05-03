"""This file implements utility functions that are useful when interacting with protein data."""

# Dependencies
from Bio.PDB import *
import numpy as np
import py3Dmol

parser = PDBParser()  # no need to instantiate this at every method call.


def parse_pdb_coordinates(directory):
    """This method reads a clean PDB file and returns an array of shape [N, b, 3] where N is the number of residues
    and b is the number of backbone atoms per residue. If there are unclean pdb files with ligands etc.,
    it will produce an error.
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


def visualize_connections(pdb_path, residue_pairs, cylinder_color='red', sphere_color='red', radius=0.2, sphere_radius=0.8):
    """A method that visualizes connections between residue pairs given residue pairs and a pdb file.
    Args:
        pdb_path: the path for the pdb file
        residue_pairs: a list of tuples (res1, res2) that indicates which residues are connected to each other
        cylinder_color: the connections are added as cylinders in py3Dmol, the color of cylinders
        sphere_color: at the end of connections, there is a sphere to indicate the residue, the color of those spheres
        radius: the radius of connecting cylinder
        sphere_radius: the radius of spheres indicating the residues
    Credit: GPT-4"""

    # Private method to add connections
    def _add_connections(in_viewer, in_residue_pairs):
        """# Function to add connections between residues in py3Dmol."""
        for pair in in_residue_pairs:
            # Add cylinder
            in_viewer.addCylinder({'start': {'resi': pair[0], 'chain': 'A', 'atom': 'CA'},
                                   'end': {'resi': pair[1], 'chain': 'A', 'atom': 'CA'},
                                   'color': cylinder_color, 'radius': radius})
            # Add spheres at the ends of the cylinder
            in_viewer.addSphere({'center': {'resi': pair[0], 'chain': 'A', 'atom': 'CA'},
                                 'color': sphere_color, 'radius': sphere_radius})
            in_viewer.addSphere({'center': {'resi': pair[1], 'chain': 'A', 'atom': 'CA'},
                                 'color': sphere_color, 'radius': sphere_radius})

    # Read the PDB file content
    with open(pdb_path, 'r') as f:
        pdb_data = f.read()

    # Create the py3Dmol viewer and add the PDB data
    viewer = py3Dmol.view()
    viewer.addModel(pdb_data, 'pdb')

    # Set protein color to gray
    viewer.setStyle({'cartoon': {'color': 'gray'}})

    # Add connections between the residues
    _add_connections(viewer, residue_pairs)

    # Render the viewer
    viewer.zoomTo()
    viewer.show()
