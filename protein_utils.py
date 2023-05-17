"""This file implements utility functions that are useful when interacting with protein data."""

# Dependencies
import jax
from Bio.PDB import *
import numpy as np
import jax.numpy as jnp
import py3Dmol
from models import Transforms


def structure_to_transforms(coordinates) -> Transforms:
    """Constructs frames using the position of three atoms from the ground truth PDB structures using a Gram-Schmidt
    process. Note: the translation vector is assigned to the centre atom. Following AlphaFold, it uses N as x1, Ca as x2
    and C as x3 for backbone frames, so the frame has Ca at the centre. This function is written without the batch dim,
    it should be jax.vmap transformed if used with batch dimension.
    For details, see AlphaFold Supplementary Information, Alg. 21.
    Args:
        coordinates: structure coordinates of shape [N, 4, 3]
    Returns: a Transforms object storing the residue transforms, each with a translation vector and a rotation matrix.
    (unit-tested)"""

    def _single_rigid_from_3_points(x1, x2, x3) -> Transforms:
        """Single example implementation fo rigid_from_3_points. The implementation is easier and less bug-prone if
        written for a single example and then jax.vmap transformed.
        Args:
            x1: coordinate array of shape [3,]
            x2: coordinate array of shape [3,]
            x3: coordinate array of shape [3,]
        Returns: a Transforms object storing the residue transforms, each with a translation vector and a rotation
                 matrix.
        """
        v1 = x3 - x2
        v2 = x1 - x2
        e1 = v1 / jnp.linalg.norm(v1)
        u2 = v2 - (e1 * jnp.dot(e1, v2))
        e2 = u2 / jnp.linalg.norm(u2)
        e3 = jnp.cross(e1, e2)
        R = jnp.stack([e1, e2, e3], axis=0)
        t = x2  # translation atom assigned to center atom x2
        return Transforms(t, R)

    N = coordinates[:, 0, :]
    Ca = coordinates[:, 1, :]
    C = coordinates[:, 2, :]
    batch_rigid_from_3_points_fn = jax.vmap(_single_rigid_from_3_points)
    return batch_rigid_from_3_points_fn(N, Ca, C)


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


def visualize_connections(pdb_path, residue_pairs, cylinder_color='red', sphere_color='red', radius=0.2,
                          sphere_radius=0.8):
    """A method that visualizes connections between residue pairs given residue pairs and a pdb file.
    Args:
        pdb_path: the path for the pdb file
        residue_pairs: a list of tuples (res1, res2) that indicates which residues are connected to each other
        cylinder_color: the connections are added as cylinders in py3Dmol, the color of cylinders
        sphere_color: at the end of connections, there is a sphere to indicate the residue, the color of spheres
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
