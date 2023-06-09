"""This file implements utility functions that are useful when interacting with protein data."""

# Dependencies
import jax
from Bio.PDB import Model, Atom, Residue, Chain
import numpy as np
import jax.numpy as jnp
import py3Dmol
from typing import NamedTuple


class Transforms(NamedTuple):
    """A residue wise transform that stores a [..., 3, 3] orientation matrix and a [..., 3] translation vector."""
    translations: jnp.array
    orientations: jnp.array


def transforms_to_structure(transforms) -> jnp.array:
    """Computes the position of backbone atoms given transforms. The coordinates are computed using the ideal bond
    lengths of Gly with the following values (taken from alphafold.common.residue_constants.py):
    'GLY': ['N': (-0.572, 1.337, 0.000)],
           ['CA': (0.000, 0.000, 0.000)],
           ['C': (1.517, -0.000, -0.000)],
           ['O': (0.626, 1.062, -0.000)]
    It is important to note that N, CA, and C are in the backbone rigid group. However, O is in the psi group, so its
    coordinates vary depending on the torsion angle psi. The Ramachandran plots show that the value of psi is changes
    depending on the secondary structures alpha helices and beta sheets. For now, I will assume that the O is part of
    the backbone rigid group. This can then be fixed using residual updates to the final coordinates predicted from
    node embeddings - the network can easily learn the correct relative position of the O atom depending on the
    secondary structure context.
    Empirically, this has a root-mean-square deviation of 1.0 Angstrom from the true coordinates when given perfect
    transforms. Importantly, the carbon alpha positions are exactly the same.
    Args:
        transforms: a Transforms object containing translation of shape [N, 3] and orientations (rotations) of
        shape [N, 3, 3]
    Returns:
        an array of shape [N, 4, 3] encoding the coordinates
    (unit-tested)
    TODO: test if gradients are actually flowing through this method
    """

    def _frame_to_coor(t, O) -> jnp.array:
        """
        Args:
            t: translation vector of shape [3,]
            O: orientation vector of shape [3, 3]
        Returns:
            a jnp.array of shape [4, 3] encoding the transformed coordinates of N, CA, C, O in that order.
        """
        N_at = jnp.array([-0.572, 1.337, 0.000])
        Ca_at = jnp.array([0.000, 0.000, 0.000])
        C_at = jnp.array([1.517, -0.000, -0.000])
        O_at = jnp.array([0.626, 1.062, -0.000])

        coor = jnp.stack([N_at, Ca_at, C_at, O_at], axis=0)

        # Apply orientation transform to each atom
        def _apply_transform(x):
            """Applies the rotation 'O' matrix to a [3,] array"""
            return jnp.matmul(O, x) + t
        apply_transform_fn = jax.vmap(_apply_transform)

        return apply_transform_fn(coor)

    frames_to_coor_fn = jax.vmap(_frame_to_coor)  # accepts [N,3] translations and [N,3,3] rotations, returns [N,4,3]
    coordinates = frames_to_coor_fn(transforms.translations, transforms.orientations)
    return coordinates


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
    rigid_from_3_points_fn = jax.vmap(_single_rigid_from_3_points)
    return rigid_from_3_points_fn(N, Ca, C)


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
