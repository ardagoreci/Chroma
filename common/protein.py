"""Protein data type."""

import collections
import dataclasses
import io
from common import residue_constants
from typing import Any, Dict, List, Mapping, Optional, Tuple
from Bio.PDB import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
import numpy as np

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.

# Data to fill the _chem_comp table when writing mmCIFs.
_CHEM_COMP: Mapping[str, Tuple[Tuple[str, str], ...]] = {
    'L-peptide linking': (
        ('ALA', 'ALANINE'),
        ('ARG', 'ARGININE'),
        ('ASN', 'ASPARAGINE'),
        ('ASP', 'ASPARTIC ACID'),
        ('CYS', 'CYSTEINE'),
        ('GLN', 'GLUTAMINE'),
        ('GLU', 'GLUTAMIC ACID'),
        ('HIS', 'HISTIDINE'),
        ('ILE', 'ISOLEUCINE'),
        ('LEU', 'LEUCINE'),
        ('LYS', 'LYSINE'),
        ('MET', 'METHIONINE'),
        ('PHE', 'PHENYLALANINE'),
        ('PRO', 'PROLINE'),
        ('SER', 'SERINE'),
        ('THR', 'THREONINE'),
        ('TRP', 'TRYPTOPHAN'),
        ('TYR', 'TYROSINE'),
        ('VAL', 'VALINE'),
    ),
    'peptide linking': (('GLY', 'GLYCINE'),),
}


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f'Cannot build an instance with more than {PDB_MAX_CHAINS} chains '
                'because these cannot be written to PDB format.')


def _from_bio_structure(structure: Structure,
                        chain_id: Optional[str] = None) -> Protein:
    """Takes a Biopython structure and creates a `Protein` instance.

      WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

      Args:
        structure: Structure from the Biopython library.
        chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
          Otherwise all chains are parsed.

      Returns:
        A new `Protein` created from the structure contents.

      Raises:
        ValueError: If the number of models included in the structure is not 1.
        ValueError: If insertion code is detected at a residue.
    """
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            'Only single model PDBs/mmCIFs are supported. Found'
            f' {len(models)} models.'
        )
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != ' ':
                raise ValueError(
                    f'PDB/mmCIF contains an insertion code at chain {chain.id} and'
                    f' residue index {res.id[1]}. These are not supported.'
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num)
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors))


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a `Protein` object.

    WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.
    Args:
        pdb_str: The contents of the pdb file
        chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
        Otherwise, all chains are parsed.

    Returns:
        A new `Protein` parsed from the pdb contents.
    """
    with io.StringIO(pdb_str) as pdb_fh:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(id='none', file=pdb_fh)
        return _from_bio_structure(structure, chain_id)


def from_mmcif_string(
        mmcif_str: str, chain_id: Optional[str] = None
) -> Protein:
    """Takes a mmCIF string and constructs a `Protein` object.

    WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.

    Args:
        mmcif_str: The contents of the mmCIF file
        chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
        Otherwise, all chains are parsed.

    Returns:
        A new `Protein` parsed from the mmCIF contents.
  """
    with io.StringIO(mmcif_str) as mmcif_fh:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(structure_id='none', filename=mmcif_fh)
        return _from_bio_structure(structure, chain_id)


def from_dict(
        prot_dict: dict
) -> Protein:
    """Takes a dict with the corresponding keys and constructs a 'Protein' object.
    Args:
        prot_dict: Dict containing the entries for the Protein dataclass.
    Returns:
        A new 'Protein' constructed from the dict contents.
    """
    assert 'atom_positions' in prot_dict.keys(), "prot_dict needs to have key 'atom_positions'"
    assert 'aatype' in prot_dict.keys(), "prot_dict needs to have key 'aatype'"
    assert 'residue_index' in prot_dict.keys(), "prot_dict needs to have key 'residue_index'"
    assert 'atom_mask' in prot_dict.keys(), "prot_dict needs to have key 'atom_mask'"
    assert 'chain_index' in prot_dict.keys(), "prot_dict needs to have key 'chain_index'"
    assert 'b_factors' in prot_dict.keys(), "prot_dict needs to have key 'b_factors'"
    return Protein(atom_positions=prot_dict['atom_positions'],
                   aatype=prot_dict['aatype'],
                   residue_index=prot_dict['residue_index'],
                   atom_mask=prot_dict['atom_mask'],
                   chain_index=prot_dict['chain_index'],
                   b_factors=prot_dict['b_factors'])


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = 'TER'
    return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
            f'{chain_name:>1}{residue_index:>4}')


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
        prot: The protein to convert to PDB.

    Returns:
        PDB string.
    """
    restypes = residue_constants.restypes + ['X']
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError('Invalid aatypes.')

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append('MODEL     1')
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(_chain_end(
                atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]],
                residue_index[i - 1]))
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
                atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            record_type = 'ATOM'
            name = atom_name if len(atom_name) == 4 else f' {atom_name}'
            alt_loc = ''
            insertion_code = ''
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ''
            # PDB is a columnar format, every space matters here!
            atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                         f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                         f'{residue_index[i]:>4}{insertion_code:>1}   '
                         f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                         f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                         f'{element:>2}{charge:>2}')
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                                chain_ids[chain_index[-1]], residue_index[-1]))
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.


def to_dict(
        protein: Protein
) -> dict:
    """Takes a dict with the corresponding keys and constructs a 'Protein' object.
    Args:
        protein: Protein object describing the protein
    Returns:
        A new dict with the keys describing the Protein
    """
    return {'atom_positions': protein.atom_positions,
            'aatype': protein.aatype,
            'residue_index': protein.residue_index,
            'atom_mask': protein.atom_mask,
            'chain_index': protein.chain_index,
            'b_factors': protein.b_factors}