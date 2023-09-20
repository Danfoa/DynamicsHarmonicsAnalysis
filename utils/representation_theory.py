import functools
from collections import OrderedDict

import numpy as np
import scipy
from escnn.group import Group, Representation, directsum
from morpho_symm.groups.isotypic_decomposition import cplx_isotypic_decomposition

def identify_isotypic_spaces(rep: Representation) -> [OrderedDict[tuple: Representation], np.ndarray]:
    """
    Identify the isotypic subspaces of a representation. See Isotypic Basis for more details (TODO).
    Args:
        rep (Representation): Input representation in any arbitrary basis.

    Returns: A `Representation` with a change of basis exposing an Isotypic Basis (a.k.a symmetry enabled basis).
        The instance of the representation contains an additional parameter `isotypic_subspaces` which is an
        `OrderedDict` of representations per each isotypic subspace. The keys are the active irreps' ids associated
        with each Isotypic subspace.
    """

    symm_group = rep.group
    potential_irreps = rep.group.irreps()
    isotypic_subspaces_indices = {irrep.id: [] for irrep in potential_irreps}

    for irrep in potential_irreps:
        for index, rep_irrep_id in enumerate(rep.irreps):
            if symm_group.irrep(*rep_irrep_id) == irrep:
                isotypic_subspaces_indices[rep_irrep_id].append(index)
        # If irreps of the same type are not consecutive numbers raise an error
        if not np.all(np.diff(isotypic_subspaces_indices[irrep.id]) == 1):
            raise NotImplementedError("TODO: Add permutations needed to handle this case")

    # Remove inactive Isotypic Spaces
    for irrep in potential_irreps:
        if len(isotypic_subspaces_indices[irrep.id]) == 0:
            del isotypic_subspaces_indices[irrep.id]

    # Each Isotypic Space will be indexed by the irrep it is associated with.
    active_isotypic_reps = {}
    for irrep_id, indices in isotypic_subspaces_indices.items():
        # if indices are not consecutive numbers raise an error
        if not np.all(np.diff(indices) == 1):
            raise NotImplementedError("TODO: Add permutations needed to handle this case")
        irrep = symm_group.irrep(*irrep_id)
        multiplicities = len(indices)
        active_isotypic_reps[irrep_id] = Representation(group=rep.group,
                                                        irreps=[irrep_id] * multiplicities,
                                                        name=f'IsoSubspace {irrep_id}',
                                                        change_of_basis=np.identity(irrep.size * multiplicities),
                                                        supported_nonlinearities=irrep.supported_nonlinearities
                                                        )

    # Impose canonical order on the Isotypic Subspaces.
    # If the trivial representation is active it will be the first Isotypic Subspace.
    # Then sort by dimension of the space from smallest to largest.
    ordered_isotypic_reps = OrderedDict(sorted(active_isotypic_reps.items(), key=lambda item: item[1].size))
    if symm_group.trivial_representation.id in ordered_isotypic_reps.keys():
        ordered_isotypic_reps.move_to_end(symm_group.trivial_representation.id, last=False)

    # Compute the decomposition of Real Irreps into Complex Irreps and store this information in irrep.attributes
    # cplx_irrep_i(g) = Q_re2cplx @ re_irrep_i(g) @ Q_re2cplx^-1`
    # for irrep_id in ordered_isotypic_reps.keys():
    #     re_irrep = symm_group.irrep(*irrep_id)
    #     cplx_subreps, Q_re2cplx = cplx_isotypic_decomposition(symm_group, re_irrep)
    #     re_irrep.is_cplx_irrep = len(cplx_subreps) == 1
    #     symm_group.irrep(*irrep_id).attributes['cplx_irreps'] = cplx_subreps
    #     symm_group.irrep(*irrep_id).attributes['Q_re2cplx'] = Q_re2cplx

    new_rep = directsum(list(ordered_isotypic_reps.values()),
                        name=rep.name + '-Iso',
                        change_of_basis=None)  # TODO: Check for additional permutations

    iso_supported_nonlinearities = [iso_rep.supported_nonlinearities for iso_rep in ordered_isotypic_reps.values()]
    new_rep.supported_nonlinearities = functools.reduce(set.intersection, iso_supported_nonlinearities)
    new_rep.attributes['isotypic_reps'] = ordered_isotypic_reps
    return new_rep, rep.change_of_basis


def isotypic_basis(representation: Representation, multiplicity: int = 1, prefix=''):
    rep, Q_iso = identify_isotypic_spaces(representation)

    iso_reps = OrderedDict()
    iso_range = OrderedDict()

    start_dim = 0
    for iso_irrep_id, reg_rep_iso in rep.attributes['isotypic_reps'].items():
        iso_reps[iso_irrep_id] = directsum([reg_rep_iso] * multiplicity,
                                           name=f"{prefix}_IsoSpace{iso_irrep_id}")
        iso_range[iso_irrep_id] = range(start_dim, start_dim + iso_reps[iso_irrep_id].size)
        start_dim += iso_reps[iso_irrep_id].size

    assert rep.size * multiplicity == sum([iso_rep.size for iso_rep in iso_reps.values()])

    if multiplicity > 1:
        Q_iso = None  # We need to handle this case. For now we just ignore the change of basis.

    return iso_reps, iso_range, Q_iso  # Dict[key:id_space -> value: rep_iso_space]
