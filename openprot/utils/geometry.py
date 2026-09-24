import torch
import numpy as np

from .rigid_utils import Rigid, Rotation
from . import residue_constants as rc
from .tensor_utils import batched_gather

@torch.cuda.amp.autocast(False)
def compute_rmsd(a, b, weights=None, reduce=True):
    B = a.shape[:-2]
    N = a.shape[-2]

    if weights is None:
        weights = a.new_ones(*B, N)

    b = rmsdalign(a, b, weights)
    sqdist = torch.square(a - b).sum(-1)
    if reduce:
        return torch.sqrt((sqdist * weights).sum(-1) / weights.sum(-1))
    else:
        return torch.sqrt(sqdist)

@torch.cuda.amp.autocast(False)
def compute_pseudo_tm(a, b, weights=None):
    B = a.shape[:-2]
    N = a.shape[-2]

    if weights is None:
        weights = a.new_ones(*B, N)

    b = rmsdalign(a, b, weights)
    L = weights.sum(-1)
    d0 = 1.24*(L-15)**(1/3)-1.8
    
    dis = torch.square(a - b).sum(-1).sqrt()
    tm = 1/(1+torch.square(dis/d0[...,None]))
    
    return (tm * weights).sum(-1) / weights.sum(-1)

# https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
@torch.cuda.amp.autocast(False)
def rmsdalign(
    a, b, weights=None, demean=True, a_origin=None, b_origin=None
):  # alignes B to A  # [*, N, 3]
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights is None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    if demean:
        a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
        a = a - a_mean
        b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
        b = b - b_mean
    if a_origin is not None:
        a = a - a_origin
    if b_origin is not None:
        b = b - b_origin
    B = torch.einsum("...ji,...jk->...ik", weights * a, b)
    u, s, vh = torch.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    sgn = torch.sign(torch.linalg.det((u @ vh).cpu())).to(u.device)  # ugly workaround
    s[..., -1] *= sgn
    u[..., :, -1] *= sgn.unsqueeze(-1)
    C = u @ vh  # c rotates B to A
    if demean:
        return b @ C.mT + a_mean
    elif a_origin is not None:
        return b @ C.mT + a_origin
    else:
        return b @ C.mT


def atom14_to_atom37(atom14: np.ndarray, aatype, atom14_mask=None):
    atom37 = batched_gather(
        atom14,
        rc.RESTYPE_ATOM37_TO_ATOM14[aatype],
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )
    atom37 *= rc.RESTYPE_ATOM37_MASK[aatype, :, None]
    if atom14_mask is not None:
        atom37_mask = batched_gather(
            atom14_mask,
            rc.RESTYPE_ATOM37_TO_ATOM14[aatype],
            dim=-1,
            no_batch_dims=len(atom14.shape[:-2]),
        )
        atom37_mask *= rc.RESTYPE_ATOM37_MASK[aatype]
        return atom37, atom37_mask
    else:
        return atom37


def atom37_to_atom14(atom37: np.ndarray, aatype, atom37_mask=None):
    atom14 = batched_gather(
        atom37,
        rc.RESTYPE_ATOM14_TO_ATOM37[aatype],
        dim=-2,
        no_batch_dims=len(atom37.shape[:-2]),
    )
    atom14 *= rc.RESTYPE_ATOM14_MASK[aatype, :, None]
    if atom37_mask is not None:
        atom14_mask = batched_gather(
            atom37_mask,
            rc.RESTYPE_ATOM14_TO_ATOM37[aatype],
            dim=-1,
            no_batch_dims=len(atom37.shape[:-2]),
        )
        atom14_mask *= rc.RESTYPE_ATOM14_MASK[aatype]
        return atom14, atom14_mask
    else:
        return atom14


def atom37_to_frames(atom37, atom37_mask=None):
    if type(atom37) is np.ndarray:
        atom37 = torch.from_numpy(atom37)
    n_coords = atom37[..., rc.atom_order["N"], :]
    ca_coords = atom37[..., rc.atom_order["CA"], :]
    c_coords = atom37[..., rc.atom_order["C"], :]
    rot_mats = gram_schmidt(origin=ca_coords, x_axis=c_coords, xy_plane=n_coords)
    rots = Rotation(rot_mats=rot_mats)
    if atom37_mask is not None:
        if type(atom37_mask) is np.ndarray:
            atom37_mask = torch.from_numpy(atom37_mask)
        frame_mask = torch.prod(
            atom37_mask[
                ..., [rc.atom_order["N"], rc.atom_order["CA"], rc.atom_order["C"]]
            ],
            dim=-1,
        )
        return Rigid(rots, ca_coords), frame_mask
    else:
        return Rigid(rots, ca_coords)


def frames_torsions_to_atom37(
    frames: Rigid,
    torsions: torch.Tensor,
    aatype: torch.Tensor,
):
    atom14 = frames_torsions_to_atom14(frames, torsions, aatype)
    return atom14_to_atom37(atom14, aatype)


def frames_torsions_to_atom14(
    frames: Rigid, torsions: torch.Tensor, aatype: torch.Tensor
):
    if type(torsions) is np.ndarray:
        torsions = torch.from_numpy(torsions)
    if type(aatype) is np.ndarray:
        aatype = torch.from_numpy(aatype)
    default_frames = torch.from_numpy(rc.restype_rigid_group_default_frame).to(
        aatype.device
    )
    lit_positions = torch.from_numpy(rc.restype_atom14_rigid_group_positions).to(
        aatype.device
    )
    group_idx = torch.from_numpy(rc.restype_atom14_to_rigid_group).to(aatype.device)
    atom_mask = torch.from_numpy(rc.restype_atom14_mask).to(aatype.device)
    frames_out = torsion_angles_to_frames(frames, torsions, aatype, default_frames)
    return frames_and_literature_positions_to_atom14_pos(
        frames_out, aatype, default_frames, group_idx, atom_mask, lit_positions
    )


def atom37_to_torsions(all_atom_positions, aatype, all_atom_mask=None):

    if type(all_atom_positions) is np.ndarray:
        all_atom_positions = torch.from_numpy(all_atom_positions)
    if type(aatype) is np.ndarray:
        aatype = torch.from_numpy(aatype)
    if all_atom_mask is None:
        all_atom_mask = torch.from_numpy(rc.RESTYPE_ATOM37_MASK[aatype]).to(
            aatype.device
        )
    if type(all_atom_mask) is np.ndarray:
        all_atom_mask = torch.from_numpy(all_atom_mask)

    pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 37, 3])
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3
    )

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(prev_all_atom_mask[..., 1:3], dim=-1) * torch.prod(
        all_atom_mask[..., :2], dim=-1
    )
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
        all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
    )
    psi_mask = (
        torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
        * all_atom_mask[..., 4]
    )

    chi_atom_indices = torch.as_tensor(get_chi_atom_indices(), device=aatype.device)

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2]), np=torch
    )

    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
        np=torch,
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = (
        torsion_angles_sin_cos
        * all_atom_mask.new_tensor(
            [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    )

    return torsion_angles_sin_cos, torsion_angles_mask


def compute_fape(
    pred_frames,
    target_frames,
    frames_mask,
    pred_positions,
    target_positions,
    positions_mask,
    length_scale,
    l1_clamp_distance=None,
    thresh=None,
    eps=1e-8,
) -> torch.Tensor:
    """
    Computes FAPE loss.

    Args:
        pred_frames:
            [*, N_frames] Rigid object of predicted frames
        target_frames:
            [*, N_frames] Rigid object of ground truth frames
        frames_mask:
            [*, N_frames] binary mask for the frames
        pred_positions:
            [*, N_pts, 3] predicted atom positions
        target_positions:
            [*, N_pts, 3] ground truth positions
        positions_mask:
            [*, N_pts] positions mask
        length_scale:
            Length scale by which the loss is divided
        pair_mask:
            [*,  N_frames, N_pts] mask to use for
            separating intra- from inter-chain losses.
        l1_clamp_distance:
            Cutoff above which distance errors are disregarded
        thresh:
            Loss will only be applied to pt-frame distances under this value
        eps:
            Small value used to regularize denominators
    Returns:
        [*] loss tensor
    """

    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale

    if thresh is not None:

        thresh_mask = torch.sqrt(torch.sum(local_target_pos**2, dim=-1)) < thresh
        mask = thresh_mask * frames_mask[..., None] * positions_mask[..., None, :]

        # normed_error = normed_error * mask
        # normed_error = torch.sum(normed_error, dim=(-1, -2))
        # normed_error = normed_error / (eps + torch.sum(mask, dim=(-1, -2)))

    else:
        normed_error = normed_error * frames_mask[..., None]
        normed_error = normed_error * positions_mask[..., None, :]
        mask = frames_mask[..., None] * positions_mask[..., None, :]

        # normed_error = torch.sum(normed_error, dim=-1)
        # normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
        # normed_error = torch.sum(normed_error, dim=-1)
        # normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    normed_error = (normed_error * mask).sum(-1) / (eps + mask.sum(-1))
    return normed_error


def compute_pade(pred_pos, gt_pos, gt_mask, eps=1e-6, cutoff=15, clamp=None):
    def get_dmat(pos):
        dmat = torch.square(pos[..., None, :] - pos[..., None, :, :]).sum(-1)
        return torch.sqrt(eps + dmat)

    pred_dmat = get_dmat(pred_pos)
    gt_dmat = get_dmat(gt_pos)

    dmat_mask = (gt_dmat < cutoff).float() * gt_mask[:, None] * gt_mask[:, :, None]
    dmat_mask = dmat_mask * (1.0 - torch.eye(gt_mask.shape[-1], device=gt_mask.device))

    score = torch.abs(pred_dmat - gt_dmat)
    if clamp is not None:
        score = torch.clamp(score, max=clamp)

    return (dmat_mask * score).sum(-1) / (eps + dmat_mask.sum(-1))  # B, L


def compute_lddt(
    pred_pos,
    gt_pos,
    gt_mask,
    cutoff=15.0,
    eps=1e-10,
    symmetric=False,
    pred_is_dmat=False,
    reduce=(-1, -2),
    soft=False,
):
    if pred_is_dmat:
        pred_dmat = pred_pos
    else:
        pred_dmat = torch.sqrt(
            eps
            + torch.sum(
                (pred_pos[..., None, :] - pred_pos[..., None, :, :]) ** 2, axis=-1
            )
        )
    gt_dmat = torch.sqrt(
        eps + torch.sum((gt_pos[..., None, :] - gt_pos[..., None, :, :]) ** 2, axis=-1)
    )
    if symmetric:
        dists_to_score = (pred_dmat < cutoff) | (gt_dmat < cutoff)
    else:
        dists_to_score = gt_dmat < cutoff
    dists_to_score = (
        dists_to_score
        * gt_mask.unsqueeze(-2)
        * gt_mask.unsqueeze(-1)
        * (1.0 - torch.eye(gt_mask.shape[-1], device=gt_mask.device))
    )
    dist_l1 = torch.abs(pred_dmat - gt_dmat)
    cutoffs = torch.tensor([0.5, 1.0, 2.0, 4.0], device=gt_mask.device)
    if soft:
        score = torch.sigmoid(cutoffs - dist_l1[..., None])
    else:
        score = dist_l1[..., None] < cutoffs
    score = score.float().mean(-1)

    if reduce:
        score = (dists_to_score * score).sum(reduce) / (
            eps + dists_to_score.sum(reduce)
        )

    return score


def gram_schmidt(origin, x_axis, xy_plane, eps=1e-8):
    x_axis = torch.unbind(x_axis, dim=-1)
    origin = torch.unbind(origin, dim=-1)
    xy_plane = torch.unbind(xy_plane, dim=-1)

    e0 = [c1 - c2 for c1, c2 in zip(x_axis, origin)]
    e1 = [c1 - c2 for c1, c2 in zip(xy_plane, origin)]

    denom = torch.sqrt(sum((c * c for c in e0)) + eps)
    e0 = [c / denom for c in e0]
    dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
    e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
    denom = torch.sqrt(sum((c * c for c in e1)) + eps)
    e1 = [c / denom for c in e1]
    e2 = [
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
    ]

    rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
    rots = rots.reshape(rots.shape[:-1] + (3, 3))
    return rots


def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 14, 4, 4]
    default_4x4 = default_frames[aatype, ...]

    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
):
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restype_1to3[residue_name]
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return chi_atom_indices
