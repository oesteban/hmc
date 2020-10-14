import numpy as np

from .registration import (affine_registration, register_series,
                           read_img_arr_or_path)
from dipy.reconst.sfm import SparseFascicleModel
from sklearn.model_selection import LeaveOneOut
import dipy.core.gradients as dpg
import collections
from fracridge import FracRidgeRegressor, fracridge
from dipy.core.optimize import SKLearnLinearSolver


def prep_sfm(gtab, data, mask=None):
    if mask is None:
        mask = np.ones(data.shape[:3])
    sfm = SparseFascicleModel(gtab)
    X = sfm.design_matrix
    b0 = np.mean(data[mask][:, gtab.b0s_mask], -1)
    dwi = data[mask][:, ~gtab.b0s_mask] / b0[np.newaxis].T
    y = dwi.T
    return X, y


def fit_sfm(clf, X, y):
    clf.fit(X, y)
    return X, y, clf


def hmc(img, gtab, mask=None, b0_ref=0):
    data, affine = read_img_arr_or_path(data, affine=affine)
    if isinstance(gtab, collections.Sequence):
        gtab = dpg.gradient_table(*gtab)

    # We fix b0 to be one volume, registered to one of the
    # b0 volumes (first, per default):
    if np.sum(gtab.b0s_mask) > 1:
        b0_img = nib.Nifti1Image(data[..., gtab.b0s_mask], affine)
        trans_b0, b0_affines = register_series(b0_img, ref=b0_ref,
                                               pipeline=pipeline)
        ref_data = np.mean(trans_b0, -1)
    else:
        # There's only one b0 and we register everything to it
        trans_b0 = ref_data = data[..., gtab.b0s_mask]
        b0_affines = np.eye(4)[..., np.newaxis]

    moving_data = data[..., ~gtab.b0s_mask]
    moving_bvals = gtab.bvals[~gtab.b0s_mask]
    moving_bvecs = gtab.bvecs[~gtab.b0s_mask]

    clf = FracRidgeRegressor()
    for loo in range(moving_data.shape[-1]):
        loo_idxer = np.ones(moving_data.shape[-1]).astype(bool)
        loo_idxer[loo] = False
        in_data = np.concatenate([ref_data, moving_data[..., loo_idxer]], -1)
        in_gtab = dpg.gradient_table(
            np.concatenate([0, moving_bvals[loo_idxer]]),
            np.concatenate([0, moving_bvecs[loo_idxer]]))
        clf = fit_sfm(in_gtab, in_data, clf)

        out_data = moving_data[..., ~loo_idxer]
        out_gtab = dpg.gradient_table(moving_bvals[~loo_idxer],
                                      moving_bvecs[~loo_idxer])
        pred_X = out_gtab
        out_pred = clf.predict()
