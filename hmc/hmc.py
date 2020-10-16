import numpy as np

from dipy.align import (affine_registration, register_series)
from dipy.io.utils import read_img_arr_or_path
from dipy.reconst.sfm import (SparseFascicleModel, SparseFascicleFit,
                              ExponentialIsotropicModel, sfm_design_matrix)
import dipy.core.gradients as dpg
import collections
from fracridge import FracRidgeRegressor
import nibabel as nib
from sklearn.base import RegressorMixin


class SFM4HMC(SparseFascicleModel):
    """
    We need to reimplement the fit, so that we can use the FRR cleverness
    under the hood
    """
    def fit(self, data, mask=None):
        """
        Fit the SparseFascicleModel object to data.

        Parameters
        ----------
        data : array
            The measured signal.

        mask : array, optional
            A boolean array used to mark the coordinates in the data that
            should be analyzed. Has the shape `data.shape[:-1]`. Default: None,
            which implies that all points should be analyzed.

        Returns
        -------
        SparseFascicleFit object
        """
        if mask is None:
            # Flatten it to 2D either way:
            data_in_mask = np.reshape(data, (-1, data.shape[-1]))
        else:
            # Check for valid shape of the mask
            if mask.shape != data.shape[:-1]:
                raise ValueError("Mask is not the same shape as data.")
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = np.reshape(data[mask], (-1, data.shape[-1]))

        # Fitting is done on the relative signal (S/S0):
        flat_S0 = np.mean(data_in_mask[..., self.gtab.b0s_mask], -1)
        if not flat_S0.size or not flat_S0.max():
            flat_S = np.zeros(data_in_mask[..., ~self.gtab.b0s_mask].shape)
        else:
            flat_S = (data_in_mask[..., ~self.gtab.b0s_mask] /
                      flat_S0[..., None])
        isotropic = self.isotropic(self.gtab).fit(data, mask)

        # We don't need to preallocate: ##
        # flat_params = np.zeros((data_in_mask.shape[0],
        #                         self.design_matrix.shape[-1]))

        isopredict = isotropic.predict()

        if mask is None:
            isopredict = np.reshape(isopredict, (-1, isopredict.shape[-1]))
        else:
            isopredict = isopredict[mask]

        # Here's where things get different: ##
        y = (flat_S - isopredict).T
        # Making sure nan voxels get 0 params:
        nan_targets = np.unique(np.where(~np.isfinite(y))[1])
        y[:, nan_targets] = 0
        flat_params = self.solver.fit(self.design_matrix, y).coef_

        # We avoid this loop over the data: ##
        # for vox, vox_data in enumerate(flat_S):
        #     # In voxels in which S0 is 0, we just want to keep the
        #     # parameters at all-zeros, and avoid nasty sklearn errors:
        #     if not (np.any(~np.isfinite(vox_data)) or np.all(vox_data == 0)):
        #         fit_it = vox_data - isopredict[vox]
        #         with warnings.catch_warnings():
        #             warnings.simplefilter("ignore")
        #             flat_params[vox] = self.solver.fit(self.design_matrix,
        #                                                fit_it).coef_

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            beta = flat_params.reshape(out_shape)
            S0 = flat_S0.reshape(data.shape[:-1])
        else:
            beta = np.zeros(data.shape[:-1] +
                            (self.design_matrix.shape[-1],))
            beta[mask, :] = flat_params
            S0 = np.zeros(data.shape[:-1])
            S0[mask] = flat_S0

        return SparseFascicleFit(self, beta, S0, isotropic)


class FRR4SFM(FracRidgeRegressor, RegressorMixin):
    def __init__(self, fracs=None, fit_intercept=False, normalize=False,
                 copy_X=True, tol=1e-10, jit=True):
        FracRidgeRegressor.__init__(
            self, fracs=fracs, fit_intercept=False, normalize=False,
            copy_X=True, tol=tol, jit=True)


def prep_data(gtab, data, mask=None):
    if mask is None:
        mask = np.ones(data.shape[:3]).astype(bool)
    b0 = np.mean(data[mask][:, gtab.b0s_mask], -1)
    dwi = data[mask][:, ~gtab.b0s_mask] / b0[np.newaxis].T
    return dwi.T


def prep_sfm(gtab, data, mask=None):
    y = prep_data(gtab, data, mask)
    isotropic = ExponentialIsotropicModel(gtab)
    sfm = SparseFascicleModel(gtab)#, isotropic=isotropic)
    X = sfm.design_matrix
    return X, y


def hmc(data, gtab, mask=None, b0_ref=0, affine=None):
    data, affine = read_img_arr_or_path(data, affine=affine)
    if isinstance(gtab, collections.Sequence):
        gtab = dpg.gradient_table(*gtab)

    # We fix b0 to be one volume, registered to one of the
    # b0 volumes (first, per default):
    if np.sum(gtab.b0s_mask) > 1:
        b0_img = nib.Nifti1Image(data[..., gtab.b0s_mask], affine)
        trans_b0, b0_affines = register_series(b0_img, ref=b0_ref)
        ref_data = np.mean(trans_b0, -1)[..., np.newaxis]
    else:
        # There's only one b0 and we register everything to it
        trans_b0 = ref_data = data[..., gtab.b0s_mask]
        b0_affines = np.eye(4)[..., np.newaxis]

    moving_data = data[..., ~gtab.b0s_mask]
    moving_bvals = gtab.bvals[~gtab.b0s_mask]
    moving_bvecs = gtab.bvecs[~gtab.b0s_mask]

    clf = FRR4SFM(fracs=0.5)
    for loo in range(moving_data.shape[-1]):
        loo_idxer = np.ones(moving_data.shape[-1]).astype(bool)
        loo_idxer[loo] = False

        in_data = np.concatenate([ref_data, moving_data[..., loo_idxer]], -1)
        in_gtab = dpg.gradient_table(
            np.concatenate([np.array([0]), moving_bvals[loo_idxer]]),
            np.concatenate([np.array([[0, 0, 0]]), moving_bvecs[loo_idxer]]))

        in_sfm = SFM4HMC(
            in_gtab,
            #isotropic=ExponentialIsotropicModel,
            solver=clf)

        in_sff = in_sfm.fit(in_data)#, mask=np.ones(in_data.shape[:3]))

        # out_data = moving_data[..., ~loo_idxer]
        out_gtab = dpg.gradient_table(moving_bvals[~loo_idxer],
                                      moving_bvecs[~loo_idxer])
        out_pred = in_sff.predict(out_gtab, S0=ref_data[..., 0])
        nib.save(nib.Nifti1Image(out_pred, affine),
                 '/Users/arokem/tmp/target.nii.gz')
        1/0.
