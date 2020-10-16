from .hmc import hmc, prep_sfm
from dipy.data import get_fnames
from dipy.core.gradients import gradient_table
import nibabel as nib
import numpy as np
from fracridge import FracRidgeRegressor, fracridge


def test_hmc():
    fdata, fbvals, fbvecs = get_fnames("sherbrooke_3shell")
    gtab = gradient_table(fbvals, fbvecs, b0_threshold=0)
    img = nib.load(fdata)
    data = img.get_fdata()
    hmc(data, gtab, affine=img.affine)


def test_fracridge_sfm():
    fdata, fbvals, fbvecs = get_fnames("sherbrooke_3shell")
    gtab = gradient_table(fbvals, fbvecs, b0_threshold=0)
    img = nib.load(fdata)
    data = img.get_fdata()
    mask = np.zeros(data.shape[:3], dtype=bool)
    mask[40:50, 40:50, 40:50] = True
    X, y = prep_sfm(gtab, data, mask=mask)
    valid_targets = np.where(np.isfinite(y))[1]
    coef, alphas = fracridge(X, y[:, np.unique(valid_targets)], [0.3])
    # Shape is n_regressors, n_alphas, n_voxels:
    assert coef.shape == (362, 1000)
