from .hmc import hmc, fit_sfm, prep_sfm
from dipy.data import get_fnames
from dipy.core.gradients import gradient_table
import nibabel as nib
import numpy as np
from fracridge import FracRidgeRegressor, fracridge

# def test_hmc():
#     fdata, fbvals, fbvecs = get_fnames("stanford_hardi")
#     gtab = gradient_table(fbvals, fbvecs, b0_threshold=0)
#     img = nib.load(fdata).get_fdata()
#     hmc(img, gtab)


def test_fracridge_sfm():
    fdata, fbvals, fbvecs = get_fnames("stanford_hardi")
    gtab = gradient_table(fbvals, fbvecs, b0_threshold=0)
    img = nib.load(fdata)
    data = img.get_fdata()
    mask = np.zeros(data.shape[:3], dtype=bool)
    mask[40:50, 40:50, 40:50] = True
    X, y = prep_sfm(gtab, data, mask=mask)
    clf = FracRidgeRegressor()
    X, y, clf = fit_sfm(clf, X, y)
    # Shape is n_regressors, n_alphas, n_voxels:
    assert clf.coef_.shape == (362, 10, 1000)
