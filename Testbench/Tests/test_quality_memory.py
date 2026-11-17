import numpy as np
import pytest
from quality_adapter import check_quality
from aieyes.vision.quality import check_quality as original

@pytest.mark.parametrize('kind',['flat','edge','noise'])
def test_tiled_quality_matches_full_resolution(kind):
    if kind=='flat':im=np.full((1101,1303,3),127,np.uint8)
    elif kind=='edge':
        im=np.zeros((1101,1303,3),np.uint8);im[127:257,501:]=255
    else:im=np.random.default_rng(43).integers(0,256,(1101,1303,3),dtype=np.uint8)
    a=original(im);b=check_quality(im)
    assert (a.width,a.height,a.labels,a.size_ok)==(b.width,b.height,b.labels,b.size_ok)
    assert b.blur_score==pytest.approx(a.blur_score,rel=2e-6,abs=.001)
    assert b.brightness==pytest.approx(a.brightness,rel=2e-6,abs=.001)

def test_no_blur_mode_is_preserved():
    im=np.full((1101,1303,3),250,np.uint8)
    a=original(im,compute_blur=False);b=check_quality(im,compute_blur=False)
    assert a.labels==b.labels and not b.blur_checked and b.blur_score==0
