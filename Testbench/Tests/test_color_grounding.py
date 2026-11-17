import numpy as np
from color_grounding import color_box

def test_color_refines_proposal_without_classifying_object():
    image=np.full((120,200,3),180,np.uint8)
    image[20:100,70:110]=(255,210,0)
    image[15:100,150:190]=(0,20,255)
    yellow=color_box(image,'黄色设备是什么？',(80,10,140,110))
    assert yellow is not None and yellow[0]<=70 and yellow[2]>=109
    assert color_box(image,'设备是什么？',(80,10,140,110)) is None
    assert color_box(image,'蓝色设备是什么？',(80,10,140,110)) is None

def test_similar_color_candidates_stay_ambiguous():
    image=np.full((100,200,3),180,np.uint8)
    image[15:85,25:65]=(255,210,0);image[15:85,125:165]=(255,210,0)
    assert color_box(image,'黄色设备是什么？',(0,0,200,100)) is None
