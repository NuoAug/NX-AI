"""显式颜色提问的局部取景：只提供像素区域，不推断物体类别或用途。"""
import numpy as np
from PIL import Image,ImageFilter
from aieyes.vision.ocr import _connected_components

_RANGES={'红':((0,15),(240,255)),'橙':((16,25),),'黄':((25,57),),
         '绿':((58,115),),'青':((116,146),),'蓝':((147,190),),'紫':((191,239),)}

def color_box(rgb,question,proposal):
    colors=[c for c in _RANGES if c+'色' in question]
    if len(colors)!=1:
        return None
    h,w=rgb.shape[:2];scale=min(1.,640/max(h,w))
    sw=max(1,round(w*scale));sh=max(1,round(h*scale))
    hsv=np.asarray(Image.fromarray(rgb).resize((sw,sh)).convert('HSV'))
    hue=np.zeros((sh,sw),bool)
    for lo,hi in _RANGES[colors[0]]:hue|=(hsv[:,:,0]>=lo)&(hsv[:,:,0]<=hi)
    mask=hue&(hsv[:,:,1]>=180)&(hsv[:,:,2]>=60)
    mask=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3)))>0
    x1,y1,x2,y2=proposal;parts=[]
    for c in _connected_components(mask,mask.astype(np.float32)):
        if c.area<sw*sh*.003:continue
        b=(round(c.x1*w/sw),round(c.y1*h/sh),round(c.x2*w/sw),round(c.y2*h/sh))
        if min(x2,b[2])>max(x1,b[0]) and min(y2,b[3])>max(y1,b[1]):parts.append((c.area,b))
    if not parts:return None
    parts.sort(reverse=True)
    if len(parts)>1 and parts[1][0]>parts[0][0]*.6:return None
    return parts[0][1]
