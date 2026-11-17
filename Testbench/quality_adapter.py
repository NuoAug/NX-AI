"""分块计算原分辨率质量指标，避免4K整帧浮点临时数组；不缩减OCR/VLM原图。"""
import numpy as np
from aieyes.vision.quality import check_quality as project_check, QualityReport
from aieyes.vision.image_ops import ensure_rgb, to_gray_float

def check_quality(rgb,*,blur_threshold=60.,dark_threshold=40.,bright_threshold=235.,min_side=32,max_side=4096,compute_blur=True):
    arr=ensure_rgb(rgb);h,w=arr.shape[:2]
    if h*w<=1_000_000:
        return project_check(arr,blur_threshold=blur_threshold,dark_threshold=dark_threshold,
            bright_threshold=bright_threshold,min_side=min_side,max_side=max_side,compute_blur=compute_blur)
    gray_sum=lap_sum=lap_sq=0.;count=h*w
    for y in range(0,h,128):
        end=min(h,y+128);top=max(0,y-1);bottom=min(h,end+1)
        gray=to_gray_float(arr[top:bottom])
        gray_sum+=float(np.sum(gray[y-top:end-top],dtype=np.float64))
        if compute_blur:
            p=np.pad(gray,((int(y==0),int(end==h)),(1,1)),mode='edge')
            lap=p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4.*p[1:-1,1:-1]
            lap_sum+=float(np.sum(lap,dtype=np.float64));lap_sq+=float(np.sum(np.square(lap,dtype=np.float64)))
    blur=max(0.,lap_sq/count-(lap_sum/count)**2) if compute_blur else 0.
    brightness=gray_sum/count;labels=[]
    if min(h,w)<min_side:labels.append('too_small')
    if max(h,w)>max_side:labels.append('too_large')
    if compute_blur and blur<blur_threshold:labels.append('blurry')
    if brightness<dark_threshold:labels.append('dark')
    elif brightness>bright_threshold:labels.append('bright')
    return QualityReport(w,h,blur,brightness,not ('too_small' in labels or 'too_large' in labels),tuple(labels),bool(compute_blur))

def install():
    import aieyes.scene.service as service
    import aieyes.scene.providers as providers
    service.check_quality=check_quality
    providers.check_quality=check_quality
