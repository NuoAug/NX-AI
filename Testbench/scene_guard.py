"""测试台找物候选局部复核；不修改正式项目的场景实现。"""
from dataclasses import replace
import threading
import ctypes
import sys
import json
import re
from aieyes.scene.service import SceneService as ProjectSceneService, _candidates_text


def confirmed_result(text, target):
    """目标不出现在图像复核提示中，只接受独立辨认出的同名实体。"""
    try:
        names=json.loads(text.strip())
    except (ValueError,TypeError,AttributeError):
        return False
    if not isinstance(names,list) or len(names)!=1 or not isinstance(names[0],str):
        return False
    name=names[0].strip().casefold()
    return bool(name) and name==target.strip().casefold()


#: 警示类文字关键词；安全提问与局部红色标识复核共用，输出仍以 OCR 原文为准
_WARNING_TEXT_RE=re.compile(r'危险|禁止|当心|小心|警告|严禁|注意|请勿|止步|高压|施工')
#: 警示文字形状过滤：中文/数字/顿号逗号/感叹号（"危险！"类标牌），防低分噪声混入安全播报
_WARNING_SHAPE_RE=re.compile(r'[\u4e00-\u9fff0-9，、！!]{2,16}')
#: 安全类整景提问：describe 不再跳过 OCR，并追加可见警示文字
_SAFETY_RE=re.compile(r'安全|危险')
#: 读字问题中的警示/彩色文字限定：允许做红色标识局部复核
_WARNING_QUERY_RE=re.compile(r'警示|警告|标语|标牌|告示|危险')
_COLOR_TEXT_RE=re.compile(r'[红橙黄绿青蓝紫粉棕灰黑白银金]色.{0,4}(?:字|文字|牌子)')
#: 实为读字的提问被项目分类误判为 describe 时改走 read（如"告示写了什么""牌上有哪些电话"）
_READ_FIX_RE=re.compile(r'写了什么|写着什么|写了哪些|写着哪些|写了些|写了啥|写着哪|有哪些电话|有哪些数字|有哪些号码|有哪些文字')
#: 提到店面/招牌的整景描述：不跳 OCR，把可见招牌原文补进回答，店名不让 VLM 猜
_SIGNBOARD_RE=re.compile(r'店面|招牌|门店|店名')
#: 危险类找物目标：抽象"危险"不是可找的物体，改走 OCR 警示证据，读不到就如实说明
_DANGER_FIND_RE=re.compile(r'^(?:有)?危险(?:的东西|物|源|隐患)?$|^警示$')
#: 电话/号码/门牌类读字问题：整帧 det 缩放压低了小字牌的分辨率，按文字簇裁剪二次 OCR
_PHONE_BOARD_RE=re.compile(r'电话|号码|门牌')


def _standalone_pos(text,sub):
    """sub 在 text 中"独立成行"出现的首个位置（前后不是汉字/数字）。
    只认独立出现，避免复扫合并后的长行（如"集美学村派出所"）被其子串
    （如低置信原文"学村派出"）的复核替换破坏。找不到返回 -1。"""
    if not sub:return -1
    for m in re.finditer(re.escape(sub),text):
        s,e=m.start(),m.end()
        if (s==0 or not re.match(r'[\u4e00-\u9fff0-9]',text[s-1])) and \
                (e==len(text) or not re.match(r'[\u4e00-\u9fff0-9]',text[e])):
            return s
    return -1


def _red_region_boxes(rgb,limit=None):
    """按 HSV 蒙版找醒目红色区域候选框；只提供像素框，不判断内容。"""
    import numpy as np
    from PIL import Image,ImageFilter
    from aieyes.vision.ocr import _connected_components
    h,w=rgb.shape[:2];scale=min(1.,1280/max(h,w))
    sw=max(1,round(w*scale));sh=max(1,round(h*scale))
    hsv=np.asarray(Image.fromarray(rgb).resize((sw,sh)).convert('HSV'))
    hue=np.zeros((sh,sw),bool)
    for lo,hi in ((0,15),(240,255)):hue|=(hsv[:,:,0]>=lo)&(hsv[:,:,0]<=hi)
    # 暗红警示字饱和度可能只有 150 左右；阈值过高会把“高压”这类窄笔画漏掉
    mask=hue&(hsv[:,:,1]>=150)&(hsv[:,:,2]>=60)
    mask=np.asarray(Image.fromarray(mask.astype(np.uint8)*255)
        .filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7)))>0
    total=sw*sh;parts=[]
    for c in _connected_components(mask,mask.astype(np.float32)):
        # 文字尺寸组件：太小是噪声，太大是墙面/大物体；两者都不值得逐块 OCR
        if not total*.0001<=c.area<=total*.03:continue
        box=(round(c.x1*w/sw),round(c.y1*h/sh),round(c.x2*w/sw),round(c.y2*h/sh))
        bw,bh=box[2]-box[0],box[3]-box[1]
        # 排除细长红色立柱/横条：文字块高度不超过 3 倍宽度，宽度不超过 8 倍高度
        if bw>=8 and bh>=8 and bh<=3*bw and bw<=8*bh:parts.append((c.area,box))
    parts.sort()
    boxes=[box for _,box in parts]
    return boxes[:limit] if limit else boxes


def _merge_row_boxes(boxes,gap_ratio=.8):
    """把同一行内水平相邻的字框并成一行（如“高”“压”），不跨行合并。"""
    out=[list(b) for b in boxes];merged=True
    while merged:
        merged=False
        for i in range(len(out)):
            for j in range(i+1,len(out)):
                a,b=out[i],out[j]
                overlap_y=min(a[3],b[3])-max(a[1],b[1])
                if overlap_y<=.5*min(a[3]-a[1],b[3]-b[1]):continue
                gap=max(a[0],b[0])-min(a[2],b[2])
                if gap<gap_ratio*max(a[3]-a[1],b[3]-b[1]):
                    out[i]=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
                    out.pop(j);merged=True;break
            if merged:break
    return [tuple(b) for b in out]


def _merge_column_boxes(boxes,max_lines=3):
    """把同列垂直相邻的框合并成整块标识区域（竖排警示字上下分行时）。
    最多合并 max_lines 行，避免沿红色立柱一路向上串成整面墙。"""
    out=sorted([[b,1] for b in boxes],key=lambda item:item[0][1],reverse=True)
    merged=True
    while merged:
        merged=False
        for i in range(len(out)):
            for j in range(i+1,len(out)):
                a,b=out[i][0],out[j][0]
                overlap=min(a[2],b[2])-max(a[0],b[0])
                gap=max(a[1],b[1])-min(a[3],b[3])
                if (out[i][1]+out[j][1]<=max_lines and overlap>.5*min(a[2]-a[0],b[2]-b[0])
                        and gap<1.2*max(a[3]-a[1],b[3]-b[1])):
                    out[i]=[[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])],
                            out[i][1]+out[j][1]]
                    out.pop(j);merged=True;break
            if merged:break
    return [tuple(item[0]) for item in out]


def _release_unused_pages():
    # 图像临时缓冲区已释放引用后，归还 glibc 保留的空闲页；不移动或清理在用对象。
    if sys.platform.startswith('linux'):
        trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
        if trim is not None:
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)

class SceneService(ProjectSceneService):
    def __init__(self,*args,**kwargs):
        from quality_adapter import install
        install()
        super().__init__(*args,**kwargs)
        self._request_state=threading.local()

    def answer(self,question,**kwargs):
        from aieyes.scene.service import classify_intent
        if kwargs.get('intent_hint')=='describe' and classify_intent(question)=='read':
            kwargs['intent_hint']='read'
        previous=getattr(self._request_state,'skip_ocr',False)
        routed=kwargs.get('intent_hint') or classify_intent(question)
        # 实为读字的提问（"写了什么/有哪些电话"）不能停在实体列举，改走 read 取 OCR 原文
        if routed=='describe' and (_READ_FIX_RE.search(question)
                or re.search(r'警示|警告|标语|告示', question)):
            routed='read';kwargs['intent_hint']='read'
        # 安全类与店面/招牌类整景提问不跳过 OCR：可见警示与招牌文字不能漏报
        self._request_state.skip_ocr=(routed=='describe' and not _SAFETY_RE.search(question)
            and not _SIGNBOARD_RE.search(question))
        # 电话/号码/门牌类问题由聚簇复扫重读牌面（分辨率更高），原始低分行的小裁剪
        # VLM 复核只会重复复扫已覆盖的区域且多被否决，跳过以省串行 VLM 调用
        self._request_state.review_ocr=routed=='read' and not _PHONE_BOARD_RE.search(question)
        self._request_state.ocr_reviews=[]
        try:
            result=super().answer(question,**kwargs)
            text=result.text
            reviews=getattr(self._request_state,'ocr_reviews',())
            if result.intent=='read' and reviews:
                from aieyes.scene.models import SourceTag
                tags=list(result.source_tags);notes=list(result.notes);uncertain=False
                for line,verified in reviews:
                    # 复核基于复扫前的原始 OCR 行；被聚簇复扫/配对合并进更长行后，
                    # 原文只剩子串（如"学村派出"在"集美学村派出所"内），不能按行替换
                    pos=_standalone_pos(text,line.text)
                    if pos<0:continue
                    notes.append('低置信OCR原文：'+line.text)
                    if verified is None:
                        text=text[:pos]+'有一行小字我还没看清'+text[pos+len(line.text):];uncertain=True
                    else:
                        text=text[:pos]+verified+text[pos+len(line.text):]
                        tags.append(SourceTag(kind='vlm',frame_id=result.frame_id,raw_text=verified,field='ocr_review'))
                        notes.append('局部两种尺寸读字一致：'+verified)
                result=replace(result,source_tags=tuple(tags),notes=tuple(notes),
                    degraded=tuple(dict.fromkeys((*result.degraded,*(['ocr_uncertain'] if uncertain else [])))),
                    confidence=min(result.confidence,.5) if uncertain else result.confidence)
            if result.intent=='describe' and result.vlm_text:
                text=re.sub(r'^(?:[红橙黄绿青蓝紫粉棕灰黑白银金]{1,2}色)?设备(?:就是|是)','这台是',text).replace('用户可以','你可以')
                if text=='没有看到这个目标。':text='我还没看到你说的这个目标。'
                elif re.fullmatch(r'有\d+[个只件辆台张把人具处]。?',text):text='我看到'+text
                elif re.fullmatch(r'(?:红|橙|黄|绿|青|蓝|紫|粉|棕|灰|黑|白|银|金)色。?',text):text='我看到是'+text.rstrip('。')+'的。'
                elif not re.match(r'我|你|这|它|未|没|暂时|无法|请|不|看不清',text):text='我看到的是'+text.rstrip('。')+'。'
            text=text.replace('包装上写的价格是','我看到标价是').replace('包装上写的生产日期是','我看到生产日期是').replace('画面上写着：','我看到上面写着：')
            return replace(result,text=text)
        finally:
            self._request_state.skip_ocr=previous
            self._request_state.review_ocr=False
            self._request_state.ocr_reviews=[]
            _release_unused_pages()

    def _board_rescan(self, frame, lines):
        """电话/号码/门牌类读字：整帧 OCR 检出的文字按空间聚簇，含数字行的簇按
        原分辨率裁剪后二次 OCR。小字牌在整帧 det 缩放下丢失细节（如"医务室"标签、
        标题误读），聚簇裁剪能读回整帧漏读的行并修正高分行。合并规则从紧：
        替换须高于原行 0.05 分，全新行须≥0.8 且不与已有行重叠，防止把噪声带进回答。"""
        if not any(re.search(r'\d{2,}', line.text) for line in lines):
            return lines
        parent = list(range(len(lines)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i

        centers = [(l.box[0] + l.box[2] / 2, l.box[1] + l.box[3] / 2) for l in lines]
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if abs(centers[i][0] - centers[j][0]) < 260 and abs(centers[i][1] - centers[j][1]) < 260:
                    parent[find(i)] = find(j)
        groups = {}
        for i, line in enumerate(lines):
            groups.setdefault(find(i), []).append(line)
        from aieyes.interfaces import OcrLine
        rescans = []
        for group in groups.values():
            if not any(re.search(r'\d{2,}', l.text) for l in group):
                continue
            xs1 = min(l.box[0] for l in group); ys1 = min(l.box[1] for l in group)
            xs2 = max(l.box[0] + l.box[2] for l in group)
            ys2 = max(l.box[1] + l.box[3] for l in group)
            px = max(12, round((xs2 - xs1) * .3)); py = max(12, round((ys2 - ys1) * .3))
            X1 = max(0, xs1 - px); Y1 = max(0, ys1 - py)
            X2 = min(frame.width, xs2 + px); Y2 = min(frame.height, ys2 + py)
            crop = frame.rgb[Y1:Y2, X1:X2]
            if not crop.size:
                continue
            try:
                result = self.ocr.recognize(crop, source_frame_id=frame.frame_id)
            except Exception:
                continue
            for line in result.lines:
                x, y, w, h = line.box
                rescans.append(OcrLine(line.text, (X1 + x, Y1 + y, w, h), line.score))
        if not rescans:
            return lines

        def iou(a, b):
            ax, ay, aw, ah = a.box; bx, by, bw, bh = b.box
            area = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
            union = aw * ah + bw * bh - area
            return area / union if union else 0

        kept = []
        used = set()
        for line in lines:
            best = None
            for index, other in enumerate(rescans):
                # 整帧 det 可能把"标签+号码"合并成一个宽框（如 集美学村派出所7797110），
                # 与复扫细分行 IoU 约 0.23，阈值 0.2 才能替换回精确框
                if iou(line, other) >= .2 and (best is None or other.score > rescans[best].score):
                    best = index
            if best is not None and rescans[best].score > line.score + .05:
                kept.append(rescans[best]); used.add(best)
            else:
                kept.append(line)
        for index, other in enumerate(rescans):
            if (index not in used and other.score >= .8
                    and all(iou(other, line) < .1 for line in kept)):
                kept.append(other)
        return kept

    def _pair_phone_units(self, question, kept, frame):
        """电话/号码类读字：把每个号码行与上方最近的中文单位行合并为"单位 号码"，
        让号码归属可见。合并行再用 VLM 两尺度复核：只有两尺度读法一致、数字与 OCR
        完全一致且中文部分与 OCR 高度相似时才采纳 VLM 的中文读法（修正个别形近字），
        否则保留 OCR 原文，不猜、不改数字。"""
        if not re.search(r'电话|号码', question):
            return kept
        from aieyes.interfaces import OcrLine
        numbers = [l for l in kept if re.fullmatch(r'\d{3,}', l.text.strip())]
        if not numbers:
            return kept
        labels = [l for l in kept if re.fullmatch(r'[\u4e00-\u9fff]{2,12}', l.text.strip())]
        pairs = []
        used_labels = set()
        for num in numbers:
            best = None
            for lab in labels:
                if id(lab) in used_labels:
                    continue
                lx = lab.box[0] + lab.box[2] / 2
                if not (num.box[0] - num.box[2] * .5 < lx < num.box[0] + num.box[2] * 1.5):
                    continue
                gap = num.box[1] - (lab.box[1] + lab.box[3])
                if gap < -num.box[3] * .2 or gap > num.box[3] * 2.5:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, lab)
            if best is not None:
                used_labels.add(id(best[1]))
                pairs.append((num, best[1]))
        if not pairs:
            return kept
        from difflib import SequenceMatcher
        from PIL import Image
        import numpy as np
        verified = {}
        for num, lab in pairs[:3]:
            x1 = min(num.box[0], lab.box[0]); y1 = min(num.box[1], lab.box[1])
            x2 = max(num.box[0] + num.box[2], lab.box[0] + lab.box[2])
            y2 = max(num.box[1] + num.box[3], lab.box[1] + lab.box[3])
            crop = frame.rgb[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4]
            if not crop.size:
                continue
            readings = []
            try:
                for scale in (1, 2):
                    im = crop if scale == 1 else np.array(
                        Image.fromarray(crop).resize((crop.shape[1] * 2, crop.shape[0] * 2)))
                    out = self.vlm.complete(
                        '逐字读取这块区域里的文字，按从左到右、从上到下的顺序。'
                        '只输出实际看清的文字，不按常见名称补全；辨认不了写无法确认。',
                        images=[im], max_tokens=64, temperature=0, timeout_s=15)
                    readings.append(re.sub(r'\s+', ' ', out).strip('。'))
                    if scale == 1 and (
                            ''.join(re.findall(r'[\u4e00-\u9fff]+', readings[0])) == lab.text.strip()
                            and ''.join(re.findall(r'\d+', readings[0])) == num.text.strip()):
                        # 1x 与 OCR 完全一致：没有修正需求，不必再串行调 2x 复核
                        break
            except Exception:
                readings = []
            if len(readings) == 2:
                # 数字两尺度都必须与 OCR 完全一致；中文部分只允许"单字替换/增删一字"
                # 级别的修正：长度差≤1 且互似≥0.8，采纳的 2x 读法还须与 OCR ≥0.75 相似。
                # 1x 小图固有形近字误读（学→校）由此放行高分辨率读法；
                # 整词截断（校园应急电话→校园应急）与乱读（医师室）仍被挡住，不猜不编。
                cjks = []
                for text in readings:
                    cjks.append(''.join(re.findall(r'[\u4e00-\u9fff]+', text)))
                digits = [''.join(re.findall(r'\d+', text)) for text in readings]
                if (all(d == num.text.strip() for d in digits) and cjks[1]
                        and abs(len(cjks[0]) - len(cjks[1])) <= 1
                        and SequenceMatcher(None, cjks[0], cjks[1]).ratio() >= .8
                        and SequenceMatcher(None, cjks[1], lab.text.strip()).ratio() >= .75):
                    verified[id(num)] = f'{cjks[1]} {num.text.strip()}'
        pair_nums = {id(num) for num, _ in pairs}
        pair_labels = {id(lab) for _, lab in pairs}
        merged = []
        for line in kept:
            if id(line) in pair_nums:
                lab_text = next(lab.text.strip() for num, lab in pairs if id(num) == id(line))
                text = verified.get(id(line), f'{lab_text} {line.text.strip()}')
                num, lab = next((num, lab) for num, lab in pairs if id(num) == id(line))
                x1 = min(num.box[0], lab.box[0]); y1 = min(num.box[1], lab.box[1])
                box = (x1, y1, max(num.box[0] + num.box[2], lab.box[0] + lab.box[2]) - x1,
                       max(num.box[1] + num.box[3], lab.box[1] + lab.box[3]) - y1)
                merged.append(OcrLine(text, box, min(num.score, lab.score)))
            elif id(line) not in pair_labels:
                merged.append(line)
        return merged

    def _answer_read(self, question, ctx):
        if ctx.ocr is not None:
            lines = list(ctx.ocr.lines)
            corner = re.search(r'(左上|右上|左下|右下)', question)
            if corner:
                width, height = ctx.ocr.image_size
                name = corner.group(1)
                lines = [line for line in lines if
                    (line.box[0] + line.box[2] / 2 < width / 3 if name.startswith('左')
                     else line.box[0] + line.box[2] / 2 > width * 2 / 3) and
                    (line.box[1] + line.box[3] / 2 < height / 3 if name.endswith('上')
                     else line.box[1] + line.box[3] / 2 > height * 2 / 3)]
                if not lines:
                    return ctx.finish(f'{name}角我没读到清楚的字。', 'read', confidence=.1,
                        extra_degraded=('no_text_in_region',))
            if _PHONE_BOARD_RE.search(question):
                lines = self._board_rescan(ctx.frame, lines)
            def overlap(a, b):
                ax, ay, aw, ah = a.box; bx, by, bw, bh = b.box
                area = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
                union = aw * ah + bw * bh - area
                return area / union if union else 0
            kept = []
            for line in sorted(lines, key=lambda item: item.score, reverse=True):
                # 低分行与低分单字多为检测噪声（如天空误检出的"FKEN"），不能当读字证据播报
                if line.score < .6 or (len(line.text.strip()) < 2 and line.score < .85):
                    continue
                if all(overlap(line, other) < .5 for other in kept):
                    kept.append(line)
            # 警示/彩色文字定向复核：整帧没读到警示词时，对红色标识区域局部放大 OCR。
            if _WARNING_QUERY_RE.search(question) or _COLOR_TEXT_RE.search(question):
                extra = self._warning_lines(ctx.frame)
                texts = [line.text for line in kept]
                kept = kept + [line for line in extra
                    if line.text not in texts
                    and not any(line.text != other.text and line.text in other.text
                                for other in extra)]
                # A partial full-frame line such as “危险” must not prevent the
                # local scan from recovering the rest of the same warning.
                kept = [line for line in kept if not any(
                    line.text != other.text and line.text in other.text
                    for other in extra)]
            kept = self._pair_phone_units(question, kept, ctx.frame)
            ctx.ocr = replace(ctx.ocr, lines=tuple(sorted(kept, key=lambda item: (item.box[1], item.box[0]))))
        elif _WARNING_QUERY_RE.search(question) or _COLOR_TEXT_RE.search(question):
            # 整帧 OCR 失败时警示问题仍可从局部标识取到证据
            from aieyes.interfaces import OcrResult
            extra = self._warning_lines(ctx.frame)
            if extra:
                ctx.ocr = OcrResult(lines=tuple(extra), source_frame_id=ctx.frame.frame_id,
                    image_size=(ctx.frame.width, ctx.frame.height))
        return super()._answer_read(question, ctx)

    def compare(self,*args,**kwargs):
        try:
            return super().compare(*args,**kwargs)
        finally:
            _release_unused_pages()

    def _run_ocr(self,frame):
        if getattr(self._request_state,'skip_ocr',False):
            from aieyes.interfaces import OcrResult
            return OcrResult(source_frame_id=frame.frame_id,image_size=(frame.width,frame.height))
        raw=super()._run_ocr(frame)
        if getattr(self._request_state,'review_ocr',False):
            self._request_state.ocr_reviews=self._review_read_lines(frame,raw)
        return raw

    def _review_read_lines(self,frame,raw):
        from difflib import SequenceMatcher
        from PIL import Image
        import numpy as np
        reviews=[];attempted=0
        for line in raw.lines:
            if line.score>=.85 or not re.fullmatch(r'[\u4e00-\u9fff]{4,64}',line.text):continue
            if attempted>=2:
                reviews.append((line,None));continue
            attempted+=1
            x,y,w,h=line.box;crop=frame.rgb[max(0,y-4):y+h+4,max(0,x-4):x+w+4]
            if not crop.size:continue
            candidates=[]
            try:
                for scale in (1,2):
                    im=crop if scale==1 else np.array(Image.fromarray(crop).resize((crop.shape[1]*2,crop.shape[0]*2)))
                    candidate=self.vlm.complete('逐字读取这块区域里的中文，按从左到右、从上到下的顺序。只输出实际看清的文字，不按常见名称补全；辨认不了写无法确认。',images=[im],max_tokens=64,temperature=0,timeout_s=15)
                    candidate=re.sub(r'\s+','',candidate).strip('。')
                    if scale==1 and candidate==line.text:break
                    if not re.fullmatch(r'[\u4e00-\u9fff]{4,64}',candidate) or candidate=='无法确认' or SequenceMatcher(None,line.text,candidate).ratio()<.75:
                        candidates=[];break
                    candidates.append(candidate)
                if not candidates and candidate==line.text:continue
                verified=candidates[0] if len(candidates)==2 and candidates[0]==candidates[1] else None
            except Exception:verified=None
            reviews.append((line,verified))
        return reviews

    def _warning_lines(self, frame):
        """对红色标识候选区域做局部 OCR；只保留高分中文且含警示词的行。
        垂直相邻行合并成整句（竖排"高压/危险"→"高压危险"），重复文本只留一处。
        返回行框已映射回整帧坐标；无证据时返回空列表，不编造文字。"""
        from PIL import Image
        import numpy as np
        from aieyes.interfaces import OcrLine
        # 先合并同一标识的上下行，再按面积取最大的几块；小块文字放大后再 OCR
        boxes = sorted(_merge_column_boxes(_merge_row_boxes(_red_region_boxes(frame.rgb))),
                       key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:6]
        picked = []
        for x1, y1, x2, y2 in boxes:
            px = max(4, round((x2 - x1) * .15)); py = max(4, round((y2 - y1) * .15))
            X1 = max(0, x1 - px); Y1 = max(0, y1 - py)
            X2 = min(frame.width, x2 + px); Y2 = min(frame.height, y2 + py)
            crop0 = frame.rgb[Y1:Y2, X1:X2]
            if not crop0.size:
                continue
            small = min(x2 - x1, y2 - y1) < 240
            box_got = False
            for up in (1, 2):
                crop = crop0 if up == 1 else np.asarray(
                    Image.fromarray(crop0).resize((crop0.shape[1] * 2, crop0.shape[0] * 2)))
                try:
                    result = self.ocr.recognize(crop, source_frame_id=frame.frame_id)
                except Exception:
                    break
                for line in result.lines:
                    text = line.text.strip()
                    if line.score < .8 or not _WARNING_SHAPE_RE.fullmatch(text):
                        continue
                    ox, oy, ow, oh = line.box
                    picked.append(OcrLine(text,
                        (X1 + round(ox / up), Y1 + round(oy / up), round(ow / up), round(oh / up)),
                        line.score))
                    box_got = True
                # 小字号在 1x 容易把“高压”读成低分错字；两块比例都试，
                # 大字号若 1x 已有高分警示行则不再放大，避免放大丢行。
                if box_got and not small:
                    break
        # 1x/2x 两轮可能读到同一行（如“危险”），先按文本保留高分一条再合并竖排
        best = {}
        for line in picked:
            previous = best.get(line.text)
            if previous is None or line.score > previous.score:
                best[line.text] = line
        picked = list(best.values())
        merged = []
        for line in sorted(picked, key=lambda item: (item.box[0], item.box[1])):
            for index, (text, box, score) in enumerate(merged):
                px, py, pw, ph = box; x, y, w, h = line.box
                if (min(px + pw, x + w) - max(px, x) > .5 * min(pw, w)
                        and 0 <= y - (py + ph) < 1.2 * max(h, ph)):
                    merged[index] = (text + line.text,
                        (min(px, x), min(py, y), max(px + pw, x + w) - min(px, x), max(py + ph, y + h) - min(py, y)),
                        min(score, line.score))
                    break
            else:
                merged.append((line.text, line.box, line.score))
        out = []
        seen = set()
        for text, box, score in merged:
            if not _WARNING_TEXT_RE.search(text) or text in seen:
                continue
            seen.add(text)
            out.append(OcrLine(text, box, score))
        return out

    def _append_safety_warnings(self, ctx, answer):
        """安全类描述回答补充画面中可见的警示文字；证据仍来自 OCR 原文。"""
        from aieyes.scene.models import SourceTag
        lines = []
        if ctx.ocr is not None:
            for line in ctx.ocr.lines:
                text = line.text.strip()
                if (_WARNING_TEXT_RE.search(text)
                        and _WARNING_SHAPE_RE.fullmatch(text)):
                    lines.append(line)
        red_scan = not lines
        if red_scan:
            lines = self._warning_lines(ctx.frame)
        if not lines:
            return answer
        # 去掉整句中已含的短行（如"危险"已被"高压危险"覆盖）与重复行
        kept = []
        for line in lines:
            text = line.text.strip()
            if any(text != other.text and text in other.text for other in lines):
                continue
            if not any(line.text == k.text for k in kept):
                kept.append(line)
        if not kept:
            return answer
        texts = list(dict.fromkeys(line.text for line in kept))
        tags = list(answer.source_tags) + [
            SourceTag(kind='ocr', frame_id=ctx.frame.frame_id, raw_text=line.text, field='warning')
            for line in kept]
        evidence = list(answer.evidence) + [
            f'警示OCR原文：{line.text} [分数{line.score:.2f}]' for line in kept]
        sentence = '画面中有' + ('红色' if red_scan else '') + '警示文字：' + '、'.join(texts) + '。'
        seen = {line.text for line in answer.ocr_lines}
        new_lines = [line for line in kept if line.text not in seen]
        return replace(answer,
            text=answer.text.rstrip('。') + '。' + sentence,
            source_tags=tuple(tags), evidence=tuple(evidence),
            ocr_lines=tuple(answer.ocr_lines) + tuple(new_lines),
            notes=tuple(answer.notes) + ('红色标识局部OCR' if red_scan else '整帧OCR',))

    def _append_signboard_text(self, ctx, answer):
        """店面/招牌类描述回答补充整帧 OCR 读到的招牌原文；不让 VLM 猜店名。"""
        from aieyes.scene.models import SourceTag
        if ctx.ocr is None:
            return answer
        kept = []
        for line in ctx.ocr.lines:
            text = line.text.strip()
            if (line.score < .8
                    or not re.fullmatch(r'[\u4e00-\u9fffA-Za-z0-9 ]{2,24}', text)):
                continue
            if any(text != other.text and text in other.text
                   for other in ctx.ocr.lines if other.text.strip()):
                continue
            if not any(line.text == k.text for k in kept):
                kept.append(line)
        if not kept:
            return answer
        texts = list(dict.fromkeys(line.text for line in kept))
        tags = list(answer.source_tags) + [
            SourceTag(kind='ocr', frame_id=ctx.frame.frame_id, raw_text=line.text, field='signboard')
            for line in kept]
        evidence = list(answer.evidence) + [
            f'招牌OCR原文：{line.text} [分数{line.score:.2f}]' for line in kept]
        sentence = '画面中有招牌文字：' + '、'.join(texts) + '。'
        seen = {line.text for line in answer.ocr_lines}
        new_lines = [line for line in kept if line.text not in seen]
        return replace(answer,
            text=answer.text.rstrip('。') + '。' + sentence,
            source_tags=tuple(tags), evidence=tuple(evidence),
            ocr_lines=tuple(answer.ocr_lines) + tuple(new_lines),
            notes=tuple(answer.notes) + ('整帧OCR',))

    @property
    def ocr(self):
        if self._ocr is None:
            from ocr_adapter import OcrEngine
            self._ocr=OcrEngine()
        return self._ocr

    @property
    def vlm(self):
        if self._vlm is None:
            from vlm_adapter import VlmClient
            self._vlm = VlmClient()
        return self._vlm

    def _vlm_complete(self, prompt, frame):
        if prompt.startswith('你是一台帮助视障用户看东西的助手。'):
            from vlm_adapter import VISUAL_PREFIX
            question = prompt.split('用户问：', 1)[-1].split('\n', 1)[0]
            if not re.search(r'分别|各自|区别|不同',question):
                from vlm_adapter import COLOR_PREFIX,COUNT_PREFIX,COLORS
                count_match=re.search(r'(?:几|多少)(个|只|件|辆|台|张|把|人|具|处)',question)
                attribute='color' if re.search(r'颜色|什么色',question) else 'count' if count_match else None
                compound=[(unit,noun) for unit,noun in
                    re.findall(r'(?:几|多少)([个只件辆台张把人具处])([\u4e00-\u9fff]{1,6})',question)] if attribute=='count' else []
                if len({noun for _,noun in compound})>1:
                    # 两类以上目标分开数：单一数量槽会漏掉一类，答案必须逐类给出
                    nouns=list(dict.fromkeys(noun for _,noun in compound))
                    rule=('分别数画面中每类目标的数量。只输出JSON对象，键为："'
                       +'"、'.join(nouns)+'"，值为整数数量；确认没有填0，无法辨认或不能可靠计数填null。')
                    raw=super()._vlm_complete(COUNT_PREFIX+'用户问：'+question+rule,frame)
                    try:
                        parsed=json.loads(raw)
                        bits=[];uncertain=[]
                        for unit,noun in dict.fromkeys(compound):
                            value=parsed.get(noun)
                            if type(value) is int and 0<=value<=1000:
                                bits.append(f'{noun}有{value}{unit}' if value else f'没有看到{noun}')
                            else:
                                bits.append(None);uncertain.append(noun)
                        if len(uncertain)==len(bits):
                            return '没数清'+'、'.join(uncertain)+'的数量。'
                        return '，'.join(
                            bit if bit else f'{noun}我没数清'
                            for (_,noun),bit in zip(dict.fromkeys(compound),bits))+'。'
                    except (ValueError,KeyError,TypeError):
                        pass
                    return '暂时无法确认数量。'
                if attribute:
                    prefix=COLOR_PREFIX if attribute=='color' else COUNT_PREFIX
                    rule='先数用户问的目标有几个，再判断该目标颜色；确认不存在时count填0、color填无法确认，不得用其他物体颜色代替。无法可靠计数时count填null。' if attribute=='color' else '只数问题中目标物体的数量；确认没有填0，无法辨认或不能可靠计数填null。'
                    raw=super()._vlm_complete(prefix+'用户问：'+question+rule+'只输出JSON对象，不添加位置、用途或周边细节。',frame)
                    try:
                        parsed=json.loads(raw);value=parsed[attribute]
                        if attribute=='color':
                            count=parsed.get('count')
                            if count is None or count==0:return '未能确认画面中有这个目标，无法判断颜色。'
                            if type(count) is not int or not 0<count<=1000:raise ValueError('invalid presence count')
                        if attribute=='color' and value in COLORS:return '暂时无法确认颜色。' if value=='无法确认' else value+'。'
                        if attribute=='count' and value is None:return '暂时无法确认数量。'
                        if attribute=='count' and type(value) is int and 0<=value<=1000:return f'有{value}{count_match.group(1)}。' if value else '没有看到这个目标。'
                    except (ValueError,KeyError,TypeError):pass
                    from aieyes.scene.errors import SceneAnalysisError
                    raise SceneAnalysisError('属性结构化回答无效',provider='vlm')
            prompt = VISUAL_PREFIX + ('只根据图片回答用户问：' + question +
                '。只回答所问内容。先用一个短语说明整体（场地、路面等是什么），'
                '再分别列出周围环境或路面上清楚可见的其他要素，最多四种实体。'
                '每个短语是简短的名词短语或"位置+有+实体"，不复述问题里的词句，不用冒号。'
                '使用常见物体名称，不重复。不要推断建筑用途、店名或未入镜的物品，不读小字。'
                '不能确认具体类型时用上位物体名，不能确认数量时说明不确定。'
                '用简短短语组成JSON数组，不输出其他内容。')
            raw = super()._vlm_complete(prompt, frame)
            try:
                parsed=json.loads(raw)
                if isinstance(parsed,list) and 1<=len(parsed)<=4 and all(isinstance(x,str) and x.strip() and not x.lstrip().startswith(('{','[','```')) for x in parsed):
                    # 短语去掉尾部顿号/逗号/句号等残留标点，避免拼接出"人行道、。"这类残句
                    # 去重按"去掉数量前缀"后的短语归一：同一实体两种说法（"一个篮球架"/"篮球架"）
                    # 不能原样播两遍；保留首次出现的原文，不改写实体名称本身
                    seen={}
                    for item in parsed:
                        part=item.strip().rstrip('、，,。；; ')
                        if not part:
                            continue
                        key=re.sub(r'^一(?:辆|个|只|件|台|张|把)', '', part) or part
                        seen.setdefault(key, part)
                    parts=list(seen.values())
                    # 明显被截断的悬垂句（8字以上仍以"为/是/有"等收尾）不是有效实体短语，不播报半句话
                    parts=[part for part in parts
                        if len(part)<8 or not re.search(r'[为是有在和与的了]$',part)]
                    answer='、'.join(parts)
                else:
                    answer=None
                if not isinstance(answer,str) or not answer.strip() or answer.lstrip().startswith(('{','[','```')):
                    raise ValueError('empty visual answer')
                return answer.strip()
            except (ValueError,KeyError,TypeError) as exc:
                from aieyes.scene.errors import SceneAnalysisError
                raise SceneAnalysisError('视觉结构化回答无效', cause=str(exc), provider='vlm') from exc
        return super()._vlm_complete(prompt, frame)

    def find_object(self, frame, target):
        if _DANGER_FIND_RE.fullmatch((target or '').strip()):
            danger = self._danger_find(frame, (target or '').strip())
            if danger is not None:
                return danger
        result = super().find_object(frame, target)
        if not result.candidates:
            return result
        kept = []
        for candidate in result.candidates:
            x, y, w, h = candidate.box
            pad = max(24, int(max(w, h) * .2))
            crop = frame.rgb[max(0, y-pad):min(frame.height, y+h+pad), max(0, x-pad):min(frame.width, x+w+pad)]
            if crop.size == 0:
                continue
            from vlm_adapter import SINGLE_PREFIX
            prompt = (SINGLE_PREFIX + '图中最主要的物体是什么？只列一项你能确认的通用物体名称，'
                      '不带颜色、位置、型号修饰，不猜用途。只输出一个元素的JSON数组。')
            try:
                text = self.vlm.complete(prompt, images=[crop], max_tokens=64, timeout_s=40)
            except Exception:
                continue
            if confirmed_result(text, target):
                kept.append(candidate)
        degraded = result.degraded
        if len(kept) != len(result.candidates):
            degraded = tuple(dict.fromkeys((*degraded, 'candidate_unverified')))
        return replace(result, found=bool(kept), candidates=tuple(kept),
                       text=_candidates_text(target, kept) if kept else f'画面中有疑似目标，但不能确认是{target}。',
                       degraded=degraded)

    def _danger_find(self, frame, target):
        """危险类找物：抽象"危险"不是可找的物体，VLM 会把行人当候选或截断输出。
        只认文字证据：整帧警示行优先，再做红色标识局部扫描；读不到就如实说明，不下安全结论。"""
        from aieyes.scene.models import ObjectCandidates, ObjectCandidate
        from aieyes.scene.service import _position_of
        lines = []
        try:
            full = self.ocr.recognize(frame.rgb, source_frame_id=frame.frame_id)
        except Exception:
            full = None
        if full is not None:
            for line in full.lines:
                text = line.text.strip()
                if (line.score >= .8 and _WARNING_SHAPE_RE.fullmatch(text)
                        and _WARNING_TEXT_RE.search(text)):
                    lines.append((line, '警示文字'))
        texts = {line.text for line, _ in lines}
        for line in self._warning_lines(frame):
            if line.text not in texts:
                lines.append((line, '红色警示文字'))
        if not lines:
            return ObjectCandidates(
                target=target, found=False,
                text='我没有在画面里读到明确的危险警示文字，可以靠近一点或换个角度再看一次。',
                frame_id=frame.frame_id)
        kept = []
        for line, kind in lines:
            text = line.text.strip()
            if any(text != other.text and text in other.text
                   for other, _ in lines):
                continue
            if not any(line.text == k[0].text for k in kept):
                kept.append((line, kind))
        candidates = [ObjectCandidate(
            label=line.text, box=line.box,
            position=_position_of(line.box, frame.width, frame.height),
            confidence=line.score, feature=kind, frame_id=frame.frame_id)
            for line, kind in kept]
        text = '；'.join(
            f'{c.position}有{c.feature}：{c.label}' for c in candidates) + '。'
        return ObjectCandidates(
            target=target, found=True, text=text,
            candidates=tuple(candidates), frame_id=frame.frame_id)

    def _answer_describe(self, question, ctx):
        if getattr(self._request_state,"skip_ocr",False):
            ctx.timings.pop("ocr",None)
        device_question = re.search(r'(?:设备|机器|柜子|装置|设施).{0,4}(?:是什么|做什么|干什么|用途)|(?:什么)(?:设备|机器|柜子|装置)', question)
        if not device_question:
            answer = super()._answer_describe(question, ctx)
            # 安全类整景提问：可见警示文字必须随回答播报，不替主控下通行结论
            if _SAFETY_RE.search(question) and answer.intent == 'describe':
                answer = self._append_safety_warnings(ctx, answer)
            # 店面/招牌类描述：把 OCR 读到的招牌原文补进回答，店名只认文字证据
            elif _SIGNBOARD_RE.search(question) and answer.intent == 'describe':
                answer = self._append_signboard_text(ctx, answer)
            return answer
        from time import monotonic
        from vlm_adapter import LOCATE_PREFIX, VISUAL_PREFIX
        from aieyes.scene.models import SourceTag
        from aieyes.interfaces import OcrResult
        started = monotonic()
        try:
            from color_grounding import color_box
            refined=color_box(ctx.frame.rgb,question,(0,0,ctx.frame.width,ctx.frame.height))
            if refined is not None:
                # 唯一明确色块先做几何取景，保留上方异色附件；不额外调用模型定位。
                rx1,ry1,rx2,ry2=refined
                x1=max(0,rx1-round((rx2-rx1)*.05));x2=min(ctx.frame.width,rx2+round((rx2-rx1)*.05))
                y1=max(0,ry1-round((ry2-ry1)*.4));y2=min(ctx.frame.height,ry2+round((ry2-ry1)*.05))
            else:
                prompt = LOCATE_PREFIX + ('请定位用户指的物体：' + question +
                    '输出JSON对象found和box。found表示画面中能否定位该物体。'
                    'box为目标完整包围框[x1,y1,x2,y2]，坐标归一化到0至1000。')
                location = json.loads(self.vlm.complete(prompt, images=[ctx.frame.rgb], max_tokens=64, temperature=0, timeout_s=30))
                ctx.timings['device_locate'] = round((monotonic()-started)*1000)
                if location.get('found') is not True:
                    raise ValueError('requested device not located')
                box = location['box']
                if len(box)!=4 or any(type(x) is not int or not 0<=x<=1000 for x in box):
                    raise ValueError('invalid device box')
                x1,y1,x2,y2=box
                x1=round(x1*ctx.frame.width/1000);x2=round(x2*ctx.frame.width/1000)
                y1=round(y1*ctx.frame.height/1000);y2=round(y2*ctx.frame.height/1000)
                if x2<=x1 or y2<=y1:
                    raise ValueError('empty device box')
                from color_grounding import color_box
                refined=color_box(ctx.frame.rgb,question,(x1,y1,x2,y2))
                if refined is not None:
                    rx1,ry1,rx2,ry2=refined
                    x1,y1,x2,y2=min(x1,rx1),min(y1,ry1),max(x2,rx2),max(y2,ry2)
            ctx.timings['device_locate']=round((monotonic()-started)*1000)
            pad=max(8,round(min(x2-x1,y2-y1)*.05))
            x1=max(0,x1-pad);y1=max(0,y1-pad);x2=min(ctx.frame.width,x2+pad);y2=min(ctx.frame.height,y2+pad)
            crop=ctx.frame.rgb[y1:y2,x1:x2]
            t=monotonic();mapped=[]
            # 对高窄设备分块读字，避免整柜缩小后漏掉底部用途标签。
            for top in range(0,max(1,crop.shape[0]-150),750):
                tile=crop[top:top+900]
                ocr=self.ocr.recognize(tile,source_frame_id=ctx.frame.frame_id)
                mapped.extend(replace(line,box=(line.box[0]+x1,line.box[1]+y1+top,line.box[2],line.box[3]),
                    polygon=tuple((x+x1,y+y1+top) for x,y in line.polygon)) for line in ocr.lines)
            ctx.timings['device_ocr']=round((monotonic()-t)*1000)
            text='、'.join(dict.fromkeys(line.text for line in mapped))
            if not text:
                raise ValueError('no readable device labels')
            ctx.ocr=OcrResult(lines=tuple(ctx.ocr.lines if ctx.ocr else ())+tuple(mapped),
                source_frame_id=ctx.frame.frame_id,image_size=(ctx.frame.width,ctx.frame.height))
            q='用户问：'+question+'\n设备局部实际OCR：'+text+'。根据整机结构和操作标识只说整台设备的通用名称，不只报里面可取出的物品，不报品牌、型号，不解释用途，不复述标语。'
            t=monotonic();raw=self.vlm.complete(q,images=[crop],max_tokens=32,temperature=0,timeout_s=30)
            answer=raw.strip();views=[('完整候选',answer)]
            if refined is not None:
                rx1,ry1,rx2,ry2=refined
                if ry1-y1 > (ry2-ry1)*.1:
                    # 颜色通常只定位到外壳；补看与外壳相连的上部，避免漏掉不同颜色的关键结构。
                    pad_x=max(8,round((rx2-rx1)*.05))
                    hx1=max(0,rx1-pad_x);hx2=min(ctx.frame.width,rx2+pad_x)
                    hy2=min(y2,ry1+round((ry2-ry1)*.2))
                    detail=ctx.frame.rgb[y1:hy2,hx1:hx2]
                    label=self.vlm.complete('画面中这个物体的通用类型是什么？根据可见外形识别，不猜品牌，不推断未入镜部分。用一个短语回答。',images=[detail],max_tokens=32,temperature=0,timeout_s=30).strip()
                    views.append(('上部结构',label))
                    if label and label!=answer:
                        choice=self.vlm.complete('同一设备两种取景得到的候选：1、'+answer+'；2、'+label+'。该设备实际标识：'+text+'。哪个候选与操作和功能标识有直接对应？不能按品牌联想。只输出1或2；证据不足输出0。',max_tokens=8,temperature=0,timeout_s=10).strip()
                        if choice in ('1','2'):answer=views[int(choice)-1][1]
                        else:answer='暂时不能确认这台设备的具体类型。'
            ctx.timings['vlm']=round((monotonic()-t)*1000)
            raw=answer
            if not isinstance(answer,str) or not answer.strip() or answer.lstrip().startswith(('{','[','```')):
                raise ValueError('invalid grounded answer')
            if re.search(r'做什么|干什么|用途',question):
                purpose=self.vlm.complete('已由画面辨认的通用名称：'+answer+'。实际可见标识：'+text+'。用户问：'+question+'。只回答使用者能在这台设备上完成什么主要事情。把它作为整台设备，不要把整机当作用户可以拿走的物品。用简短一句口语，不罗列标识，不写如何操作，不补充收费或未确认功能。',max_tokens=48,temperature=0,timeout_s=15).strip()
                if not purpose or purpose.startswith(('{','[','```')):raise ValueError('purpose answer format invalid')
                answer=purpose
            return ctx.finish(answer,'describe',vlm_text=raw,confidence=.7,
                tags=(SourceTag(kind='ocr',frame_id=ctx.frame.frame_id,raw_text=text,field='device_detail'),),
                evidence=(f'局部框 {x1},{y1},{x2},{y2}',),extra_notes=tuple(f'{name}候选：{label}' for name,label in views))
        except Exception as exc:
            return ctx.finish('暂时不能确认这台设备的用途，请对准设备标识再问一次。','describe',
                confidence=.2,extra_degraded=('device_unverified',),extra_notes=(str(exc),))
