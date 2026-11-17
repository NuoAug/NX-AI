"""测试台将聊天上下文按API角色传递；图像任务沿用项目请求。"""
import json
from aieyes.vision.vlm import VlmClient as ProjectVlmClient
CHAT_PREFIX = '__BENCH_CHAT_MESSAGES__:'
VISUAL_PREFIX = '__BENCH_VISUAL_ANSWER__:'
#: 单体辨认（找物候选局部复核）：只允许一个元素的 JSON 数组，与整景描述的 4 实体区分开
SINGLE_PREFIX = '__BENCH_SINGLE_OBJECT__:'
LOCATE_PREFIX = '__BENCH_LOCATE__:'
COLOR_PREFIX = '__BENCH_COLOR__:'
COUNT_PREFIX = '__BENCH_COUNT__:'
COLORS = ('红色','橙色','黄色','绿色','青色','蓝色','紫色','粉色','棕色','灰色','黑色','白色','银色','金色','透明','多种颜色','其他颜色','无法确认')

class VlmClient(ProjectVlmClient):
    def __init__(self, *args, cancel_event=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cancel_event = cancel_event
        if cancel_event is not None and self._owns_session and self.llm.base_url.startswith('http://'):
            from cancel_http import LocalModelSession
            self._session.close()
            self._session = LocalModelSession(cancel_event)

    def complete_stream(self, *args, **kwargs):
        import requests
        from concurrent.futures import CancelledError
        try:
            yield from super().complete_stream(*args, **kwargs)
        except requests.RequestException:
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise CancelledError('Model request cancelled') from None
            raise

    def _build_payload(self, prompt, images, **kwargs):
        if not prompt.startswith(CHAT_PREFIX):
            attribute = None
            if prompt.startswith(COLOR_PREFIX):attribute='color';prompt=prompt[len(COLOR_PREFIX):]
            elif prompt.startswith(COUNT_PREFIX):attribute='count';prompt=prompt[len(COUNT_PREFIX):]
            locate = prompt.startswith(LOCATE_PREFIX)
            if locate:
                prompt = prompt[len(LOCATE_PREFIX):]
            compact = prompt.startswith(VISUAL_PREFIX)
            single = prompt.startswith(SINGLE_PREFIX)
            if compact:
                prompt = prompt[len(VISUAL_PREFIX):]
            elif single:
                prompt = prompt[len(SINGLE_PREFIX):]
            if compact or single:
                from aieyes.vision.image_ops import resize_keep_aspect
                import numpy as np
                images = [resize_keep_aspect(im, max_side_px=640)[0] if isinstance(im,np.ndarray) else im for im in images]
            payload = super()._build_payload(prompt, images, **kwargs)
            if images:
                payload['messages'].insert(0, {'role':'system', 'content':
                    '你是图像助手。只依据图片中清楚可见的内容回答。'
                    '不要猜测设备用途、品牌、看不清的文字或精确数量。'
                    '不确定时只描述可见的外形，不补全细节。'})
            if compact or single:
                # 场景/整体描述问题需要覆盖问题问到的多个方面（如"场地+周围环境"），
                # 实体数上限 4、单短语 24 字、96 token 预算与之匹配；
                # 单体辨认（找物候选复核）只允许 1 个元素，多列会把候选名拆成多解
                payload['response_format'] = {'type':'json_schema','schema':{
                    'type':'array','items':{'type':'string','minLength':1,'maxLength':24},
                    'minItems':1,'maxItems':1 if single else 4}}
                payload['max_tokens'] = 48 if single else 96
            if locate:
                payload['response_format'] = {'type':'json_schema','schema':{
                    'type':'object','properties':{'found':{'type':'boolean'},'box':{
                        'type':'array','items':{'type':'integer','minimum':0,'maximum':1000},
                        'minItems':4,'maxItems':4}},'required':['found','box'],'additionalProperties':False}}
                payload['max_tokens'] = 64
            if attribute:
                value_schema={'type':'string','enum':list(COLORS)} if attribute=='color' else {'anyOf':[{'type':'integer','minimum':0,'maximum':1000},{'type':'null'}]}
                properties={attribute:value_schema}
                if attribute=='color':properties={'count':{'anyOf':[{'type':'integer','minimum':0,'maximum':1000},{'type':'null'}]},'color':value_schema}
                payload['response_format']={'type':'json_schema','schema':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}
                payload['max_tokens']=24
            return payload
        messages = json.loads(prompt[len(CHAT_PREFIX):])
        payload = super()._build_payload('', images, **kwargs)
        image_parts = payload['messages'][0]['content'][1:]
        if image_parts:
            messages[-1]['content'] = [{'type':'text','text':messages[-1]['content']}, *image_parts]
        payload['messages'] = messages
        return payload
