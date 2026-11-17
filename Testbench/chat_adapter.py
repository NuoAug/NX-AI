"""测试台聊天提示：区分普通聊天与需要画面/实时数据的问题。"""
import json
import re
from aieyes.chat import ChatSession, Answer
from aieyes.clock import mono_ms
from calculator import answer_for
from vlm_adapter import CHAT_PREFIX

class RoleChatSession(ChatSession):
    # 用户主动提供的称呼只存在当前对话内，不参与声纹和权限判定。
    _name_declaration=re.compile(r'(?:^|[，,。；;])\s*(?:我叫|我的名字是|请叫我|叫我|你可以叫我)([\u4e00-\u9fffA-Za-z·]{1,16})(?=[，,。；;！!]|$)')
    _name_question=re.compile(r'(?:我(?:现在|到底|刚才)?叫(?:什么|啥)|我的名字(?:是)?(?:什么|啥)|记得我(?:的名字|叫什么))')
    # 无图像事实时的“看现场”问法：只要句中指向当前画面，就不能交给模型自由描述。
    # 处所不枚举成封闭词表，而是“泛指处所 | 家具/容器/房间名词(+可选方位) | 单字处所+必需方位”：
    # 换一个处所词（柜子上、抽屉里、窗外、桌子上、床上…）仍然命中；单字处所必须带方位，
    # 才不会把“起床以后有什么安排”这类句子当成现场问句。
    _place_generic=(
        r'面前|眼前|前面|后面|背后|画面|镜头|照片|那边|这边|这儿|这里|对面|旁边|附近|周围|四周'
        r'|这一带|这一片|上面|下面|外面|里面|室内|室外|帮我看|你看看'
    )
    _place_noun=(
        r'柜子|桌子|茶几|书架|架子|台面|桌面|墙面|衣柜|碗柜|鞋柜|冰箱|电视柜|床头柜|沙发|椅子|凳子'
        r'|枕头|箱子|袋子|垃圾桶|花盆|阳台|走廊|楼道|院子|房间|天花板|地板|地面|墙角|抽屉|窗户'
        r'|门框|床头|窗台|扶手|柜门|镜框|屏幕|背景|前景|衣服|裤子|外套|鞋子|背包|挎包'
    )
    _place_short=r'床|门|窗|桌|柜|架|墙|地|包|车|屋|手|身|楼|箱|袋'
    _orient=r'(?:上面|上|里面|里|下面|下|底下|旁边|边|后面|后|外面|外|内侧|内|中间|尽头|口|周围|四周)'
    _view_missing=re.compile(
        r'(?:'+_place_generic+
        r'|(?:'+_place_noun+r')(?:'+_orient+r')?'
        r'|(?:'+_place_short+r')(?:'+_orient+r'))'+
        r'[^。！？]{0,16}?(?:有什么|是什么|什么颜色|几个|安全吗|有人|有没有|亮着|开着|关着|在不在'
        r'|有(?:什么|没有)?(?:东西|水|人|字|字迹|动静|声音)|有啥东西|(?:挂|摆|放|搁|贴)着(?:什么|啥)'
        r'|(?:写|印)着(?:什么|啥))'
        r'|(?:这|那)(?:盏|个|件|台|张|块|条|瓶)[^。！？]{0,8}?(?:亮着|开着|关着|在不在|是什么)'
        r'|你(?:能)?(?:看见|看到|看清)[^。！？]{0,20}(?:吗|么)'
    )
    # 同一句式里问的不是“当前现场”，而是消息、文件、知识或推荐：豁免，不当作无图现场问句。
    _view_exempt=re.compile(
        r'消息|信息|短信|邮件|微信|简历|文件|文档|通知|记录|我说的|我说过|我刚才说|发的|发过|说过'
        r'|毛病|症状|不舒服|生病|什么病|有病吗'
        r'|有(?:什么|没有)?(?:好吃的|好喝的|好玩的|好玩的地方|推荐的|值得去|景点|餐厅|饭店|超市|商场|医院|学校|银行|公交|地铁)'
        r'|有(?:什么|没有)?(?:成语|诗|词|典故|办法|建议|句子|故事|歌|电影|书)'
        r'|有(?:什么)?(?:原因|含义|意思|道理|讲究|寓意|说法|设定)'
    )

    def record(self,user,assistant,**kwargs):
        super().record(user,assistant,**kwargs)
        if self._name_question.search(user):return
        known=getattr(self,'_user_name',None)
        if known and re.search(r'(?:我不叫|别叫我)'+re.escape(known),user):
            self._user_name=None
        matches=list(self._name_declaration.finditer(user))
        if matches:
            # “我叫李敏不是陈晨”：纠正短语不能整体记成名字，取“不是/不叫”之前的名字。
            name=re.split(r'不是|不叫',matches[-1].group(1))[0]
            if name and not re.search(r'什么|啥|谁|是不是|对不对',name):self._user_name=name

    def reset(self):
        super().reset();self._user_name=None

    def _grounded_answer(self,user_text):
        from intent_adapter import control_feedback
        feedback=control_feedback(user_text)
        if feedback is not None:return feedback
        exact=answer_for(user_text)
        if exact is not None:return exact
        if not self._view_exempt.search(user_text) and self._view_missing.search(user_text):
            return '我还没看到画面，让我看一眼再告诉你，好吗？'
        if re.search(r'(?:测试|检测|验收).{0,8}(?:通过|正常|结果|没问题).*(?:吗|么|呢|怎么样|如何|[？?])',user_text):
            return '我还不能确定，得实际测过，再看结果。'
        if self._name_question.search(user_text):
            name=getattr(self,'_user_name',None)
            return f'你叫{name}。' if name else '我还不能确认你的名字，请再告诉我一次。'
        return None

    def ask(self,user_text,*,images=(),**kwargs):
        exact=self._grounded_answer(user_text) if not images else None
        if exact is None:return super().ask(user_text,images=images,**kwargs)
        now=mono_ms()
        # record=False：调用方（交互引擎）按问答代次提交历史，固定答句也不能抢先落盘
        if kwargs.get('record',True):self.record(user_text,exact,t_ms=now)
        return Answer(exact,now,now)

    def ask_stream(self,user_text,*,images=(),**kwargs):
        exact=self._grounded_answer(user_text) if not images else None
        if exact is not None:return iter([exact])
        return super().ask_stream(user_text,images=images,**kwargs)

    def build_prompt(self, user_text):
        messages = [{"role":"system","content":self.system_prompt}]
        for turn in self.turns[-self.max_history_turns:]:
            # 没有回答的旧轮次（模型失败或未定稿）不重放：只留 user 会形成连续两条 user，并把旧问句塞到当前问题前面。
            if not turn.assistant:
                continue
            messages.append({"role":"user","content":turn.user})
            messages.append({"role":"assistant","content":turn.assistant})
        messages.append({"role":"user","content":user_text})
        return CHAT_PREFIX + json.dumps(messages,ensure_ascii=False)


def make_chat(vlm):
    return RoleChatSession(vlm, system_prompt=(
        '你叫小冰，是陪在用户身边的中文辅助伙伴。用我和你自然交流，通常一句或两句，直接回应用户当前说的话。'
        '像熟悉的同伴一样温和简洁，不用客服腔，不重复自我介绍，不每次反问。'
        '只依据用户已明确说出的事实和可靠常识，不补充未知的现场情况。建议可以用如果或要是，不能把假设说成事实。'
        '用户只说正在测试时，就自然接话，不主动评判结果、转去看周围或讨论查数据。用户说想出门，可以陪他聊，不编造风雨冷热或保证安全。'
        '用户说我指用户，回答时称你。准确记住用户自己的名字、否定和纠正，不把他的经历当作你的。'
    ))
