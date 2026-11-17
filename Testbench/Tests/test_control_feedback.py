import pytest
from chat_adapter import make_chat


class NoModel:
    def complete(self,*args,**kwargs):
        raise AssertionError('Known control feedback must not invent model facts')


@pytest.mark.parametrize(('text','reply'),[
    ('不要关闭麦克风','好，我不更改麦克风设置。'),
    ('我没有让你关闭麦克风','好，我不更改麦克风设置。'),
    ('不要取消静音','好，我不更改静音设置。'),
    ('不要取消精英','好，我不取消。'),
    ('怎么清空资料','直接对我说“清空资料”就可以了。'),
    ('怎样清空资料','直接对我说“清空资料”就可以了。'),
    ('怎么才能清空资料','直接对我说“清空资料”就可以了。'),
    ('清空资料怎么操作','直接对我说“清空资料”就可以了。'),
    ('我该如何删除这轮对话','直接对我说“清空资料”就可以了。'),
])
def test_no_change_acknowledgment_and_real_feature_help(text,reply):
    assert make_chat(NoModel()).ask(text).text==reply


@pytest.mark.parametrize('text',['我想继续上学','不要继续上学','关闭麦克风是什么意思',
                                 '不要关闭麦克风，请继续说刚才的故事',
                                 '不要清空资料，怎么继续聊天'])
def test_help_does_not_replace_general_chat_or_mixed_requests(text):
    assert make_chat(NoModel())._grounded_answer(text) is None
