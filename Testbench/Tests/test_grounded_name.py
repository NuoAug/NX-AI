from unittest.mock import Mock
from chat_adapter import make_chat

def make():
 v=Mock();v.complete.return_value='收到。'
 return make_chat(v)

def test_user_name_and_assistant_roles_stay_separate_after_correction():
 c=make();c.ask('我叫陈晨，请记住。');c.ask('刚才说错了，我叫李敏，不叫陈晨。')
 assert c.ask('我现在叫什么？').text=='你叫李敏。'
 assert ''.join(c.ask_stream('我的名字是什么？'))=='你叫李敏。'

def test_nickname_and_relative_are_not_confused():
 c=make();c.ask('请叫我阿杰。');assert c.ask('我叫什么名字？').text=='你叫阿杰。'
 c.ask('我妈妈叫李兰，我叫周宁。');assert c.ask('我叫什么？').text=='你叫周宁。'

def test_name_is_not_question_text_and_reset_clears_it():
 c=make();c.ask('我叫李敏。');c.ask('我叫什么？');assert c.ask('我叫什么？').text=='你叫李敏。'
 c.ask('我不叫李敏。');assert '不能确认' in c.ask('我叫什么？').text
 c.ask('我叫周宁。');c.reset();assert '不能确认' in c.ask('我叫什么？').text

def test_name_survives_history_window_within_current_conversation():
 c=make();c.ask('我的名字是小林。')
 for _ in range(12):c.record('聊点别的','好的')
 assert c.ask('记得我的名字吗？').text=='你叫小林。'

def test_correction_inside_declaration_keeps_only_the_name():
 c=make();c.ask('我叫李敏不是陈晨。');assert c.ask('我叫什么？').text=='你叫李敏。'
 c=make();c.ask('我叫王五不叫王六');assert c.ask('我叫什么？').text=='你叫王五。'
