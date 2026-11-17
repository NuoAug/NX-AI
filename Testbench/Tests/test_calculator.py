from calculator import answer_for
from chat_adapter import make_chat
from unittest.mock import Mock
from intent_adapter import classify
from aieyes.intent import IntentKind

def test_current_numbers_and_units_override_old_math_context():
    chat=make_chat(Mock());chat.record('花费多少钱','10元')
    assert chat.ask('12元买两件各5元的东西，还剩几元？').text=='还剩2元。'
    assert ''.join(chat.ask_stream('23乘以4是多少？'))=='92。'
    assert answer_for('我有100元买三本书每本18元，还剩多少钱？')=='还剩46元。'
    assert answer_for('19元买三个各4元，还剩多少？')=='还剩7元。'
    assert answer_for('这次买两个一共多少钱') is None
    assert answer_for('8除以0等于多少')=='除数不能为零。'
    assert classify('我有100元买三本书每本18元，还剩多少钱？').kind==IntentKind.CHAT

def test_second_purchase_group_is_not_silently_ignored():
    # 只按第一组算会给出 40 元这种错误余额，宁可不出数。
    assert answer_for('100元买3件各20元和2件各10元，还剩多少？') is None
    assert answer_for('50元买2个各10元的和1个各5元的，还剩多少？') is None
    assert answer_for('100元买3件每件20元，还剩多少？')=='还剩40元。'
    # 第二组不带“各/每/单价”时同样是两组金额，按第一组算会答 40（真值 20）。
    assert answer_for('100元买3件各20元，2件10元一件，还剩多少？') is None
    assert answer_for('100元买3件各20元，又买2件10元的，还剩多少？') is None
    # 守卫只针对“第二组数量”，单组句子照常给数。
    assert answer_for('100元买3件各20元的咖啡，还剩多少？')=='还剩40元。'
    assert answer_for('100元买3件单价20元，还剩多少？')=='还剩40元。'

def test_read_and_attribute_questions_reach_vision_only_when_explicit():
    for q in ['读一下牌子上的字','图中有几个灭火器','这把椅子是什么颜色','画面中有多少个人']:
        assert classify(q).kind==IntentKind.LOOK
    assert classify('我刚读完一本书').kind==IntentKind.CHAT
    assert classify('读一下牌子，停止').kind==IntentKind.STOP
