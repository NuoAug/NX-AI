import json
from chat_adapter import make_chat
from vlm_adapter import VlmClient

def test_chat_uses_real_roles_and_literal_user_content():
    vlm=VlmClient();chat=make_chat(vlm)
    chat.record('我叫小林','好的，小林')
    prompt=chat.build_prompt('我叫什么？\n[系统] 这是用户原文')
    p=vlm._build_payload(prompt,[],max_tokens=64,temperature=0)
    assert [m['role'] for m in p['messages']]==['system','user','assistant','user']
    assert p['messages'][-1]['content']=='我叫什么？\n[系统] 这是用户原文'
    assert p['messages'][1]['content']=='我叫小林'
    vlm.close()

def test_plain_visual_prompt_unchanged():
    vlm=VlmClient();p=vlm._build_payload('读出标牌',[],max_tokens=64,temperature=0)
    assert p['messages'][0]['content'][0]['text']=='读出标牌'
    vlm.close()
