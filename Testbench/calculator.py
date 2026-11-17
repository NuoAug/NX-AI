"""仅处理本句给全数字的明确算式和购物余额；不从历史猜测缺失金额。"""
import re
from decimal import Decimal
from aieyes.scene.parse import format_amount_cn
_NUM=r'(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千]+)'
_REMAIN=re.compile(r'(?P<budget>'+_NUM+r')(?:元|块)[^。！？]*?(?:买|购买)(?P<count>'+_NUM+r')(?:个|件|本|瓶|袋|份|支|盒)[^。！？]*?(?:各|每(?:个|件|本|瓶|袋|份|支|盒)?|单价)(?P<price>'+_NUM+r')(?:元|块)[^。！？]*?(?:还剩|剩下|余额|找回|找零)')
_EXPR=re.compile(r'^(?:请问|请|帮我|算一下|计算)?\s*('+_NUM+r')\s*(加上|加|减去|减|乘以|乘|除以|[+×*÷/−-])\s*('+_NUM+r')\s*(?:是多少|等于多少|等于几|得多少|是多少呢|等于|是)?[？?。=\s]*$')

def number(s):
    try:return Decimal(s)
    except Exception:pass
    digits=dict(zip('零〇一二两三四五六七八九',[0,0,1,2,2,3,4,5,6,7,8,9]));units={'十':10,'百':100,'千':1000};total=last=0
    for c in s:
        if c in digits:last=digits[c]
        elif c in units:total+=(last or 1)*units[c];last=0
        else:raise ValueError('unsupported numeral')
    return Decimal(total+last)

def answer_for(text):
    q=(text or '').strip();m=_REMAIN.search(q)
    if m:
        # 一句里出现第二组“数量+量词”（如“3件各20元，2件10元”）时，只按第一组算会给出错误余额，本句不给数。
        if len(re.findall(r'[0-9零〇一二两三四五六七八九十百千]+(?:个|件|本|瓶|袋|份|支|盒)',m.group(0)))>1:return None
        balance=number(m['budget'])-number(m['count'])*number(m['price'])
        return ('还剩' if balance>=0 else '还差')+format_amount_cn(abs(balance))+'。'
    m=_EXPR.fullmatch(q)
    if not m:return None
    a,b=number(m[1]),number(m[3]);op=m[2]
    if op in ('加上','加','+'):result=a+b
    elif op in ('减去','减','−','-'):result=a-b
    elif op in ('乘以','乘','×','*'):result=a*b
    elif b==0:return '除数不能为零。'
    else:result=a/b
    shown=format(result.quantize(Decimal('.000001')),'f').rstrip('0').rstrip('.') if result!=result.to_integral() else str(int(result))
    return shown+'。'
