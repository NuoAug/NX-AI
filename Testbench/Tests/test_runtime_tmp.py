"""关键词缓存必须落在系统临时根（tempfile.gettempdir()），不能落在 CWD/仓库。

本用例显式使用仓库里**现有的** KWS 模型资产，不依赖目标机的正式路径
（``/home/mylin/Desktop/Nuo/models``）。本机没有资产时记 ``xfail``（“本机不具备该环境”），
不用 ``skip`` 冒充“已跳过检查”；``xfail`` 会在报告里单列，不会被算成通过。

为了让这条断言真的有约束力，用例把 ``tempfile.gettempdir`` 指到一个可识别的
临时根：若产品代码改成写 ``Path.cwd()``（变异检查的锚点），断言会失败。
"""
from pathlib import Path
import tempfile

import numpy as np
import pytest

from aieyes.speech.kws import SherpaKws
from aieyes.interfaces import AudioFrame

_REPO = Path(__file__).resolve().parents[2]

#: 仓库内可用的 KWS 资产（按优先级）。两者都含 tokens.txt + onnx。
_ASSET_DIRS = (
    _REPO / 'testbench' / 'data' / 'models' / 'kws-zh-en-20251220-chunk8',
    _REPO / 'models' / 'kws' / 'sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01',
    _REPO / 'Testbench' / 'Data' / 'Models' / 'kws-zh-en-20251220-chunk8',
    _REPO / 'NX-AI' / 'Models' / 'kws' / 'sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01',
)


def _existing_asset() -> Path:
    for path in _ASSET_DIRS:
        if (path / 'tokens.txt').is_file():
            return path
    pytest.xfail(
        '本机无可用 KWS 资产（tokens.txt 不存在）：'
        + '; '.join(str(p) for p in _ASSET_DIRS)
    )


def test_keywords_follow_system_temp_root(tmp_path, monkeypatch):
    system_temp = tmp_path / 'systmp'
    system_temp.mkdir()
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(system_temp))

    kws=SherpaKws(model_dir=_existing_asset())
    try:
        kws.accept(AudioFrame(np.zeros(320,np.float32),0))
        assert kws.ready
        path=kws.keywords_path
        assert path and path.is_file()
        # 必须落在系统临时根下，而不是 CWD/仓库
        assert path.is_relative_to(system_temp), f'{path} 不在系统临时根 {system_temp} 下'
        assert 'x iǎo b īng @小冰' in path.read_text()
    finally:kws.close()
