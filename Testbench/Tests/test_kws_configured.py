from pathlib import Path
import tempfile
import unittest
from tempfile import TemporaryDirectory
from unittest import mock

from aieyes.config import KwsConfig
from aieyes.speech.kws import SherpaKws as ProjectKws
from kws_adapter import SherpaKws


class ConfiguredWakeTests(unittest.TestCase):
    def test_model_stock_phrases_are_not_enabled_implicitly(self):
        with TemporaryDirectory() as d:
            p=Path(d)
            (p/'tokens.txt').write_text('x 0\niǎo 1\nb 2\nīng 3\nn 4\nǐ 5\nh 6\nǎo 7\n')
            stock='n ǐ h ǎo @你好\n'
            (p/'keywords.txt').write_text(stock)
            kws=SherpaKws(KwsConfig(wake_phrases=['小冰']), model_dir=p)
            target=kws._prepare_keywords()
            self.assertEqual(target.read_text(), 'x iǎo b īng @小冰\n')
            self.assertEqual((p/'keywords.txt').read_text(), stock)
            self.assertTrue(target.is_relative_to(Path(__file__).resolve().parents[1]/'.runtime'))

    def test_project_base_class_still_uses_the_system_temp_root(self):
        """产品基类（Src/aieyes/speech/kws.py）的落点仍是系统临时根 + keywords_ 前缀。

        这一条只锚定“不写模型目录、不写仓库”的布局；本机 tempfile.gettempdir()
        在本会话会被沙箱拒写而退化到 CWD（见 B_speech2.md 的待主控项），所以这里把
        tempdir 显式钉到临时目录，避免测试自己在仓库根留下 aieyes_kws/。
        """
        with TemporaryDirectory() as model_d, TemporaryDirectory() as temp_d:
            model = Path(model_d)
            (model/'tokens.txt').write_text('x 0\niǎo 1\nb 2\nīng 3\n')
            with mock.patch.object(tempfile, 'tempdir', temp_d):
                target = ProjectKws(KwsConfig(wake_phrases=['小冰']),
                                    model_dir=model)._prepare_keywords()
                self.assertEqual(tempfile.gettempdir(), temp_d)
            self.assertEqual(target.parent, Path(temp_d)/'aieyes_kws')
            self.assertTrue(target.name.startswith('keywords_'))
            self.assertTrue(target.name.endswith('.txt'))
            self.assertEqual(target.read_text(encoding='utf-8'), 'x iǎo b īng @小冰\n')
            self.assertEqual(sorted(p.name for p in model.iterdir()), ['tokens.txt'])

    def test_keywords_never_land_in_cwd_when_temp_root_is_unusable(self):
        """tempdir 退化到 CWD 时，关键词文件不得写进当前工作目录。

        本机 %TEMP% 被拒写时 tempfile.gettempdir() 会静默回落到 CWD，历史上
        因此把 aieyes_kws/ 建到项目根。修复后应退到 <cwd>/.runtime/aieyes_kws。
        """
        import os
        tokens = os.linesep.join(['x 0', 'iǎo 1', 'b 2', 'īng 3']) + os.linesep
        with TemporaryDirectory() as work, TemporaryDirectory() as model_d:
            work_path = Path(work)
            model = Path(model_d)
            (model / 'tokens.txt').write_text(tokens, encoding='utf-8')
            old_cwd = Path.cwd()
            os.chdir(work_path)
            try:
                with mock.patch.object(tempfile, 'tempdir', str(work_path)):
                    self.assertEqual(tempfile.gettempdir(), str(work_path))
                    target = ProjectKws(KwsConfig(wake_phrases=['小冰']),
                                        model_dir=model)._prepare_keywords()
                self.assertFalse((work_path / 'aieyes_kws').exists(),
                                 '不得在 CWD 建 aieyes_kws/')
                self.assertEqual(target.parent, work_path / '.runtime' / 'aieyes_kws')
                self.assertEqual(target.read_text(encoding='utf-8').split(),
                                 ['x', 'iǎo', 'b', 'īng', '@小冰'])
            finally:
                os.chdir(old_cwd)

    def test_keywords_fall_back_when_temp_root_cannot_be_created(self):
        """tempdir 指向无法 mkdir 的路径（这里是文件）时同样不得污染 CWD。"""
        import os
        tokens = os.linesep.join(['x 0', 'iǎo 1', 'b 2', 'īng 3']) + os.linesep
        with TemporaryDirectory() as work, TemporaryDirectory() as model_d:
            work_path = Path(work)
            model = Path(model_d)
            (model / 'tokens.txt').write_text(tokens, encoding='utf-8')
            not_a_dir = work_path / 'not_a_dir'
            not_a_dir.write_text('', encoding='utf-8')
            old_cwd = Path.cwd()
            os.chdir(work_path)
            try:
                with mock.patch.object(tempfile, 'tempdir', str(not_a_dir)):
                    target = ProjectKws(KwsConfig(wake_phrases=['小冰']),
                                        model_dir=model)._prepare_keywords()
                self.assertFalse((work_path / 'aieyes_kws').exists())
                self.assertEqual(target.parent, work_path / '.runtime' / 'aieyes_kws')
            finally:
                os.chdir(old_cwd)

    def test_keywords_cache_dir_env_override(self):
        """AIEYES_KWS_CACHE 直接指定缓存目录（部署侧显式落点）。"""
        from aieyes.speech.kws import keywords_cache_dir
        with TemporaryDirectory() as d:
            with mock.patch.dict('os.environ', {'AIEYES_KWS_CACHE': d}):
                self.assertEqual(keywords_cache_dir(), Path(d))
