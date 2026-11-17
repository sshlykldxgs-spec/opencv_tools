import csv, os, shutil
import pytest
from conftest import PRE, run


@pytest.fixture(scope='module')
def raw(tmp_path_factory):
    import synth
    r = str(tmp_path_factory.mktemp('raw') / 'raw')
    synth.make_raw(r)
    return r


def q(path):
    return {r[0]: r for r in list(csv.reader(open(path, encoding='utf-8-sig')))[1:]}


def test_help_works_and_does_not_run():
    code, out = run(PRE, '--help')
    assert code == 0 and '命令行参数' in out and '不存在' not in out
    code, out = run(PRE, '-h')
    assert code == 0 and '命令行参数' in out


def test_pre_basic(raw, tmp_path):
    out = str(tmp_path / 'o')
    code, log = run(PRE, '--in', raw, '--out', out, '--workers', 2, '--max-side', 2000)
    assert code == 0, log
    d = q(out + '/quality.csv')
    assert d['corrupt.jpg'][1] == 'rejected' and 'corrupt' in d['corrupt.jpg'][2]
    assert 'too_small' in d['tiny.jpg'][2] and d['tiny.jpg'][1] == 'rejected'
    assert d['p3.jpg'][1] == 'kept' and d['blur.jpg'][1] == 'rejected'          # 近似重复保留更清晰的
    assert d['视频中文.mp4#frame0'][1] == 'kept'                                   # 中文名视频能抽帧
    import cv2
    from PIL import Image
    big = [r for r in d.values() if r[0] == 'big.jpg'][0]
    assert cv2.imread(os.path.join(out, 'images', big[4])).shape[:2] == (1500, 2000)
    ex = [r for r in d.values() if r[0] == 'exif6.jpg'][0]
    im = Image.open(os.path.join(out, 'images', ex[4]))
    assert im.size == (600, 800) and im.getexif().get(274) in (None, 1)          # EXIF 旋转已烘焙
    assert os.path.exists(out + '/group_manifest.csv') and not os.path.exists(out + '.partial')


def test_scene_mode_picks_shots(raw, tmp_path):
    out = str(tmp_path / 'o')
    assert run(PRE, '--in', raw, '--out', out, '--mode', 'scene')[0] == 0
    kept = [r for r in q(out + '/quality.csv').values() if r[0].startswith('video1') and r[1] == 'kept']
    assert len(kept) == 3


def test_overwrite_protection_and_bad_config(raw, tmp_path):
    out = str(tmp_path / 'o')
    assert run(PRE, '--in', raw, '--out', out)[0] == 0
    code, log = run(PRE, '--in', raw, '--out', out)
    assert code == 2 and '已存在' in log
    assert run(PRE, '--in', raw, '--out', out, '--overwrite')[0] == 0
    code, log = run(PRE, '--in', str(tmp_path / 'nope'), '--out', str(tmp_path / 'x'), '--mode', 'bad')
    assert code == 2 and '配置有 2 处问题' in log and 'Traceback' not in log
    assert not os.path.exists(str(tmp_path / 'x'))


def test_empty_input_is_a_clean_error(tmp_path):
    e = str(tmp_path / 'empty'); os.makedirs(e)
    code, log = run(PRE, '--in', e, '--out', str(tmp_path / 'o'))
    assert code == 2 and '没有可用的素材' in log and not os.path.exists(str(tmp_path / 'o'))
