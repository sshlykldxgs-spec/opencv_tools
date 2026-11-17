import os, subprocess, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.environ.get('PREP_SCRIPTS', os.path.dirname(HERE))
sys.path.insert(0, HERE)
POST = os.path.join(SCRIPTS, '标注后-数据集生成.py')
PRE = os.path.join(SCRIPTS, '标注前-素材整理.py')


def run(script, *args, env=None, cwd=None):
    e = {**os.environ, **(env or {})}
    p = subprocess.run([sys.executable, script, *map(str, args)], capture_output=True, text=True, encoding='utf-8', env=e, cwd=cwd, timeout=900)
    return p.returncode, p.stdout + p.stderr


@pytest.fixture(scope='session')
def det(tmp_path_factory):
    import synth
    root = str(tmp_path_factory.mktemp('det') / 'd')
    truth = synth.make_det(root)
    return root, truth
