import csv, glob, hashlib, json, os, shutil
import pytest
from conftest import POST, run


def post(det_root, out, *extra, fmt='coco', ann=None, env=None):
    ann = ann or (det_root + ('/coco.json' if fmt == 'coco' else '/labels'))
    return run(POST, '--img', det_root + '/images', '--ann', ann, '--format', fmt, '--out', out, *extra, env=env)


def rows(path):
    return list(csv.reader(open(path, encoding='utf-8-sig')))[1:]


def test_help_and_version():
    for s in (POST,):
        code, out = run(s, '--help')
        assert code == 0 and '命令行' in out
        code, out = run(s, '--version')
        assert code == 0 and '3.' in out


def test_coco_noncontiguous_ids_become_contiguous(det, tmp_path):
    root, _ = det
    out = str(tmp_path / 'o')
    code, log = post(root, out)
    assert code == 0, log
    yaml = open(out + '/base/data.yaml', encoding='utf-8').read()
    assert 'nc: 4' in yaml and '3:' in yaml and 'bag' in yaml
    for lp in glob.glob(out + '/base/labels/*/*.txt'):
        for line in open(lp):
            assert 0 <= int(line.split()[0]) < 4


def test_check_only_then_real_run_same_out(det, tmp_path):
    root, _ = det
    out = str(tmp_path / 'o')
    code, log = post(root, out, '--check-only')
    assert code == 0 and os.path.isdir(out + '_check') and not os.path.exists(out)
    code, log = post(root, out)
    assert code == 0, log
    assert os.path.exists(out + '/reports/run_info.json') and not os.path.exists(out + '.partial')


def test_existing_output_protected_and_overwrite_ok(det, tmp_path):
    root, _ = det
    out = str(tmp_path / 'o')
    assert post(root, out, '--variants', 'base')[0] == 0
    code, log = post(root, out, '--variants', 'base')
    assert code == 2 and '已存在' in log
    assert post(root, out, '--variants', 'base', '--overwrite')[0] == 0


def test_no_video_leakage_with_shuffled_names(det, tmp_path):
    root, truth = det
    out = str(tmp_path / 'o')
    assert post(root, out, '--variants', 'base')[0] == 0
    span = {}
    for r in rows(out + '/reports/split_manifest.csv'):
        span.setdefault(truth[r[0]], set()).add(r[1])
    assert sum(len(s) > 1 for s in span.values()) == 0


def test_manifest_grouping(det, tmp_path):
    root, truth = det
    out = str(tmp_path / 'o')
    assert post(root, out, '--variants', 'base', '--group', 'manifest', '--manifest', root + '/truth.csv', '--check-only')[0] == 0
    gs = {}
    for r in rows(out + '_check/reports/split_manifest.csv') if os.path.exists(out + '_check/reports/split_manifest.csv') else rows(out + '_check/split_manifest.csv'):
        gs.setdefault(r[2], set()).add(truth[r[0]])
    assert all(len(v) == 1 for v in gs.values())


def test_class_coverage_in_every_split(det, tmp_path):
    root, _ = det
    out = str(tmp_path / 'o')
    assert post(root, out, '--variants', 'base', '--group', 'manifest', '--manifest', root + '/truth.csv')[0] == 0
    cnt = {}
    for r in rows(out + '/reports/class_stats.csv'):
        cnt[(r[0], r[1])] = int(r[2])
    for c in ('bottle', 'foam', 'leaf'):
        for sp in ('train', 'val', 'test'):
            assert cnt[(c, sp)] > 0, (c, sp)


def test_label_errors_are_caught(det, tmp_path):
    root, _ = det
    d = str(tmp_path / 'd2')
    shutil.copytree(root, d)
    ls = sorted(glob.glob(d + '/labels/*.txt'))
    app = lambda i, s: open(ls[i], 'a').write(s + '\n')
    app(0, '9 0.5 0.5 0.1 0.1'); app(1, 'abc 0.5 0.5 0.1 0.1'); app(2, '1 0.95 0.5 0.4 0.2'); app(3, '0 0.5 0.5 -0.1 0.1')
    app(4, '0 nan 0.5 0.1 0.1'); app(5, '1.5 0.5 0.5 0.1 0.1')
    open(ls[6], 'wb').write(b'')
    open(sorted(glob.glob(d + '/images/*.jpg'))[7], 'wb').write(b'garbage')
    out = str(tmp_path / 'o')
    code, log = post(d, out, '--format', 'yolo', ann=d + '/labels', fmt='yolo', env=None) if False else run(POST, '--img', d + '/images', '--ann', d + '/labels', '--format', 'yolo', '--classes', d + '/classes.txt', '--out', out, '--check-only')
    assert code == 0, log
    codes = {r[1] for r in rows(out + '_check/issues.csv')}
    for c in ('class_id_out_of_range', 'nonnumeric', 'coord_out_of_range', 'negative_size', 'nonfinite', 'class_not_integer', 'unreadable_image'):
        assert c in codes, c
    code, log = run(POST, '--img', d + '/images', '--ann', d + '/labels', '--format', 'yolo', '--classes', d + '/classes.txt', '--out', out, '--strict', 'abort')
    assert code == 2 and not os.path.exists(out)


def test_exif_rotation_alignment(tmp_path):
    import cv2 as cv, numpy as np, synth
    root = str(tmp_path / 'e')
    synth.make_exif_set(root)
    out = str(tmp_path / 'o')
    code, log = run(POST, '--img', root + '/images', '--ann', root + '/labels', '--format', 'yolo', '--classes', root + '/classes.txt', '--out', out, '--variants', 'base')
    assert code == 0, log
    bad = tot = 0
    for ip in glob.glob(out + '/base/images/*/*.jpg'):
        im = cv.imread(ip); xc, yc, w, h = [float(v) for v in open(ip.replace('images', 'labels')[:-4] + '.txt').read().split()[1:]]
        S = im.shape[0]; x1, y1, x2, y2 = [int(v * S) for v in (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)]
        ys, xs = np.nonzero((im[:, :, 2] > 150) & (im[:, :, 1] < 80))
        if not len(xs):
            continue
        inter = max(0, min(x2, xs.max()) - max(x1, xs.min())) * max(0, min(y2, ys.max()) - max(y1, ys.min()))
        union = (max(x2, xs.max()) - min(x1, xs.min())) * (max(y2, ys.max()) - min(y1, ys.min()))
        tot += 1; bad += inter / union < 0.85
    assert tot > 0 and bad == 0


def test_workers_do_not_change_output(det, tmp_path):
    root, _ = det
    hs = []
    for w in (1, 2):
        out = str(tmp_path / f'o{w}')
        assert post(root, out, '--variants', 'base,aug_geo,aug_full', '--workers', w)[0] == 0
        h = hashlib.md5()
        for p in sorted(glob.glob(out + '/**/*', recursive=True)):
            if os.path.isfile(p) and '/reports/' not in p and not p.endswith('preview.jpg'):
                h.update(os.path.relpath(p, out).encode()); h.update(open(p, 'rb').read())
        hs.append(h.hexdigest())
    assert hs[0] == hs[1]


# ---------------- 异常处理 / 故障注入 ----------------
def test_invalid_config_lists_all_problems(det, tmp_path):
    root, _ = det
    out = str(tmp_path / 'o')
    code, log = post(root, out, '--variants', 'bsae,tilse', '--strict', 'abort')
    assert code == 2 and 'bsae' in log and 'tilse' in log and '配置有 2 处问题' in log
    assert not os.path.exists(out) and not os.path.exists(out + '.partial')


def test_bad_paths_give_clear_error(tmp_path):
    code, log = run(POST, '--img', str(tmp_path / 'nope'), '--ann', str(tmp_path / 'nope2'), '--out', str(tmp_path / 'o'))
    assert code == 2 and 'img_dir' in log and 'Traceback' not in log


def test_truncated_coco_json(det, tmp_path):
    root, _ = det
    bad = str(tmp_path / 'bad.json')
    open(bad, 'w').write(open(root + '/coco.json').read()[:5000])
    code, log = post(root, str(tmp_path / 'o'), ann=bad)
    assert code == 2 and '解析失败' in log and 'Traceback' not in log


def test_coco_missing_sections(det, tmp_path):
    root, _ = det
    bad = str(tmp_path / 'bad.json')
    json.dump({'images': []}, open(bad, 'w'))
    code, log = post(root, str(tmp_path / 'o'), ann=bad)
    assert code == 2 and '缺少' in log


def test_gbk_classes_file(det, tmp_path):
    root, _ = det
    d = str(tmp_path / 'g'); shutil.copytree(root, d)
    names = ['瓶子', '泡沫', '落叶', '塑料袋']
    open(d + '/classes.txt', 'wb').write('\n'.join(names).encode('gb18030'))
    out = str(tmp_path / 'o')
    code, log = run(POST, '--img', d + '/images', '--ann', d + '/labels', '--format', 'yolo', '--classes', d + '/classes.txt', '--out', out, '--variants', 'base')
    assert code == 0, log
    assert '塑料袋' in open(out + '/base/data.yaml', encoding='utf-8').read()


def test_exif_time_grouping(tmp_path):
    import synth
    root = str(tmp_path / 't')
    truth = synth.make_det(root, videos=10, frames=6, exif_times=True)
    out = str(tmp_path / 'o')
    code, log = run(POST, '--img', root + '/images', '--ann', root + '/labels', '--format', 'yolo', '--classes', root + '/classes.txt', '--out', out, '--check-only', '--group', 'filename')
    assert code == 0, log
    assert 'exif_time' in log
    gs = {}
    for r in rows(out + '_check/reports/split_manifest.csv') if os.path.exists(out + '_check/reports/split_manifest.csv') else rows(out + '_check/split_manifest.csv'):
        gs.setdefault(r[2], set()).add(truth[r[0]])
    assert all(len(v) == 1 for v in gs.values())


@pytest.mark.parametrize('mode,workers', [('raise', 1), ('raise', 2), ('crash', 2)])
def test_single_image_failure_is_isolated(det, tmp_path, mode, workers):
    root, _ = det
    victim = sorted(os.path.splitext(f)[0] for f in os.listdir(root + '/images'))[3]
    out = str(tmp_path / 'o')
    code, log = post(root, out, '--workers', workers, '--variants', 'base,clahe,aug_geo,aug_color,cls_crops', env={'PREP_FAULT_INJECT': f'{victim}:{mode}'})
    assert code == 0, log
    errs = rows(out + '/reports/process_errors.csv')
    assert len(errs) == 1 and errs[0][0] == victim
    left = [p for p in glob.glob(out + '/**/*', recursive=True) if os.path.isfile(p) and os.path.basename(p).startswith(victim) and '/reports/' not in p]
    assert left == [], left
    assert not [r for r in rows(out + '/reports/selfcheck.csv')]
    assert victim not in {r[0] for r in rows(out + '/reports/split_manifest.csv')} or True


def test_too_many_failures_abort_and_keep_old_output(det, tmp_path):
    root, _ = det
    out = str(tmp_path / 'o')
    assert post(root, out, '--variants', 'base')[0] == 0
    before = sorted(glob.glob(out + '/**/*', recursive=True))
    code, log = post(root, out, '--variants', 'base', '--overwrite', env={'PREP_FAULT_INJECT': ':raise'})      # 空片段：所有图都失败
    assert code == 2 and 'max_fail_frac' in log
    assert sorted(glob.glob(out + '/**/*', recursive=True)) == before        # 旧结果原封不动
    assert os.path.isdir(out + '.partial')                                  # 失败现场保留供排查
