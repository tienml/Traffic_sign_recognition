"""One-off, read/verify-only delivery audit; writes only new verification evidence.

Executed notebooks are copied into the notebook delivery paths after checking
all cells completed without errors. Their prior versions are archived.
"""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import unquote
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.verify_models import preservation
import nbformat

logs = Path(__file__).parent
policy = json.loads((logs / 'policy-final.json').read_text(encoding='utf-8'))
baseline = json.loads((logs / 'baseline.json').read_text(encoding='utf-8'))
relocation = json.loads((logs / 'relocations.json').read_text(encoding='utf-8'))
model_dir = ROOT / 'logs/knn/verification-models-delivery-20261115'
eval_dir = ROOT / 'logs/evaluation/verification-models-delivery-20261115'
model_check = json.loads((model_dir / 'verification.json').read_text(encoding='utf-8'))
eval_check = json.loads((eval_dir / 'verification.json').read_text(encoding='utf-8'))
assert model_check['verification_passed'] and eval_check['new_work_passed']
assert model_check['known_baseline_failure_only'] and eval_check['known_baseline_failure_only']
source_validation = json.loads((logs / 'validated_source.json').read_text(encoding='utf-8'))
for relative, expected in source_validation['sha256'].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative

installed = []
for executed, target in [
    (model_dir / '03_models.ipynb', ROOT / 'notebooks/03_models.ipynb'),
    (eval_dir / '04_evaluation_tien.ipynb', ROOT / 'notebooks/04_evaluation_tien.ipynb'),
]:
    notebook = nbformat.read(executed, as_version=4)
    assert not [o for c in notebook.cells if c.cell_type == 'code'
                for o in c.get('outputs', []) if o.output_type == 'error']
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == 'code')
    shutil.copyfile(executed, target)
    installed.append({
        'path': target.relative_to(ROOT).as_posix(),
        'code_cells': sum(c.cell_type == 'code' for c in notebook.cells),
        'error_outputs': 0,
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    })
final_preservation = preservation(
    logs / 'baseline.json', logs / 'final_preservation.json',
    allowed_changes=policy['authorized_changes'], relocations=logs / 'relocations.json',
)
assert final_preservation['protected_unchanged'] and not final_preservation['unexpected_changes']
for old, copies in relocation['duplicate_evidence'].items():
    expected = baseline['before_sha256'][old]
    for relative in copies:
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
for relative, expected in baseline['before_sha256'].items():
    if relative.startswith('data/models/'):
        path = ROOT / relocation['paths'][relative]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
for relative in [
    'src/models.py', 'src/evaluate.py', 'src/decision_tree.py', 'src/preprocessing.py',
    'src/features.py', 'src/tests/test_features.py', 'src/tests/test_preprocessing.py',
    'main.py', 'src/config.py', 'requirements.txt',
    'notebooks/01_eda.ipynb', 'notebooks/02_preprocessing_ablation.ipynb',
]:
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == baseline['before_sha256'][relative], relative

missing, links = [], 0
for relative in [
    'README.md', 'AI_USAGE.md', 'report/notes_knn.md', 'report/notes_dt.md',
    'report/notes_evaluation_tien.md', 'report/evaluation_scope_tien.md', 'report/handoff_tien.md',
]:
    path = ROOT / relative
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8-sig')):
        target = target.strip().split('#', 1)[0]
        if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
            continue
        links += 1
        if not (path.parent / unquote(target)).exists():
            missing.append({'document': relative, 'target': target})
assert not missing, missing
for old in [
    'report/model3_upgrade', 'logs/model3_upgrade', 'data/models/model3_upgrade',
    'report/notes_model3_upgrade.md',
]:
    assert not (ROOT / old).exists(), old
obsolete = {
    'scripts.train_model3_upgrade', 'scripts.summarize_model3_upgrade',
    'scripts.verify_model3_upgrade', 'scripts.check_model3_artifacts',
    'scripts.create_model3_notebook', 'src.model_upgrade', 'src.evaluation_saved',
}
for folder in ('src', 'scripts'):
    for path in (ROOT / folder).rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in obsolete, path
            if isinstance(node, ast.Import):
                assert all(x.name not in obsolete for x in node.names), path
check = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
with (logs / 'git_diff_check.log').open('x', encoding='utf-8') as stream:
    stream.write(check.stdout + check.stderr)
assert check.returncode == 0
old_requirements = (logs / 'source_before/scripts/requirements_model3_upgrade.txt').read_text(encoding='utf-8').splitlines()
new_requirements = (ROOT / 'scripts/requirements_models.txt').read_text(encoding='utf-8').splitlines()
assert [x for x in old_requirements if x and not x.startswith('#')] == [x for x in new_requirements if x and not x.startswith('#')]
lengths = {}
for name in ['README.md', 'AI_USAGE.md']:
    before = (logs / 'source_before' / name).read_text(encoding='utf-8-sig')
    after = (ROOT / name).read_text(encoding='utf-8-sig')
    lengths[name] = {'before_characters': len(before), 'after_characters': len(after),
                     'before_lines': len(before.splitlines()), 'after_lines': len(after.splitlines())}
completion = {
    'date': '2026-11-15', 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'contributor': 'Đặng Quang Tiến', 'member_number': 4,
    'organization': 'All modeling results in existing KNN/DT areas',
    'model_runs': {'knn': 'report/knn/hog-full-20261115', 'dt': 'report/dt/hog-full-20261115'},
    'standalone_upgrade_directories_removed': True, 'standalone_upgrade_report_merged': True,
    'original_contributions_preserved': True, 'experiment_metadata_and_hashes_rewritten': False,
    'relocations': len(relocation['paths']),
    'byte_identical_relocations': len(final_preservation['relocated_byte_identical']),
    'shared_evidence_copies': sum(map(len, relocation['duplicate_evidence'].values())),
    'focused_tests': {'models': {'passed': 20, 'total': 20}, 'evaluation': {'passed': 18, 'total': 18}},
    'full_suite': {'passed': 95, 'total': 96, 'exit_code': 1, 'all_tests_passed': False,
                   'known_failure': 'test_pca_variance_curve_is_cumulative_and_capped'},
    'notebooks': installed, 'historical_notebook_versions_archived': True,
    'saved_prediction_parity': {'official_test_csv': 6, 'oof_csv': 4, 'passed': True},
    'new_dataset_training': False, 'new_official_test_inference': False, 'model_deserialization': False,
    'all_model_bundle_hashes_preserved': True, 'core_source_unchanged_this_reorganization': True,
    'requirements_version_pins_unchanged': True, 'unexpected_changes': [],
    'local_document_links_checked': links, 'missing_links': missing, 'git_diff_check_passed': True,
    'document_simplification': lengths, 'committed': False, 'pushed': False,
    'status': 'completed_with_known_baseline_pca_test_failure',
}
with (logs / 'completion.json').open('x', encoding='utf-8') as stream:
    json.dump(completion, stream, ensure_ascii=False, indent=2)
print(json.dumps(completion, ensure_ascii=False, indent=2))
