"""Complete Đặng Quang Tiến's evaluation after a completed model upgrade.

No polling, training or new inference. Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.verify_model3_upgrade import execute, preservation


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upgrade-run',type=Path,required=True)
    parser.add_argument('--run-id',default='tien-evaluation-20261115')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):
        parser.error('Invalid run-id')
    upgrade=args.upgrade_run.resolve()
    if not (ROOT/'logs/model3_upgrade'/upgrade.name/'completion.json').is_file():
        raise ValueError('Finish and verify model upgrade before member4 evaluation')
    orchestration=ROOT/'logs/evaluation'/f'orchestration-{args.run_id}'
    orchestration.mkdir(parents=True,exist_ok=False)
    run=ROOT/'report/evaluation'/args.run_id
    tag=f'{args.run_id}-final'
    commands=[
        ('evaluation.log',[sys.executable,'-B','scripts/evaluate_saved_models.py','--dataset-dir','data',
                           '--upgrade-run',str(upgrade),'--run-id',args.run_id]),
        ('notebook_create.log',[sys.executable,'-B','scripts/create_evaluation_notebook.py','--run-dir',str(run)]),
        ('verification.log',[sys.executable,'-B','scripts/verify_evaluation.py','--run-dir',str(run),'--tag',tag]),
        ('report.log',[sys.executable,'-B','scripts/write_evaluation_report.py','--run-dir',str(run),
                       '--verification-dir',str(ROOT/'logs/evaluation'/f'verification-{tag}')]),
    ]
    for name,command in commands:
        result=execute(command,orchestration/name)
        print(name,result,flush=True)
        if result['exit_code']!=0:
            raise RuntimeError(f'Step failed: {name}; inspect real logs before rerunning')
    # Append only the new member4 declaration to the current user-edited document.
    ai=ROOT/'AI_USAGE.md'
    marker='### Đặng Quang Tiến — hoàn thiện đánh giá và phân tích lỗi (2026-11-15)'
    current=ai.read_text(encoding='utf-8')
    if marker in current:
        raise FileExistsError('Evaluation AI declaration already exists')
    declaration=(f'\n{marker}\n\n'
        '- Người phụ trách: **Đặng Quang Tiến (người4)**. Trợ lý hỗ trợ module đánh giá saved predictions, '
        'confusion43lớp, per-class/paired/group diagnostics, ảnh lỗi, audit trùng byte, tests, logs/report và notebook04. '
        'Giữ nguyên phần gốc của người1/2/3 và hàm evaluate_and_analyze cũ; chỉ thêm files/output có namespace mới.\n'
        '- Đối chiếu sáu bộ predictions lịch sử với Test.csv hiện tại, selection/claim và metrics; không fit/predict, '
        'không nạp joblib hoặc chạy official test mới. Kết quả OOF nâng cấp tách riêng, có selection bias, '
        'không gọi là nested/unbiased estimate hoặc điểm test mới.\n'
        '- Log và verification thực chạy nằm logs/evaluation/, báo cáo report/notes_evaluation_tien.md, '
        'bảng/ảnh evaluation_<run-id>_* và notebook notebooks/04_evaluation_tien.ipynb đã lưu output. '
        'Giữ nguyên lỗi PCA float32 người2; không tuyên bố toàn suite pass.\n'
        '- Phân tích lỗi/ảnh minh họa chỉ mô tả, không khẳng định nguyên nhân mờ/tối hoặc significance. '
        'Test đã xem ở EDA; trùng byte không chứng minh ảnh hưởng đến metrics và không loại trừ near-duplicate. '
        'Người học tự xác minh và chịu trách nhiệm; không commit/push hoặc xuất bản ra dịch vụ ngoài.\n')
    with ai.open('a',encoding='utf-8',newline='') as stream:
        stream.write(declaration)
    evidence=preservation(ROOT/'data/model3_preservation_before.json',orchestration/'final_preservation.json')
    if not evidence['protected_unchanged']:
        raise RuntimeError('Protected source/history changed')
    with (orchestration/'completion.json').open('x',encoding='utf-8') as stream:
        json.dump({'run_id':args.run_id,'contributor':'Đặng Quang Tiến','member_number':4,
                   'finished_utc':datetime.now(timezone.utc).isoformat(),'evaluation':'passed',
                   'notebook':'executed','report':'written','protected_unchanged':True,
                   'new_official_test_inference':False},stream,ensure_ascii=False,indent=2)
    print('Member4 evaluation completed; all original source and historical results preserved.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
