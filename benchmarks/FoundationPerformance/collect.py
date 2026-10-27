"""Normalize runner output, validate checksum agreement, and produce the measured table."""
import json, pathlib, sys
source=pathlib.Path(sys.argv[1]); target=pathlib.Path(__file__).parent/'results';target.mkdir(exist_ok=True)
lines=['| Device / workload | Shape | Before ms | After ms | PyTorch ms | Before/after |', '|---|---|---:|---:|---:|---:|']
for device in ['cpu','gpu']:
    reports=[]
    for version in ['before','after','pytorch']:
        raw=(source/f'foundation-{version}-{device}.json').read_text(encoding='utf-8-sig')
        report=json.loads(raw[raw.index('{'):]);reports.append(report)
        (target/f'{version}-{device}.json').write_text(json.dumps(report,indent=2)+'\n')
    assert len({len(r['results']) for r in reports})==1
    for rows in zip(*(r['results'] for r in reports)):
        assert all(r['name']==rows[0]['name'] and r['shape']==rows[0]['shape'] for r in rows)
        checks=[r['checksum'] for r in rows]
        assert max(checks)-min(checks)<max(1e-5,abs(checks[0])*1e-5),(rows[0]['name'],checks)
        before,after,torch=[r['median_ms'] for r in rows]
        lines.append(f"| {device.upper()} {rows[0]['name']} | {'×'.join(map(str,rows[0]['shape']))} | {before:.4f} | {after:.4f} | {torch:.4f} | {before/after:.2f}× |")
(target/'table.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Validated all benchmark checksums; wrote normalized reports and table.')
