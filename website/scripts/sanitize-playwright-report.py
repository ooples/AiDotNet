"""Redact credential-bearing diagnostic text before a report can be uploaded."""
import base64, io, json, os, pathlib, re, sys, zipfile

secret_names = ['SUPABASE_SERVICE_ROLE_KEY', 'STRIPE_SECRET_KEY', 'STRIPE_WEBHOOK_SECRET',
                'VERCEL_AUTOMATION_BYPASS_SECRET', 'PLAYWRIGHT_USER_PASSWORD', 'PLAYWRIGHT_ADMIN_PASSWORD']
values = [os.environ[k] for k in secret_names if os.environ.get(k)]
state_dir = os.environ.get('AIDOTNET_AUTH_RUN_DIR')
if state_dir:
    for name in ['deployment.json', 'user.json', 'admin.json']:
        path = pathlib.Path(state_dir) / name
        if path.exists():
            state = json.loads(path.read_text(encoding='utf-8'))
            values.extend(cookie['value'] for cookie in state.get('cookies', []))
values = sorted({v for v in values if len(v) > 3}, key=len, reverse=True)

def redact(text):
    for value in values:
        text = text.replace(value, '[redacted]').replace(json.dumps(value)[1:-1], '[redacted]')
    text = re.sub(r'(?im)(authorization|apikey|cookie|x-vercel-protection-bypass)\s*:[^\r\n]*', r'\1: [redacted]', text)
    text = re.sub(r'''(?i)([?&#](?:access_token|refresh_token|token_hash|token|code|secret|client_secret)=)[^&\s"'<>]+''', r'\1[redacted]', text)
    text = re.sub(r'''(?i)(https?://billing\.stripe\.com/p/session/)[^/?#\s"'<>]+''', r'\1[redacted]', text)
    text = re.sub(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', '[redacted token]', text)
    text = re.sub(r'\badn_[0-9a-f]{64}\b', '[redacted API key]', text, flags=re.I)
    text = re.sub(r'\bAIDN-[A-Z0-9-]+-[A-F0-9]{8,}\b', '[redacted license key]', text, flags=re.I)
    text = re.sub(r'E2e![A-Za-z0-9_-]{12,}', '[redacted fixture password]', text)
    return text

def clean(value):
    if isinstance(value, str): return redact(value)
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    return value

def clean_archive(encoded):
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as source, zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if pathlib.Path(item.filename).suffix == '.json':
                data = json.dumps(clean(json.loads(data.decode('utf-8'))), ensure_ascii=False).encode('utf-8')
            elif pathlib.Path(item.filename).suffix in ['.txt', '.md']:
                data = redact(data.decode('utf-8')).encode('utf-8')
            target.writestr(item, data)
    return base64.b64encode(output.getvalue()).decode('ascii')

folder = pathlib.Path(sys.argv[1])
if not (folder / 'index.html').exists():
    raise SystemExit('No complete Playwright HTML report was produced; nothing is safe to upload.')
for path in folder.rglob('*'):
    if not path.is_file():
        continue
    if path.suffix == '.zip':
        raise SystemExit('Unexpected trace archive in report; privileged runs must disable tracing.')
    if path.suffix not in ['.json', '.txt', '.md', '.html']:
        continue
    text = path.read_text(encoding='utf-8')
    if path.name == 'index.html':
        text = re.sub(r'(data:application/zip;base64,)([A-Za-z0-9+/=]+)', lambda m: m[1] + clean_archive(m[2]), text)
    # JSON diagnostics must be redacted as string values, preserving document structure.
    if path.suffix == '.json':
        text = json.dumps(clean(json.loads(text)), ensure_ascii=False)
    elif path.suffix != '.html':
        text = redact(text)
    path.write_text(text, encoding='utf-8')
print('Playwright diagnostic text sanitized; no credential values were printed.')
