"""Reject a Functions release that would silently remove either public route."""
import json
import pathlib
import sys


def verify(folder, execution=False):
    folder = pathlib.Path(folder)
    for name in ('worker.config.json', 'extensions.json'):
        if not (folder / name).is_file():
            raise ValueError(f'Missing isolated-worker artifact: {name}')
    if not (folder / '.azurefunctions').is_dir():
        raise ValueError('Missing isolated-worker directory: .azurefunctions')
    functions = json.loads((folder / 'functions.metadata').read_text(encoding='utf-8-sig'))
    names = [item['name'] for item in functions]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate function names in package')
    required = ('execute',) if execution else ('execute', 'stripe-webhook')
    if any(name not in required for name in names):
        raise ValueError('Unexpected function in isolated package')
    for name in required:
        if name not in names:
            raise ValueError(f'Missing required function: {name}')
        item = next(item for item in functions if item['name'] == name)
        trigger = next((b for b in item['bindings'] if b['type'].lower() == 'httptrigger'), None)
        if not trigger or 'post' not in [m.lower() for m in trigger.get('methods', [])]:
            raise ValueError(f'{name} needs a POST HTTP trigger')
        if trigger.get('route') not in (None, name) or trigger.get('authLevel', '').lower() != 'anonymous':
            raise ValueError(f'{name} must retain its public route and application-level authentication')
        assembly = (folder / item['scriptFile']).resolve()
        if not assembly.is_relative_to(folder.resolve()) or not assembly.is_file():
            raise ValueError(f'{name} assembly is missing or outside the package')
    if execution and (folder / 'Stripe.net.dll').exists():
        raise ValueError('Execution package must not contain Stripe')
    if not execution:
        execute = next(item for item in functions if item['name'] == 'execute')
        if execute.get('entryPoint') != 'AiDotNet.Playground.Functions.ExecuteCodeProxy.Run':
            raise ValueError('Billing worker must only proxy execution')
        if any(folder.glob('Microsoft.CodeAnalysis*.dll')) or (folder / 'AiDotNet.dll').exists():
            raise ValueError('Billing package must not contain the code compiler or execution library')
    if not execution and not (folder / 'Stripe.net.dll').is_file():
        raise ValueError('Stripe runtime assembly is missing')
    host = json.loads((folder / 'host.json').read_text(encoding='utf-8-sig'))
    if host.get('extensions', {}).get('http', {}).get('routePrefix', 'api') != 'api':
        raise ValueError('The existing /api route prefix must be preserved')
    print('Verified isolated execution package' if execution else 'Verified webhook and execution proxy package')


if __name__ == '__main__':
    verify(sys.argv[1], execution='--execution' in sys.argv[2:])
