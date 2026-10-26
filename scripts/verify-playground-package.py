"""Reject a Functions release that would silently remove either public route."""
import json
import pathlib
import sys


def verify(folder):
    folder = pathlib.Path(folder)
    functions = json.loads((folder / 'functions.metadata').read_text(encoding='utf-8-sig'))
    names = [item['name'] for item in functions]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate function names in package')
    for name in ('execute', 'stripe-webhook'):
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
    if not (folder / 'Stripe.net.dll').is_file():
        raise ValueError('Stripe runtime assembly is missing')
    host = json.loads((folder / 'host.json').read_text(encoding='utf-8-sig'))
    if host.get('extensions', {}).get('http', {}).get('routePrefix', 'api') != 'api':
        raise ValueError('The existing /api route prefix must be preserved')
    print('Verified package: POST /api/execute and POST /api/stripe-webhook, assemblies and host configuration')


if __name__ == '__main__':
    verify(sys.argv[1])
