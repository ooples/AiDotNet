"""Fail deployment if the code runner can access billing credentials or shared storage."""
import json
import shutil
import subprocess
import sys


def verify(billing, execution, billing_settings, execution_settings):
    if billing['id'].lower() == execution['id'].lower():
        raise ValueError('Execution requires a separate Function App')
    if execution.get('identity') and execution['identity'].get('type', 'None') != 'None':
        raise ValueError('Execution app must not have a managed identity')
    # Fail closed for new settings. This app intentionally has no telemetry credentials,
    # billing secrets, shared API keys, or access to another app through a managed identity.
    allowed = {
        'AzureWebJobsStorage', 'WEBSITE_CONTENTAZUREFILECONNECTIONSTRING',
        'WEBSITE_CONTENTSHARE', 'WEBSITE_RUN_FROM_PACKAGE', 'FUNCTIONS_EXTENSION_VERSION',
        'FUNCTIONS_WORKER_RUNTIME', 'WEBSITE_USE_PLACEHOLDER_DOTNETISOLATED',
        'WEBSITE_NODE_DEFAULT_VERSION', 'WEBSITE_ENABLE_SYNC_UPDATE_SITE',
        'SCM_DO_BUILD_DURING_DEPLOYMENT', 'ENABLE_ORYX_BUILD',
    }
    if set(execution_settings) - allowed:
        raise ValueError('Execution app has unapproved settings; inspect names without exposing values')
    if execution_settings.get('WEBSITE_RUN_FROM_PACKAGE', '1') != '1':
        raise ValueError('Execution package must be local, not a credential-bearing URL')

    def accounts(settings):
        result = set()
        for key in ('AzureWebJobsStorage', 'WEBSITE_CONTENTAZUREFILECONNECTIONSTRING'):
            # Run-from-package apps may not mount an Azure Files content share.
            # Host storage is always required; validate a content share whenever present.
            if key == 'WEBSITE_CONTENTAZUREFILECONNECTIONSTRING' and key not in settings:
                continue
            parts = dict(part.split('=', 1) for part in settings.get(key, '').split(';') if '=' in part)
            account = parts.get('AccountName')
            if not account:
                raise ValueError('Cannot verify Function App storage account isolation')
            result.add(account.lower())
        return result

    if accounts(billing_settings) & accounts(execution_settings):
        raise ValueError('Execution and billing apps must not share storage accounts')


def main():
    group, billing_name, execution_name = sys.argv[1:]
    az = shutil.which('az') or shutil.which('az.cmd')
    if not az:
        raise RuntimeError('Azure CLI is required')

    def read(args):
        result = subprocess.run([az, *args, '-o', 'json', '--only-show-errors'],
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def app(name):
        return read(['functionapp', 'show', '-g', group, '-n', name])

    def settings(name):
        return {item['name']: item['value'] for item in
                read(['functionapp', 'config', 'appsettings', 'list', '-g', group, '-n', name])}

    verify(app(billing_name), app(execution_name), settings(billing_name), settings(execution_name))
    print('Verified separate apps/storage, no execution managed identity, and approved execution settings only')


if __name__ == '__main__':
    main()
