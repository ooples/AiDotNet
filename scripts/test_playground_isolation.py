import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location('isolation', pathlib.Path(__file__).with_name('verify-playground-isolation.py'))
isolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolation)


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.billing = {'id': '/apps/billing'}
        self.execution = {'id': '/apps/execution', 'identity': None}
        self.billing_settings = self.storage('billing')
        self.execution_settings = self.storage('execution')

    @staticmethod
    def storage(name):
        return {key: f'AccountName={name};AccountKey=fake' for key in
                ('AzureWebJobsStorage', 'WEBSITE_CONTENTAZUREFILECONNECTIONSTRING')}

    def verify(self):
        isolation.verify(self.billing, self.execution, self.billing_settings, self.execution_settings)

    def test_separate_apps_pass(self):
        self.verify()

    def test_run_from_package_without_content_share_passes(self):
        del self.billing_settings['WEBSITE_CONTENTAZUREFILECONNECTIONSTRING']
        self.verify()

    def test_each_shared_storage_connection_fails(self):
        for key in self.execution_settings:
            with self.subTest(key=key):
                original = self.execution_settings[key]
                self.execution_settings[key] = self.billing_settings[key]
                with self.assertRaisesRegex(ValueError, 'share storage'):
                    self.verify()
                self.execution_settings[key] = original

    def test_same_app_or_identity_fails(self):
        self.execution['id'] = self.billing['id'].upper()
        with self.assertRaisesRegex(ValueError, 'separate Function App'):
            self.verify()
        self.execution['id'] = '/apps/execution'
        self.execution['identity'] = {'type': 'SystemAssigned'}
        with self.assertRaisesRegex(ValueError, 'managed identity'):
            self.verify()

    def test_unapproved_credentials_fail_without_printing_values(self):
        for key in ('STRIPE_SECRET_KEY', 'STRIPE_WEBHOOK_SECRET_SECONDARY', 'SUPABASE_SECRET_KEY', 'NEW_SECRET'):
            self.execution_settings[key] = 'do-not-expose'
            with self.assertRaisesRegex(ValueError, 'unapproved settings') as error:
                self.verify()
            self.assertNotIn('do-not-expose', str(error.exception))
            del self.execution_settings[key]

    def test_unverifiable_storage_and_remote_package_fail(self):
        self.execution_settings['AzureWebJobsStorage'] = ''
        with self.assertRaisesRegex(ValueError, 'Cannot verify'):
            self.verify()
        self.execution_settings = self.storage('execution')
        self.execution_settings['WEBSITE_RUN_FROM_PACKAGE'] = 'https://storage/package?secret'
        with self.assertRaisesRegex(ValueError, 'must be local'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
