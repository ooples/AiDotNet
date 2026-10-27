import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('package_check', pathlib.Path(__file__).with_name('verify-playground-package.py'))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PlaygroundPackage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.functions = [{'name': name, 'scriptFile': 'App.dll', 'bindings': [
            {'type': 'httpTrigger', 'methods': ['post'], 'authLevel': 'anonymous'}
        ]} for name in ('execute', 'stripe-webhook')]
        (self.root/'App.dll').write_bytes(b'fixture')
        (self.root/'Stripe.net.dll').write_bytes(b'fixture')
        (self.root/'worker.config.json').write_text('{}')
        (self.root/'extensions.json').write_text('{}')
        (self.root/'.azurefunctions').mkdir()
        (self.root/'host.json').write_text('{"version":"2.0","extensions":{"http":{"routePrefix":"api"}}}')

    def verify(self):
        (self.root/'functions.metadata').write_text(json.dumps(self.functions))
        with contextlib.redirect_stdout(io.StringIO()):
            checker.verify(self.root)

    def test_complete_package(self):
        self.verify()

    def test_missing_worker_artifacts_are_rejected(self):
        for name in ('worker.config.json', 'extensions.json'):
            with self.subTest(name=name):
                artifact = self.root/name
                artifact.unlink()
                with self.assertRaisesRegex(ValueError, 'Missing isolated-worker artifact'):
                    self.verify()
                artifact.write_text('{}')

    def test_worker_directory_must_exist_and_be_a_directory(self):
        artifact = self.root/'.azurefunctions'
        artifact.rmdir()
        with self.assertRaisesRegex(ValueError, 'Missing isolated-worker directory'):
            self.verify()
        artifact.write_text('not a directory')
        with self.assertRaisesRegex(ValueError, 'Missing isolated-worker directory'):
            self.verify()

    def test_execute_only_package_is_rejected(self):
        self.functions.pop()
        with self.assertRaisesRegex(ValueError, 'Missing required function: stripe-webhook'):
            self.verify()

    def test_changed_route_or_authentication_is_rejected(self):
        trigger = self.functions[1]['bindings'][0]
        for key, value in [('methods', ['get']), ('route', 'other'), ('authLevel', 'function')]:
            with self.subTest(key=key):
                original = trigger.get(key)
                trigger[key] = value
                with self.assertRaises(ValueError):
                    self.verify()
                if original is None:
                    del trigger[key]
                else:
                    trigger[key] = original

    def test_missing_stripe_dependency_is_rejected(self):
        (self.root/'Stripe.net.dll').unlink()
        with self.assertRaisesRegex(ValueError, 'Stripe runtime'):
            self.verify()

    def test_wrong_prefix_or_external_assembly_is_rejected(self):
        (self.root/'host.json').write_text('{"extensions":{"http":{"routePrefix":""}}}')
        with self.assertRaisesRegex(ValueError, 'prefix'):
            self.verify()
        (self.root/'host.json').write_text('{}')
        self.functions[1]['scriptFile'] = '../outside.dll'
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
