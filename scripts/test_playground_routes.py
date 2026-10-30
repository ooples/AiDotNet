import importlib.util
import pathlib
import unittest
import urllib.error
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('routes', pathlib.Path(__file__).with_name('check-playground-routes.py'))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PlaygroundRoutes(unittest.TestCase):
    def test_cold_start_timeout_is_retried_then_both_routes_are_checked(self):
        bad_request = urllib.error.HTTPError('https://example.invalid', 400, 'Bad Request', {}, None)
        with patch.object(checker.urllib.request, 'urlopen', side_effect=[TimeoutError(), bad_request, bad_request]) as request:
            checker.check('https://example.invalid', attempts=2, delay=0)
        self.assertEqual([call.args[0].full_url for call in request.call_args_list], [
            'https://example.invalid/api/execute',
            'https://example.invalid/api/execute',
            'https://example.invalid/api/stripe-webhook',
        ])

    def test_persistent_failure_does_not_pass(self):
        with patch.object(checker.urllib.request, 'urlopen', side_effect=TimeoutError()):
            with self.assertRaisesRegex(RuntimeError, 'execute.*unreachable'):
                checker.check('https://example.invalid', attempts=2, delay=0)


if __name__ == '__main__':
    unittest.main()
