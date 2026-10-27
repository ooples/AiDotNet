"""Smoke-check routing without creating Stripe events or executing user code."""
import sys
import time
import urllib.error
import urllib.request


def check(base, attempts=12, delay=5):
    for route in ('execute', 'stripe-webhook'):
        status = None
        for attempt in range(attempts):
            request = urllib.request.Request(base.rstrip('/') + '/api/' + route, data=b'{}',
                                             headers={'Content-Type': 'application/json'}, method='POST')
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    status = response.status
            except urllib.error.HTTPError as error:
                status = error.code
            except (urllib.error.URLError, TimeoutError):
                status = 'unreachable'
            if status == 400:
                break
            if attempt + 1 < attempts:
                time.sleep(delay)
        if status != 400:
            raise RuntimeError(f'/api/{route}: expected 400 for an empty unsigned request, received {status}')
        print(f'/api/{route}: 400 as expected; handler is reachable and rejects the empty request')


if __name__ == '__main__':
    check(sys.argv[1])
