import json
import subprocess

query = '''
{
  repository(owner: "ooples", name: "AiDotNet") {
    pullRequest(number: 1368) {
      reviewThreads(first: 100) {
        nodes {
          id
          isResolved
          path
          line
          comments(first: 3) {
            nodes {
              body
              author { login }
            }
          }
        }
      }
    }
  }
}
'''
result = subprocess.run(['gh', 'api', 'graphql', '-f', f'query={query}'],
                       capture_output=True, encoding='utf-8', errors='replace')
data = json.loads(result.stdout)
threads = data['data']['repository']['pullRequest']['reviewThreads']['nodes']
unresolved = [t for t in threads if not t['isResolved']]
print(f"Total unresolved: {len(unresolved)}\n")
for i, t in enumerate(unresolved):
    print(f"=== {i+1}: {t['id']} {t['path']}:{t['line']} ===")
    for c in t['comments']['nodes'][:1]:
        body = c['body']
        if '<details>' in body:
            body = body.split('<details>')[0]
        # Encode-safe print
        print(body[:500].encode('ascii', 'replace').decode('ascii'))
    print()
