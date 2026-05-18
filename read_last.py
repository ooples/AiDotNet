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
          comments(first: 1) {
            nodes { body }
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
    body = t['comments']['nodes'][0]['body']
    if '<details>' in body:
        body = body.split('<details>')[0]
    print(body[:500].encode('ascii', 'replace').decode('ascii'))
    print()
