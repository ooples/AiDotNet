import json
import sys

with open('unresolved-comments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

threads = data['data']['repository']['pullRequest']['reviewThreads']['nodes']

unresolved = []
for idx, thread in enumerate(threads, 1):
    if not thread['isResolved'] and not thread['isOutdated']:
        comment = thread['comments']['nodes'][0] if thread['comments']['nodes'] else None
        if comment:
            unresolved.append({
                'num': idx,
                'id': thread['id'],
                'path': thread['path'] or 'N/A',
                'line': thread['line'] or 'N/A',
                'author': comment['author']['login'],
                'body': comment['body'][:200] + '...' if len(comment['body']) > 200 else comment['body']
            })

print(f"\n{'='*80}")
print(f"UNRESOLVED COMMENTS ON PR #393: {len(unresolved)} total")
print(f"{'='*80}\n")

for item in unresolved:
    print(f"#{item['num']} | File: {item['path']}")
    print(f"   Line: {item['line']} | Author: {item['author']}")
    print(f"   Thread ID: {item['id']}")
    print(f"   Issue: {item['body'][:150]}...")
    print(f"{'-'*80}\n")
