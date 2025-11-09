import json

with open('unresolved-comments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

threads = data['data']['repository']['pullRequest']['reviewThreads']['nodes']

print("UNRESOLVED COMMENTS (16 total):\n")
num = 0
for thread in threads:
    if not thread['isResolved'] and not thread['isOutdated']:
        num += 1
        comment = thread['comments']['nodes'][0] if thread['comments']['nodes'] else None
        if comment:
            path = thread['path'] or 'N/A'
            line = thread['line'] or 'N/A'
            body = comment['body']
            # Extract first sentence or first 100 chars
            summary = body.split('\n')[0][:100]
            print(f"{num}. {path}:{line}")
            print(f"   ID: {thread['id']}")
            print(f"   {summary}")
            print()
