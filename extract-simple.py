import json

with open('unresolved-comments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

threads = data['data']['repository']['pullRequest']['reviewThreads']['nodes']

unresolved_items = []
for thread in threads:
    if not thread['isResolved'] and not thread['isOutdated']:
        path = thread['path'] or 'N/A'
        line = thread['line'] or 'N/A'
        thread_id = thread['id']
        unresolved_items.append((path, line, thread_id))

# Output as JSON
import json
with open('unresolved-simple.json', 'w') as f:
    json.dump(unresolved_items, f, indent=2)

# Also output simple list
with open('unresolved-files.txt', 'w') as f:
    for path, line, tid in unresolved_items:
        f.write(f"{path}:{line} ({tid})\n")

print(f"Found {len(unresolved_items)} unresolved comments")
for path, line, tid in unresolved_items:
    print(f"  {path}:{line}")
