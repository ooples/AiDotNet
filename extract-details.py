import json

with open('unresolved-comments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

threads = data['data']['repository']['pullRequest']['reviewThreads']['nodes']

with open('unresolved-details.txt', 'w', encoding='utf-8') as out:
    num = 0
    for thread in threads:
        if not thread['isResolved'] and not thread['isOutdated']:
            num += 1
            comment = thread['comments']['nodes'][0] if thread['comments']['nodes'] else None
            if comment:
                path = thread['path'] or 'N/A'
                line = thread['line'] or 'N/A'
                author = comment['author']['login']
                body = comment['body']
                
                out.write(f"\n{'='*80}\n")
                out.write(f"ISSUE #{num}\n")
                out.write(f"File: {path}\n")
                out.write(f"Line: {line}\n")
                out.write(f"Author: {author}\n")
                out.write(f"Thread ID: {thread['id']}\n")
                out.write(f"{'='*80}\n\n")
                out.write(body)
                out.write(f"\n\n")

print(f"Extracted {num} unresolved comment details to unresolved-details.txt")
