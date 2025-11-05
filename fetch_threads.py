import subprocess
import json

def get_all_threads():
    all_threads = []
    cursor = None
    page = 1
    
    while page <= 5:
        print(f"Fetching page {page}...")
        
        after = f', after: "{cursor}"' if cursor else ''
        query = f'{{ repository(owner: "ooples", name: "AiDotNet") {{ pullRequest(number: 304) {{ reviewThreads(first: 100{after}) {{ nodes {{ id isResolved comments(first: 3) {{ nodes {{ id body path createdAt }} }} }} pageInfo {{ hasNextPage endCursor }} }} }} }} }}'
        
        try:
            result = subprocess.run(
                ['gh', 'api', 'graphql', '-f', f'query={query}'],
                capture_output=True,
                text=True,
                check=True
            )
            data = json.loads(result.stdout)
            threads = data['data']['repository']['pullRequest']['reviewThreads']
            
            all_threads.extend(threads['nodes'])
            if not threads['pageInfo']['hasNextPage']:
                break
            cursor = threads['pageInfo']['endCursor']
            page += 1
        except Exception as e:
            print(f"Error: {e}")
            break
    
    unresolved = [t for t in all_threads if not t['isResolved']]
    print(f"\nTotal threads: {len(all_threads)}")
    print(f"Unresolved: {len(unresolved)}")
    
    # Show sample
    print("\n=== FIRST 5 UNRESOLVED ===")
    for i, t in enumerate(unresolved[:5]):
        if t['comments']['nodes']:
            comment = t['comments']['nodes'][0]
            print(f"\n[{i+1}] {t['id']}")
            print(f"File: {comment['path']}")
            print(f"Preview: {comment['body'][:100]}...")
    
    # Save
    with open('unresolved_final.json', 'w') as f:
        json.dump(unresolved, f, indent=2)
    print(f"\nSaved {len(unresolved)} unresolved threads")

get_all_threads()
