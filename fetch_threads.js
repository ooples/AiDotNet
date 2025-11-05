const { execSync } = require('child_process');

async function getAllThreads() {
    let allThreads = [];
    let hasNext = true;
    let cursor = null;
    let page = 1;
    
    while (hasNext && page <= 5) {
        console.log(`Fetching page ${page}...`);
        
        const afterClause = cursor ? `, after: "${cursor}"` : '';
        const query = `{
            repository(owner: "ooples", name: "AiDotNet") {
                pullRequest(number: 304) {
                    reviewThreads(first: 100${afterClause}) {
                        nodes {
                            id
                            isResolved
                            comments(first: 3) {
                                nodes {
                                    id
                                    body
                                    path
                                    createdAt
                                }
                            }
                        }
                        pageInfo {
                            hasNextPage
                            endCursor
                        }
                    }
                }
            }
        }`;
        
        try {
            const result = JSON.parse(execSync(`gh api graphql -f query='${query}'`).toString());
            const threads = result.data.repository.pullRequest.reviewThreads;
            
            allThreads = allThreads.concat(threads.nodes);
            hasNext = threads.pageInfo.hasNextPage;
            cursor = threads.pageInfo.endCursor;
            page++;
        } catch (e) {
            console.error('Error:', e.message);
            break;
        }
    }
    
    const unresolved = allThreads.filter(t => !t.isResolved);
    console.log(`\nTotal threads: ${allThreads.length}`);
    console.log(`Unresolved: ${unresolved.length}`);
    
    // Show first 10 unresolved
    console.log('\n=== FIRST 10 UNRESOLVED ===');
    unresolved.slice(0, 10).forEach((t, i) => {
        const comment = t.comments.nodes[0];
        if (comment) {
            console.log(`\n[${i+1}] ${t.id}`);
            console.log(`File: ${comment.path}`);
            console.log(`Preview: ${comment.body.substring(0, 100)}...`);
        }
    });
    
    // Save to file
    require('fs').writeFileSync('unresolved_threads_final.json', JSON.stringify(unresolved, null, 2));
    console.log(`\nSaved ${unresolved.length} threads to unresolved_threads_final.json`);
}

getAllThreads();
