#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');

// Read the review threads JSON
const data = JSON.parse(fs.readFileSync('pr497_review_threads.json', 'utf8'));
const threads = data.data.repository.pullRequest.reviewThreads.nodes;

// Filter unresolved threads
const unresolvedThreads = threads.filter(thread => !thread.isResolved);

console.log(`Found ${unresolvedThreads.length} unresolved review threads`);
console.log('');

// Resolve each thread
let resolved = 0;
let failed = 0;

for (const thread of unresolvedThreads) {
    const threadId = thread.id;
    const comment = thread.comments.nodes[0];
    const path = comment.path;
    const body = comment.body.substring(0, 60) + '...';

    console.log(`Resolving thread ${threadId}`);
    console.log(`  File: ${path}`);
    console.log(`  Comment: ${body}`);

    try {
        const mutation = `
mutation {
  resolveReviewThread(input: {threadId: "${threadId}"}) {
    thread {
      id
      isResolved
    }
  }
}`;

        const result = execSync(`gh api graphql -f query='${mutation}'`, { encoding: 'utf8' });
        const resultJson = JSON.parse(result);

        if (resultJson.data.resolveReviewThread.thread.isResolved) {
            console.log(`  ✓ Resolved successfully`);
            resolved++;
        } else {
            console.log(`  ✗ Failed to resolve`);
            failed++;
        }
    } catch (error) {
        console.log(`  ✗ Error: ${error.message}`);
        failed++;
    }

    console.log('');
}

console.log('');
console.log('='.repeat(80));
console.log(`SUMMARY: ${resolved} resolved, ${failed} failed out of ${unresolvedThreads.length} total`);
console.log('='.repeat(80));
