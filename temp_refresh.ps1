$allThreads = @()
$hasNextPage = $true
$cursor = $null

while ($hasNextPage) {
    if ($cursor) {
        $query = 'query { repository(owner: "ooples", name: "AiDotNet") { pullRequest(number: 304) { reviewThreads(first: 100, after: "' + $cursor + '") { nodes { id isResolved isOutdated comments(first: 10) { nodes { id body path position originalPosition createdAt author { login } } } } pageInfo { hasNextPage endCursor } } } } }'
    } else {
        $query = 'query { repository(owner: "ooples", name: "AiDotNet") { pullRequest(number: 304) { reviewThreads(first: 100) { nodes { id isResolved isOutdated comments(first: 10) { nodes { id body path position originalPosition createdAt author { login } } } } pageInfo { hasNextPage endCursor } } } } }'
    }
    
    $result = gh api graphql -f query=$query | ConvertFrom-Json
    $threads = $result.data.repository.pullRequest.reviewThreads.nodes
    $allThreads += $threads
    $hasNextPage = $result.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage
    $cursor = $result.data.repository.pullRequest.reviewThreads.pageInfo.endCursor
}

$unresolvedThreads = $allThreads | Where-Object { -not $_.isResolved -and -not $_.isOutdated }
Write-Host "Total unresolved threads: $($unresolvedThreads.Count)"

$unresolvedThreads | ConvertTo-Json -Depth 10 | Out-File -FilePath "all_unresolved_pr304.json" -Encoding utf8

# Show next 10 threads  
for ($i = 0; $i -lt [Math]::Min(10, $unresolvedThreads.Count); $i++) {
    $thread = $unresolvedThreads[$i]
    $firstComment = $thread.comments.nodes[0]
    Write-Host "`n#$($i+1): $($firstComment.path) - $($firstComment.createdAt)"
    Write-Host "Thread ID: $($thread.id)"
}
