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
    Write-Host "Fetched $($threads.Count) threads, hasNextPage: $hasNextPage"
}

$unresolvedThreads = $allThreads | Where-Object { -not $_.isResolved -and -not $_.isOutdated }
Write-Host "`nTotal unresolved threads: $($unresolvedThreads.Count)"

$unresolvedThreads | ConvertTo-Json -Depth 10 | Out-File -FilePath "all_unresolved_threads.json" -Encoding utf8
Write-Host "`nFirst 3 unresolved threads:"
$unresolvedThreads | Select-Object -First 3 | ForEach-Object {
    $firstComment = $_.comments.nodes[0]
    Write-Host "`nThread ID: $($_.id)"
    Write-Host "File: $($firstComment.path)"
    Write-Host "Position: $($firstComment.position)"
    Write-Host "Comment preview: $($firstComment.body.Substring(0, [Math]::Min(200, $firstComment.body.Length)))..."
}
