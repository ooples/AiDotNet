# Fix all remaining PRs with the same exception handling pattern
$owner = "ooples"
$repo = "AiDotNet"

$prsToFix = @(
    @{ number = 147; branch = "fix/us-bf-007-savemodel-loadmodel-timeseriesmodelbase"; file = "src/TimeSeries/TimeSeriesModelBase.cs" }
    @{ number = 149; branch = "fix/us-bf-002-savemodel-loadmodel-neuralnetworkmodel"; file = "src/NeuralNetworks/NeuralNetworkModel.cs" }
    @{ number = 153; branch = "fix/us-bf-009-savemodel-loadmodel-nonlinearregressionbase"; file = "src/Regression/NonLinearRegressionBase.cs" }
    @{ number = 154; branch = "fix/us-bf-001-savemodel-loadmodel-modelindividual"; file = "src/Genetics/ModelIndividual.cs" }
)

$saveModelTemplate = @'
public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must not be null or empty.", nameof(filePath));
        }

        try
        {
            var data = Serialize();
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }
            File.WriteAllBytes(filePath, data);
        }
        catch (IOException ex)
        {
            throw new InvalidOperationException($"Failed to save model to '{filePath}': {ex.Message}", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new InvalidOperationException($"Access denied when saving model to '{filePath}': {ex.Message}", ex);
        }
        catch (System.Security.SecurityException ex)
        {
            throw new InvalidOperationException($"Security error when saving model to '{filePath}': {ex.Message}", ex);
        }
    }
'@

$loadModelTemplate = @'
public virtual void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must not be null or empty.", nameof(filePath));
        }

        try
        {
            var data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }
        catch (FileNotFoundException ex)
        {
            throw new FileNotFoundException($"The specified model file does not exist: {filePath}", filePath, ex);
        }
        catch (IOException ex)
        {
            throw new InvalidOperationException($"File I/O error while loading model from '{filePath}': {ex.Message}", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new InvalidOperationException($"Access denied when loading model from '{filePath}': {ex.Message}", ex);
        }
        catch (System.Security.SecurityException ex)
        {
            throw new InvalidOperationException($"Security error when loading model from '{filePath}': {ex.Message}", ex);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to deserialize model from file '{filePath}': {ex.Message}", ex);
        }
    }
'@

foreach ($pr in $prsToFix) {
    Write-Host "Fixing PR #$($pr.number)..."

    $branchSha = (gh api "repos/$owner/$repo/git/ref/heads/$($pr.branch)" | ConvertFrom-Json).object.sha
    $treeSha = (gh api "repos/$owner/$repo/git/commits/$branchSha" | ConvertFrom-Json).tree.sha
    $tree = gh api "repos/$owner/$repo/git/trees/$treeSha" | ConvertFrom-Json
    $fileEntry = $tree.tree | Where-Object { $_.path -eq $pr.file }

    if (!$fileEntry) {
        Write-Host "File not found: $($pr.file)"
        continue
    }

    $fileBlobSha = $fileEntry.sha
    $blob = gh api "repos/$owner/$repo/git/blobs/$fileBlobSha" | ConvertFrom-Json
    $currentContent = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($blob.content))

    $pattern1 = '(?s)public virtual void SaveModel\(string filePath\)\s*\{[^}]*File\.WriteAllBytes\(filePath, data\);[^}]*\}'
    $newContent = $currentContent -replace $pattern1, $saveModelTemplate

    $pattern2 = '(?s)public virtual void LoadModel\(string filePath\)\s*\{[^}]*Deserialize\(data\);[^}]*\}'
    $newContent = $newContent -replace $pattern2, $loadModelTemplate

    $newContentBytes = [System.Text.Encoding]::UTF8.GetBytes($newContent)
    $newContentBase64 = [Convert]::ToBase64String($newContentBytes)

    $blobJson = @{ content = $newContentBase64; encoding = "base64" } | ConvertTo-Json
    $newBlobSha = ($blobJson | gh api "repos/$owner/$repo/git/blobs" -X POST --input - | ConvertFrom-Json).sha

    $treeJson = @{ base_tree = $treeSha; tree = @(@{ path = $pr.file; mode = "100644"; type = "blob"; sha = $newBlobSha }) } | ConvertTo-Json -Depth 10
    $newTreeSha = ($treeJson | gh api "repos/$owner/$repo/git/trees" -X POST --input - | ConvertFrom-Json).sha

    $message = @"
fix: add exception handling to SaveModel/LoadModel

Address Copilot review comments with production-ready exception handling.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
"@

    $commitJson = @{ message = $message; tree = $newTreeSha; parents = @($branchSha) } | ConvertTo-Json -Depth 10
    $newCommitSha = ($commitJson | gh api "repos/$owner/$repo/git/commits" -X POST --input - | ConvertFrom-Json).sha

    gh api "repos/$owner/$repo/git/refs/heads/$($pr.branch)" -X PATCH -f sha=$newCommitSha | Out-Null
    Write-Host "  Fixed - Commit: $newCommitSha"
}

Write-Host "All SaveModel/LoadModel PRs fixed!"
