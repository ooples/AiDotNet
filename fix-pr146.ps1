# Fix PR #146 - Add exception handling to RegressionBase
$owner = "ooples"
$repo = "AiDotNet"
$branch = "fix/us-bf-006-savemodel-loadmodel-regressionbase"
$filePath = "src/Regression/RegressionBase.cs"

$branchSha = (gh api "repos/$owner/$repo/git/ref/heads/$branch" | ConvertFrom-Json).object.sha
$treeSha = (gh api "repos/$owner/$repo/git/commits/$branchSha" | ConvertFrom-Json).tree.sha
$tree = gh api "repos/$owner/$repo/git/trees/$treeSha" | ConvertFrom-Json
$fileEntry = $tree.tree | Where-Object { $_.path -eq $filePath }
$fileBlobSha = $fileEntry.sha
$blob = gh api "repos/$owner/$repo/git/blobs/$fileBlobSha" | ConvertFrom-Json
$currentContent = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($blob.content))

$saveModelFixed = @'
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

$loadModelFixed = @'
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

$pattern1 = '(?s)public virtual void SaveModel\(string filePath\)\s*\{[^}]*File\.WriteAllBytes\(filePath, data\);[^}]*\}'
$newContent = $currentContent -replace $pattern1, $saveModelFixed

$pattern2 = '(?s)public virtual void LoadModel\(string filePath\)\s*\{[^}]*Deserialize\(data\);[^}]*\}'
$newContent = $newContent -replace $pattern2, $loadModelFixed

$newContentBytes = [System.Text.Encoding]::UTF8.GetBytes($newContent)
$newContentBase64 = [Convert]::ToBase64String($newContentBytes)

$blobJson = @{ content = $newContentBase64; encoding = "base64" } | ConvertTo-Json
$newBlobSha = ($blobJson | gh api "repos/$owner/$repo/git/blobs" -X POST --input - | ConvertFrom-Json).sha

$treeJson = @{ base_tree = $treeSha; tree = @(@{ path = $filePath; mode = "100644"; type = "blob"; sha = $newBlobSha }) } | ConvertTo-Json -Depth 10
$newTreeSha = ($treeJson | gh api "repos/$owner/$repo/git/trees" -X POST --input - | ConvertFrom-Json).sha

$message = @"
fix: add exception handling to RegressionBase SaveModel/LoadModel

Address Copilot review comments:
- Add try-catch blocks for I/O exceptions in SaveModel
- Add directory creation before writing
- Distinguish I/O errors from deserialization errors in LoadModel

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
"@

$commitJson = @{ message = $message; tree = $newTreeSha; parents = @($branchSha) } | ConvertTo-Json -Depth 10
$newCommitSha = ($commitJson | gh api "repos/$owner/$repo/git/commits" -X POST --input - | ConvertFrom-Json).sha

gh api "repos/$owner/$repo/git/refs/heads/$branch" -X PATCH -f sha=$newCommitSha | Out-Null
Write-Host "PR #146 fixed - Commit: $newCommitSha"
