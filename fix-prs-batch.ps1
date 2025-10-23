$owner = "ooples"
$repo = "AiDotNet"

$prs = @(
    @{ n = 147; b = "fix/us-bf-007-savemodel-loadmodel-timeseriesmodelbase"; f = "src/TimeSeries/TimeSeriesModelBase.cs" }
    @{ n = 149; b = "fix/us-bf-002-savemodel-loadmodel-neuralnetworkmodel"; f = "src/Models/NeuralNetworkModel.cs" }
    @{ n = 153; b = "fix/us-bf-009-savemodel-loadmodel-nonlinearregressionbase"; f = "src/Regression/NonLinearRegressionBase.cs" }
    @{ n = 154; b = "fix/us-bf-001-savemodel-loadmodel-modelindividual"; f = "src/Genetics/ModelIndividual.cs" }
)

$saveFix = @'
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

$loadFix = @'
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
            throw new InvalidOperationException($"Failed to deserialize model from file '{filePath}'. The file may be corrupted or incompatible: {ex.Message}", ex);
        }
    }
'@

foreach ($pr in $prs) {
    Write-Host "PR #$($pr.n)..."

    $sha1 = (gh api "repos/$owner/$repo/git/ref/heads/$($pr.b)" | ConvertFrom-Json).object.sha
    $sha2 = (gh api "repos/$owner/$repo/git/commits/$sha1" | ConvertFrom-Json).tree.sha
    $tree = gh api "repos/$owner/$repo/git/trees/$sha2" | ConvertFrom-Json
    $entry = $tree.tree | Where-Object { $_.path -eq $pr.f }

    if (!$entry) { Write-Host "  Skip - file not in tree"; continue }

    $blob = gh api "repos/$owner/$repo/git/blobs/$($entry.sha)" | ConvertFrom-Json
    $content = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($blob.content))

    $new = $content -replace '(?s)public virtual void SaveModel\(string filePath\)\s*\{[^}]*File\.WriteAllBytes\(filePath, data\);[^}]*\}', $saveFix
    $new = $new -replace '(?s)public virtual void LoadModel\(string filePath\)\s*\{[^}]*Deserialize\(data\);[^}]*\}', $loadFix

    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($new))
    $blobSha = ((@{ content = $b64; encoding = "base64" } | ConvertTo-Json) | gh api "repos/$owner/$repo/git/blobs" -X POST --input - | ConvertFrom-Json).sha
    $treeSha = ((@{ base_tree = $sha2; tree = @(@{ path = $pr.f; mode = "100644"; type = "blob"; sha = $blobSha }) } | ConvertTo-Json -Depth 10) | gh api "repos/$owner/$repo/git/trees" -X POST --input - | ConvertFrom-Json).sha

    $msg = "fix: add exception handling`n`nAddress Copilot review comments.`n`n🤖 Generated with [Claude Code](https://claude.com/claude-code)`n`nCo-Authored-By: Claude <noreply@anthropic.com>"
    $commitSha = ((@{ message = $msg; tree = $treeSha; parents = @($sha1) } | ConvertTo-Json -Depth 10) | gh api "repos/$owner/$repo/git/commits" -X POST --input - | ConvertFrom-Json).sha

    gh api "repos/$owner/$repo/git/refs/heads/$($pr.b)" -X PATCH -f sha=$commitSha | Out-Null
    Write-Host "  Done: $commitSha"
}
