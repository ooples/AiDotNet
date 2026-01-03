using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Interfaces;
using Newtonsoft.Json.Linq;

namespace AiDotNet.ModelLoading;

/// <summary>
/// Downloads and caches models from HuggingFace Hub.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> HuggingFace Hub is like a library of pretrained AI models.
///
/// Instead of training models yourself (which requires huge amounts of data and compute),
/// you can download models that others have already trained. This class handles:
///
/// 1. Downloading model files from HuggingFace
/// 2. Caching them locally so you don't re-download every time
/// 3. Loading the weights into your model
///
/// Example usage:
/// ```csharp
/// var loader = new HuggingFaceModelLoader&lt;float&gt;();
///
/// // Download and cache a pretrained VAE
/// var files = await loader.DownloadModelAsync("stabilityai/sd-vae-ft-mse");
///
/// // Load weights into your model
/// var vae = new VAEEncoder&lt;float&gt;();
/// loader.LoadWeights(vae, files["diffusion_pytorch_model.safetensors"]);
/// ```
/// </para>
/// </remarks>
public class HuggingFaceModelLoader<T>
{
    /// <summary>
    /// Default HuggingFace Hub API URL.
    /// </summary>
    private const string HF_API_URL = "https://huggingface.co";

    /// <summary>
    /// HTTP client for API requests.
    /// </summary>
    private readonly HttpClient _httpClient;

    /// <summary>
    /// Local cache directory for downloaded models.
    /// </summary>
    private readonly string _cacheDir;

    /// <summary>
    /// Optional HuggingFace API token for accessing private/gated models.
    /// </summary>
    private readonly string? _apiToken;

    /// <summary>
    /// Whether to log download progress.
    /// </summary>
    private readonly bool _verbose;

    /// <summary>
    /// SafeTensors loader for loading weights.
    /// </summary>
    private readonly SafeTensorsLoader<T> _safeTensorsLoader;

    /// <summary>
    /// Pretrained model loader for applying weights.
    /// </summary>
    private readonly PretrainedModelLoader<T> _pretrainedLoader;

    /// <summary>
    /// Initializes a new instance of the HuggingFaceModelLoader class.
    /// </summary>
    /// <param name="cacheDir">Directory to cache downloaded models. Uses ~/.cache/huggingface/hub by default.</param>
    /// <param name="apiToken">Optional HuggingFace API token for private models.</param>
    /// <param name="verbose">Whether to log download progress.</param>
    public HuggingFaceModelLoader(
        string? cacheDir = null,
        string? apiToken = null,
        bool verbose = false)
    {
        _cacheDir = cacheDir ?? GetDefaultCacheDir();
        _apiToken = apiToken ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        _verbose = verbose;
        _safeTensorsLoader = new SafeTensorsLoader<T>();
        _pretrainedLoader = new PretrainedModelLoader<T>(verbose);

        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.UserAgent.Add(
            new ProductInfoHeaderValue("AiDotNet", "1.0"));

        if (!string.IsNullOrEmpty(_apiToken))
        {
            _httpClient.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", _apiToken);
        }

        // Ensure cache directory exists
        Directory.CreateDirectory(_cacheDir);
    }

    /// <summary>
    /// Gets the default cache directory.
    /// </summary>
    private static string GetDefaultCacheDir()
    {
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(home, ".cache", "huggingface", "hub");
    }

    /// <summary>
    /// Downloads a model from HuggingFace Hub.
    /// </summary>
    /// <param name="repoId">Repository ID (e.g., "stabilityai/sd-vae-ft-mse").</param>
    /// <param name="revision">Git revision/branch (default: "main").</param>
    /// <param name="filePatterns">Optional patterns to filter files (e.g., "*.safetensors").</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Dictionary mapping file names to local paths.</returns>
    public async Task<Dictionary<string, string>> DownloadModelAsync(
        string repoId,
        string revision = "main",
        IEnumerable<string>? filePatterns = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(repoId))
            throw new ArgumentNullException(nameof(repoId));

        var result = new Dictionary<string, string>();
        var patterns = filePatterns?.ToList() ?? new List<string> { "*.safetensors", "*.json" };

        if (_verbose)
        {
            Console.WriteLine($"Downloading model: {repoId}@{revision}");
        }

        // Get list of files in repository
        var files = await GetRepoFilesAsync(repoId, revision, cancellationToken);

        // Filter files by patterns
        var filteredFiles = files.Where(f => MatchesAnyPattern(f, patterns)).ToList();

        if (_verbose)
        {
            Console.WriteLine($"  Found {filteredFiles.Count} files to download");
        }

        // Download each file
        foreach (var fileName in filteredFiles)
        {
            var localPath = await DownloadFileAsync(repoId, revision, fileName, cancellationToken);
            result[fileName] = localPath;
        }

        return result;
    }

    /// <summary>
    /// Gets the list of files in a HuggingFace repository.
    /// </summary>
    private async Task<List<string>> GetRepoFilesAsync(
        string repoId,
        string revision,
        CancellationToken cancellationToken)
    {
        var url = $"{HF_API_URL}/api/models/{repoId}/tree/{revision}";

        try
        {
            var response = await _httpClient.GetStringAsync(url, cancellationToken);
            var files = JArray.Parse(response);

            return files
                .Where(f => f["type"]?.ToString() == "file")
                .Select(f => f["path"]?.ToString() ?? string.Empty)
                .Where(p => !string.IsNullOrEmpty(p))
                .ToList();
        }
        catch (HttpRequestException ex)
        {
            throw new InvalidOperationException(
                $"Failed to get file list for {repoId}: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Downloads a single file from HuggingFace Hub.
    /// </summary>
    /// <param name="repoId">Repository ID.</param>
    /// <param name="revision">Git revision.</param>
    /// <param name="fileName">File name within the repository.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Local path to the downloaded file.</returns>
    public async Task<string> DownloadFileAsync(
        string repoId,
        string revision,
        string fileName,
        CancellationToken cancellationToken = default)
    {
        // Create local path with cache structure
        var repoHash = ComputeHash($"{repoId}@{revision}");
        var localDir = Path.Combine(_cacheDir, $"models--{repoId.Replace("/", "--")}", "snapshots", repoHash);
        var localPath = Path.Combine(localDir, fileName);

        // Check if already cached
        if (File.Exists(localPath))
        {
            if (_verbose)
            {
                Console.WriteLine($"  Using cached: {fileName}");
            }
            return localPath;
        }

        // Ensure directory exists
        var fileDir = Path.GetDirectoryName(localPath);
        if (!string.IsNullOrEmpty(fileDir))
        {
            Directory.CreateDirectory(fileDir);
        }

        // Download file
        var url = $"{HF_API_URL}/{repoId}/resolve/{revision}/{fileName}";

        if (_verbose)
        {
            Console.WriteLine($"  Downloading: {fileName}");
        }

        try
        {
            using var response = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? -1;

            // Download to temp file first
            var tempPath = localPath + ".tmp";

            using (var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken))
            using (var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, true))
            {
                if (_verbose && totalBytes > 0)
                {
                    await CopyWithProgressAsync(contentStream, fileStream, totalBytes, fileName, cancellationToken);
                }
                else
                {
                    await contentStream.CopyToAsync(fileStream, cancellationToken);
                }
            }

            // Move temp to final location
            File.Move(tempPath, localPath, overwrite: true);
        }
        catch (HttpRequestException ex)
        {
            throw new InvalidOperationException(
                $"Failed to download {fileName}: {ex.Message}", ex);
        }

        return localPath;
    }

    /// <summary>
    /// Copies stream with progress reporting.
    /// </summary>
    private async Task CopyWithProgressAsync(
        Stream source,
        Stream destination,
        long totalBytes,
        string fileName,
        CancellationToken cancellationToken)
    {
        var buffer = new byte[81920];
        long bytesRead = 0;
        int lastPercent = -1;

        while (true)
        {
            var count = await source.ReadAsync(buffer, cancellationToken);
            if (count == 0)
                break;

            await destination.WriteAsync(buffer.AsMemory(0, count), cancellationToken);
            bytesRead += count;

            var percent = (int)(bytesRead * 100 / totalBytes);
            if (percent != lastPercent && percent % 10 == 0)
            {
                Console.WriteLine($"    {fileName}: {percent}% ({bytesRead / 1024 / 1024:N0} MB / {totalBytes / 1024 / 1024:N0} MB)");
                lastPercent = percent;
            }
        }
    }

    /// <summary>
    /// Loads weights from a downloaded SafeTensors file into a model.
    /// </summary>
    /// <param name="model">The model to load weights into.</param>
    /// <param name="weightsPath">Path to the .safetensors file.</param>
    /// <param name="mapping">Optional weight mapping.</param>
    /// <param name="strict">If true, fails when weights can't be loaded.</param>
    /// <returns>Load result with statistics.</returns>
    public LoadResult LoadWeights(
        IWeightLoadable<T> model,
        string weightsPath,
        Func<string, string?>? mapping = null,
        bool strict = false)
    {
        return _pretrainedLoader.LoadWeights(model, weightsPath, mapping, strict);
    }

    /// <summary>
    /// Loads weights using a WeightMapping instance.
    /// </summary>
    public LoadResult LoadWeights(
        IWeightLoadable<T> model,
        string weightsPath,
        WeightMapping mapping,
        bool strict = false)
    {
        return _pretrainedLoader.LoadWeights(model, weightsPath, mapping, strict);
    }

    /// <summary>
    /// Downloads a model and loads its weights in one operation.
    /// </summary>
    /// <param name="model">The model to load weights into.</param>
    /// <param name="repoId">HuggingFace repository ID.</param>
    /// <param name="weightsFile">Name of the weights file (e.g., "diffusion_pytorch_model.safetensors").</param>
    /// <param name="mapping">Optional weight mapping.</param>
    /// <param name="revision">Git revision (default: "main").</param>
    /// <param name="strict">If true, fails when weights can't be loaded.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Load result with statistics.</returns>
    public async Task<LoadResult> DownloadAndLoadAsync(
        IWeightLoadable<T> model,
        string repoId,
        string weightsFile = "diffusion_pytorch_model.safetensors",
        Func<string, string?>? mapping = null,
        string revision = "main",
        bool strict = false,
        CancellationToken cancellationToken = default)
    {
        var localPath = await DownloadFileAsync(repoId, revision, weightsFile, cancellationToken);
        return LoadWeights(model, localPath, mapping, strict);
    }

    /// <summary>
    /// Gets the local cache path for a repository.
    /// </summary>
    /// <param name="repoId">Repository ID.</param>
    /// <param name="revision">Git revision.</param>
    /// <returns>Local cache directory path.</returns>
    public string GetCachePath(string repoId, string revision = "main")
    {
        var repoHash = ComputeHash($"{repoId}@{revision}");
        return Path.Combine(_cacheDir, $"models--{repoId.Replace("/", "--")}", "snapshots", repoHash);
    }

    /// <summary>
    /// Checks if a model is already cached locally.
    /// </summary>
    /// <param name="repoId">Repository ID.</param>
    /// <param name="weightsFile">Name of the weights file to check.</param>
    /// <param name="revision">Git revision.</param>
    /// <returns>True if the weights file exists in cache.</returns>
    public bool IsCached(string repoId, string weightsFile, string revision = "main")
    {
        var cachePath = GetCachePath(repoId, revision);
        return File.Exists(Path.Combine(cachePath, weightsFile));
    }

    /// <summary>
    /// Clears the cache for a specific model.
    /// </summary>
    /// <param name="repoId">Repository ID.</param>
    public void ClearCache(string repoId)
    {
        var cacheDir = Path.Combine(_cacheDir, $"models--{repoId.Replace("/", "--")}");
        if (Directory.Exists(cacheDir))
        {
            Directory.Delete(cacheDir, recursive: true);

            if (_verbose)
            {
                Console.WriteLine($"Cleared cache for {repoId}");
            }
        }
    }

    /// <summary>
    /// Clears all cached models.
    /// </summary>
    public void ClearAllCache()
    {
        if (Directory.Exists(_cacheDir))
        {
            Directory.Delete(_cacheDir, recursive: true);
            Directory.CreateDirectory(_cacheDir);

            if (_verbose)
            {
                Console.WriteLine("Cleared all model cache");
            }
        }
    }

    /// <summary>
    /// Computes a short hash for cache directory naming.
    /// </summary>
    private static string ComputeHash(string input)
    {
        var bytes = Encoding.UTF8.GetBytes(input);
        var hashBytes = SHA256.HashData(bytes);
        return Convert.ToHexString(hashBytes).Substring(0, 12).ToLowerInvariant();
    }

    /// <summary>
    /// Checks if a file name matches any of the given patterns.
    /// </summary>
    private static bool MatchesAnyPattern(string fileName, List<string> patterns)
    {
        foreach (var pattern in patterns)
        {
            if (MatchesWildcard(fileName, pattern))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Simple wildcard matching (supports * only).
    /// </summary>
    private static bool MatchesWildcard(string input, string pattern)
    {
        if (pattern == "*")
            return true;

        if (pattern.StartsWith("*"))
        {
            var suffix = pattern.Substring(1);
            return input.EndsWith(suffix, StringComparison.OrdinalIgnoreCase);
        }

        if (pattern.EndsWith("*"))
        {
            var prefix = pattern.Substring(0, pattern.Length - 1);
            return input.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);
        }

        return string.Equals(input, pattern, StringComparison.OrdinalIgnoreCase);
    }
}
