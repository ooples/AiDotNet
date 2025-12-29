using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json;
using AiDotNet.Interfaces;

namespace AiDotNet.Onnx;

/// <summary>
/// Downloads ONNX models from HuggingFace Hub and other repositories.
/// </summary>
/// <remarks>
/// <para>
/// This class provides functionality to download pre-trained ONNX models from
/// HuggingFace Hub with automatic caching, progress reporting, and resume support.
/// </para>
/// <para><b>For Beginners:</b> Instead of manually downloading model files, use this class:
/// <code>
/// var downloader = new OnnxModelDownloader();
///
/// // Download Whisper model from HuggingFace
/// var modelPath = await downloader.DownloadAsync(
///     "openai/whisper-base",
///     "model.onnx",
///     progress: new Progress&lt;double&gt;(p => Console.WriteLine($"{p:P0}")));
///
/// var model = new OnnxModel&lt;float&gt;(modelPath);
/// </code>
/// </para>
/// </remarks>
public class OnnxModelDownloader : IOnnxModelDownloader
{
    private readonly string _cacheDirectory;
    private readonly HttpClient _httpClient;
    private readonly bool _ownsHttpClient;

    /// <summary>
    /// The base URL for HuggingFace Hub.
    /// </summary>
    public const string HuggingFaceBaseUrl = "https://huggingface.co";

    /// <summary>
    /// Creates a new OnnxModelDownloader with default settings.
    /// </summary>
    /// <param name="cacheDirectory">Optional custom cache directory. Defaults to ~/.aidotnet/models</param>
    public OnnxModelDownloader(string? cacheDirectory = null)
        : this(new HttpClient(), cacheDirectory, ownsHttpClient: true)
    {
    }

    /// <summary>
    /// Creates a new OnnxModelDownloader with a custom HttpClient.
    /// </summary>
    /// <param name="httpClient">The HttpClient to use for downloads.</param>
    /// <param name="cacheDirectory">Optional custom cache directory.</param>
    /// <param name="ownsHttpClient">Whether to dispose the HttpClient when this downloader is disposed.</param>
    public OnnxModelDownloader(HttpClient httpClient, string? cacheDirectory = null, bool ownsHttpClient = false)
    {
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
        _ownsHttpClient = ownsHttpClient;

        if (string.IsNullOrWhiteSpace(cacheDirectory))
        {
            var userHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            _cacheDirectory = Path.Combine(userHome, ".aidotnet", "models");
        }
        else
        {
            _cacheDirectory = cacheDirectory;
        }

        Directory.CreateDirectory(_cacheDirectory);
    }

    /// <inheritdoc/>
    public async Task<string> DownloadAsync(
        string modelId,
        string? fileName = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(modelId);

        // Check cache first
        var cachedPath = GetCachedPath(modelId, fileName);
        if (cachedPath is not null && File.Exists(cachedPath))
        {
            progress?.Report(1.0);
            return cachedPath;
        }

        // Determine file to download
        var targetFileName = fileName ?? "model.onnx";
        var downloadUrl = BuildDownloadUrl(modelId, targetFileName);

        // Create cache directory for this model
        var modelCacheDir = GetModelCacheDirectory(modelId);
        Directory.CreateDirectory(modelCacheDir);

        var localPath = Path.Combine(modelCacheDir, targetFileName);

        // Download the file
        await DownloadFileAsync(downloadUrl, localPath, progress, cancellationToken);

        return localPath;
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyDictionary<string, string>> DownloadMultipleAsync(
        string modelId,
        IEnumerable<string> fileNames,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(modelId);
        ArgumentNullException.ThrowIfNull(fileNames);

        var fileList = fileNames.ToList();
        if (fileList.Count == 0)
        {
            throw new ArgumentException("At least one file name must be specified.", nameof(fileNames));
        }

        var results = new Dictionary<string, string>();
        int completed = 0;

        foreach (var fileName in fileList)
        {
            var fileProgress = progress is null
                ? null
                : new Progress<double>(p =>
                {
                    var totalProgress = (completed + p) / fileList.Count;
                    progress.Report(totalProgress);
                });

            var localPath = await DownloadAsync(modelId, fileName, fileProgress, cancellationToken);
            results[fileName] = localPath;
            completed++;
        }

        return results;
    }

    /// <inheritdoc/>
    public string? GetCachedPath(string modelId, string? fileName = null)
    {
        var modelCacheDir = GetModelCacheDirectory(modelId);
        var targetFileName = fileName ?? "model.onnx";
        var localPath = Path.Combine(modelCacheDir, targetFileName);

        return File.Exists(localPath) ? localPath : null;
    }

    /// <inheritdoc/>
    public void ClearCache(string? modelId = null)
    {
        if (modelId is null)
        {
            // Clear entire cache
            if (Directory.Exists(_cacheDirectory))
            {
                Directory.Delete(_cacheDirectory, recursive: true);
                Directory.CreateDirectory(_cacheDirectory);
            }
        }
        else
        {
            // Clear specific model cache
            var modelCacheDir = GetModelCacheDirectory(modelId);
            if (Directory.Exists(modelCacheDir))
            {
                Directory.Delete(modelCacheDir, recursive: true);
            }
        }
    }

    /// <inheritdoc/>
    public long GetCacheSize()
    {
        if (!Directory.Exists(_cacheDirectory))
            return 0;

        return Directory.EnumerateFiles(_cacheDirectory, "*", SearchOption.AllDirectories)
            .Sum(file => new FileInfo(file).Length);
    }

    /// <summary>
    /// Lists all cached models.
    /// </summary>
    /// <returns>A list of cached model IDs.</returns>
    public IReadOnlyList<string> ListCachedModels()
    {
        if (!Directory.Exists(_cacheDirectory))
            return [];

        return Directory.GetDirectories(_cacheDirectory)
            .Select(dir => Path.GetFileName(dir))
            .Where(name => !string.IsNullOrEmpty(name))
            .ToList()!;
    }

    /// <summary>
    /// Gets the files cached for a specific model.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>A list of cached file names.</returns>
    public IReadOnlyList<string> GetCachedFiles(string modelId)
    {
        var modelCacheDir = GetModelCacheDirectory(modelId);

        if (!Directory.Exists(modelCacheDir))
            return [];

        return Directory.GetFiles(modelCacheDir)
            .Select(Path.GetFileName)
            .Where(name => !string.IsNullOrEmpty(name))
            .ToList()!;
    }

    private string GetModelCacheDirectory(string modelId)
    {
        // Convert model ID to safe directory name
        var safeName = modelId.Replace("/", "__").Replace("\\", "__");
        return Path.Combine(_cacheDirectory, safeName);
    }

    private static string BuildDownloadUrl(string modelId, string fileName)
    {
        // HuggingFace Hub URL format: https://huggingface.co/{model_id}/resolve/main/{file_name}
        var encodedFileName = Uri.EscapeDataString(fileName);
        return $"{HuggingFaceBaseUrl}/{modelId}/resolve/main/{encodedFileName}";
    }

    private async Task DownloadFileAsync(
        string url,
        string localPath,
        IProgress<double>? progress,
        CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("*/*"));

        // Check for existing partial download
        var tempPath = localPath + ".partial";
        long existingLength = 0;

        if (File.Exists(tempPath))
        {
            existingLength = new FileInfo(tempPath).Length;
            request.Headers.Range = new RangeHeaderValue(existingLength, null);
        }

        using var response = await _httpClient.SendAsync(
            request,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        // Handle non-resumable download
        if (response.StatusCode == System.Net.HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            existingLength = 0;
            File.Delete(tempPath);
        }
        else
        {
            response.EnsureSuccessStatusCode();
        }

        var totalBytes = response.Content.Headers.ContentLength;
        if (totalBytes.HasValue && existingLength > 0)
        {
            totalBytes += existingLength;
        }

        using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);

        var mode = existingLength > 0 ? FileMode.Append : FileMode.Create;
        using var fileStream = new FileStream(tempPath, mode, FileAccess.Write, FileShare.None, 8192, true);

        var buffer = new byte[8192];
        long totalBytesRead = existingLength;
        int bytesRead;

        while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
            totalBytesRead += bytesRead;

            if (totalBytes.HasValue && totalBytes.Value > 0)
            {
                progress?.Report((double)totalBytesRead / totalBytes.Value);
            }
        }

        // Move completed download to final location
        fileStream.Close();

        if (File.Exists(localPath))
        {
            File.Delete(localPath);
        }

        File.Move(tempPath, localPath);
        progress?.Report(1.0);
    }

    /// <summary>
    /// Downloads a model from a direct URL (not HuggingFace).
    /// </summary>
    /// <param name="url">The direct URL to the ONNX model file.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The local path to the downloaded model.</returns>
    public async Task<string> DownloadFromUrlAsync(
        string url,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(url);

        // Create a cache key from the URL hash
        var urlHash = Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(url)))[..16];
        var fileName = Path.GetFileName(new Uri(url).LocalPath);
        if (string.IsNullOrEmpty(fileName))
        {
            fileName = "model.onnx";
        }

        var cacheDir = Path.Combine(_cacheDirectory, "_urls", urlHash);
        Directory.CreateDirectory(cacheDir);

        var localPath = Path.Combine(cacheDir, fileName);

        if (File.Exists(localPath))
        {
            progress?.Report(1.0);
            return localPath;
        }

        await DownloadFileAsync(url, localPath, progress, cancellationToken);
        return localPath;
    }
}

/// <summary>
/// Common ONNX model repositories and their identifiers.
/// </summary>
public static class OnnxModelRepositories
{
    /// <summary>
    /// OpenAI Whisper speech recognition models.
    /// </summary>
    public static class Whisper
    {
        /// <summary>Whisper tiny model (39M parameters).</summary>
        public const string Tiny = "openai/whisper-tiny";

        /// <summary>Whisper base model (74M parameters).</summary>
        public const string Base = "openai/whisper-base";

        /// <summary>Whisper small model (244M parameters).</summary>
        public const string Small = "openai/whisper-small";

        /// <summary>Whisper medium model (769M parameters).</summary>
        public const string Medium = "openai/whisper-medium";

        /// <summary>Whisper large model (1.5B parameters).</summary>
        public const string Large = "openai/whisper-large-v3";
    }

    /// <summary>
    /// Text-to-speech models.
    /// </summary>
    public static class Tts
    {
        /// <summary>FastSpeech2 ONNX model.</summary>
        public const string FastSpeech2 = "microsoft/speecht5_tts";

        /// <summary>HiFi-GAN vocoder.</summary>
        public const string HiFiGan = "facebook/hifigan";
    }

    /// <summary>
    /// Audio generation models.
    /// </summary>
    public static class AudioGen
    {
        /// <summary>AudioGen small model.</summary>
        public const string Small = "facebook/audiogen-medium";

        /// <summary>MusicGen small model.</summary>
        public const string MusicGenSmall = "facebook/musicgen-small";
    }
}
