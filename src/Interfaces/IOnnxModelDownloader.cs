namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for downloading ONNX models from remote sources.
/// </summary>
/// <remarks>
/// <para>
/// This interface provides a way to download ONNX models from repositories
/// like HuggingFace Hub or ONNX Model Zoo. It supports progress reporting
/// and caching of downloaded models.
/// </para>
/// <para><b>For Beginners:</b> Instead of manually downloading model files,
/// you can use implementers of this interface to automatically fetch models:
/// <code>
/// var downloader = new OnnxModelDownloader();
/// var modelPath = await downloader.DownloadAsync("openai/whisper-base");
/// var model = new OnnxModel&lt;float&gt;(modelPath);
/// </code>
/// </para>
/// </remarks>
public interface IOnnxModelDownloader
{
    /// <summary>
    /// Downloads an ONNX model from a remote repository.
    /// </summary>
    /// <param name="modelId">The model identifier (e.g., "openai/whisper-base").</param>
    /// <param name="fileName">Optional specific file name within the model repository.</param>
    /// <param name="progress">Optional progress reporter (0.0 to 1.0).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The local file path to the downloaded model.</returns>
    Task<string> DownloadAsync(
        string modelId,
        string? fileName = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Downloads multiple ONNX files from a model repository.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <param name="fileNames">The file names to download.</param>
    /// <param name="progress">Optional progress reporter (0.0 to 1.0).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Dictionary mapping file names to local paths.</returns>
    Task<IReadOnlyDictionary<string, string>> DownloadMultipleAsync(
        string modelId,
        IEnumerable<string> fileNames,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Checks if a model is already cached locally.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <param name="fileName">Optional specific file name.</param>
    /// <returns>The local path if cached, null otherwise.</returns>
    string? GetCachedPath(string modelId, string? fileName = null);

    /// <summary>
    /// Clears the local cache for a specific model or all models.
    /// </summary>
    /// <param name="modelId">Optional model identifier. If null, clears all cached models.</param>
    void ClearCache(string? modelId = null);

    /// <summary>
    /// Gets the total size of the local cache in bytes.
    /// </summary>
    long GetCacheSize();
}

/// <summary>
/// Progress information for model downloads.
/// </summary>
public class ModelDownloadProgress
{
    /// <summary>
    /// Gets or sets the current file being downloaded.
    /// </summary>
    public string CurrentFile { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the number of bytes downloaded for the current file.
    /// </summary>
    public long BytesDownloaded { get; set; }

    /// <summary>
    /// Gets or sets the total bytes for the current file (-1 if unknown).
    /// </summary>
    public long TotalBytes { get; set; } = -1;

    /// <summary>
    /// Gets or sets the overall progress (0.0 to 1.0).
    /// </summary>
    public double OverallProgress { get; set; }

    /// <summary>
    /// Gets or sets the current file progress (0.0 to 1.0).
    /// </summary>
    public double FileProgress => TotalBytes > 0 ? (double)BytesDownloaded / TotalBytes : 0;
}
