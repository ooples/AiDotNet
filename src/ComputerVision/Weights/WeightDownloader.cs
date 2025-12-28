using System.Net.Http;

namespace AiDotNet.ComputerVision.Weights;

/// <summary>
/// Downloads and caches pre-trained model weights from remote URLs.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Pre-trained weights are model parameters that have been
/// trained on large datasets (like COCO or ImageNet). Instead of training from scratch,
/// you can download these weights and either use them directly or fine-tune them on
/// your own data. This class handles downloading and caching these weight files.</para>
/// </remarks>
public class WeightDownloader
{
    private static readonly HttpClient _httpClient = new HttpClient();
    private readonly string _cacheDirectory;

    /// <summary>
    /// Creates a new weight downloader with the default cache directory.
    /// </summary>
    public WeightDownloader() : this(GetDefaultCacheDirectory())
    {
    }

    /// <summary>
    /// Creates a new weight downloader with a custom cache directory.
    /// </summary>
    /// <param name="cacheDirectory">Directory to store downloaded weights.</param>
    public WeightDownloader(string cacheDirectory)
    {
        _cacheDirectory = cacheDirectory;
        Directory.CreateDirectory(_cacheDirectory);
    }

    /// <summary>
    /// Gets the default cache directory for storing weights.
    /// </summary>
    /// <returns>The default cache directory path.</returns>
    public static string GetDefaultCacheDirectory()
    {
        var appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return Path.Combine(appData, "AiDotNet", "weights");
    }

    /// <summary>
    /// Downloads weights if not already cached.
    /// </summary>
    /// <param name="url">URL to download from.</param>
    /// <param name="fileName">Local filename to save as.</param>
    /// <param name="progress">Optional progress callback (0.0 to 1.0).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Path to the downloaded or cached weight file.</returns>
    public async Task<string> DownloadIfNeededAsync(
        string url,
        string fileName,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var localPath = Path.Combine(_cacheDirectory, fileName);

        if (File.Exists(localPath))
        {
            return localPath;
        }

        await DownloadAsync(url, localPath, progress, cancellationToken);
        return localPath;
    }

    /// <summary>
    /// Downloads a weight file from a URL.
    /// </summary>
    /// <param name="url">URL to download from.</param>
    /// <param name="localPath">Local path to save to.</param>
    /// <param name="progress">Optional progress callback.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task DownloadAsync(
        string url,
        string localPath,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var tempPath = localPath + ".downloading";

        try
        {
            using var response = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? -1L;
            var downloadedBytes = 0L;

            using var contentStream = await response.Content.ReadAsStreamAsync();
            using var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);

            var buffer = new byte[8192];
            int bytesRead;

            while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length, cancellationToken)) > 0)
            {
                await fileStream.WriteAsync(buffer, 0, bytesRead, cancellationToken);
                downloadedBytes += bytesRead;

                if (totalBytes > 0 && progress is not null)
                {
                    progress.Report((double)downloadedBytes / totalBytes);
                }
            }

            // Move completed download to final location
            File.Move(tempPath, localPath);
        }
        finally
        {
            // Clean up temp file if download failed
            if (File.Exists(tempPath))
            {
                try { File.Delete(tempPath); } catch { /* Ignore cleanup errors */ }
            }
        }
    }

    /// <summary>
    /// Checks if weights are already cached.
    /// </summary>
    /// <param name="fileName">The filename to check for.</param>
    /// <returns>True if the file exists in cache.</returns>
    public bool IsCached(string fileName)
    {
        var localPath = Path.Combine(_cacheDirectory, fileName);
        return File.Exists(localPath);
    }

    /// <summary>
    /// Gets the cached path for a weight file.
    /// </summary>
    /// <param name="fileName">The filename.</param>
    /// <returns>Full path to the cached file (may not exist).</returns>
    public string GetCachePath(string fileName)
    {
        return Path.Combine(_cacheDirectory, fileName);
    }

    /// <summary>
    /// Removes a weight file from cache.
    /// </summary>
    /// <param name="fileName">The filename to remove.</param>
    /// <returns>True if the file was removed.</returns>
    public bool RemoveFromCache(string fileName)
    {
        var localPath = Path.Combine(_cacheDirectory, fileName);
        if (File.Exists(localPath))
        {
            File.Delete(localPath);
            return true;
        }
        return false;
    }

    /// <summary>
    /// Clears all cached weights.
    /// </summary>
    public void ClearCache()
    {
        if (Directory.Exists(_cacheDirectory))
        {
            foreach (var file in Directory.GetFiles(_cacheDirectory))
            {
                try { File.Delete(file); } catch { /* Ignore errors */ }
            }
        }
    }

    /// <summary>
    /// Gets the total size of cached weights in bytes.
    /// </summary>
    /// <returns>Total cache size in bytes.</returns>
    public long GetCacheSize()
    {
        if (!Directory.Exists(_cacheDirectory)) return 0;

        return Directory.GetFiles(_cacheDirectory)
            .Select(f => new FileInfo(f).Length)
            .Sum();
    }
}

/// <summary>
/// Registry of pre-trained model weights with their download URLs.
/// </summary>
public static class PretrainedRegistry
{
    // Note: These are placeholder URLs. In production, these would point to actual weight files.
    // Users can also provide their own URLs via the options.

    private static readonly Dictionary<string, string> _weightUrls = new()
    {
        // YOLO weights (placeholder URLs - would need actual hosting)
        ["yolov8n-coco"] = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        ["yolov8s-coco"] = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        ["yolov8m-coco"] = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        ["yolov8l-coco"] = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        ["yolov8x-coco"] = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",

        // ResNet backbones (ImageNet pretrained)
        ["resnet50-imagenet"] = "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        ["resnet101-imagenet"] = "https://download.pytorch.org/models/resnet101-63fe2227.pth",

        // Swin Transformer backbones
        ["swin-t-imagenet"] = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        ["swin-s-imagenet"] = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",

        // DETR weights
        ["detr-resnet50-coco"] = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
        ["detr-resnet101-coco"] = "https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth",

        // Text detection
        ["craft-general"] = "https://github.com/clovaai/CRAFT-pytorch/releases/download/v1.0/craft_mlt_25k.pth",
        ["dbnet-general"] = "https://github.com/MhLiao/DB/releases/download/v1.0/db_resnet50-pretrained.pth",

        // OCR recognition
        ["crnn-english"] = "https://github.com/clovaai/deep-text-recognition-benchmark/releases/download/v1.0/crnn.pth",
    };

    /// <summary>
    /// Gets the URL for a pretrained model.
    /// </summary>
    /// <param name="modelKey">The model key (e.g., "yolov8m-coco").</param>
    /// <returns>The download URL, or null if not found.</returns>
    public static string? GetUrl(string modelKey)
    {
        return _weightUrls.TryGetValue(modelKey, out var url) ? url : null;
    }

    /// <summary>
    /// Gets all available pretrained model keys.
    /// </summary>
    /// <returns>Collection of model keys.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return _weightUrls.Keys;
    }

    /// <summary>
    /// Registers a custom weight URL.
    /// </summary>
    /// <param name="modelKey">The model key.</param>
    /// <param name="url">The download URL.</param>
    public static void Register(string modelKey, string url)
    {
        _weightUrls[modelKey] = url;
    }

    /// <summary>
    /// Generates the standard model key for detection models.
    /// </summary>
    /// <param name="architecture">The detection architecture.</param>
    /// <param name="size">The model size.</param>
    /// <param name="dataset">The training dataset (default: "coco").</param>
    /// <returns>The model key.</returns>
    public static string GetDetectionModelKey(
        Models.Options.DetectionArchitecture architecture,
        Models.Options.ModelSize size,
        string dataset = "coco")
    {
        var archName = architecture.ToString().ToLowerInvariant();
        var sizeSuffix = size switch
        {
            Models.Options.ModelSize.Nano => "n",
            Models.Options.ModelSize.Small => "s",
            Models.Options.ModelSize.Medium => "m",
            Models.Options.ModelSize.Large => "l",
            Models.Options.ModelSize.XLarge => "x",
            _ => "m"
        };

        return $"{archName}{sizeSuffix}-{dataset}";
    }
}
