using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Loads CLIP models from HuggingFace Hub or local directories.
/// </summary>
/// <remarks>
/// <para>
/// This loader handles downloading and caching CLIP model files from HuggingFace,
/// including ONNX model files for image and text encoders, tokenizer files,
/// and model configuration.
/// </para>
/// <para><b>For Beginners:</b> Instead of manually downloading files, this loader
/// automatically fetches CLIP models from HuggingFace Hub:
///
/// <code>
/// // Load a pretrained CLIP model
/// var clip = await ClipModelLoader.FromPretrainedAsync&lt;float&gt;(
///     "openai/clip-vit-base-patch32");
///
/// // Use the model
/// var textEmb = clip.EncodeText("a photo of a cat");
/// var imageEmb = clip.EncodeImage(imageTensor);
/// var similarity = clip.CalculateSimilarity(textEmb, imageEmb);
/// </code>
///
/// The model files are cached locally so subsequent loads are fast.
/// </para>
/// </remarks>
public static class ClipModelLoader
{
    private static readonly HttpClient HttpClient = new HttpClient();
    private const string HuggingFaceHubUrl = "https://huggingface.co";
    private const string DefaultCacheDir = ".cache/huggingface/clip";

    /// <summary>
    /// Validates that a combined path does not escape the base directory (path traversal protection).
    /// </summary>
    /// <param name="baseDirectory">The base directory that should contain the result.</param>
    /// <param name="relativePath">The relative path to combine with the base.</param>
    /// <returns>The validated full path.</returns>
    /// <exception cref="InvalidOperationException">If path traversal is detected.</exception>
    private static string ValidateAndCombinePath(string baseDirectory, string relativePath)
    {
        // Reject obviously malicious patterns early
        if (relativePath.Contains("..") || relativePath.StartsWith("/", StringComparison.Ordinal) ||
            relativePath.StartsWith("\\", StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Invalid path component detected: '{relativePath}'. Path traversal is not allowed.");
        }

        string baseFullPath = Path.GetFullPath(baseDirectory);
        if (!baseFullPath.EndsWith(Path.DirectorySeparatorChar.ToString(), StringComparison.Ordinal))
        {
            baseFullPath += Path.DirectorySeparatorChar;
        }

        string combinedPath = Path.Combine(baseDirectory, relativePath);
        string combinedFullPath = Path.GetFullPath(combinedPath);

        if (!combinedFullPath.StartsWith(baseFullPath, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                $"Path traversal detected. The path '{relativePath}' would escape the base directory.");
        }

        return combinedFullPath;
    }

    /// <summary>
    /// Sanitizes a model ID for use as a directory name.
    /// </summary>
    /// <param name="modelId">The model ID to sanitize.</param>
    /// <returns>A sanitized directory name.</returns>
    /// <exception cref="ArgumentException">If the model ID contains invalid characters.</exception>
    private static string SanitizeModelIdForPath(string modelId)
    {
        // Replace forward slash with double dash (standard HuggingFace convention)
        // and validate no path traversal characters remain
        var sanitized = modelId.Replace("/", "--");

        if (sanitized.Contains("..") || sanitized.Contains("\\") || sanitized.Contains(":"))
        {
            throw new ArgumentException(
                $"Model ID '{modelId}' contains invalid characters for use as a directory name.",
                nameof(modelId));
        }

        return sanitized;
    }

    /// <summary>
    /// Known CLIP model configurations on HuggingFace Hub.
    /// </summary>
    public static readonly IReadOnlyDictionary<string, ClipModelConfig> KnownModels = new Dictionary<string, ClipModelConfig>
    {
        ["openai/clip-vit-base-patch32"] = new ClipModelConfig
        {
            ImageEncoderFile = "visual_encoder.onnx",
            TextEncoderFile = "text_encoder.onnx",
            EmbeddingDimension = 512,
            ImageSize = 224,
            MaxSequenceLength = 77
        },
        ["openai/clip-vit-base-patch16"] = new ClipModelConfig
        {
            ImageEncoderFile = "visual_encoder.onnx",
            TextEncoderFile = "text_encoder.onnx",
            EmbeddingDimension = 512,
            ImageSize = 224,
            MaxSequenceLength = 77
        },
        ["openai/clip-vit-large-patch14"] = new ClipModelConfig
        {
            ImageEncoderFile = "visual_encoder.onnx",
            TextEncoderFile = "text_encoder.onnx",
            EmbeddingDimension = 768,
            ImageSize = 224,
            MaxSequenceLength = 77
        },
        ["openai/clip-vit-large-patch14-336"] = new ClipModelConfig
        {
            ImageEncoderFile = "visual_encoder.onnx",
            TextEncoderFile = "text_encoder.onnx",
            EmbeddingDimension = 768,
            ImageSize = 336,
            MaxSequenceLength = 77
        }
    };

    /// <summary>
    /// Loads a CLIP model from HuggingFace Hub asynchronously.
    /// </summary>
    /// <typeparam name="T">The numeric type for the model.</typeparam>
    /// <param name="modelId">The HuggingFace model ID (e.g., "openai/clip-vit-base-patch32").</param>
    /// <param name="cacheDir">Optional custom cache directory.</param>
    /// <param name="progress">Optional progress reporter (0.0 to 1.0).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A configured CLIP neural network.</returns>
    /// <exception cref="ArgumentException">If model ID is invalid.</exception>
    /// <exception cref="InvalidOperationException">If model files cannot be loaded.</exception>
    public static async Task<ClipNeuralNetwork<T>> FromPretrainedAsync<T>(
        string modelId,
        string? cacheDir = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(modelId))
            throw new ArgumentException("Model ID cannot be empty.", nameof(modelId));

        // Get or create model configuration
        var config = GetModelConfig(modelId);

        // Set up cache directory with path traversal protection
        cacheDir ??= GetDefaultCacheDir();
        var sanitizedModelId = SanitizeModelIdForPath(modelId);
        var modelCacheDir = ValidateAndCombinePath(cacheDir, sanitizedModelId);

        if (!Directory.Exists(modelCacheDir))
            Directory.CreateDirectory(modelCacheDir);

        // Download model files
        var filesToDownload = GetRequiredFiles(config);
        var downloadedFiles = new Dictionary<string, string>();
        int fileIndex = 0;

        foreach (var fileName in filesToDownload)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var localPath = ValidateAndCombinePath(modelCacheDir, fileName);
            downloadedFiles[fileName] = localPath;

            if (!File.Exists(localPath))
            {
                await DownloadFileAsync(modelId, fileName, localPath, cancellationToken);
            }

            fileIndex++;
            progress?.Report((double)fileIndex / filesToDownload.Count);
        }

        // Load tokenizer
        var tokenizer = LoadTokenizer(modelCacheDir);

        // Create architecture
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.TwoDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputSize: config.EmbeddingDimension,
            outputSize: config.EmbeddingDimension,
            inputHeight: config.ImageSize,
            inputWidth: config.ImageSize
        );

        // Create and return the CLIP network
        return new ClipNeuralNetwork<T>(
            architecture,
            downloadedFiles[config.ImageEncoderFile],
            downloadedFiles[config.TextEncoderFile],
            tokenizer,
            embeddingDimension: config.EmbeddingDimension,
            maxSequenceLength: config.MaxSequenceLength,
            imageSize: config.ImageSize
        );
    }

    /// <summary>
    /// Loads a CLIP model from HuggingFace Hub synchronously.
    /// </summary>
    /// <remarks>
    /// <para><b>Warning:</b> This method uses sync-over-async internally and may cause deadlocks
    /// in UI applications or ASP.NET contexts with synchronization contexts.
    /// Prefer using <see cref="FromPretrainedAsync{T}"/> when possible.</para>
    /// </remarks>
    public static ClipNeuralNetwork<T> FromPretrained<T>(
        string modelId,
        string? cacheDir = null)
    {
        return Task.Run(() => FromPretrainedAsync<T>(modelId, cacheDir)).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Loads a CLIP model from a local directory.
    /// </summary>
    /// <typeparam name="T">The numeric type for the model.</typeparam>
    /// <param name="modelPath">Path to the directory containing model files.</param>
    /// <param name="config">Optional model configuration. If null, attempts to auto-detect.</param>
    /// <returns>A configured CLIP neural network.</returns>
    public static ClipNeuralNetwork<T> FromDirectory<T>(
        string modelPath,
        ClipModelConfig? config = null)
    {
        if (!Directory.Exists(modelPath))
            throw new DirectoryNotFoundException($"Model directory not found: {modelPath}");

        config ??= DetectModelConfig(modelPath);

        // Validate paths to prevent path traversal attacks
        var imageEncoderPath = ValidateAndCombinePath(modelPath, config.ImageEncoderFile);
        var textEncoderPath = ValidateAndCombinePath(modelPath, config.TextEncoderFile);

        if (!File.Exists(imageEncoderPath))
            throw new FileNotFoundException($"Image encoder not found: {imageEncoderPath}");
        if (!File.Exists(textEncoderPath))
            throw new FileNotFoundException($"Text encoder not found: {textEncoderPath}");

        var tokenizer = LoadTokenizer(modelPath);

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.TwoDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputSize: config.EmbeddingDimension,
            outputSize: config.EmbeddingDimension,
            inputHeight: config.ImageSize,
            inputWidth: config.ImageSize
        );

        return new ClipNeuralNetwork<T>(
            architecture,
            imageEncoderPath,
            textEncoderPath,
            tokenizer,
            embeddingDimension: config.EmbeddingDimension,
            maxSequenceLength: config.MaxSequenceLength,
            imageSize: config.ImageSize
        );
    }

    /// <summary>
    /// Checks if a model is downloaded and cached locally.
    /// </summary>
    /// <param name="modelId">The HuggingFace model ID.</param>
    /// <param name="cacheDir">Optional custom cache directory.</param>
    /// <returns>True if all required files are cached.</returns>
    public static bool IsModelCached(string modelId, string? cacheDir = null)
    {
        if (string.IsNullOrWhiteSpace(modelId))
            return false;

        try
        {
            var config = GetModelConfig(modelId);
            cacheDir ??= GetDefaultCacheDir();
            var sanitizedModelId = SanitizeModelIdForPath(modelId);
            var modelCacheDir = ValidateAndCombinePath(cacheDir, sanitizedModelId);

            if (!Directory.Exists(modelCacheDir))
                return false;

            var requiredFiles = GetRequiredFiles(config);
            foreach (var fileName in requiredFiles)
            {
                var localPath = ValidateAndCombinePath(modelCacheDir, fileName);
                if (!File.Exists(localPath))
                    return false;
            }

            return true;
        }
        catch (InvalidOperationException)
        {
            // Path traversal attempt or invalid path - model is not properly cached
            return false;
        }
        catch (ArgumentException)
        {
            // Invalid model ID format - model is not properly cached
            return false;
        }
    }

    /// <summary>
    /// Clears the cached model files.
    /// </summary>
    /// <param name="modelId">The HuggingFace model ID.</param>
    /// <param name="cacheDir">Optional custom cache directory.</param>
    public static void ClearCache(string modelId, string? cacheDir = null)
    {
        if (string.IsNullOrWhiteSpace(modelId))
            return;

        cacheDir ??= GetDefaultCacheDir();
        var sanitizedModelId = SanitizeModelIdForPath(modelId);
        var modelCacheDir = ValidateAndCombinePath(cacheDir, sanitizedModelId);

        if (Directory.Exists(modelCacheDir))
        {
            Directory.Delete(modelCacheDir, recursive: true);
        }
    }

    private static ClipModelConfig GetModelConfig(string modelId)
    {
        if (KnownModels.TryGetValue(modelId, out var config))
            return config;

        // Default configuration for unknown models
        return new ClipModelConfig
        {
            ImageEncoderFile = "visual_encoder.onnx",
            TextEncoderFile = "text_encoder.onnx",
            EmbeddingDimension = 512,
            ImageSize = 224,
            MaxSequenceLength = 77
        };
    }

    private static ClipModelConfig DetectModelConfig(string modelPath)
    {
        // Try to load config.json if it exists
        var configPath = ValidateAndCombinePath(modelPath, "config.json");
        if (File.Exists(configPath))
        {
            try
            {
                var configJson = File.ReadAllText(configPath);
                var configObj = JObject.Parse(configJson);

                return new ClipModelConfig
                {
                    ImageEncoderFile = FindOnnxFile(modelPath, "visual", "image") ?? "visual_encoder.onnx",
                    TextEncoderFile = FindOnnxFile(modelPath, "text") ?? "text_encoder.onnx",
                    EmbeddingDimension = configObj["projection_dim"]?.Value<int>() ?? 512,
                    ImageSize = configObj["vision_config"]?["image_size"]?.Value<int>() ?? 224,
                    MaxSequenceLength = configObj["text_config"]?["max_position_embeddings"]?.Value<int>() ?? 77
                };
            }
            catch (JsonException)
            {
                // Invalid JSON - fall through to defaults
            }
            catch (IOException)
            {
                // File read error - fall through to defaults
            }
        }

        // Auto-detect ONNX files
        return new ClipModelConfig
        {
            ImageEncoderFile = FindOnnxFile(modelPath, "visual", "image") ?? "visual_encoder.onnx",
            TextEncoderFile = FindOnnxFile(modelPath, "text") ?? "text_encoder.onnx",
            EmbeddingDimension = 512,
            ImageSize = 224,
            MaxSequenceLength = 77
        };
    }

    private static string? FindOnnxFile(string directory, params string[] keywords)
    {
        var onnxFiles = Directory.GetFiles(directory, "*.onnx");
        foreach (var file in onnxFiles)
        {
            var fileName = Path.GetFileName(file).ToLowerInvariant();
            foreach (var keyword in keywords)
            {
                if (fileName.Contains(keyword.ToLowerInvariant()))
                    return Path.GetFileName(file);
            }
        }
        return onnxFiles.Length > 0 ? Path.GetFileName(onnxFiles[0]) : null;
    }

    private static List<string> GetRequiredFiles(ClipModelConfig config)
    {
        return new List<string>
        {
            config.ImageEncoderFile,
            config.TextEncoderFile,
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "config.json"
        };
    }

    private static string GetDefaultCacheDir()
    {
        var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(userProfile, DefaultCacheDir);
    }

    private static ITokenizer LoadTokenizer(string modelPath)
    {
        try
        {
            // Try to load using HuggingFace tokenizer loader
            return HuggingFaceTokenizerLoader.LoadFromDirectory(modelPath);
        }
        catch (FileNotFoundException)
        {
            // Tokenizer files not found - fall back to CLIP factory
        }
        catch (IOException)
        {
            // File read error - fall back to CLIP factory
        }
        catch (JsonException)
        {
            // Invalid tokenizer JSON - fall back to CLIP factory
        }

        // Fall back to CLIP factory
        var vocabPath = ValidateAndCombinePath(modelPath, "vocab.json");
        var mergesPath = ValidateAndCombinePath(modelPath, "merges.txt");

        if (File.Exists(vocabPath) && File.Exists(mergesPath))
        {
            return ClipTokenizerFactory.FromPretrained(vocabPath, mergesPath);
        }

        // Last resort: create simple tokenizer
        return ClipTokenizerFactory.CreateSimple();
    }

    private static async Task DownloadFileAsync(
        string modelId,
        string fileName,
        string localPath,
        CancellationToken cancellationToken)
    {
        var url = $"{HuggingFaceHubUrl}/{modelId}/resolve/main/{fileName}";

        try
        {
            // Use streaming to handle large files without loading entire file into memory
            using var response = await HttpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);

            if (response.IsSuccessStatusCode)
            {
                using var contentStream = await response.Content.ReadAsStreamAsync();
                using var fileStream = new FileStream(localPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 81920, useAsync: true);
                await contentStream.CopyToAsync(fileStream, 81920, cancellationToken);
            }
            else if (response.StatusCode == System.Net.HttpStatusCode.NotFound &&
                     fileName.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            {
                // Required ONNX file not found - throw for ONNX files only (optional files are ignored)
                throw new FileNotFoundException(
                    $"Required model file not found on HuggingFace: {fileName}. " +
                    $"The model '{modelId}' may not have ONNX exports available. " +
                    "Please export the model to ONNX format or use a model with ONNX files.");
            }
        }
        catch (HttpRequestException ex)
        {
            throw new InvalidOperationException(
                $"Failed to download model file '{fileName}' from HuggingFace. " +
                "Please check your internet connection and model ID.", ex);
        }
    }
}

/// <summary>
/// Configuration for a CLIP model variant.
/// </summary>
public class ClipModelConfig
{
    /// <summary>
    /// The filename of the image encoder ONNX model.
    /// </summary>
    public string ImageEncoderFile { get; set; } = "visual_encoder.onnx";

    /// <summary>
    /// The filename of the text encoder ONNX model.
    /// </summary>
    public string TextEncoderFile { get; set; } = "text_encoder.onnx";

    /// <summary>
    /// The embedding dimension (e.g., 512 for ViT-B, 768 for ViT-L).
    /// </summary>
    public int EmbeddingDimension { get; set; } = 512;

    /// <summary>
    /// The expected image size (e.g., 224 or 336).
    /// </summary>
    public int ImageSize { get; set; } = 224;

    /// <summary>
    /// The maximum text sequence length (typically 77 for CLIP).
    /// </summary>
    public int MaxSequenceLength { get; set; } = 77;
}
