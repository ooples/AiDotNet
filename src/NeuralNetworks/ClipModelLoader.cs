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

        // Set up cache directory
        cacheDir ??= GetDefaultCacheDir();
        var modelCacheDir = Path.Combine(cacheDir, modelId.Replace("/", "--"));

        if (!Directory.Exists(modelCacheDir))
            Directory.CreateDirectory(modelCacheDir);

        // Download model files
        var filesToDownload = GetRequiredFiles(config);
        var downloadedFiles = new Dictionary<string, string>();
        int fileIndex = 0;

        foreach (var fileName in filesToDownload)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var localPath = Path.Combine(modelCacheDir, fileName);
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

        var imageEncoderPath = Path.Combine(modelPath, config.ImageEncoderFile);
        var textEncoderPath = Path.Combine(modelPath, config.TextEncoderFile);

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

        var config = GetModelConfig(modelId);
        cacheDir ??= GetDefaultCacheDir();
        var modelCacheDir = Path.Combine(cacheDir, modelId.Replace("/", "--"));

        if (!Directory.Exists(modelCacheDir))
            return false;

        var requiredFiles = GetRequiredFiles(config);
        foreach (var fileName in requiredFiles)
        {
            var localPath = Path.Combine(modelCacheDir, fileName);
            if (!File.Exists(localPath))
                return false;
        }

        return true;
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
        var modelCacheDir = Path.Combine(cacheDir, modelId.Replace("/", "--"));

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
        var configPath = Path.Combine(modelPath, "config.json");
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
            catch
            {
                // Fall through to defaults
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
        catch
        {
            // Fall back to CLIP factory
            var vocabPath = Path.Combine(modelPath, "vocab.json");
            var mergesPath = Path.Combine(modelPath, "merges.txt");

            if (File.Exists(vocabPath) && File.Exists(mergesPath))
            {
                return ClipTokenizerFactory.FromPretrained(vocabPath, mergesPath);
            }

            // Last resort: create simple tokenizer
            return ClipTokenizerFactory.CreateSimple();
        }
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
            using var response = await HttpClient.GetAsync(url, cancellationToken);

            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsByteArrayAsync();
                File.WriteAllBytes(localPath, content);
            }
            else if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                // File not found - this might be okay for optional files
                // Only throw for required ONNX files
                if (fileName.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
                {
                    throw new FileNotFoundException(
                        $"Required model file not found on HuggingFace: {fileName}. " +
                        $"The model '{modelId}' may not have ONNX exports available. " +
                        "Please export the model to ONNX format or use a model with ONNX files.");
                }
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
