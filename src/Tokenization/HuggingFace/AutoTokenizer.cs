using System;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Tokenization.HuggingFace
{
    /// <summary>
    /// AutoTokenizer provides HuggingFace-style automatic tokenizer loading.
    /// This class automatically detects and loads the appropriate tokenizer type
    /// based on the model configuration.
    /// </summary>
    /// <remarks>
    /// Usage mirrors the HuggingFace transformers library:
    /// <code>
    /// // Load from HuggingFace Hub
    /// var tokenizer = AutoTokenizer.FromPretrained("bert-base-uncased");
    ///
    /// // Load from local directory
    /// var tokenizer = AutoTokenizer.FromPretrained("./my-model");
    /// </code>
    /// </remarks>
    public static class AutoTokenizer
    {
        /// <summary>
        /// Loads a tokenizer from a pretrained model name or path.
        /// </summary>
        /// <param name="modelNameOrPath">
        /// Either a HuggingFace model name (e.g., "bert-base-uncased", "gpt2")
        /// or a local directory path containing tokenizer files.
        /// </param>
        /// <param name="cacheDir">
        /// Optional cache directory for downloaded files.
        /// Defaults to ~/.cache/huggingface/tokenizers
        /// </param>
        /// <returns>The loaded tokenizer.</returns>
        /// <exception cref="ArgumentException">Thrown when modelNameOrPath is empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when tokenizer cannot be loaded.</exception>
        public static ITokenizer FromPretrained(string modelNameOrPath, string? cacheDir = null)
        {
            if (string.IsNullOrWhiteSpace(modelNameOrPath))
                throw new ArgumentException("Model name or path cannot be empty", nameof(modelNameOrPath));

            // Check if it's a local path
            if (Directory.Exists(modelNameOrPath))
            {
                return HuggingFaceTokenizerLoader.LoadFromDirectory(modelNameOrPath);
            }

            // Otherwise, load from HuggingFace Hub
            return HuggingFaceTokenizerLoader.LoadFromHub(modelNameOrPath, cacheDir);
        }

        /// <summary>
        /// Asynchronously loads a tokenizer from a pretrained model name or path.
        /// </summary>
        /// <param name="modelNameOrPath">
        /// Either a HuggingFace model name (e.g., "bert-base-uncased", "gpt2")
        /// or a local directory path containing tokenizer files.
        /// </param>
        /// <param name="cacheDir">
        /// Optional cache directory for downloaded files.
        /// Defaults to ~/.cache/huggingface/tokenizers
        /// </param>
        /// <returns>The loaded tokenizer.</returns>
        public static async Task<ITokenizer> FromPretrainedAsync(string modelNameOrPath, string? cacheDir = null)
        {
            if (string.IsNullOrWhiteSpace(modelNameOrPath))
                throw new ArgumentException("Model name or path cannot be empty", nameof(modelNameOrPath));

            // Check if it's a local path
            if (Directory.Exists(modelNameOrPath))
            {
                return HuggingFaceTokenizerLoader.LoadFromDirectory(modelNameOrPath);
            }

            // Otherwise, load from HuggingFace Hub
            return await HuggingFaceTokenizerLoader.LoadFromHubAsync(modelNameOrPath, cacheDir);
        }

        /// <summary>
        /// Gets the default cache directory for tokenizer files.
        /// </summary>
        /// <returns>The default cache directory path.</returns>
        public static string GetDefaultCacheDir()
        {
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".cache",
                "huggingface",
                "tokenizers"
            );
        }

        /// <summary>
        /// Checks if a tokenizer is cached locally.
        /// </summary>
        /// <param name="modelName">The model name to check.</param>
        /// <param name="cacheDir">Optional cache directory. Uses default if not specified.</param>
        /// <returns>True if the tokenizer is cached, false otherwise.</returns>
        public static bool IsCached(string modelName, string? cacheDir = null)
        {
            if (string.IsNullOrWhiteSpace(modelName))
                return false;

            cacheDir ??= GetDefaultCacheDir();
            var modelCacheDir = Path.Combine(cacheDir, modelName.Replace("/", "--"));

            if (!Directory.Exists(modelCacheDir))
                return false;

            // Check if essential files exist
            var tokenizerJsonPath = Path.Combine(modelCacheDir, "tokenizer.json");
            var vocabPath = Path.Combine(modelCacheDir, "vocab.json");
            var configPath = Path.Combine(modelCacheDir, "tokenizer_config.json");

            return File.Exists(tokenizerJsonPath) ||
                   (File.Exists(vocabPath) && File.Exists(configPath));
        }

        /// <summary>
        /// Clears the cache for a specific model or all models.
        /// </summary>
        /// <param name="modelName">
        /// Optional model name to clear. If null, clears all cached tokenizers.
        /// </param>
        /// <param name="cacheDir">Optional cache directory. Uses default if not specified.</param>
        public static void ClearCache(string? modelName = null, string? cacheDir = null)
        {
            cacheDir ??= GetDefaultCacheDir();

            if (!Directory.Exists(cacheDir))
                return;

            if (string.IsNullOrWhiteSpace(modelName))
            {
                // Clear all cached tokenizers
                Directory.Delete(cacheDir, true);
                Directory.CreateDirectory(cacheDir);
            }
            else if (modelName is not null)
            {
                // Clear specific model cache
                var modelCacheDir = Path.Combine(cacheDir, modelName.Replace("/", "--"));
                if (Directory.Exists(modelCacheDir))
                {
                    Directory.Delete(modelCacheDir, true);
                }
            }
        }

        /// <summary>
        /// Lists all cached tokenizer models.
        /// </summary>
        /// <param name="cacheDir">Optional cache directory. Uses default if not specified.</param>
        /// <returns>Array of cached model names.</returns>
        public static string[] ListCachedModels(string? cacheDir = null)
        {
            cacheDir ??= GetDefaultCacheDir();

            if (!Directory.Exists(cacheDir))
                return Array.Empty<string>();

            var directories = Directory.GetDirectories(cacheDir);
            var models = new string[directories.Length];

            for (int i = 0; i < directories.Length; i++)
            {
                var dirName = Path.GetFileName(directories[i]);
                // Convert back from filesystem-safe format
                models[i] = dirName.Replace("--", "/");
            }

            return models;
        }
    }
}
