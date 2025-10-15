using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Logging;

namespace AiDotNet.FoundationModels.Providers
{
    /// <summary>
    /// Provider for loading foundation models from local file system.
    /// Supports various model formats and manages local model repository.
    /// </summary>
    public class LocalModelProvider : IFoundationModelProvider
    {
        private readonly string _modelsDirectory;
        private readonly ILogging _logger;
        private readonly Dictionary<string, ModelMetadata> _modelRegistry;
        private readonly Dictionary<string, object> _loadedModels;
        private readonly object _lockObject = new object();

        /// <summary>
        /// Initializes a new instance of the LocalModelProvider class
        /// </summary>
        /// <param name="modelsDirectory">Directory containing model files</param>
        /// <param name="logger">Optional logger</param>
        public LocalModelProvider(string modelsDirectory, ILogging? logger = null)
        {
            _modelsDirectory = modelsDirectory ?? throw new ArgumentNullException(nameof(modelsDirectory));
            _logger = logger ?? new AiDotNetLogger();
            _modelRegistry = new Dictionary<string, ModelMetadata>(StringComparer.OrdinalIgnoreCase);
            _loadedModels = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
            
            // Create directory if it doesn't exist
            Directory.CreateDirectory(_modelsDirectory);
            
            // Scan for available models
            ScanForModels();
        }

        #region IFoundationModelProvider Implementation

        /// <inheritdoc/>
        public string ProviderName => "Local";

        /// <inheritdoc/>
        public IReadOnlyList<string> SupportedArchitectures => new[]
        {
            "GPT2", "BERT", "T5", "LLAMA", "Custom"
        };

        /// <inheritdoc/>
        public async Task<bool> IsModelAvailableAsync(string modelId)
        {
            await Task.CompletedTask;
            return _modelRegistry.ContainsKey(modelId) || 
                   File.Exists(GetModelPath(modelId));
        }

        /// <inheritdoc/>
        async Task<IFoundationModel<T>> IFoundationModelProvider.GetModelAsync<T>(
            string modelId,
            FoundationModelConfig? config,
            CancellationToken cancellationToken)
        {
            // Check if already loaded
            lock (_lockObject)
            {
                var cacheKey = $"{modelId}_{typeof(T).Name}";
                if (_loadedModels.TryGetValue(cacheKey, out var loadedModel) && loadedModel is IFoundationModel<T> typedModel)
                {
                    _logger.Debug("Returning already loaded model: {ModelId}", modelId);
                    return typedModel;
                }
            }

            // Load model metadata
            var metadata = await GetModelMetadataAsync(modelId);
            if (metadata == null)
            {
                throw new InvalidOperationException($"Model not found: {modelId}");
            }

            // Create appropriate model instance based on architecture
            var model = await CreateModelInstanceAsync<T>(metadata, config ?? FoundationModelConfig.CreateDefault(modelId));
            
            // Cache the loaded model
            lock (_lockObject)
            {
                var cacheKey = $"{modelId}_{typeof(T).Name}";
                _loadedModels[cacheKey] = model;
            }

            return model;
        }

        /// <inheritdoc/>
        public async Task<IReadOnlyList<FoundationModelInfo>> ListAvailableModelsAsync(ModelFilter? filter = null)
        {
            var models = new List<FoundationModelInfo>();
            
            foreach (var kvp in _modelRegistry)
            {
                var modelId = kvp.Key;
                var metadata = kvp.Value;
                
                var info = new FoundationModelInfo
                {
                    ModelId = modelId,
                    Name = metadata.Name,
                    Architecture = metadata.Architecture,
                    ParameterCount = metadata.ParameterCount,
                    Description = metadata.Description,
                    SupportedTasks = metadata.SupportedTasks,
                    MaxContextLength = metadata.MaxContextLength,
                    ModelSizeBytes = metadata.ModelSizeBytes,
                    IsOfflineAvailable = true,
                    License = metadata.License,
                    Metadata = metadata.AdditionalMetadata
                };

                if (MatchesFilter(info, filter))
                {
                    models.Add(info);
                }
            }

            return await Task.FromResult(models);
        }

        /// <inheritdoc/>
        public async Task<string> DownloadModelAsync(
            string modelId,
            Action<DownloadProgress>? progressCallback = null,
            CancellationToken cancellationToken = default)
        {
            // For local provider, models are already available
            var modelPath = GetModelPath(modelId);
            
            if (!File.Exists(modelPath))
            {
                throw new InvalidOperationException($"Model file not found: {modelPath}");
            }

            progressCallback?.Invoke(new DownloadProgress
                {
                    TotalBytes = 0,
                    BytesDownloaded = 0,
                    CurrentFile = modelPath
                });

            return await Task.FromResult(modelPath);
        }

        /// <inheritdoc/>
        public async Task<bool> ValidateConnectionAsync()
        {
            try
            {
                var exists = Directory.Exists(_modelsDirectory);
                if (!exists)
                {
                    _logger.Warning("Models directory does not exist: {Directory}", _modelsDirectory);
                }
                return await Task.FromResult(exists);
            }
            catch (Exception ex)
            {
                _logger.Error("Failed to validate local model provider: {Error}", ex.Message);
                return false;
            }
        }

        /// <inheritdoc/>
        public async Task<FoundationModelConfig> GetDefaultConfigAsync(string modelId)
        {
            var metadata = await GetModelMetadataAsync(modelId);
            if (metadata == null)
            {
                return FoundationModelConfig.CreateDefault(modelId);
            }

            var config = FoundationModelConfig.CreateDefault(modelId);
            
            // Apply model-specific defaults
            if (metadata.DefaultConfig != null)
            {
                config.MaxSequenceLength = metadata.DefaultConfig.MaxSequenceLength;
                config.MaxBatchSize = metadata.DefaultConfig.MaxBatchSize ?? 1;
                config.DataType = metadata.DefaultConfig.DataType ?? "float32";
            }

            return config;
        }

        /// <inheritdoc/>
        public async Task ReleaseModelAsync(string modelId)
        {
            lock (_lockObject)
            {
                if (_loadedModels.Remove(modelId))
                {
                    _logger.Information("Released model from memory: {ModelId}", modelId);
                }
            }
            
            await Task.CompletedTask;
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Scans the models directory for available models
        /// </summary>
        private void ScanForModels()
        {
            try
            {
                // Look for model metadata files
                var metadataFiles = Directory.GetFiles(_modelsDirectory, "model_metadata.json", SearchOption.AllDirectories);
                
                foreach (var metadataFile in metadataFiles)
                {
                    try
                    {
                        var json = File.ReadAllText(metadataFile);
                        var metadata = JsonSerializer.Deserialize<ModelMetadata>(json);
                        
                        if (metadata != null && !string.IsNullOrEmpty(metadata.ModelId))
                        {
                            _modelRegistry[metadata.ModelId] = metadata;
                            _logger.Debug("Discovered model: {ModelId} ({Architecture})", 
                                metadata.ModelId, metadata.Architecture);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.Warning("Failed to load model metadata from {File}: {Error}", 
                            metadataFile, ex.Message);
                    }
                }

                // Also scan for common model file patterns
                ScanForCommonModelFormats();
                
                _logger.Information("Local model provider initialized with {Count} models", _modelRegistry.Count);
            }
            catch (Exception ex)
            {
                _logger.Error("Failed to scan for models: {Error}", ex.Message);
            }
        }

        /// <summary>
        /// Scans for common model file formats
        /// </summary>
        private void ScanForCommonModelFormats()
        {
            // ONNX models
            var onnxFiles = Directory.GetFiles(_modelsDirectory, "*.onnx", SearchOption.AllDirectories);
            foreach (var onnxFile in onnxFiles)
            {
                var modelId = Path.GetFileNameWithoutExtension(onnxFile);
                if (!_modelRegistry.ContainsKey(modelId))
                {
                    _modelRegistry[modelId] = CreateMetadataFromFile(onnxFile, "ONNX");
                }
            }

            // PyTorch models
            var ptFiles = Directory.GetFiles(_modelsDirectory, "*.pt", SearchOption.AllDirectories);
            foreach (var ptFile in ptFiles)
            {
                var modelId = Path.GetFileNameWithoutExtension(ptFile);
                if (!_modelRegistry.ContainsKey(modelId))
                {
                    _modelRegistry[modelId] = CreateMetadataFromFile(ptFile, "PyTorch");
                }
            }

            // TensorFlow models
            var pbFiles = Directory.GetFiles(_modelsDirectory, "*.pb", SearchOption.AllDirectories);
            foreach (var pbFile in pbFiles)
            {
                var modelId = Path.GetFileNameWithoutExtension(pbFile);
                if (!_modelRegistry.ContainsKey(modelId))
                {
                    _modelRegistry[modelId] = CreateMetadataFromFile(pbFile, "TensorFlow");
                }
            }
        }

        /// <summary>
        /// Creates metadata from a model file
        /// </summary>
        private ModelMetadata CreateMetadataFromFile(string filePath, string format)
        {
            var fileInfo = new FileInfo(filePath);
            var modelId = Path.GetFileNameWithoutExtension(filePath);
            
            return new ModelMetadata
            {
                ModelId = modelId,
                Name = modelId,
                Architecture = DetermineArchitecture(modelId),
                ModelPath = filePath,
                Format = format,
                ModelSizeBytes = fileInfo.Length,
                Description = $"Auto-discovered {format} model",
                SupportedTasks = new List<string> { "text-generation", "embeddings" },
                MaxContextLength = 512, // Default
                ParameterCount = EstimateParameterCount(fileInfo.Length)
            };
        }

        /// <summary>
        /// Determines architecture from model name
        /// </summary>
        private string DetermineArchitecture(string modelId)
        {
            var lowerModelId = modelId.ToLowerInvariant();
            
            if (lowerModelId.Contains("gpt")) return "GPT2";
            if (lowerModelId.Contains("bert")) return "BERT";
            if (lowerModelId.Contains("t5")) return "T5";
            if (lowerModelId.Contains("llama")) return "LLAMA";
            
            return "Custom";
        }

        /// <summary>
        /// Estimates parameter count from file size
        /// </summary>
        private long EstimateParameterCount(long fileSizeBytes)
        {
            // Rough estimate: 4 bytes per parameter (float32)
            return fileSizeBytes / 4;
        }

        /// <summary>
        /// Gets the model path
        /// </summary>
        private string GetModelPath(string modelId)
        {
            return Path.Combine(_modelsDirectory, modelId);
        }

        /// <summary>
        /// Gets model metadata
        /// </summary>
        private async Task<ModelMetadata?> GetModelMetadataAsync(string modelId)
        {
            await Task.CompletedTask;
            
            if (_modelRegistry.TryGetValue(modelId, out var metadata))
            {
                return metadata;
            }

            // Try to load metadata file directly
            var metadataPath = Path.Combine(GetModelPath(modelId), "model_metadata.json");
            if (File.Exists(metadataPath))
            {
                try
                {
                    var json = File.ReadAllText(metadataPath);
                    return JsonSerializer.Deserialize<ModelMetadata>(json);
                }
                catch (Exception ex)
                {
                    _logger.Warning("Failed to load metadata for {ModelId}: {Error}", modelId, ex.Message);
                }
            }

            return null;
        }

        /// <summary>
        /// Creates a model instance based on metadata
        /// </summary>
        private async Task<IFoundationModel<T>> CreateModelInstanceAsync<T>(ModelMetadata metadata, FoundationModelConfig config)
            where T : struct, IComparable<T>, IConvertible
        {
            _logger.Information("Creating model instance: {ModelId} ({Architecture})", metadata.ModelId, metadata.Architecture);

            // Create tokenizer
            var tokenizer = CreateTokenizer(metadata);
            await tokenizer.InitializeAsync();

            // Create model based on architecture
            switch (metadata.Architecture.ToUpperInvariant())
            {
                case "GPT2":
                    return new Models.GPT2Model<T>(metadata.ModelPath, tokenizer, config, _logger);
                    
                case "BERT":
                    return new Models.BERTModel<T>(metadata.ModelPath, tokenizer, config, _logger);
                    
                default:
                    throw new NotSupportedException($"Architecture not yet implemented: {metadata.Architecture}");
            }
        }

        /// <summary>
        /// Creates appropriate tokenizer for the model
        /// </summary>
        private ITokenizer CreateTokenizer(ModelMetadata metadata)
        {
            var vocabPath = Path.Combine(Path.GetDirectoryName(metadata.ModelPath) ?? "", "vocab.txt");
            var mergesPath = Path.Combine(Path.GetDirectoryName(metadata.ModelPath) ?? "", "merges.txt");

            return metadata.Architecture.ToUpperInvariant() switch
            {
                "GPT2" => new Tokenizers.BPETokenizer(vocabPath, mergesPath),
                "BERT" => new Tokenizers.WordPieceTokenizer(vocabPath),
                _ => new Tokenizers.BPETokenizer(vocabPath, mergesPath)
            };
        }

        /// <summary>
        /// Checks if a model matches the filter
        /// </summary>
        private bool MatchesFilter(FoundationModelInfo info, ModelFilter? filter)
        {
            if (filter == null) return true;

            if (!string.IsNullOrEmpty(filter.Architecture) && 
                !info.Architecture.Equals(filter.Architecture, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (!string.IsNullOrEmpty(filter.Task) && 
                !info.SupportedTasks.Contains(filter.Task, StringComparer.OrdinalIgnoreCase))
            {
                return false;
            }

            if (filter.MinParameters.HasValue && info.ParameterCount < filter.MinParameters.Value)
            {
                return false;
            }

            if (filter.MaxParameters.HasValue && info.ParameterCount > filter.MaxParameters.Value)
            {
                return false;
            }

            if (filter.OfflineOnly == true && !info.IsOfflineAvailable)
            {
                return false;
            }

            if (filter.SearchQuery != null && filter.SearchQuery.Length > 0)
            {
                var query = filter.SearchQuery.ToLowerInvariant();
                return info.Name.ToLowerInvariant().Contains(query) ||
                       info.Description.ToLowerInvariant().Contains(query);
            }

            return true;
        }

        #endregion

        #region Nested Classes

        /// <summary>
        /// Model metadata stored in JSON files
        /// </summary>
        private class ModelMetadata
        {
            public string ModelId { get; set; } = string.Empty;
            public string Name { get; set; } = string.Empty;
            public string Architecture { get; set; } = string.Empty;
            public string ModelPath { get; set; } = string.Empty;
            public string Format { get; set; } = string.Empty;
            public long ParameterCount { get; set; }
            public string Description { get; set; } = string.Empty;
            public List<string> SupportedTasks { get; set; } = new List<string>();
            public int MaxContextLength { get; set; }
            public long ModelSizeBytes { get; set; }
            public string License { get; set; } = string.Empty;
            public Dictionary<string, object> AdditionalMetadata { get; set; } = new Dictionary<string, object>();
            public DefaultConfig? DefaultConfig { get; set; }
        }

        /// <summary>
        /// Default configuration from metadata
        /// </summary>
        private class DefaultConfig
        {
            public int? MaxSequenceLength { get; set; }
            public int? MaxBatchSize { get; set; }
            public string? DataType { get; set; }
        }

        #endregion
    }
}