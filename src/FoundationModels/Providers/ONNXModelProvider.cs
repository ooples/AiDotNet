using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Logging;
using AiDotNet.FoundationModels.Models;
using AiDotNet.MultimodalAI;

namespace AiDotNet.FoundationModels.Providers
{
    /// <summary>
    /// Provider for loading ONNX models using ONNX Runtime
    /// </summary>
    public class ONNXModelProvider : FoundationModelProviderBase
    {
        private readonly string _modelsDirectory;
        private readonly Dictionary<string, ONNXModelInfo> _modelRegistry;

        /// <inheritdoc/>
        public override string ProviderName => "ONNX";

        /// <inheritdoc/>
        public override IReadOnlyList<string> SupportedArchitectures => new[]
        {
            "bert", "gpt2", "t5", "roberta", "distilbert", "albert",
            "xlm", "bart", "mbart", "pegasus", "marian", "whisper",
            "clip", "vision-transformer", "resnet", "efficientnet"
        };

        /// <summary>
        /// Initializes a new instance of the ONNXModelProvider class
        /// </summary>
        /// <param name="modelsDirectory">Directory containing ONNX models</param>
        /// <param name="logger">Logger instance</param>
        public ONNXModelProvider(string? modelsDirectory = null, ILogging? logger = null)
            : base(logger)
        {
            _modelsDirectory = modelsDirectory ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".aidotnet", "onnx_models");
            
            _modelRegistry = new Dictionary<string, ONNXModelInfo>(StringComparer.OrdinalIgnoreCase);
            
            // Ensure directory exists
            Directory.CreateDirectory(_modelsDirectory);
            
            // Scan for ONNX models
            ScanForONNXModels();
        }

        /// <inheritdoc/>
        public override async Task<bool> IsModelAvailableAsync(string modelId)
        {
            // Check registry
            if (_modelRegistry.ContainsKey(modelId))
            {
                return true;
            }

            // Check if ONNX file exists
            var modelPath = GetONNXModelPath(modelId);
            if (File.Exists(modelPath))
            {
                // Register the model
                _modelRegistry[modelId] = new ONNXModelInfo
                {
                    ModelId = modelId,
                    ModelPath = modelPath,
                    Architecture = GetArchitectureFromModelId(modelId)
                };
                return true;
            }

            // Check subdirectories
            var modelDir = GetLocalModelPath(modelId);
            if (Directory.Exists(modelDir))
            {
                var onnxFiles = Directory.GetFiles(modelDir, "*.onnx", SearchOption.AllDirectories);
                if (onnxFiles.Length > 0)
                {
                    _modelRegistry[modelId] = new ONNXModelInfo
                    {
                        ModelId = modelId,
                        ModelPath = onnxFiles[0],
                        Architecture = GetArchitectureFromModelId(modelId)
                    };
                    return true;
                }
            }

            return false;
        }

        /// <inheritdoc/>
        public override async Task<IReadOnlyList<FoundationModelInfo>> ListAvailableModelsAsync(
            ModelFilter? filter = null)
        {
            var models = new List<FoundationModelInfo>();

            foreach (var modelInfo in _modelRegistry.Values)
            {
                var info = new FoundationModelInfo
                {
                    ModelId = modelInfo.ModelId,
                    Name = modelInfo.Name ?? modelInfo.ModelId,
                    Architecture = modelInfo.Architecture,
                    ParameterCount = modelInfo.ParameterCount,
                    Description = $"ONNX optimized {modelInfo.Architecture} model",
                    SupportedTasks = GetSupportedTasks(modelInfo.Architecture),
                    MaxContextLength = modelInfo.MaxSequenceLength ?? 512,
                    ModelSizeBytes = GetModelSize(modelInfo.ModelPath),
                    IsOfflineAvailable = true,
                    License = modelInfo.License ?? "unknown",
                    Metadata = new Dictionary<string, object>
                    {
                        ["format"] = "onnx",
                        ["opset_version"] = modelInfo.OpsetVersion ?? 14,
                        ["provider"] = "onnx"
                    }
                };

                if (MatchesFilter(info, filter))
                {
                    models.Add(info);
                }
            }

            return await Task.FromResult(models);
        }

        /// <inheritdoc/>
        public override async Task<bool> ValidateConnectionAsync()
        {
            try
            {
                // Check if directory is accessible
                if (!Directory.Exists(_modelsDirectory))
                {
                    Directory.CreateDirectory(_modelsDirectory);
                }

                // In a real implementation, check if ONNX Runtime is available
                // For now, just return true
                return await Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.Error("Failed to validate ONNX provider connection: {Error}", ex.Message);
                return false;
            }
        }

        /// <inheritdoc/>
        protected override async Task<IFoundationModel<T>> LoadModelInternalAsync<T>(
            string modelId,
            FoundationModelConfig config,
            CancellationToken cancellationToken)
        {
            if (!_modelRegistry.TryGetValue(modelId, out var modelInfo))
            {
                throw new InvalidOperationException($"Model {modelId} not found in ONNX registry");
            }

            if (!File.Exists(modelInfo.ModelPath))
            {
                throw new FileNotFoundException($"ONNX model file not found: {modelInfo.ModelPath}");
            }

            _logger.Information("Loading ONNX model from {Path}", modelInfo.ModelPath);

            // In a real implementation, this would use ONNX Runtime to load the model
            // For now, return appropriate mock based on architecture
            return modelInfo.Architecture.ToLower() switch
            {
                "bert" or "distilbert" or "roberta" or "albert" => new BERTFoundationModel<T>(),
                "clip" => new CLIPMultimodalModel<T>(),
                _ => new BERTFoundationModel<T>()
            };
        }

        /// <inheritdoc/>
        protected override async Task<string> DownloadModelInternalAsync(
            string modelId,
            Action<DownloadProgress>? progressCallback,
            CancellationToken cancellationToken)
        {
            // In a real implementation, this would download from ONNX Model Zoo or HuggingFace
            _logger.Information("Downloading ONNX model {ModelId}", modelId);

            var modelPath = GetONNXModelPath(modelId);
            var modelDir = Path.GetDirectoryName(modelPath);
            
            if (!string.IsNullOrEmpty(modelDir))
            {
                Directory.CreateDirectory(modelDir);
            }

            // Simulate download progress
            for (int i = 0; i <= 100; i += 10)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    throw new OperationCanceledException();
                }

                ReportProgress(progressCallback, 100 * 1024 * 1024, i * 1024 * 1024, 
                    1024 * 1024, $"{modelId}.onnx");
                
                await Task.Delay(100, cancellationToken);
            }

            // Create placeholder file
            await File.WriteAllTextAsync(modelPath, "ONNX_MODEL_PLACEHOLDER", cancellationToken);

            return modelPath;
        }

        /// <inheritdoc/>
        protected override string GetLocalModelPath(string modelId)
        {
            return Path.Combine(_modelsDirectory, modelId.Replace('/', '_'));
        }

        /// <inheritdoc/>
        protected override bool IsModelDownloaded(string modelId)
        {
            return _modelRegistry.ContainsKey(modelId) || 
                   File.Exists(GetONNXModelPath(modelId));
        }

        /// <inheritdoc/>
        protected override async Task ApplyProviderDefaultsAsync(FoundationModelConfig config, string modelId)
        {
            await base.ApplyProviderDefaultsAsync(config, modelId);

            // ONNX-specific defaults
            config.OptimizationLevel = Enums.OptimizationLevel.O2;
            config.UseMixedPrecision = true;
            
            // Enable graph optimization for ONNX
            config.CustomConfig["enable_graph_optimization"] = true;
            config.CustomConfig["graph_optimization_level"] = "all";
            
            // Set execution providers based on device
            if (config.Device.StartsWith("cuda"))
            {
                config.CustomConfig["execution_providers"] = new[] { "CUDAExecutionProvider", "CPUExecutionProvider" };
            }
            else if (config.Device.StartsWith("directml"))
            {
                config.CustomConfig["execution_providers"] = new[] { "DmlExecutionProvider", "CPUExecutionProvider" };
            }
            else
            {
                config.CustomConfig["execution_providers"] = new[] { "CPUExecutionProvider" };
            }
        }

        #region Private Methods

        private void ScanForONNXModels()
        {
            if (!Directory.Exists(_modelsDirectory))
            {
                return;
            }

            // Scan for .onnx files
            var onnxFiles = Directory.GetFiles(_modelsDirectory, "*.onnx", SearchOption.AllDirectories);
            
            foreach (var onnxFile in onnxFiles)
            {
                var modelId = Path.GetFileNameWithoutExtension(onnxFile);
                var relativePath = Path.GetRelativePath(_modelsDirectory, onnxFile);
                
                // Try to infer architecture from path or filename
                var architecture = InferArchitecture(relativePath);
                
                _modelRegistry[modelId] = new ONNXModelInfo
                {
                    ModelId = modelId,
                    Name = modelId,
                    ModelPath = onnxFile,
                    Architecture = architecture,
                    MaxSequenceLength = 512 // Default
                };

                _logger.Debug("Found ONNX model: {ModelId} at {Path}", modelId, onnxFile);
            }
        }

        private string GetONNXModelPath(string modelId)
        {
            // First check if it's already a full path
            if (Path.IsPathRooted(modelId) && modelId.EndsWith(".onnx"))
            {
                return modelId;
            }

            // Standard path
            return Path.Combine(GetLocalModelPath(modelId), "model.onnx");
        }

        private string InferArchitecture(string path)
        {
            var lowerPath = path.ToLower();
            
            if (lowerPath.Contains("bert")) return "bert";
            if (lowerPath.Contains("gpt")) return "gpt2";
            if (lowerPath.Contains("t5")) return "t5";
            if (lowerPath.Contains("clip")) return "clip";
            if (lowerPath.Contains("roberta")) return "roberta";
            if (lowerPath.Contains("distilbert")) return "distilbert";
            if (lowerPath.Contains("albert")) return "albert";
            if (lowerPath.Contains("resnet")) return "resnet";
            if (lowerPath.Contains("efficientnet")) return "efficientnet";
            
            return "unknown";
        }

        private List<string> GetSupportedTasks(string architecture)
        {
            return architecture.ToLower() switch
            {
                "bert" or "roberta" or "distilbert" or "albert" => 
                    new List<string> { "text-classification", "token-classification", "question-answering", "embeddings" },
                "gpt2" => 
                    new List<string> { "text-generation" },
                "t5" or "bart" or "mbart" => 
                    new List<string> { "text2text-generation", "summarization", "translation" },
                "clip" => 
                    new List<string> { "zero-shot-image-classification", "image-text-similarity" },
                "resnet" or "efficientnet" => 
                    new List<string> { "image-classification" },
                _ => new List<string> { "unknown" }
            };
        }

        private long GetModelSize(string modelPath)
        {
            try
            {
                if (File.Exists(modelPath))
                {
                    return new FileInfo(modelPath).Length;
                }
            }
            catch (Exception ex)
            {
                _logger.Warning("Failed to get model size for {Path}: {Error}", modelPath, ex.Message);
            }
            
            return 0;
        }

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
                var query = filter.SearchQuery.ToLower();
                return info.Name.ToLower().Contains(query) || 
                       info.Description.ToLower().Contains(query) ||
                       info.ModelId.ToLower().Contains(query);
            }

            return true;
        }

        #endregion

        /// <summary>
        /// ONNX model information
        /// </summary>
        private class ONNXModelInfo
        {
            public string ModelId { get; set; } = string.Empty;
            public string? Name { get; set; }
            public string ModelPath { get; set; } = string.Empty;
            public string Architecture { get; set; } = "unknown";
            public long ParameterCount { get; set; }
            public int? MaxSequenceLength { get; set; }
            public int? OpsetVersion { get; set; }
            public string? License { get; set; }
        }
    }
}