using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Logging;
using AiDotNet.FoundationModels.Models;
using AiDotNet.MultimodalAI;

namespace AiDotNet.FoundationModels.Providers;

/// <summary>
/// Provider for loading models from HuggingFace Hub
/// </summary>
public class HuggingFaceModelProvider : FoundationModelProviderBase
{
    private readonly string _cacheDirectory;
    private readonly string _apiToken;
    private readonly HttpClient _httpClient;
    private const string HF_API_BASE = "https://huggingface.co/api";
    private const string HF_HUB_BASE = "https://huggingface.co";

    /// <inheritdoc/>
    public override string ProviderName => "HuggingFace";

    /// <inheritdoc/>
    public override IReadOnlyList<string> SupportedArchitectures => new[]
    {
        "bert", "gpt2", "gpt", "t5", "roberta", "distilbert", "albert",
        "xlm", "xlm-roberta", "bart", "mbart", "pegasus", "marian",
        "llama", "mistral", "mixtral", "falcon", "opt", "bloom",
        "whisper", "clip", "vit", "wav2vec2", "layoutlm", "deberta"
    };

    /// <summary>
    /// Initializes a new instance of the HuggingFaceModelProvider class
    /// </summary>
    /// <param name="apiToken">HuggingFace API token</param>
    /// <param name="cacheDirectory">Directory to cache downloaded models</param>
    /// <param name="logger">Logger instance</param>
    public HuggingFaceModelProvider(
        string? apiToken = null, 
        string? cacheDirectory = null, 
        ILogging? logger = null)
        : base(logger)
    {
        _apiToken = apiToken ?? Environment.GetEnvironmentVariable("HUGGINGFACE_TOKEN") ?? "";
        _cacheDirectory = cacheDirectory ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface");
        
        _httpClient = new HttpClient();
        if (!string.IsNullOrEmpty(_apiToken))
        {
            _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {_apiToken}");
        }
        
        _modelCache = new Dictionary<string, HuggingFaceModelInfo>(StringComparer.OrdinalIgnoreCase);
        
        // Ensure cache directory exists
        Directory.CreateDirectory(_cacheDirectory);
    }

    /// <inheritdoc/>
    public override async Task<bool> IsModelAvailableAsync(string modelId)
    {
        try
        {
            // Check local cache first
            if (IsModelCached(modelId))
            {
                return true;
            }

            // Check HuggingFace Hub
            var response = await _httpClient.GetAsync($"{HF_API_BASE}/models/{modelId}");
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            _logger.Warning("Failed to check model availability for {ModelId}: {Error}", 
                modelId, ex.Message);
            return false;
        }
    }

    /// <inheritdoc/>
    public override async Task<IReadOnlyList<FoundationModelInfo>> ListAvailableModelsAsync(
        ModelFilter? filter = null)
    {
        var models = new List<FoundationModelInfo>();

        try
        {
            // Build query parameters
            var queryParams = new List<string>();
            
            if (filter != null && !string.IsNullOrEmpty(filter.Task))
            {
                queryParams.Add($"filter={filter.Task}");
            }
            
            if (filter != null && !string.IsNullOrEmpty(filter.SearchQuery))
            {
                queryParams.Add($"search={Uri.EscapeDataString(filter.SearchQuery)}");
            }

            var query = queryParams.Count > 0 ? $"?{string.Join("&", queryParams)}" : "";
            var response = await _httpClient.GetAsync($"{HF_API_BASE}/models{query}");
            
            if (response.IsSuccessStatusCode)
            {
                var json = await response.Content.ReadAsStringAsync();
                var hfModels = JsonSerializer.Deserialize<List<HuggingFaceApiModel>>(json);
                
                if (hfModels != null)
                {
                    foreach (var hfModel in hfModels)
                    {
                        var info = ConvertToModelInfo(hfModel);
                        if (MatchesFilter(info, filter))
                        {
                            models.Add(info);
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.Error("Failed to list models from HuggingFace: {Error}", ex.Message);
        }

        // Add cached models
        var cachedModels = GetCachedModels();
        foreach (var cachedModel in cachedModels)
        {
            if (MatchesFilter(cachedModel, filter) && 
                !models.Any(m => m.ModelId == cachedModel.ModelId))
            {
                models.Add(cachedModel);
            }
        }

        return models;
    }

    /// <inheritdoc/>
    public override async Task<bool> ValidateConnectionAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync($"{HF_API_BASE}/models?limit=1");
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            _logger.Error("Failed to validate HuggingFace connection: {Error}", ex.Message);
            return false;
        }
    }

    /// <inheritdoc/>
    protected override async Task<IFoundationModel<T>> LoadModelInternalAsync<T>(
        string modelId,
        FoundationModelConfig config,
        CancellationToken cancellationToken)
    {
        // Check if model is cached
        var modelPath = GetLocalModelPath(modelId);
        
        if (!IsModelCached(modelId))
        {
            // Download model if not cached
            await DownloadModelInternalAsync(modelId, null, cancellationToken);
        }

        // Get model info
        var modelInfo = await GetModelInfoAsync(modelId);
        
        // In a real implementation, this would load the actual model
        // For now, return appropriate mock based on architecture
        var architecture = modelInfo?.Architecture ?? GetArchitectureFromModelId(modelId);
        
        return architecture.ToLower() switch
        {
            "bert" or "roberta" or "distilbert" or "albert" => new BERTFoundationModel<T>(),
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
        var modelPath = GetLocalModelPath(modelId);
        Directory.CreateDirectory(modelPath);

        _logger.Information("Downloading model {ModelId} from HuggingFace", modelId);

        try
        {
            // Get model files list
            var files = await GetModelFilesAsync(modelId);
            var totalSize = files.Sum(f => f.Size);
            var downloadedSize = 0L;

            foreach (var file in files)
            {
                var filePath = Path.Combine(modelPath, file.Name);
                var fileDir = Path.GetDirectoryName(filePath);
                
                if (!string.IsNullOrEmpty(fileDir))
                {
                    Directory.CreateDirectory(fileDir);
                }

                // Download file
                await DownloadFileAsync(
                    modelId, 
                    file.Name, 
                    filePath,
                    (bytesDownloaded) =>
                    {
                        var currentTotal = downloadedSize + bytesDownloaded;
                        ReportProgress(
                            progressCallback,
                            totalSize,
                            currentTotal,
                            1024 * 1024, // 1 MB/s placeholder
                            file.Name
                        );
                    },
                    cancellationToken
                );

                downloadedSize += file.Size;
            }

            // Save model metadata
            await SaveModelMetadataAsync(modelId, modelPath);

            return modelPath;
        }
        catch (Exception ex)
        {
            _logger.Error("Failed to download model {ModelId}: {Error}", modelId, ex.Message);
            
            // Clean up partial download
            if (Directory.Exists(modelPath))
            {
                Directory.Delete(modelPath, true);
            }
            
            throw;
        }
    }

    /// <inheritdoc/>
    protected override string GetLocalModelPath(string modelId)
    {
        return Path.Combine(_cacheDirectory, "hub", modelId.Replace('/', '_'));
    }

    /// <inheritdoc/>
    protected override bool IsModelDownloaded(string modelId)
    {
        return IsModelCached(modelId);
    }

    #region Private Methods

    private bool IsModelCached(string modelId)
    {
        var modelPath = GetLocalModelPath(modelId);
        return Directory.Exists(modelPath) && 
               File.Exists(Path.Combine(modelPath, "config.json"));
    }

    private async Task<HuggingFaceModelInfo?> GetModelInfoAsync(string modelId)
    {
        if (_modelCache.TryGetValue(modelId, out var cachedInfo))
        {
            return cachedInfo;
        }

        try
        {
            var response = await _httpClient.GetAsync($"{HF_API_BASE}/models/{modelId}");
            if (response.IsSuccessStatusCode)
            {
                var json = await response.Content.ReadAsStringAsync();
                var apiModel = JsonSerializer.Deserialize<HuggingFaceApiModel>(json);
                
                if (apiModel != null)
                {
                    var info = new HuggingFaceModelInfo
                    {
                        ModelId = apiModel.ModelId ?? modelId,
                        Architecture = InferArchitecture(apiModel),
                        Tags = apiModel.Tags ?? new List<string>(),
                        Downloads = apiModel.Downloads ?? 0
                    };
                    
                    _modelCache[modelId] = info;
                    return info;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.Warning("Failed to get model info for {ModelId}: {Error}", modelId, ex.Message);
        }

        return null;
    }

    private async Task<List<ModelFile>> GetModelFilesAsync(string modelId)
    {
        var files = new List<ModelFile>();

        try
        {
            var response = await _httpClient.GetAsync($"{HF_API_BASE}/models/{modelId}/tree/main");
            if (response.IsSuccessStatusCode)
            {
                var json = await response.Content.ReadAsStringAsync();
                var tree = JsonSerializer.Deserialize<List<TreeItem>>(json);
                
                if (tree != null)
                {
                    foreach (var item in tree.Where(t => t.Type == "file"))
                    {
                        files.Add(new ModelFile
                        {
                            Name = item.Path ?? "",
                            Size = item.Size ?? 0
                        });
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.Warning("Failed to get model files for {ModelId}: {Error}", modelId, ex.Message);
        }

        // If API fails, return minimal set of files
        if (files.Count == 0)
        {
            files.Add(new ModelFile { Name = "config.json", Size = 1024 });
            files.Add(new ModelFile { Name = "model.safetensors", Size = 100 * 1024 * 1024 });
        }

        return files;
    }

    private async Task DownloadFileAsync(
        string modelId,
        string fileName,
        string localPath,
        Action<long> progressCallback,
        CancellationToken cancellationToken)
    {
        var url = $"{HF_HUB_BASE}/{modelId}/resolve/main/{fileName}";
        
        // Simulate download for now
        await Task.Delay(100, cancellationToken);
        progressCallback(1024 * 1024);
        
        // Create placeholder file
        await File.WriteAllTextAsync(localPath, $"PLACEHOLDER_{fileName}", cancellationToken);
    }

    private async Task SaveModelMetadataAsync(string modelId, string modelPath)
    {
        var metadata = new
        {
            modelId = modelId,
            downloadDate = DateTime.UtcNow,
            provider = "huggingface"
        };

        var metadataPath = Path.Combine(modelPath, ".metadata.json");
        var json = JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(metadataPath, json);
    }

    private List<FoundationModelInfo> GetCachedModels()
    {
        var models = new List<FoundationModelInfo>();

        if (Directory.Exists(Path.Combine(_cacheDirectory, "hub")))
        {
            foreach (var modelDir in Directory.GetDirectories(Path.Combine(_cacheDirectory, "hub")))
            {
                try
                {
                    var modelId = Path.GetFileName(modelDir).Replace('_', '/');
                    var configPath = Path.Combine(modelDir, "config.json");
                    
                    if (File.Exists(configPath))
                    {
                        models.Add(new FoundationModelInfo
                        {
                            ModelId = modelId,
                            Name = modelId,
                            Architecture = GetArchitectureFromModelId(modelId),
                            Description = "Cached HuggingFace model",
                            IsOfflineAvailable = true,
                            SupportedTasks = new List<string>()
                        });
                    }
                }
                catch (Exception ex)
                {
                    _logger.Warning("Failed to load cached model info: {Error}", ex.Message);
                }
            }
        }

        return models;
    }

    private FoundationModelInfo ConvertToModelInfo(HuggingFaceApiModel apiModel)
    {
        return new FoundationModelInfo
        {
            ModelId = apiModel.ModelId ?? "",
            Name = apiModel.ModelId ?? "",
            Architecture = InferArchitecture(apiModel),
            Description = $"HuggingFace model with {apiModel.Downloads ?? 0} downloads",
            SupportedTasks = apiModel.Tags ?? new List<string>(),
            IsOfflineAvailable = IsModelCached(apiModel.ModelId ?? ""),
            License = apiModel.License ?? "unknown",
            Metadata = new Dictionary<string, object>
            {
                ["downloads"] = apiModel.Downloads ?? 0,
                ["likes"] = apiModel.Likes ?? 0,
                ["tags"] = apiModel.Tags ?? new List<string>()
            }
        };
    }

    private string InferArchitecture(HuggingFaceApiModel apiModel)
    {
        // Check tags first
        if (apiModel.Tags != null)
        {
            foreach (var tag in apiModel.Tags)
            {
                if (SupportedArchitectures.Contains(tag, StringComparer.OrdinalIgnoreCase))
                {
                    return tag;
                }
            }
        }

        // Fall back to model ID
        return GetArchitectureFromModelId(apiModel.ModelId ?? "");
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

    #region Private Classes

    private class HuggingFaceApiModel
    {
        public string? ModelId { get; set; }
        public List<string>? Tags { get; set; }
        public int? Downloads { get; set; }
        public int? Likes { get; set; }
        public string? License { get; set; }
    }

    private class TreeItem
    {
        public string? Type { get; set; }
        public string? Path { get; set; }
        public long? Size { get; set; }
    }

    private class ModelFile
    {
        public string Name { get; set; } = "";
        public long Size { get; set; }
    }

    private class HuggingFaceModelInfo
    {
        public string ModelId { get; set; } = "";
        public string Architecture { get; set; } = "";
        public List<string> Tags { get; set; } = new();
        public int Downloads { get; set; }
    }

    #endregion
}