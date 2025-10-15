namespace AiDotNet.FoundationModels.Providers;

/// <summary>
/// Base class for foundation model providers
/// </summary>
public abstract class FoundationModelProviderBase : IFoundationModelProvider
{
    protected readonly ILogging _logger;
    protected readonly Dictionary<string, object> _modelCache;
    protected readonly object _cacheLock = new object();

    /// <inheritdoc/>
    public abstract string ProviderName { get; }

    /// <inheritdoc/>
    public abstract IReadOnlyList<string> SupportedArchitectures { get; }

    /// <summary>
    /// Initializes a new instance of the FoundationModelProviderBase class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    protected FoundationModelProviderBase(ILogging? logger = null)
    {
        _logger = logger ?? LoggingFactory.Current;
        _modelCache = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
    }

    /// <inheritdoc/>
    public abstract Task<bool> IsModelAvailableAsync(string modelId);

    /// <inheritdoc/>
    public virtual async Task<IFoundationModel<T>> GetModelAsync<T>(
        string modelId, 
        FoundationModelConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            throw new ArgumentException("Model ID cannot be null or empty", nameof(modelId));
        }

        // Check cache first
        lock (_cacheLock)
        {
            var cacheKey = $"{modelId}_{typeof(T).Name}";
            if (_modelCache.TryGetValue(cacheKey, out var cachedModel) && cachedModel is IFoundationModel<T> typedModel)
            {
                _logger.Debug("Returning cached model: {ModelId}", modelId);
                return typedModel;
            }
        }

        // Validate model availability
        if (!await IsModelAvailableAsync(modelId))
        {
            throw new InvalidOperationException($"Model {modelId} is not available from provider {ProviderName}");
        }

        // Create configuration if not provided
        config ??= await GetDefaultConfigAsync(modelId);
        config.Validate();

        // Load the model
        _logger.Information("Loading model {ModelId} from provider {Provider}", modelId, ProviderName);
        var model = await LoadModelInternalAsync<T>(modelId, config, cancellationToken);

        // Cache the model
        lock (_cacheLock)
        {
            var cacheKey = $"{modelId}_{typeof(T).Name}";
            _modelCache[cacheKey] = model;
        }

        return model;
    }

    /// <inheritdoc/>
    public abstract Task<IReadOnlyList<FoundationModelInfo>> ListAvailableModelsAsync(
        ModelFilter? filter = null);

    /// <inheritdoc/>
    public virtual async Task<string> DownloadModelAsync(
        string modelId,
        Action<DownloadProgress>? progressCallback = null,
        CancellationToken cancellationToken = default)
    {
        _logger.Information("Downloading model {ModelId} from {Provider}", modelId, ProviderName);
        
        // Check if already downloaded
        var localPath = GetLocalModelPath(modelId);
        if (IsModelDownloaded(modelId))
        {
            _logger.Information("Model {ModelId} already downloaded at {Path}", modelId, localPath);
            return localPath;
        }

        // Download the model
        return await DownloadModelInternalAsync(modelId, progressCallback, cancellationToken);
    }

    /// <inheritdoc/>
    public abstract Task<bool> ValidateConnectionAsync();

    /// <inheritdoc/>
    public virtual async Task<FoundationModelConfig> GetDefaultConfigAsync(string modelId)
    {
        var config = FoundationModelConfig.CreateDefault(modelId);
        
        // Apply provider-specific defaults
        await ApplyProviderDefaultsAsync(config, modelId);
        
        return config;
    }

    /// <inheritdoc/>
    public virtual Task ReleaseModelAsync(string modelId)
    {
        lock (_cacheLock)
        {
            if (_modelCache.TryGetValue(modelId, out var model))
            {
                _modelCache.Remove(modelId);
                
                // Dispose if model implements IDisposable
                if (model is IDisposable disposable)
                {
                    disposable.Dispose();
                }
                
                _logger.Information("Released model {ModelId} from {Provider}", modelId, ProviderName);
            }
        }

        return Task.CompletedTask;
    }

    #region Protected Abstract Methods

    /// <summary>
    /// Loads a model internally
    /// </summary>
    protected abstract Task<IFoundationModel<T>> LoadModelInternalAsync<T>(
        string modelId,
        FoundationModelConfig config,
        CancellationToken cancellationToken);

    /// <summary>
    /// Downloads a model internally
    /// </summary>
    protected abstract Task<string> DownloadModelInternalAsync(
        string modelId,
        Action<DownloadProgress>? progressCallback,
        CancellationToken cancellationToken);

    /// <summary>
    /// Gets the local path for a model
    /// </summary>
    protected abstract string GetLocalModelPath(string modelId);

    /// <summary>
    /// Checks if a model is downloaded
    /// </summary>
    protected abstract bool IsModelDownloaded(string modelId);

    /// <summary>
    /// Applies provider-specific defaults to configuration
    /// </summary>
    protected virtual Task ApplyProviderDefaultsAsync(FoundationModelConfig config, string modelId)
    {
        // Override in derived classes to apply provider-specific defaults
        return Task.CompletedTask;
    }

    #endregion

    #region Protected Helper Methods

    /// <summary>
    /// Reports download progress
    /// </summary>
    protected void ReportProgress(
        Action<DownloadProgress>? callback,
        long totalBytes,
        long bytesDownloaded,
        long bytesPerSecond,
        string currentFile)
    {
        if (callback == null) return;

        var remainingBytes = totalBytes - bytesDownloaded;
        var estimatedSeconds = bytesPerSecond > 0 ? remainingBytes / bytesPerSecond : 0;

        callback(new DownloadProgress
        {
            TotalBytes = totalBytes,
            BytesDownloaded = bytesDownloaded,
            BytesPerSecond = bytesPerSecond,
            EstimatedTimeRemaining = TimeSpan.FromSeconds(estimatedSeconds),
            CurrentFile = currentFile
        });
    }

    /// <summary>
    /// Validates model architecture
    /// </summary>
    protected bool IsArchitectureSupported(string architecture)
    {
        return SupportedArchitectures.Contains(architecture, StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Gets model architecture from model ID
    /// </summary>
    protected virtual string GetArchitectureFromModelId(string modelId)
    {
        // Common patterns
        if (modelId.IndexOf("bert", StringComparison.OrdinalIgnoreCase) >= 0)
            return "bert";
        if (modelId.IndexOf("gpt", StringComparison.OrdinalIgnoreCase) >= 0)
            return "gpt";
        if (modelId.IndexOf("t5", StringComparison.OrdinalIgnoreCase) >= 0)
            return "t5";
        if (modelId.IndexOf("llama", StringComparison.OrdinalIgnoreCase) >= 0)
            return "llama";
        
        return "unknown";
    }

    #endregion
}