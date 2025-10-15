using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Models.Options;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for providers that can create and manage foundation models.
    /// Handles model loading, caching, and lifecycle management.
    /// </summary>
    public interface IFoundationModelProvider
    {
        /// <summary>
        /// Gets the name of the provider (e.g., "HuggingFace", "OpenAI", "Local")
        /// </summary>
        string ProviderName { get; }

        /// <summary>
        /// Gets the list of supported model architectures
        /// </summary>
        IReadOnlyList<string> SupportedArchitectures { get; }

        /// <summary>
        /// Checks if a specific model is available
        /// </summary>
        /// <param name="modelId">The model identifier</param>
        /// <returns>True if the model is available</returns>
        Task<bool> IsModelAvailableAsync(string modelId);

        /// <summary>
        /// Gets a foundation model instance
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations</typeparam>
        /// <param name="modelId">The model identifier</param>
        /// <param name="config">Optional configuration for the model</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>The foundation model instance</returns>
        Task<IFoundationModel<T>> GetModelAsync<T>(
            string modelId, 
            FoundationModelConfig? config = null,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Lists all available models from this provider
        /// </summary>
        /// <param name="filter">Optional filter for model types</param>
        /// <returns>List of available model information</returns>
        Task<IReadOnlyList<FoundationModelInfo>> ListAvailableModelsAsync(
            ModelFilter? filter = null);

        /// <summary>
        /// Downloads a model for offline use
        /// </summary>
        /// <param name="modelId">The model identifier</param>
        /// <param name="progressCallback">Progress callback</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Path where the model was downloaded</returns>
        Task<string> DownloadModelAsync(
            string modelId,
            System.Action<DownloadProgress>? progressCallback = null,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Validates provider configuration and connectivity
        /// </summary>
        /// <returns>True if provider is properly configured</returns>
        Task<bool> ValidateConnectionAsync();

        /// <summary>
        /// Gets the default configuration for a model
        /// </summary>
        /// <param name="modelId">The model identifier</param>
        /// <returns>Default configuration for the model</returns>
        Task<FoundationModelConfig> GetDefaultConfigAsync(string modelId);

        /// <summary>
        /// Releases resources associated with a model
        /// </summary>
        /// <param name="modelId">The model identifier</param>
        Task ReleaseModelAsync(string modelId);
    }

    /// <summary>
    /// Information about an available foundation model
    /// </summary>
    public class FoundationModelInfo
    {
        /// <summary>
        /// Unique identifier for the model
        /// </summary>
        public string ModelId { get; set; } = string.Empty;

        /// <summary>
        /// Display name of the model
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Model architecture type
        /// </summary>
        public string Architecture { get; set; } = string.Empty;

        /// <summary>
        /// Number of parameters in the model
        /// </summary>
        public long ParameterCount { get; set; }

        /// <summary>
        /// Model description
        /// </summary>
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// Supported tasks (e.g., "text-generation", "embeddings")
        /// </summary>
        public List<string> SupportedTasks { get; set; } = new List<string>();

        /// <summary>
        /// Maximum context length
        /// </summary>
        public int MaxContextLength { get; set; }

        /// <summary>
        /// Model size in bytes
        /// </summary>
        public long ModelSizeBytes { get; set; }

        /// <summary>
        /// Whether the model is available offline
        /// </summary>
        public bool IsOfflineAvailable { get; set; }

        /// <summary>
        /// License information
        /// </summary>
        public string License { get; set; } = string.Empty;

        /// <summary>
        /// Additional metadata
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
    }

    /// <summary>
    /// Filter for querying available models
    /// </summary>
    public class ModelFilter
    {
        /// <summary>
        /// Filter by architecture type
        /// </summary>
        public string? Architecture { get; set; }

        /// <summary>
        /// Filter by supported task
        /// </summary>
        public string? Task { get; set; }

        /// <summary>
        /// Minimum parameter count
        /// </summary>
        public long? MinParameters { get; set; }

        /// <summary>
        /// Maximum parameter count
        /// </summary>
        public long? MaxParameters { get; set; }

        /// <summary>
        /// Only show offline available models
        /// </summary>
        public bool? OfflineOnly { get; set; }

        /// <summary>
        /// Search query for model name/description
        /// </summary>
        public string? SearchQuery { get; set; }
    }

    /// <summary>
    /// Progress information for model downloads
    /// </summary>
    public class DownloadProgress
    {
        /// <summary>
        /// Total bytes to download
        /// </summary>
        public long TotalBytes { get; set; }

        /// <summary>
        /// Bytes downloaded so far
        /// </summary>
        public long BytesDownloaded { get; set; }

        /// <summary>
        /// Download progress percentage (0-100)
        /// </summary>
        public double ProgressPercentage => TotalBytes > 0 ? (BytesDownloaded * 100.0 / TotalBytes) : 0;

        /// <summary>
        /// Current download speed in bytes per second
        /// </summary>
        public long BytesPerSecond { get; set; }

        /// <summary>
        /// Estimated time remaining
        /// </summary>
        public System.TimeSpan EstimatedTimeRemaining { get; set; }

        /// <summary>
        /// Current file being downloaded
        /// </summary>
        public string CurrentFile { get; set; } = string.Empty;
    }
}