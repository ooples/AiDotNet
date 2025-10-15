using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Logging;
using AiDotNet.FoundationModels.Models;
using AiDotNet.FoundationModels.Tokenizers;
using AiDotNet.FoundationModels.Providers;

namespace AiDotNet.Factories
{
    /// <summary>
    /// Factory for creating foundation model instances.
    /// Manages model providers and handles model instantiation.
    /// </summary>
    public class FoundationModelFactory
    {
        private readonly Dictionary<string, IFoundationModelProvider> _providers;
        private readonly ILogging _logger;
        private readonly Dictionary<string, object> _modelCache;
        private readonly object _cacheLock = new object();

        /// <summary>
        /// Initializes a new instance of the FoundationModelFactory class
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public FoundationModelFactory(ILogging? logger = null)
        {
            _providers = new Dictionary<string, IFoundationModelProvider>(StringComparer.OrdinalIgnoreCase);
            _logger = logger ?? new AiDotNetLogger();
            _modelCache = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
            
            // Register default providers
            RegisterDefaultProviders();
        }

        #region Public Methods

        /// <summary>
        /// Registers a foundation model provider
        /// </summary>
        /// <param name="provider">The provider to register</param>
        public void RegisterProvider(IFoundationModelProvider provider)
        {
            if (provider == null)
            {
                throw new ArgumentNullException(nameof(provider));
            }

            _providers[provider.ProviderName] = provider;
            _logger.Information("Registered foundation model provider: {Provider}", provider.ProviderName);
        }

        /// <summary>
        /// Creates a foundation model instance
        /// </summary>
        /// <param name="modelId">Model identifier (e.g., "bert-base-uncased", "gpt2")</param>
        /// <param name="config">Optional model configuration</param>
        /// <param name="useCache">Whether to use cached instances</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Foundation model instance</returns>
        public async Task<IFoundationModel<T>> CreateModelAsync<T>(
            string modelId,
            FoundationModelConfig? config = null,
            bool useCache = true,
            CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(modelId))
            {
                throw new ArgumentException("Model ID cannot be null or empty", nameof(modelId));
            }

            // Check cache first
            if (useCache)
            {
                lock (_cacheLock)
                {
                    if (_modelCache.TryGetValue(modelId, out var cachedModel))
                    {
                        _logger.Debug("Returning cached model: {ModelId}", modelId);
                        return cachedModel;
                    }
                }
            }

            // Find appropriate provider
            var provider = await FindProviderForModelAsync(modelId);
            if (provider == null)
            {
                // Try to create a mock model for development/testing
                if (IsMockModel(modelId))
                {
                    return CreateMockModel(modelId);
                }

                throw new InvalidOperationException($"No provider found for model: {modelId}");
            }

            // Create model instance
            _logger.Information("Creating model {ModelId} using provider {Provider}", 
                modelId, provider.ProviderName);

            var model = await provider.GetModelAsync(modelId, config, cancellationToken);

            // Cache if requested
            if (useCache)
            {
                lock (_cacheLock)
                {
                    _modelCache[modelId] = model;
                }
            }

            return model;
        }

        /// <summary>
        /// Lists all available models across all providers
        /// </summary>
        /// <param name="filter">Optional filter</param>
        /// <returns>List of available models</returns>
        public async Task<IReadOnlyList<FoundationModelInfo>> ListAvailableModelsAsync(ModelFilter? filter = null)
        {
            var allModels = new List<FoundationModelInfo>();

            foreach (var provider in _providers.Values)
            {
                try
                {
                    var models = await provider.ListAvailableModelsAsync(filter);
                    allModels.AddRange(models);
                }
                catch (Exception ex)
                {
                    _logger.Warning("Failed to list models from provider {Provider}: {Error}", 
                        provider.ProviderName, ex.Message);
                }
            }

            return allModels;
        }

        /// <summary>
        /// Gets a specific provider by name
        /// </summary>
        /// <param name="providerName">Provider name</param>
        /// <returns>Provider instance or null if not found</returns>
        public IFoundationModelProvider? GetProvider(string providerName)
        {
            return _providers.TryGetValue(providerName, out var provider) ? provider : null;
        }

        /// <summary>
        /// Lists all registered providers
        /// </summary>
        /// <returns>List of provider names</returns>
        public IReadOnlyList<string> GetRegisteredProviders()
        {
            return _providers.Keys.ToList();
        }

        /// <summary>
        /// Clears the model cache
        /// </summary>
        public void ClearCache()
        {
            lock (_cacheLock)
            {
                _modelCache.Clear();
            }
            
            _logger.Information("Model cache cleared");
        }

        /// <summary>
        /// Releases a specific model from cache
        /// </summary>
        /// <param name="modelId">Model ID to release</param>
        public async Task ReleaseModelAsync(string modelId)
        {
            lock (_cacheLock)
            {
                _modelCache.Remove(modelId);
            }

            // Also release from provider
            foreach (var provider in _providers.Values)
            {
                try
                {
                    await provider.ReleaseModelAsync(modelId);
                }
                catch (Exception ex)
                {
                    _logger.Warning("Failed to release model {ModelId} from provider {Provider}: {Error}",
                        modelId, provider.ProviderName, ex.Message);
                }
            }
        }

        #endregion

        #region Private Methods

        private void RegisterDefaultProviders()
        {
            // Register built-in providers
            
            // Local model provider for models stored on disk
            RegisterProvider(new LocalModelProvider(logger: _logger));
            
            // ONNX Runtime provider
            RegisterProvider(new ONNXModelProvider(logger: _logger));
            
            // HuggingFace provider (only if API token is available)
            var hfToken = Environment.GetEnvironmentVariable("HUGGINGFACE_TOKEN");
            if (!string.IsNullOrEmpty(hfToken))
            {
                RegisterProvider(new HuggingFaceModelProvider(apiToken: hfToken, logger: _logger));
            }
        }

        private async Task<IFoundationModelProvider?> FindProviderForModelAsync(string modelId)
        {
            foreach (var provider in _providers.Values)
            {
                try
                {
                    if (await provider.IsModelAvailableAsync(modelId))
                    {
                        return provider;
                    }
                }
                catch (Exception ex)
                {
                    _logger.Warning("Error checking model availability with provider {Provider}: {Error}",
                        provider.ProviderName, ex.Message);
                }
            }

            return null;
        }

        private bool IsMockModel(string modelId)
        {
            var mockModels = new[] { "bert-base-uncased", "gpt2", "mock-model", "test-model" };
            return mockModels.Contains(modelId.ToLowerInvariant());
        }

        private IFoundationModel<T> CreateMockModel<T>(string modelId)
        {
            _logger.Information("Creating mock model for: {ModelId}", modelId);
            
            // Return the existing BERTFoundationModel
            return new BERTFoundationModel<double>();
        }

        #endregion

        #region Static Factory Methods

        /// <summary>
        /// Creates a foundation model with default configuration
        /// </summary>
        /// <param name="modelId">Model identifier</param>
        /// <returns>Foundation model instance</returns>
        public static async Task<IFoundationModel<T>> CreateAsync<T>(string modelId)
        {
            var factory = new FoundationModelFactory();
            return await factory.CreateModelAsync(modelId);
        }

        /// <summary>
        /// Creates a foundation model with custom configuration
        /// </summary>
        /// <param name="modelId">Model identifier</param>
        /// <param name="configureOptions">Configuration action</param>
        /// <returns>Foundation model instance</returns>
        public static async Task<IFoundationModel<T>> CreateAsync<T>(
            string modelId, 
            Action<FoundationModelConfig> configureOptions)
        {
            var config = FoundationModelConfig.CreateDefault(modelId);
            configureOptions(config);
            
            var factory = new FoundationModelFactory();
            return await factory.CreateModelAsync(modelId, config);
        }

        /// <summary>
        /// Creates a tokenizer for the specified model
        /// </summary>
        /// <param name="modelId">Model identifier</param>
        /// <param name="tokenizerPath">Optional custom tokenizer path</param>
        /// <returns>Tokenizer instance</returns>
        public static async Task<ITokenizer> CreateTokenizerAsync(
            string modelId,
            string? tokenizerPath = null)
        {
            var tokenizer = TokenizerFactory.CreateTokenizerForModel(modelId, tokenizerPath ?? "");
            await tokenizer.InitializeAsync();
            return tokenizer;
        }

        /// <summary>
        /// Creates a foundation model with its associated tokenizer
        /// </summary>
        /// <param name="modelId">Model identifier</param>
        /// <param name="config">Optional model configuration</param>
        /// <returns>Tuple of model and tokenizer</returns>
        public static async Task<(IFoundationModel<T> Model, ITokenizer Tokenizer)> CreateWithTokenizerAsync<T>(
            string modelId,
            FoundationModelConfig? config = null)
        {
            var factory = new FoundationModelFactory();
            var modelTask = factory.CreateModelAsync(modelId, config);
            var tokenizerTask = CreateTokenizerAsync(modelId, config?.CustomTokenizerPath);
            
            await Task.WhenAll(modelTask, tokenizerTask);
            
            return (modelTask.Result, tokenizerTask.Result);
        }

        #endregion
    }
}