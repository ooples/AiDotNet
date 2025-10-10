global using System.Security.Cryptography;

namespace AiDotNet.Caching;

/// <summary>
/// Provides a context-aware implementation of model caching for optimization step data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix<double>).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector<double>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A model cache is like a storage box that keeps track of the progress made during 
/// machine learning model training, but with additional intelligence to know exactly which data belongs together.
/// 
/// This enhanced cache uses a "context-aware" approach, meaning it considers both:
/// 1. The model's current state (parameters, active features, etc.)
/// 2. The characteristics of the data being used (dimensions, sample values, etc.)
/// 
/// This prevents confusion between similar models trained on different data, and ensures accurate
/// retrieval of cached results.
/// 
/// The cache also automatically removes old entries to prevent memory problems during long training sessions.
/// </para>
/// </remarks>
public class DefaultModelCache<T, TInput, TOutput> : IModelCache<T, TInput, TOutput>, IDisposable
{
    /// <summary>
    /// The internal cache storing optimization step data with expiration timestamps.
    /// </summary>
    private readonly ConcurrentDictionary<string, (DateTime Expiration, OptimizationStepData<T, TInput, TOutput> Data)> _cache = new();

    /// <summary>
    /// Configuration options for the cache.
    /// </summary>
    private readonly ModelCacheOptions _options = default!;

    /// <summary>
    /// Tracks the last time the cache was cleaned up.
    /// </summary>
    private DateTime _lastCleanupTime = DateTime.UtcNow;

    /// <summary>
    /// Cancellation token source for background cleanup.
    /// </summary>
    private readonly CancellationTokenSource _cleanupCts = default!;

    /// <summary>
    /// Lock object for cleanup synchronization.
    /// </summary>
    private readonly SemaphoreSlim _cleanupLock = new SemaphoreSlim(1, 1);

    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Initializes a new instance of the <see cref="DefaultModelCache{T, TInput, TOutput}"/> class with default options.
    /// </summary>
    public DefaultModelCache() : this(new ModelCacheOptions())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DefaultModelCache{T, TInput, TOutput}"/> class with the specified options.
    /// </summary>
    /// <param name="options">The cache configuration options.</param>
    /// <exception cref="ArgumentNullException">Thrown if options is null.</exception>
    public DefaultModelCache(ModelCacheOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _cleanupCts = new CancellationTokenSource();

        if (_options.EnableBackgroundCleanup)
        {
            StartBackgroundCleanup();
        }
    }

    /// <summary>
    /// Removes all cached optimization step data from the cache.
    /// </summary>
    public void ClearCache()
    {
        _cache.Clear();
        _lastCleanupTime = DateTime.UtcNow;
    }

    /// <summary>
    /// Retrieves cached optimization step data using the specified key.
    /// </summary>
    /// <param name="key">The unique identifier for the optimization step data.</param>
    /// <returns>The cached optimization step data if found and not expired; otherwise, null.</returns>
    public OptimizationStepData<T, TInput, TOutput>? GetCachedStepData(string key)
    {
        if (_cache.TryGetValue(key, out var cachedItem))
        {
            if (cachedItem.Expiration > DateTime.UtcNow)
            {
                return cachedItem.Data;
            }

            // Remove expired item
            _cache.TryRemove(key, out _);
        }

        return null;
    }

    /// <summary>
    /// Stores optimization step data in the cache with the specified key and default expiration.
    /// </summary>
    /// <param name="key">The unique identifier for the optimization step data.</param>
    /// <param name="stepData">The optimization step data to cache.</param>
    public void CacheStepData(string key, OptimizationStepData<T, TInput, TOutput> stepData)
    {
        CacheStepData(key, stepData, _options.DefaultExpiration);
    }

    /// <summary>
    /// Stores optimization step data in the cache with the specified key and expiration time.
    /// </summary>
    /// <param name="key">The unique identifier for the optimization step data.</param>
    /// <param name="stepData">The optimization step data to cache.</param>
    /// <param name="expiration">How long the data should remain in the cache.</param>
    public void CacheStepData(string key, OptimizationStepData<T, TInput, TOutput> stepData, TimeSpan expiration)
    {
        if (stepData == null) throw new ArgumentNullException(nameof(stepData));

        var expirationTime = DateTime.UtcNow.Add(expiration);
        _cache[key] = (expirationTime, stepData);

        TriggerCleanupIfNeeded().GetAwaiter().GetResult();
    }

    /// <summary>
    /// Generates a unique cache key based on feature selection and input data characteristics.
    /// </summary>
    /// <param name="model">The model being evaluated (pre-training).</param>
    /// <param name="inputData">The data used for evaluation.</param>
    /// <returns>A unique string identifier for caching.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method creates a unique ID that represents a specific combination of
    /// selected features and training data. It allows us to recognize if we've already trained a model
    /// with this exact feature set before, so we don't have to repeat the training process.
    /// </remarks>
    public string GenerateCacheKey(IFullModel<T, TInput, TOutput> model, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (inputData == null) throw new ArgumentNullException(nameof(inputData));

        // Focus on active features and model type for the cache key
        using var sha = SHA256.Create();
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write model type (different model types should have different caches)
        writer.Write(model.GetType().FullName ?? "UnknownType");

        // Write selected features (the primary component for our cache key)
        var activeFeatures = model.GetActiveFeatureIndices().OrderBy(f => f).ToList();
        writer.Write(activeFeatures.Count);
        foreach (var feature in activeFeatures)
        {
            writer.Write(feature);
        }

        // Include basic data characteristics (dimensions and sample values)
        var dataFingerprint = GetDataFingerprint(inputData);
        writer.Write(dataFingerprint);

        ms.Flush();
        var hash = sha.ComputeHash(ms.ToArray());

        return Convert.ToBase64String(hash);
    }

    /// <summary>
    /// Creates a fingerprint of the dataset characteristics.
    /// </summary>
    /// <param name="data">The data to fingerprint.</param>
    /// <returns>A string representation of the data's key characteristics.</returns>
    private string GetDataFingerprint(OptimizationInputData<T, TInput, TOutput> data)
    {
        using var sha = SHA256.Create();
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Include data dimensions
        var trainColumns = InputHelper<T, TInput>.GetInputSize(data.XTrain);
        var trainRows = InputHelper<T, TInput>.GetBatchSize(data.XTrain);
        writer.Write(trainRows);
        writer.Write(trainColumns);

        if (data.XValidation != null && data.YValidation != null)
        {
            var validateColumns = InputHelper<T, TInput>.GetInputSize(data.XValidation);
            var validateRows = InputHelper<T, TInput>.GetBatchSize(data.XValidation);
            writer.Write(validateRows);
            writer.Write(validateColumns);
        }

        // Sample values from training data
        if (data.XTrain != null)
        {
            SampleAndWriteData(writer, data.XTrain);
        }

        if (data.YTrain != null)
        {
            SampleAndWriteData(writer, data.YTrain);
        }

        ms.Flush();
        var hash = sha.ComputeHash(ms.ToArray());
        return Convert.ToBase64String(hash);
    }

    /// <summary>
    /// Samples and writes a small subset of data to help fingerprint the dataset.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="data">The data to sample.</param>
    private void SampleAndWriteData(BinaryWriter writer, dynamic data)
    {
        if (data is Matrix<T> matrix)
        {
            var rows = matrix.Rows;
            var cols = matrix.Columns;
            // Sample corners of the matrix
            if (rows > 0 && cols > 0)
            {
                writer.Write(Convert.ToString(matrix[0, 0]) ?? "null");
                if (cols > 1)
                    writer.Write(Convert.ToString(matrix[0, cols - 1]) ?? "null");
                if (rows > 1)
                    writer.Write(Convert.ToString(matrix[rows - 1, 0]) ?? "null");
                if (rows > 1 && cols > 1)
                    writer.Write(Convert.ToString(matrix[rows - 1, cols - 1]) ?? "null");
                // Sample a middle value if possible
                if (rows > 2 && cols > 2)
                    writer.Write(Convert.ToString(matrix[rows / 2, cols / 2]) ?? "null");
            }
        }
        else if (data is Vector<T> vector)
        {
            var length = vector.Length;
            if (length > 0)
            {
                writer.Write(Convert.ToString(vector[0]) ?? "null");
                if (length > 1)
                    writer.Write(Convert.ToString(vector[length - 1]) ?? "null");
                if (length > 2)
                    writer.Write(Convert.ToString(vector[length / 2]) ?? "null");
            }
        }
        else if (data is Tensor<T> tensor)
        {
            var dimensions = tensor.Shape;

            // Write the dimensions themselves as part of the fingerprint
            writer.Write(dimensions.Length);
            foreach (var dim in dimensions)
            {
                writer.Write(dim);
            }

            if (dimensions.Length == 0 || dimensions.Any(d => d == 0))
                return; // Empty tensor

            // Sample the first element
            var firstIndices = new int[dimensions.Length];
            writer.Write(Convert.ToString(tensor[firstIndices]) ?? "null");

            // Sample the last element
            var lastIndices = dimensions.Select(d => d - 1).ToArray();
            writer.Write(Convert.ToString(tensor[lastIndices]) ?? "null");

            // Sample center element if tensor is big enough
            if (dimensions.All(d => d > 2))
            {
                var centerIndices = dimensions.Select(d => d / 2).ToArray();
                writer.Write(Convert.ToString(tensor[centerIndices]) ?? "null");
            }

            // Sample a few strategic points if it's a 3D tensor
            if (dimensions.Length == 3)
            {
                // Sample from different corners
                if (dimensions[0] > 1 && dimensions[1] > 1 && dimensions[2] > 1)
                {
                    writer.Write(Convert.ToString(tensor[[0, 0, dimensions[2] - 1]]) ?? "null");
                    writer.Write(Convert.ToString(tensor[[0, dimensions[1] - 1, 0]]) ?? "null");
                    writer.Write(Convert.ToString(tensor[[dimensions[0] - 1, 0, 0]]) ?? "null");
                }
            }

            // For 4D or higher tensors, sample additional strategic points
            if (dimensions.Length > 3)
            {
                // Create a few more sampling indices by alternating 0 and max in each dimension
                for (int i = 0; i < Math.Min(3, dimensions.Length); i++)
                {
                    var indices = new int[dimensions.Length];
                    for (int j = 0; j < dimensions.Length; j++)
                    {
                        indices[j] = (i + j) % 2 == 0 ? 0 : dimensions[j] - 1;
                    }
                    writer.Write(Convert.ToString(tensor[indices]) ?? "null");
                }
            }
        }
    }

    /// <summary>
    /// Triggers cleanup if the cache exceeds the threshold size or if sufficient time has elapsed.
    /// </summary>
    private async Task TriggerCleanupIfNeeded()
    {
        var now = DateTime.UtcNow;
        var cacheSize = _cache.Count;

        if ((cacheSize > _options.CleanupThreshold && _lastCleanupTime.Add(_options.CleanupInterval) < now) ||
            (cacheSize > _options.CleanupThreshold * 2))
        {
            await CleanupCache();
        }
    }

    /// <summary>
    /// Starts a background task to periodically clean up expired items.
    /// </summary>
    private void StartBackgroundCleanup()
    {
        Task.Run(async () =>
        {
            while (!_cleanupCts.Token.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(_options.CleanupInterval, _cleanupCts.Token);
                    await CleanupCache();
                }
                catch (TaskCanceledException)
                {
                    // Normal cancellation, exit the loop
                    break;
                }
                catch (Exception)
                {
                    // Log error if desired, but continue cleanup loop
                    await Task.Delay(TimeSpan.FromSeconds(30), _cleanupCts.Token);
                }
            }
        }, _cleanupCts.Token);
    }

    /// <summary>
    /// Removes expired items from the cache.
    /// </summary>
    private async Task CleanupCache()
    {
        // Ensure only one cleanup runs at a time
        if (!await _cleanupLock.WaitAsync(0))
            return;

        try
        {
            var now = DateTime.UtcNow;
            _lastCleanupTime = now;

            var expiredKeys = _cache
                .Where(kvp => kvp.Value.Expiration < now)
                .Select(kvp => kvp.Key)
                .ToList();

            foreach (var key in expiredKeys)
            {
                _cache.TryRemove(key, out _);
            }
        }
        finally
        {
            _cleanupLock.Release();
        }
    }

    /// <summary>
    /// Gets the number of items currently in the cache.
    /// </summary>
    /// <returns>The number of cached items.</returns>
    public int GetCacheSize() => _cache.Count;

    /// <summary>
    /// Gets statistics about the cache usage.
    /// </summary>
    /// <returns>A dictionary containing cache statistics.</returns>
    public Dictionary<string, object> GetCacheStatistics()
    {
        var now = DateTime.UtcNow;
        var expiredCount = _cache.Count(kvp => kvp.Value.Expiration < now);
        var validCount = _cache.Count - expiredCount;

        return new Dictionary<string, object>
        {
            ["TotalItems"] = _cache.Count,
            ["ExpiredItems"] = expiredCount,
            ["ValidItems"] = validCount,
            ["LastCleanupTime"] = _lastCleanupTime
        };
    }

    /// <summary>
    /// Disposes the cache resources.
    /// </summary>
    public void Dispose()
    {
        _cleanupCts?.Cancel();
        _cleanupCts?.Dispose();
        _cleanupLock?.Dispose();

        GC.SuppressFinalize(this);
    }
}