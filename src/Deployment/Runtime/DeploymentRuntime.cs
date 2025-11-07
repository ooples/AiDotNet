using System.Collections.Concurrent;
using System.Diagnostics;

namespace AiDotNet.Deployment.Runtime;

/// <summary>
/// Runtime environment for deployed models with warm-up, versioning, A/B testing, and telemetry.
/// </summary>
/// <typeparam name="T">The numeric type for input/output tensors</typeparam>
public class DeploymentRuntime<T> where T : struct
{
    private readonly RuntimeConfiguration _config;
    private readonly ConcurrentDictionary<string, ModelVersion<T>> _models;
    private readonly ConcurrentDictionary<string, ABTestConfig> _abTests;
    private readonly TelemetryCollector _telemetry;
    private readonly ModelCache<T> _cache;

    public DeploymentRuntime(RuntimeConfiguration config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _models = new ConcurrentDictionary<string, ModelVersion<T>>();
        _abTests = new ConcurrentDictionary<string, ABTestConfig>();
        _telemetry = new TelemetryCollector(config.EnableTelemetry);
        _cache = new ModelCache<T>(config.EnableCaching);
    }

    /// <summary>
    /// Registers a model version with the runtime.
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="version">Version identifier</param>
    /// <param name="modelPath">Path to the model file</param>
    /// <param name="metadata">Optional metadata</param>
    public void RegisterModel(string modelName, string version, string modelPath, Dictionary<string, object>? metadata = null)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
        if (string.IsNullOrWhiteSpace(version))
            throw new ArgumentException("Version cannot be null or empty", nameof(version));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        var modelVersion = new ModelVersion<T>
        {
            Name = modelName,
            Version = version,
            ModelPath = modelPath,
            LoadedAt = DateTime.UtcNow,
            Metadata = metadata ?? new Dictionary<string, object>()
        };

        var key = GetModelKey(modelName, version);
        _models[key] = modelVersion;

        _telemetry.RecordEvent("ModelRegistered", new Dictionary<string, object>
        {
            ["ModelName"] = modelName,
            ["Version"] = version,
            ["Timestamp"] = DateTime.UtcNow
        });
    }

    /// <summary>
    /// Warms up a model by running inference on dummy data.
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="version">Version identifier</param>
    /// <param name="numIterations">Number of warm-up iterations (default: 10)</param>
    public void WarmUpModel(string modelName, string version, int numIterations = 10)
    {
        var key = GetModelKey(modelName, version);
        if (!_models.TryGetValue(key, out var modelVersion))
            throw new InvalidOperationException($"Model {modelName} version {version} not registered");

        var stopwatch = Stopwatch.StartNew();

        // Create dummy input
        var dummyInput = CreateDummyInput();

        // Run warm-up iterations
        for (int i = 0; i < numIterations; i++)
        {
            // Simulate inference
            Thread.Sleep(1);
        }

        stopwatch.Stop();
        modelVersion.IsWarmedUp = true;
        modelVersion.WarmUpTimeMs = stopwatch.ElapsedMilliseconds;

        _telemetry.RecordEvent("ModelWarmedUp", new Dictionary<string, object>
        {
            ["ModelName"] = modelName,
            ["Version"] = version,
            ["WarmUpTimeMs"] = stopwatch.ElapsedMilliseconds,
            ["Iterations"] = numIterations
        });
    }

    /// <summary>
    /// Performs inference with the specified model version.
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="version">Version identifier (use "latest" for latest version)</param>
    /// <param name="input">Input tensor data</param>
    /// <returns>Output tensor data</returns>
    public async Task<T[]> InferAsync(string modelName, string version, T[] input)
    {
        var stopwatch = Stopwatch.StartNew();

        try
        {
            // Resolve version
            var resolvedVersion = ResolveVersion(modelName, version);
            var key = GetModelKey(modelName, resolvedVersion);

            if (!_models.TryGetValue(key, out var modelVersion))
                throw new InvalidOperationException($"Model {modelName} version {resolvedVersion} not registered");

            // Check cache
            if (_config.EnableCaching)
            {
                var cachedResult = _cache.Get(key, input);
                if (cachedResult != null)
                {
                    _telemetry.RecordInference(modelName, resolvedVersion, stopwatch.ElapsedMilliseconds, true);
                    return cachedResult;
                }
            }

            // Perform inference
            var result = await PerformInferenceAsync(modelVersion, input);

            // Cache result
            if (_config.EnableCaching)
            {
                _cache.Put(key, input, result);
            }

            stopwatch.Stop();
            _telemetry.RecordInference(modelName, resolvedVersion, stopwatch.ElapsedMilliseconds, false);

            return result;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            _telemetry.RecordError(modelName, version, ex);
            throw;
        }
    }

    /// <summary>
    /// Sets up A/B testing between two model versions.
    /// </summary>
    /// <param name="testName">Name of the A/B test</param>
    /// <param name="modelName">Name of the model</param>
    /// <param name="versionA">Version A identifier</param>
    /// <param name="versionB">Version B identifier</param>
    /// <param name="trafficSplit">Percentage of traffic for version A (0.0 to 1.0, default: 0.5)</param>
    public void SetupABTest(string testName, string modelName, string versionA, string versionB, double trafficSplit = 0.5)
    {
        if (trafficSplit < 0.0 || trafficSplit > 1.0)
            throw new ArgumentException("Traffic split must be between 0.0 and 1.0", nameof(trafficSplit));

        var abTest = new ABTestConfig
        {
            TestName = testName,
            ModelName = modelName,
            VersionA = versionA,
            VersionB = versionB,
            TrafficSplit = trafficSplit,
            StartedAt = DateTime.UtcNow
        };

        _abTests[testName] = abTest;

        _telemetry.RecordEvent("ABTestStarted", new Dictionary<string, object>
        {
            ["TestName"] = testName,
            ["ModelName"] = modelName,
            ["VersionA"] = versionA,
            ["VersionB"] = versionB,
            ["TrafficSplit"] = trafficSplit
        });
    }

    /// <summary>
    /// Performs inference with A/B testing (automatically selects version based on traffic split).
    /// </summary>
    /// <param name="testName">Name of the A/B test</param>
    /// <param name="input">Input tensor data</param>
    /// <returns>Output tensor data and selected version</returns>
    public async Task<(T[] output, string selectedVersion)> InferWithABTestAsync(string testName, T[] input)
    {
        if (!_abTests.TryGetValue(testName, out var abTest))
            throw new InvalidOperationException($"A/B test {testName} not configured");

        // Select version based on traffic split
        var random = Random.Shared.NextDouble();
        var selectedVersion = random < abTest.TrafficSplit ? abTest.VersionA : abTest.VersionB;

        var output = await InferAsync(abTest.ModelName, selectedVersion, input);

        _telemetry.RecordEvent("ABTestInference", new Dictionary<string, object>
        {
            ["TestName"] = testName,
            ["SelectedVersion"] = selectedVersion,
            ["TrafficSplit"] = abTest.TrafficSplit
        });

        return (output, selectedVersion);
    }

    /// <summary>
    /// Gets telemetry statistics for a model.
    /// </summary>
    public ModelStatistics GetModelStatistics(string modelName, string? version = null)
    {
        return _telemetry.GetStatistics(modelName, version);
    }

    /// <summary>
    /// Gets all registered model versions.
    /// </summary>
    public List<ModelVersionInfo> GetRegisteredModels()
    {
        return _models.Values.Select(m => new ModelVersionInfo
        {
            Name = m.Name,
            Version = m.Version,
            LoadedAt = m.LoadedAt,
            IsWarmedUp = m.IsWarmedUp,
            WarmUpTimeMs = m.WarmUpTimeMs
        }).ToList();
    }

    private string ResolveVersion(string modelName, string version)
    {
        if (version.Equals("latest", StringComparison.OrdinalIgnoreCase))
        {
            // Find latest version
            var versions = _models.Keys
                .Where(k => k.StartsWith($"{modelName}:"))
                .Select(k => k.Split(':')[1])
                .OrderByDescending(v => v)
                .FirstOrDefault();

            if (versions == null)
                throw new InvalidOperationException($"No versions found for model {modelName}");

            return versions;
        }

        return version;
    }

    private string GetModelKey(string modelName, string version) => $"{modelName}:{version}";

    private T[] CreateDummyInput()
    {
        // Create dummy input for warm-up
        return new T[224 * 224 * 3]; // Example: standard image input
    }

    private async Task<T[]> PerformInferenceAsync(ModelVersion<T> modelVersion, T[] input)
    {
        // Placeholder for actual inference
        // In production, this would load and execute the model
        await Task.Delay(10); // Simulate inference time

        var output = new T[1000]; // Example output size
        return output;
    }
}

/// <summary>
/// Represents a versioned model in the runtime.
/// </summary>
internal class ModelVersion<T> where T : struct
{
    public string Name { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public string ModelPath { get; set; } = string.Empty;
    public DateTime LoadedAt { get; set; }
    public bool IsWarmedUp { get; set; }
    public long WarmUpTimeMs { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Configuration for A/B testing.
/// </summary>
internal class ABTestConfig
{
    public string TestName { get; set; } = string.Empty;
    public string ModelName { get; set; } = string.Empty;
    public string VersionA { get; set; } = string.Empty;
    public string VersionB { get; set; } = string.Empty;
    public double TrafficSplit { get; set; }
    public DateTime StartedAt { get; set; }
}

/// <summary>
/// Public model version information.
/// </summary>
public class ModelVersionInfo
{
    public string Name { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public DateTime LoadedAt { get; set; }
    public bool IsWarmedUp { get; set; }
    public long WarmUpTimeMs { get; set; }
}
