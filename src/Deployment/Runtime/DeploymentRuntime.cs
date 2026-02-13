using System.Collections.Concurrent;
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using AiDotNet.Validation;

namespace AiDotNet.Deployment.Runtime;

/// <summary>
/// Runtime environment for deployed models with warm-up, versioning, A/B testing, and telemetry.
/// </summary>
/// <typeparam name="T">The numeric type for input/output tensors</typeparam>
public class DeploymentRuntime<T>
{
    private readonly RuntimeConfiguration _config;
    private readonly ConcurrentDictionary<string, ModelVersion<T>> _models;
    private readonly ConcurrentDictionary<string, ABTestConfig> _abTests;
    private readonly TelemetryCollector _telemetry;
    private readonly ModelCache<T> _cache;
    private readonly ConcurrentDictionary<string, InferenceSession> _sessions;
    private readonly Random _random;

    public DeploymentRuntime(RuntimeConfiguration config)
    {
        Guard.NotNull(config);
        _config = config;
        _models = new ConcurrentDictionary<string, ModelVersion<T>>();
        _abTests = new ConcurrentDictionary<string, ABTestConfig>();
        _telemetry = new TelemetryCollector(config.EnableTelemetry);
        _cache = new ModelCache<T>(config.EnableCaching);
        _sessions = new ConcurrentDictionary<string, InferenceSession>();
        _random = RandomHelper.CreateSecureRandom();
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
    public async Task WarmUpModelAsync(string modelName, string version, int numIterations = 10)
    {
        var key = GetModelKey(modelName, version);
        if (!_models.TryGetValue(key, out var modelVersion))
            throw new InvalidOperationException($"Model {modelName} version {version} not registered");

        var stopwatch = Stopwatch.StartNew();

        // Get or create inference session to determine input shape
        var session = GetOrCreateSession(key, modelVersion.ModelPath);
        var dummyInput = CreateDummyInput(session);

        // Run warm-up iterations with dummy input
        for (int i = 0; i < numIterations; i++)
        {
            await PerformInferenceAsync(modelVersion, dummyInput);
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
        var randomValue = _random.NextDouble();
        var selectedVersion = randomValue < abTest.TrafficSplit ? abTest.VersionA : abTest.VersionB;

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
            // Find latest version using semantic version comparison
            var versions = _models.Keys
                .Where(k => k.StartsWith($"{modelName}:"))
                .Select(k => k.Split(':')[1])
                .ToList();

            if (!versions.Any())
                throw new InvalidOperationException($"No versions found for model {modelName}");

            // Sort by semantic version: parse version numbers and compare numerically
            var latestVersion = versions
                .OrderByDescending(v =>
                {
                    // Strip 'v' prefix and split/prerelease suffix
                    var sanitized = v.TrimStart('v').Split('-', '+')[0];

                    // Try to parse as System.Version
                    if (Version.TryParse(sanitized, out var parsed))
                    {
                        return parsed;
                    }

                    // Fallback: try parsing major.minor.patch manually
                    var parts = sanitized.Split('.');
                    if (parts.Length >= 2)
                    {
                        // Build version from available parts (pad with zeros if needed)
                        var major = int.TryParse(parts[0], out var maj) ? maj : 0;
                        var minor = int.TryParse(parts[1], out var min) ? min : 0;
                        var patch = parts.Length > 2 && int.TryParse(parts[2], out var pat) ? pat : 0;
                        return new Version(major, minor, patch);
                    }

                    // Last resort: use version 0.0 for unparseable versions
                    return new Version(0, 0);
                })
                .FirstOrDefault();

            if (latestVersion == null)
                throw new InvalidOperationException($"No versions found for model {modelName}");

            return latestVersion;
        }

        return version;
    }

    private string GetModelKey(string modelName, string version) => $"{modelName}:{version}";

    private InferenceSession GetOrCreateSession(string key, string modelPath)
    {
        return _sessions.GetOrAdd(key, _ =>
        {
            var sessionOptions = new SessionOptions();

            // Configure session based on runtime configuration
            if (_config.EnableGpuAcceleration)
            {
                // Try to use GPU acceleration if available
                try
                {
                    sessionOptions.AppendExecutionProvider_CUDA(0);
                }
                catch
                {
                    // Fall back to CPU if CUDA is not available
                }
            }

            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            return new InferenceSession(modelPath, sessionOptions);
        });
    }

    private T[] CreateDummyInput(InferenceSession session)
    {
        // Get input metadata to determine shape
        var inputMeta = session.InputMetadata.First();
        var shape = inputMeta.Value.Dimensions;

        // Calculate total size (skip batch dimension if dynamic)
        int totalSize = 1;
        foreach (var dim in shape)
        {
            if (dim > 0) // Skip dynamic dimensions (-1)
                totalSize *= dim;
        }

        // Default to 224x224x3 if shape is fully dynamic
        if (totalSize <= 1)
            totalSize = 224 * 224 * 3;

        return new T[totalSize];
    }

    private async Task<T[]> PerformInferenceAsync(ModelVersion<T> modelVersion, T[] input)
    {
        return await Task.Run(() =>
        {
            var key = GetModelKey(modelVersion.Name, modelVersion.Version);
            var session = GetOrCreateSession(key, modelVersion.ModelPath);

            // Get input metadata
            var inputMeta = session.InputMetadata.First();
            var inputName = inputMeta.Key;
            var inputShape = CalculateInputShape(session, input.Length);

            // Create input tensor based on type T
            var inputs = new List<NamedOnnxValue>();

            try
            {
                if (typeof(T) == typeof(float))
                {
                    var floatInput = ConvertToFloatArray(input);
                    var tensor = new DenseTensor<float>(floatInput, inputShape);
                    inputs.Add(NamedOnnxValue.CreateFromTensor(inputName, tensor));
                }
                else if (typeof(T) == typeof(double))
                {
                    var doubleInput = ConvertToDoubleArray(input);
                    var tensor = new DenseTensor<double>(doubleInput, inputShape);
                    inputs.Add(NamedOnnxValue.CreateFromTensor(inputName, tensor));
                }
                else if (typeof(T) == typeof(int))
                {
                    var intInput = ConvertToIntArray(input);
                    var tensor = new DenseTensor<int>(intInput, inputShape);
                    inputs.Add(NamedOnnxValue.CreateFromTensor(inputName, tensor));
                }
                else if (typeof(T) == typeof(long))
                {
                    var longInput = ConvertToLongArray(input);
                    var tensor = new DenseTensor<long>(longInput, inputShape);
                    inputs.Add(NamedOnnxValue.CreateFromTensor(inputName, tensor));
                }
                else
                {
                    throw new NotSupportedException($"Type {typeof(T).Name} is not supported for ONNX inference. Supported types: float, double, int, long");
                }

                // Run inference
                using var results = session.Run(inputs);

                // Extract output (first output tensor)
                var outputTensor = results.First().Value;
                return ConvertOutputToT(outputTensor);
            }
            finally
            {
                // Dispose input tensors to avoid native memory leaks
                // Note: NamedOnnxValue in ONNX Runtime 1.x may not implement IDisposable
                // For newer versions that do, this prevents memory leaks
                foreach (var inputValue in inputs)
                {
                    if (inputValue is IDisposable disposable)
                    {
                        disposable.Dispose();
                    }
                }
            }
        });
    }

    private int[] CalculateInputShape(InferenceSession session, int inputLength)
    {
        var inputMeta = session.InputMetadata.First();
        var shape = inputMeta.Value.Dimensions.ToArray();

        // Replace dynamic dimensions with calculated values
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] <= 0) // Dynamic dimension
            {
                if (i == 0)
                {
                    // Batch dimension - default to 1
                    shape[i] = 1;
                }
                else
                {
                    // Calculate based on remaining input length
                    var remainingDims = shape.Skip(i).Count(d => d > 0);
                    var knownSize = shape.Where(d => d > 0).Aggregate(1, (a, b) => a * b);
                    shape[i] = remainingDims > 0 ? inputLength / knownSize : inputLength;
                }
            }
        }

        return shape;
    }

    private float[] ConvertToFloatArray(T[] input)
    {
        if (typeof(T) == typeof(float))
            return (float[])(object)input;

        var result = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            result[i] = Convert.ToSingle(input[i]);
        return result;
    }

    private double[] ConvertToDoubleArray(T[] input)
    {
        if (typeof(T) == typeof(double))
            return (double[])(object)input;

        var result = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
            result[i] = Convert.ToDouble(input[i]);
        return result;
    }

    private int[] ConvertToIntArray(T[] input)
    {
        if (typeof(T) == typeof(int))
            return (int[])(object)input;

        var result = new int[input.Length];
        for (int i = 0; i < input.Length; i++)
            result[i] = Convert.ToInt32(input[i]);
        return result;
    }

    private long[] ConvertToLongArray(T[] input)
    {
        if (typeof(T) == typeof(long))
            return (long[])(object)input;

        var result = new long[input.Length];
        for (int i = 0; i < input.Length; i++)
            result[i] = Convert.ToInt64(input[i]);
        return result;
    }

    private T[] ConvertOutputToT(object outputTensor)
    {
        // Handle different tensor types
        if (outputTensor is Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> floatTensor)
        {
            var flatArray = floatTensor.ToArray();
            if (typeof(T) == typeof(float))
                return (T[])(object)flatArray;

            var result = new T[flatArray.Length];
            for (int i = 0; i < flatArray.Length; i++)
                result[i] = (T)Convert.ChangeType(flatArray[i], typeof(T));
            return result;
        }
        else if (outputTensor is Microsoft.ML.OnnxRuntime.Tensors.Tensor<double> doubleTensor)
        {
            var flatArray = doubleTensor.ToArray();
            if (typeof(T) == typeof(double))
                return (T[])(object)flatArray;

            var result = new T[flatArray.Length];
            for (int i = 0; i < flatArray.Length; i++)
                result[i] = (T)Convert.ChangeType(flatArray[i], typeof(T));
            return result;
        }
        else if (outputTensor is Microsoft.ML.OnnxRuntime.Tensors.Tensor<int> intTensor)
        {
            var flatArray = intTensor.ToArray();
            if (typeof(T) == typeof(int))
                return (T[])(object)flatArray;

            var result = new T[flatArray.Length];
            for (int i = 0; i < flatArray.Length; i++)
                result[i] = (T)Convert.ChangeType(flatArray[i], typeof(T));
            return result;
        }
        else if (outputTensor is Microsoft.ML.OnnxRuntime.Tensors.Tensor<long> longTensor)
        {
            var flatArray = longTensor.ToArray();
            if (typeof(T) == typeof(long))
                return (T[])(object)flatArray;

            var result = new T[flatArray.Length];
            for (int i = 0; i < flatArray.Length; i++)
                result[i] = (T)Convert.ChangeType(flatArray[i], typeof(T));
            return result;
        }

        throw new NotSupportedException($"Output tensor type {outputTensor.GetType().Name} is not supported");
    }
}

/// <summary>
/// Represents a versioned model in the runtime.
/// </summary>
internal class ModelVersion<T>
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
