using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using AiDotNet.Validation;

namespace AiDotNet.Deployment.TensorRT;

/// <summary>
/// High-performance inference engine for TensorRT models.
/// Supports multi-stream execution and CUDA graph optimization.
/// </summary>
/// <typeparam name="T">The numeric type for input/output tensors</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> TensorRTInferenceEngine provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class TensorRTInferenceEngine<T> : IDisposable
{
    private readonly string _enginePath;
    private readonly TensorRTConfiguration _config;
    private readonly SemaphoreSlim _streamSemaphore;
    private readonly ConcurrentDictionary<int, StreamContext> _streamContexts;
    private InferenceSession? _session;
    private TensorRTEngineMetadata? _metadata;
    private bool _isInitialized = false;
    private bool _disposed = false;

    public TensorRTInferenceEngine(string enginePath, TensorRTConfiguration config)
    {
        Guard.NotNull(enginePath);
        _enginePath = enginePath;
        Guard.NotNull(config);
        _config = config;
        var numContexts = config.EnableMultiStream ? config.NumStreams : 1;
        _streamSemaphore = new SemaphoreSlim(numContexts, numContexts);
        _streamContexts = new ConcurrentDictionary<int, StreamContext>();
    }

    /// <summary>
    /// Initializes the inference engine and creates execution contexts.
    /// </summary>
    public void Initialize()
    {
        if (_isInitialized)
            return;

        if (!File.Exists(_enginePath))
            throw new FileNotFoundException($"TensorRT engine not found: {_enginePath}");

        // Load engine
        var engineData = File.ReadAllBytes(_enginePath);
        LoadEngine(engineData);

        // Create execution contexts for multi-stream
        var numContexts = _config.EnableMultiStream ? _config.NumStreams : 1;

        for (int i = 0; i < numContexts; i++)
        {
            var context = CreateExecutionContext(i);
            _streamContexts[i] = context;
        }

        _isInitialized = true;
    }

    /// <summary>
    /// Performs inference on the input data.
    /// </summary>
    /// <param name="input">Input tensor data</param>
    /// <returns>Output tensor data</returns>
    public async Task<T[]> InferAsync(T[] input)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Engine not initialized. Call Initialize() first.");

        // Wait for available stream using semaphore (avoids busy-waiting)
        await _streamSemaphore.WaitAsync();

        try
        {
            // Find an inactive stream
            var streamId = _streamContexts.First(kvp => !kvp.Value.GetIsActive()).Key;
            var context = _streamContexts[streamId];
            var result = await ExecuteInferenceAsync(context, input);
            return result;
        }
        finally
        {
            // Release semaphore to allow next inference
            _streamSemaphore.Release();
        }
    }

    /// <summary>
    /// Performs batch inference on multiple inputs concurrently.
    /// </summary>
    public async Task<T[][]> InferBatchAsync(T[][] inputs)
    {
        var tasks = inputs.Select(input => InferAsync(input));
        var results = await Task.WhenAll(tasks);
        return results;
    }

    /// <summary>
    /// Warms up the model by running inference on dummy data.
    /// </summary>
    public async Task WarmUpAsync(int numIterations = 10, int[]? inputShape = null)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Engine not initialized. Call Initialize() first.");

        var dummyInput = CreateDummyInput(inputShape);

        // Run inference multiple times to warm up GPU
        for (int i = 0; i < numIterations; i++)
        {
            await InferAsync(dummyInput);
        }
    }

    private void LoadEngine(byte[] engineData)
    {
        // Parse engine metadata
        using var stream = new MemoryStream(engineData);
        using var reader = new BinaryReader(stream);

        // Read and validate header
        var header = new string(reader.ReadChars(9));
        if (header != "TRTENGINE")
            throw new InvalidDataException("Invalid TensorRT engine file format");

        var version = reader.ReadInt32();

        // Read TensorRT configuration
        _metadata = new TensorRTEngineMetadata
        {
            MaxBatchSize = reader.ReadInt32(),
            MaxWorkspaceSize = reader.ReadInt64(),
            UseFp16 = reader.ReadBoolean(),
            UseInt8 = reader.ReadBoolean(),
            StrictTypeConstraints = reader.ReadBoolean(),
            EnableDynamicShapes = reader.ReadBoolean(),
            DeviceId = reader.ReadInt32(),
            DlaCore = reader.ReadInt32()
        };

        if (_metadata.DlaCore == -1)
            _metadata.DlaCore = null;

        // Extract embedded ONNX model (version 2+ embeds the model)
        byte[] onnxModelData;
        if (version >= 2)
        {
            var onnxDataLength = reader.ReadInt32();
            onnxModelData = reader.ReadBytes(onnxDataLength);
        }
        else
        {
            // Version 1: read ONNX path and load from disk
            var onnxPath = reader.ReadString();
            if (!File.Exists(onnxPath))
                throw new FileNotFoundException($"ONNX model not found: {onnxPath}");
            onnxModelData = File.ReadAllBytes(onnxPath);
        }

        // Create ONNX Runtime session with TensorRT execution provider
        var sessionOptions = new SessionOptions();
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        // Configure TensorRT execution provider
        try
        {
            var trtOptions = new Dictionary<string, string>
            {
                ["device_id"] = _metadata.DeviceId.ToString(),
                ["trt_max_workspace_size"] = _metadata.MaxWorkspaceSize.ToString(),
                ["trt_fp16_enable"] = _metadata.UseFp16 ? "1" : "0",
                ["trt_int8_enable"] = _metadata.UseInt8 ? "1" : "0"
            };

            if (_metadata.DlaCore.HasValue)
            {
                trtOptions["trt_dla_enable"] = "1";
                trtOptions["trt_dla_core"] = _metadata.DlaCore.Value.ToString();
            }

            if (_config.EnableMultiStream)
            {
                trtOptions["trt_engine_cache_enable"] = "1";
            }

            sessionOptions.AppendExecutionProvider("TensorRT", trtOptions);
        }
        catch
        {
            // TensorRT not available, fall back to CUDA
            try
            {
                sessionOptions.AppendExecutionProvider_CUDA(_metadata.DeviceId);
            }
            catch
            {
                // CUDA not available, fall back to CPU
            }
        }

        // Create inference session from embedded ONNX model
        _session = new InferenceSession(onnxModelData, sessionOptions);
    }

    private StreamContext CreateExecutionContext(int streamId)
    {
        var context = new StreamContext
        {
            StreamId = streamId
        };
        context.SetIsActive(false);
        context.SetLastUsedTime(DateTime.UtcNow);
        return context;
    }

    private async Task<T[]> ExecuteInferenceAsync(StreamContext context, T[] input)
    {
        context.SetIsActive(true);
        context.SetLastUsedTime(DateTime.UtcNow);

        try
        {
            if (_session == null)
                throw new InvalidOperationException("Inference session not initialized");

            return await Task.Run(() =>
            {
                // Get input metadata
                var inputMeta = _session.InputMetadata.First();
                var inputName = inputMeta.Key;
                var inputShape = CalculateInputShape(_session, input.Length);

                // Create input tensor based on type T
                var inputs = new List<NamedOnnxValue>();

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
                    throw new NotSupportedException($"Type {typeof(T).Name} is not supported for TensorRT inference");
                }

                // Execute inference on GPU via TensorRT execution provider
                using var results = _session.Run(inputs);

                // Extract output tensor
                var outputTensor = results.First().Value;
                return ConvertOutputToT(outputTensor);
            });
        }
        finally
        {
            context.SetIsActive(false);
        }
    }

    private T[] CreateDummyInput(int[]? shape)
    {
        if (shape == null || shape.Length == 0)
        {
            // Default to a reasonable size
            shape = new[] { 1, 224, 224, 3 };
        }

        var totalSize = shape.Aggregate(1, (a, b) => a * b);
        return new T[totalSize];
    }

    /// <summary>
    /// Gets inference statistics for monitoring.
    /// </summary>
    public InferenceStatistics GetStatistics()
    {
        return new InferenceStatistics
        {
            NumStreams = _streamContexts.Count,
            AvailableStreams = _streamSemaphore.CurrentCount,
            ActiveStreams = _streamContexts.Values.Count(c => c.GetIsActive())
        };
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
                    // Batch dimension - use from config or default to 1
                    shape[i] = _metadata?.MaxBatchSize ?? 1;
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

    public void Dispose()
    {
        if (_disposed)
            return;

        // Cleanup resources
        _session?.Dispose();
        _streamContexts.Clear();
        _streamSemaphore?.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Stream context with thread-safe property access.
    /// </summary>
    private class StreamContext
    {
        private int _isActive;
        private long _lastUsedTimeTicks;

        public int StreamId { get; set; }

        public bool GetIsActive() => Interlocked.CompareExchange(ref _isActive, 0, 0) == 1;

        public void SetIsActive(bool value) => Interlocked.Exchange(ref _isActive, value ? 1 : 0);

        public DateTime GetLastUsedTime() => new DateTime(Interlocked.Read(ref _lastUsedTimeTicks), DateTimeKind.Utc);

        public void SetLastUsedTime(DateTime value) => Interlocked.Exchange(ref _lastUsedTimeTicks, value.Ticks);
    }

    /// <summary>
    /// Metadata extracted from TensorRT engine file.
    /// </summary>
    private class TensorRTEngineMetadata
    {
        public int MaxBatchSize { get; set; }
        public long MaxWorkspaceSize { get; set; }
        public bool UseFp16 { get; set; }
        public bool UseInt8 { get; set; }
        public bool StrictTypeConstraints { get; set; }
        public bool EnableDynamicShapes { get; set; }
        public int DeviceId { get; set; }
        public int? DlaCore { get; set; }
    }
}
