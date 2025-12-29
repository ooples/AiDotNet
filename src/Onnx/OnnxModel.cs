using System.Net.Http;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Onnx;

/// <summary>
/// A wrapper for ONNX models that provides easy-to-use inference with AiDotNet Tensor types.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class wraps the ONNX Runtime InferenceSession and provides:
/// <list type="bullet">
/// <item>Automatic tensor conversion between AiDotNet and ONNX formats</item>
/// <item>Support for multiple execution providers (CPU, CUDA, TensorRT, DirectML)</item>
/// <item>Multi-input/multi-output model support</item>
/// <item>Warm-up and async inference</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Use this class to run pre-trained ONNX models:
/// <code>
/// // Load a model
/// var model = new OnnxModel&lt;float&gt;("model.onnx");
///
/// // Run inference
/// var input = new Tensor&lt;float&gt;([1, 3, 224, 224]);
/// var output = model.Run(input);
///
/// // Don't forget to dispose
/// model.Dispose();
/// </code>
/// </para>
/// </remarks>
public class OnnxModel<T> : IOnnxModel<T>
{
    private readonly InferenceSession _session;
    private readonly OnnxModelOptions _options;
    private readonly INumericOperations<T> _numOps;
    private bool _disposed;

    /// <inheritdoc/>
    public IOnnxModelMetadata Metadata { get; }

    /// <inheritdoc/>
    public bool IsLoaded => _session is not null && !_disposed;

    /// <inheritdoc/>
    public string ExecutionProvider { get; }

    /// <summary>
    /// Creates a new OnnxModel from a file path.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional configuration options.</param>
    public OnnxModel(string modelPath, OnnxModelOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentNullException(nameof(modelPath));

        if (!File.Exists(modelPath))
            throw new FileNotFoundException("ONNX model file not found.", modelPath);

        _options = options ?? new OnnxModelOptions();
        _numOps = MathHelper.GetNumericOperations<T>();

        var (session, provider) = CreateSession(modelPath);
        _session = session;
        ExecutionProvider = provider;

        Metadata = ExtractMetadata(_session);

        if (_options.AutoWarmUp)
        {
            WarmUp();
        }
    }

    /// <summary>
    /// Creates a new OnnxModel from a byte array.
    /// </summary>
    /// <param name="modelBytes">The ONNX model as a byte array.</param>
    /// <param name="options">Optional configuration options.</param>
    public OnnxModel(byte[] modelBytes, OnnxModelOptions? options = null)
    {
        if (modelBytes is null)
            throw new ArgumentNullException(nameof(modelBytes));

        _options = options ?? new OnnxModelOptions();
        _numOps = MathHelper.GetNumericOperations<T>();

        var (session, provider) = CreateSessionFromBytes(modelBytes);
        _session = session;
        ExecutionProvider = provider;

        Metadata = ExtractMetadata(_session);

        if (_options.AutoWarmUp)
        {
            WarmUp();
        }
    }

    /// <inheritdoc/>
    public Tensor<T> Run(Tensor<T> input)
    {
        ThrowIfDisposed();

        var primaryInputName = Metadata.Inputs.First().Name;
        var inputs = new Dictionary<string, Tensor<T>> { [primaryInputName] = input };
        var outputs = Run(inputs);

        return outputs.Values.First();
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, Tensor<T>> Run(IReadOnlyDictionary<string, Tensor<T>> inputs)
    {
        ThrowIfDisposed();

        var onnxInputs = new List<NamedOnnxValue>();

        foreach (var (name, tensor) in inputs)
        {
            var onnxTensor = OnnxTensorConverter.ToOnnxFloat(tensor);
            onnxInputs.Add(NamedOnnxValue.CreateFromTensor(name, onnxTensor));
        }

        using var results = _session.Run(onnxInputs);

        var outputs = new Dictionary<string, Tensor<T>>();
        foreach (var result in results)
        {
            var onnxTensor = result.AsTensor<float>();
            if (onnxTensor is not null)
            {
                outputs[result.Name] = OnnxTensorConverter.FromOnnxFloat<T>(onnxTensor);
            }
        }

        return outputs;
    }

    /// <inheritdoc/>
    public Task<Tensor<T>> RunAsync(Tensor<T> input, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Run(input), cancellationToken);
    }

    /// <inheritdoc/>
    public Task<IReadOnlyDictionary<string, Tensor<T>>> RunAsync(
        IReadOnlyDictionary<string, Tensor<T>> inputs,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Run(inputs), cancellationToken);
    }

    /// <inheritdoc/>
    public void WarmUp()
    {
        ThrowIfDisposed();

        // Create dummy inputs based on model metadata
        var inputs = new Dictionary<string, Tensor<T>>();

        foreach (var inputInfo in Metadata.Inputs)
        {
            var shape = inputInfo.Shape.Select(d => d < 0 ? 1 : d).ToArray();
            inputs[inputInfo.Name] = new Tensor<T>(shape);
        }

        // Run inference to warm up
        Run(inputs);
    }

    /// <inheritdoc/>
    public Task WarmUpAsync(CancellationToken cancellationToken = default)
    {
        return Task.Run(WarmUp, cancellationToken);
    }

    /// <summary>
    /// Runs inference with specific output names.
    /// </summary>
    /// <param name="inputs">The input tensors.</param>
    /// <param name="outputNames">The names of outputs to retrieve.</param>
    /// <returns>Dictionary of requested outputs.</returns>
    public IReadOnlyDictionary<string, Tensor<T>> Run(
        IReadOnlyDictionary<string, Tensor<T>> inputs,
        IEnumerable<string> outputNames)
    {
        ThrowIfDisposed();

        var onnxInputs = new List<NamedOnnxValue>();

        foreach (var (name, tensor) in inputs)
        {
            var onnxTensor = OnnxTensorConverter.ToOnnxFloat(tensor);
            onnxInputs.Add(NamedOnnxValue.CreateFromTensor(name, onnxTensor));
        }

        var outputNamesList = outputNames.ToList();
        using var results = _session.Run(onnxInputs, outputNamesList);

        var outputs = new Dictionary<string, Tensor<T>>();
        foreach (var result in results)
        {
            var onnxTensor = result.AsTensor<float>();
            if (onnxTensor is not null)
            {
                outputs[result.Name] = OnnxTensorConverter.FromOnnxFloat<T>(onnxTensor);
            }
        }

        return outputs;
    }

    /// <summary>
    /// Runs inference with long integer inputs (useful for token IDs).
    /// </summary>
    /// <param name="inputName">The input tensor name.</param>
    /// <param name="tokenIds">The token IDs to process.</param>
    /// <returns>The output tensor.</returns>
    public Tensor<T> RunWithLongInput(string inputName, long[] tokenIds)
    {
        ThrowIfDisposed();

        var shape = new[] { 1, tokenIds.Length };
        var onnxTensor = new OnnxTensors.DenseTensor<long>(shape);

        for (int i = 0; i < tokenIds.Length; i++)
        {
            onnxTensor[0, i] = tokenIds[i];
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxTensor)
        };

        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();

        return output is not null
            ? OnnxTensorConverter.FromOnnxFloat<T>(output)
            : new Tensor<T>([0]);
    }

    private (InferenceSession session, string provider) CreateSession(string modelPath)
    {
        var providers = GetProvidersToTry();

        foreach (var provider in providers)
        {
            try
            {
                var sessionOptions = CreateSessionOptions(provider);
                var session = new InferenceSession(modelPath, sessionOptions);
                return (session, provider.ToString());
            }
            catch (Exception)
            {
                // Try next provider
                continue;
            }
        }

        // Fallback to CPU
        var cpuOptions = CreateSessionOptions(OnnxExecutionProvider.Cpu);
        return (new InferenceSession(modelPath, cpuOptions), "CPU");
    }

    private (InferenceSession session, string provider) CreateSessionFromBytes(byte[] modelBytes)
    {
        var providers = GetProvidersToTry();

        foreach (var provider in providers)
        {
            try
            {
                var sessionOptions = CreateSessionOptions(provider);
                var session = new InferenceSession(modelBytes, sessionOptions);
                return (session, provider.ToString());
            }
            catch (Exception)
            {
                // Try next provider
                continue;
            }
        }

        // Fallback to CPU
        var cpuOptions = CreateSessionOptions(OnnxExecutionProvider.Cpu);
        return (new InferenceSession(modelBytes, cpuOptions), "CPU");
    }

    private List<OnnxExecutionProvider> GetProvidersToTry()
    {
        var providers = new List<OnnxExecutionProvider>();

        if (_options.ExecutionProvider == OnnxExecutionProvider.Auto)
        {
            // Try providers in order of preference
            providers.Add(OnnxExecutionProvider.TensorRT);
            providers.Add(OnnxExecutionProvider.Cuda);
            providers.Add(OnnxExecutionProvider.DirectML);
            providers.Add(OnnxExecutionProvider.Cpu);
        }
        else
        {
            providers.Add(_options.ExecutionProvider);
            providers.AddRange(_options.FallbackProviders);
        }

        return providers;
    }

    private SessionOptions CreateSessionOptions(OnnxExecutionProvider provider)
    {
        var options = new SessionOptions();

        // Set optimization level
        options.GraphOptimizationLevel = _options.OptimizationLevel switch
        {
            GraphOptimizationLevel.None => Microsoft.ML.OnnxRuntime.GraphOptimizationLevel.ORT_DISABLE_ALL,
            GraphOptimizationLevel.Basic => Microsoft.ML.OnnxRuntime.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            GraphOptimizationLevel.Extended => Microsoft.ML.OnnxRuntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            GraphOptimizationLevel.All => Microsoft.ML.OnnxRuntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
            _ => Microsoft.ML.OnnxRuntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        // Set threading
        if (_options.IntraOpNumThreads > 0)
        {
            options.IntraOpNumThreads = _options.IntraOpNumThreads;
        }

        if (_options.InterOpNumThreads > 0)
        {
            options.InterOpNumThreads = _options.InterOpNumThreads;
        }

        // Enable memory pattern and arena
        options.EnableMemoryPattern = _options.EnableMemoryPattern;

        // Set profiling
        if (_options.EnableProfiling)
        {
            options.EnableProfiling = true;
            if (!string.IsNullOrEmpty(_options.ProfileOutputPath))
            {
                options.ProfileOutputPathPrefix = _options.ProfileOutputPath;
            }
        }

        // Set log level
        options.LogSeverityLevel = (OrtLoggingLevel)_options.LogLevel;

        // Add execution provider
        switch (provider)
        {
            case OnnxExecutionProvider.Cuda:
                options.AppendExecutionProvider_CUDA(_options.GpuDeviceId);
                break;

            case OnnxExecutionProvider.TensorRT:
                options.AppendExecutionProvider_Tensorrt(_options.GpuDeviceId);
                options.AppendExecutionProvider_CUDA(_options.GpuDeviceId);
                break;

            case OnnxExecutionProvider.DirectML:
                options.AppendExecutionProvider_DML(_options.GpuDeviceId);
                break;

            case OnnxExecutionProvider.CoreML:
                // CoreML is only available on macOS
                try
                {
                    options.AppendExecutionProvider_CoreML();
                }
                catch (Exception)
                {
                    // Not available on this platform
                }
                break;

            case OnnxExecutionProvider.Cpu:
            default:
                // CPU is default, no need to append
                break;
        }

        return options;
    }

    private static OnnxModelMetadata ExtractMetadata(InferenceSession session)
    {
        var inputs = new List<IOnnxTensorInfo>();
        foreach (var input in session.InputMetadata)
        {
            inputs.Add(new OnnxTensorInfo
            {
                Name = input.Key,
                Shape = input.Value.Dimensions,
                ElementType = GetElementTypeName(input.Value.ElementType)
            });
        }

        var outputs = new List<IOnnxTensorInfo>();
        foreach (var output in session.OutputMetadata)
        {
            outputs.Add(new OnnxTensorInfo
            {
                Name = output.Key,
                Shape = output.Value.Dimensions,
                ElementType = GetElementTypeName(output.Value.ElementType)
            });
        }

        var modelMetadata = session.ModelMetadata;
        var customMetadata = new Dictionary<string, string>();

        try
        {
            foreach (var kvp in modelMetadata.CustomMetadataMap)
            {
                customMetadata[kvp.Key] = kvp.Value;
            }
        }
        catch
        {
            // Custom metadata might not be available
        }

        return new OnnxModelMetadata
        {
            ModelName = modelMetadata.GraphName ?? "Unknown",
            Description = modelMetadata.Description,
            ProducerName = modelMetadata.ProducerName,
            ProducerVersion = null, // Not directly available
            OpsetVersion = 0, // Would need to parse from model file
            GraphName = modelMetadata.GraphName,
            Domain = modelMetadata.Domain,
            Inputs = inputs,
            Outputs = outputs,
            CustomMetadata = customMetadata
        };
    }

    private static string GetElementTypeName(Type elementType)
    {
        return OnnxTensorConverter.GetOnnxTypeName(elementType);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }

    /// <summary>
    /// Disposes the ONNX session and releases resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes managed and unmanaged resources.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(), false if from finalizer.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
        {
            _session?.Dispose();
        }

        _disposed = true;
    }

    /// <summary>
    /// Creates an OnnxModel asynchronously, optionally downloading from a URL.
    /// </summary>
    /// <param name="modelPath">Local path or URL to the ONNX model.</param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional download progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The loaded OnnxModel.</returns>
    public static async Task<OnnxModel<T>> CreateAsync(
        string modelPath,
        OnnxModelOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (modelPath.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            modelPath.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            // Download model from URL
            using var httpClient = new HttpClient();
#if NET6_0_OR_GREATER
            var response = await httpClient.GetAsync(modelPath, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
#else
            var response = await httpClient.GetAsync(modelPath, cancellationToken);
#endif
            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? -1;
            var bytes = new List<byte>();

#if NET6_0_OR_GREATER
            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
#else
            using var stream = await response.Content.ReadAsStreamAsync();
#endif
            var buffer = new byte[8192];
            int bytesRead;
            long totalBytesRead = 0;

#if NET6_0_OR_GREATER
            while ((bytesRead = await stream.ReadAsync(buffer.AsMemory(0, buffer.Length), cancellationToken)) > 0)
#else
            while ((bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length, cancellationToken)) > 0)
#endif
            {
                bytes.AddRange(buffer.Take(bytesRead));
                totalBytesRead += bytesRead;

                if (totalBytes > 0)
                {
                    progress?.Report((double)totalBytesRead / totalBytes);
                }
            }

            progress?.Report(1.0);
            return new OnnxModel<T>([.. bytes], options);
        }

        // Local file
        return await Task.Run(() => new OnnxModel<T>(modelPath, options), cancellationToken);
    }
}
