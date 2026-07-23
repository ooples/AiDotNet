namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for ONNX model wrappers that provide cross-platform model inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This interface provides a unified way to work with ONNX models in AiDotNet.
/// It supports loading models from files, byte arrays, or URLs, and provides
/// methods for running inference with AiDotNet Tensor types.
/// </para>
/// <para><b>For Beginners:</b> ONNX (Open Neural Network Exchange) is a universal format
/// for neural network models. This interface allows you to:
/// <list type="bullet">
/// <item>Load models trained in PyTorch, TensorFlow, or other frameworks</item>
/// <item>Run inference using CPU, GPU (CUDA), TensorRT, or DirectML</item>
/// <item>Convert between AiDotNet tensors and ONNX tensors automatically</item>
/// </list>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("OnnxModel")]
public interface IOnnxModel<T> : IDisposable
{
    /// <summary>
    /// Gets the metadata about the loaded ONNX model.
    /// </summary>
    IOnnxModelMetadata Metadata { get; }

    /// <summary>
    /// Gets whether the model has been successfully loaded and is ready for inference.
    /// </summary>
    bool IsLoaded { get; }

    /// <summary>
    /// Gets the execution provider currently being used (CPU, CUDA, TensorRT, DirectML).
    /// </summary>
    string ExecutionProvider { get; }

    /// <summary>
    /// Runs inference with a single input tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor from the model.</returns>
    Tensor<T> Run(Tensor<T> input);

    /// <summary>
    /// Runs inference with named inputs.
    /// </summary>
    /// <param name="inputs">Dictionary mapping input names to tensors.</param>
    /// <returns>Dictionary mapping output names to tensors.</returns>
    IReadOnlyDictionary<string, Tensor<T>> Run(IReadOnlyDictionary<string, Tensor<T>> inputs);

    /// <summary>
    /// Runs inference asynchronously with a single input tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The output tensor from the model.</returns>
    Task<Tensor<T>> RunAsync(Tensor<T> input, CancellationToken cancellationToken = default);

    /// <summary>
    /// Runs inference asynchronously with named inputs.
    /// </summary>
    /// <param name="inputs">Dictionary mapping input names to tensors.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Dictionary mapping output names to tensors.</returns>
    Task<IReadOnlyDictionary<string, Tensor<T>>> RunAsync(
        IReadOnlyDictionary<string, Tensor<T>> inputs,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Warms up the model by running a single inference with dummy data.
    /// This helps ensure consistent inference times by initializing lazy resources.
    /// </summary>
    void WarmUp();

    /// <summary>
    /// Warms up the model asynchronously.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmUpAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Metadata about an ONNX model including inputs, outputs, and opset version.
/// </summary>
public interface IOnnxModelMetadata
{
    /// <summary>
    /// Gets the model name from the ONNX file.
    /// </summary>
    string ModelName { get; }

    /// <summary>
    /// Gets the model description if available.
    /// </summary>
    string? Description { get; }

    /// <summary>
    /// Gets the producer name (e.g., "pytorch", "tensorflow").
    /// </summary>
    string? ProducerName { get; }

    /// <summary>
    /// Gets the producer version.
    /// </summary>
    string? ProducerVersion { get; }

    /// <summary>
    /// Gets the ONNX opset version used by this model.
    /// </summary>
    long OpsetVersion { get; }

    /// <summary>
    /// Gets information about all input tensors.
    /// </summary>
    IReadOnlyList<IOnnxTensorInfo> Inputs { get; }

    /// <summary>
    /// Gets information about all output tensors.
    /// </summary>
    IReadOnlyList<IOnnxTensorInfo> Outputs { get; }
}

/// <summary>
/// Information about an ONNX tensor (input or output).
/// </summary>
public interface IOnnxTensorInfo
{
    /// <summary>
    /// Gets the name of the tensor.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the shape of the tensor. -1 indicates a dynamic dimension.
    /// </summary>
    int[] Shape { get; }

    /// <summary>
    /// Gets the element type name (e.g., "float", "int64", "string").
    /// </summary>
    string ElementType { get; }
}
