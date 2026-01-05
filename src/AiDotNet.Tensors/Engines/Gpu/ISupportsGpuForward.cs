namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Interface for layers that support GPU-resident forward pass.
/// Layers implementing this interface can chain GPU operations without
/// CPU round-trips, significantly improving performance for multi-layer networks.
/// </summary>
/// <typeparam name="T">The element type of the tensor.</typeparam>
/// <remarks>
/// <para><b>Usage Pattern for GPU-Resident Inference:</b></para>
/// <code>
/// // For multi-layer inference, use ForwardGpu to avoid CPU round-trips
/// using var gpuInput = engine.UploadToGpu(input);
///
/// IGpuTensor&lt;T&gt; current = gpuInput;
/// foreach (var layer in layers)
/// {
///     if (layer is ISupportsGpuForward&lt;T&gt; gpuLayer)
///     {
///         var next = gpuLayer.ForwardGpu(current);
///         if (current != gpuInput) current.Dispose(); // Dispose intermediate
///         current = next;
///     }
///     else
///     {
///         // Fallback to CPU for non-GPU layers
///         var cpuResult = layer.Forward(current.ToTensor());
///         current.Dispose();
///         current = engine.UploadToGpu(cpuResult);
///     }
/// }
///
/// // Only download final result
/// var output = current.ToTensor();
/// current.Dispose();
/// </code>
/// </remarks>
public interface ISupportsGpuForward<T>
{
    /// <summary>
    /// Performs a forward pass with GPU-resident input and output.
    /// The input tensor remains on GPU and the result stays on GPU.
    /// </summary>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <returns>GPU-resident output tensor. Caller is responsible for disposal.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown if no GPU backend is available or the layer is not initialized.
    /// </exception>
    IGpuTensor<T> ForwardGpu(IGpuTensor<T> input);

    /// <summary>
    /// Gets whether this layer can execute on GPU.
    /// May return false if no GPU backend is available or initialization failed.
    /// </summary>
    bool CanExecuteOnGpu { get; }
}
