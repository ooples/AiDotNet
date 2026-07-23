using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Helper methods for creating GPU-resident Tensor instances from CPU data.
/// Replaces the deleted GpuTensor&lt;T&gt; constructors with unified Tensor&lt;T&gt; approach.
/// </summary>
internal static class GpuTensorHelper
{
    /// <summary>
    /// Creates a GPU-resident tensor by uploading a CPU tensor's data to the GPU.
    /// Replaces: new GpuTensor&lt;T&gt;(backend, tensor, role)
    /// </summary>
    public static Tensor<T> UploadToGpu<T>(IDirectGpuBackend backend, Tensor<T> cpuTensor, GpuTensorRole role = GpuTensorRole.General)
    {
        var floatData = DirectGpuEngine.ToFloatArray(cpuTensor.GetDataArray());
        var buffer = backend.AllocateBuffer(floatData);
        return Tensor<T>.FromGpuBuffer(backend, buffer, cpuTensor._shape, role);
    }

    /// <summary>
    /// Creates a GPU-resident tensor by uploading a CPU array to the GPU.
    /// Replaces: new GpuTensor&lt;T&gt;(backend, data, shape, role)
    /// </summary>
    public static Tensor<T> UploadToGpu<T>(IDirectGpuBackend backend, T[] data, int[] shape, GpuTensorRole role = GpuTensorRole.General)
    {
        var floatData = DirectGpuEngine.ToFloatArray(data);
        var buffer = backend.AllocateBuffer(floatData);
        return Tensor<T>.FromGpuBuffer(backend, buffer, shape, role);
    }

    /// <summary>
    /// Creates a GPU-resident tensor from an existing GPU buffer.
    /// Replaces: new GpuTensor&lt;T&gt;(backend, buffer, shape, role)
    /// </summary>
    public static Tensor<T> UploadToGpu<T>(IDirectGpuBackend backend, IGpuBuffer buffer, int[] shape, GpuTensorRole role = GpuTensorRole.General, bool ownsBuffer = true)
    {
        return Tensor<T>.FromGpuBuffer(backend, buffer, shape, role, ownsBuffer);
    }
}

/// <summary>
/// Extension methods for Tensor to replace deleted GpuTensor methods.
/// </summary>
internal static class TensorGpuExtensions
{
    /// <summary>
    /// Creates a view of the tensor starting at the given flat offset with the specified shape.
    /// Replaces GpuTensor.CreateView(offset, shape).
    /// </summary>
    public static Tensor<T> CreateView<T>(this Tensor<T> tensor, int offset, int[] shape)
    {
        if (offset == 0)
            return tensor.Reshape(shape);

        int viewSize = 1;
        foreach (var d in shape) viewSize *= d;
        var data = new T[viewSize];
        for (int i = 0; i < viewSize; i++)
            data[i] = tensor.GetFlat(offset + i);
        return new Tensor<T>(data, shape);
    }
}
