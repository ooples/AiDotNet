using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Gpu;

/// <summary>
/// Interface for GPU backend implementations.
/// </summary>
/// <typeparam name="T">The numeric type for GPU operations.</typeparam>
/// <remarks>
/// <para>
/// This interface defines the contract for GPU acceleration backends.
/// Implementations provide GPU-accelerated tensor operations and memory management.
/// </para>
/// <para><b>For Beginners:</b> This is the blueprint for how we talk to the GPU.
///
/// Think of it like a universal remote control:
/// - Different GPU brands (NVIDIA, AMD, Intel) are like different TV brands
/// - This interface is like the standard buttons (volume, channel, etc.)
/// - Each implementation knows how to actually communicate with specific hardware
///
/// This abstraction lets us write code once and run on any GPU!
/// </para>
/// </remarks>
public interface IGpuBackend<T> : IDisposable
{
    /// <summary>
    /// Gets the type of GPU device this backend uses.
    /// </summary>
    GpuDeviceType DeviceType { get; }

    /// <summary>
    /// Gets a value indicating whether the GPU is available and initialized.
    /// </summary>
    bool IsAvailable { get; }

    /// <summary>
    /// Gets the name of the GPU device.
    /// </summary>
    string DeviceName { get; }

    /// <summary>
    /// Gets the total memory available on the GPU in bytes.
    /// </summary>
    long TotalMemory { get; }

    /// <summary>
    /// Gets the amount of free memory on the GPU in bytes.
    /// </summary>
    long FreeMemory { get; }

    /// <summary>
    /// Initializes the GPU backend.
    /// </summary>
    void Initialize();

    /// <summary>
    /// Synchronizes the GPU, waiting for all operations to complete.
    /// </summary>
    void Synchronize();

    #region Memory Management

    /// <summary>
    /// Allocates a GPU tensor with the specified shape.
    /// </summary>
    /// <param name="shape">The shape of the tensor to allocate.</param>
    /// <returns>A new GPU tensor.</returns>
    GpuTensor<T> Allocate(int[] shape);

    /// <summary>
    /// Transfers a CPU tensor to GPU memory.
    /// </summary>
    /// <param name="cpuTensor">The CPU tensor to transfer.</param>
    /// <returns>A GPU tensor containing the same data.</returns>
    GpuTensor<T> ToGpu(Tensor<T> cpuTensor);

    /// <summary>
    /// Transfers a GPU tensor to CPU memory.
    /// </summary>
    /// <param name="gpuTensor">The GPU tensor to transfer.</param>
    /// <returns>A CPU tensor containing the same data.</returns>
    Tensor<T> ToCpu(GpuTensor<T> gpuTensor);

    /// <summary>
    /// Frees GPU memory occupied by a tensor.
    /// </summary>
    /// <param name="gpuTensor">The GPU tensor to free.</param>
    void Free(GpuTensor<T> gpuTensor);

    #endregion

    #region Basic Operations

    /// <summary>
    /// Performs element-wise addition of two GPU tensors.
    /// </summary>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A new GPU tensor containing the sum.</returns>
    GpuTensor<T> Add(GpuTensor<T> a, GpuTensor<T> b);

    /// <summary>
    /// Performs element-wise subtraction of two GPU tensors.
    /// </summary>
    /// <param name="a">The tensor to subtract from.</param>
    /// <param name="b">The tensor to subtract.</param>
    /// <returns>A new GPU tensor containing the difference.</returns>
    GpuTensor<T> Subtract(GpuTensor<T> a, GpuTensor<T> b);

    /// <summary>
    /// Performs element-wise multiplication of two GPU tensors.
    /// </summary>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A new GPU tensor containing the product.</returns>
    GpuTensor<T> Multiply(GpuTensor<T> a, GpuTensor<T> b);

    /// <summary>
    /// Performs element-wise division of two GPU tensors.
    /// </summary>
    /// <param name="a">The numerator tensor.</param>
    /// <param name="b">The denominator tensor.</param>
    /// <returns>A new GPU tensor containing the quotient.</returns>
    GpuTensor<T> Divide(GpuTensor<T> a, GpuTensor<T> b);

    #endregion

    #region Linear Algebra

    /// <summary>
    /// Performs matrix multiplication of two GPU tensors.
    /// </summary>
    /// <param name="a">The first matrix (M x K).</param>
    /// <param name="b">The second matrix (K x N).</param>
    /// <returns>A new GPU tensor containing the result (M x N).</returns>
    GpuTensor<T> MatMul(GpuTensor<T> a, GpuTensor<T> b);

    /// <summary>
    /// Transposes a GPU tensor.
    /// </summary>
    /// <param name="a">The tensor to transpose.</param>
    /// <returns>A new GPU tensor containing the transposed result.</returns>
    GpuTensor<T> Transpose(GpuTensor<T> a);

    #endregion

    #region Activations

    /// <summary>
    /// Applies ReLU activation function element-wise.
    /// </summary>
    /// <param name="a">The input tensor.</param>
    /// <returns>A new GPU tensor with ReLU applied.</returns>
    GpuTensor<T> ReLU(GpuTensor<T> a);

    /// <summary>
    /// Applies Sigmoid activation function element-wise.
    /// </summary>
    /// <param name="a">The input tensor.</param>
    /// <returns>A new GPU tensor with Sigmoid applied.</returns>
    GpuTensor<T> Sigmoid(GpuTensor<T> a);

    /// <summary>
    /// Applies Tanh activation function element-wise.
    /// </summary>
    /// <param name="a">The input tensor.</param>
    /// <returns>A new GPU tensor with Tanh applied.</returns>
    GpuTensor<T> Tanh(GpuTensor<T> a);

    #endregion

    #region Reductions

    /// <summary>
    /// Computes the sum of all elements in a GPU tensor.
    /// </summary>
    /// <param name="a">The input tensor.</param>
    /// <returns>A scalar GPU tensor containing the sum.</returns>
    GpuTensor<T> Sum(GpuTensor<T> a);

    /// <summary>
    /// Computes the mean of all elements in a GPU tensor.
    /// </summary>
    /// <param name="a">The input tensor.</param>
    /// <returns>A scalar GPU tensor containing the mean.</returns>
    GpuTensor<T> Mean(GpuTensor<T> a);

    #endregion
}
