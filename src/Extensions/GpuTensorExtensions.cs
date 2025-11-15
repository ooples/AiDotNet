using AiDotNet.Gpu;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Extensions;

/// <summary>
/// Extension methods for GPU tensor operations on existing Tensor, Matrix, and Vector types.
/// </summary>
/// <remarks>
/// <para>
/// These extensions provide seamless integration between existing CPU-based types
/// and GPU-accelerated operations. They allow you to easily move data to/from GPU
/// while maintaining compatibility with your existing codebase.
/// </para>
/// <para><b>For Beginners:</b> These extensions let you use GPU acceleration with your existing code!
///
/// Instead of rewriting everything, you can now do:
/// <code>
/// // Your existing CPU code
/// var tensor = new Tensor&lt;float&gt;(shape);
///
/// // Move to GPU for acceleration
/// var gpuTensor = tensor.ToGpu(backend);
///
/// // Do fast GPU operations
/// var result = backend.Add(gpuTensor, gpuTensor);
///
/// // Move back to CPU
/// var cpuResult = result.ToCpu(backend);
/// </code>
///
/// This means you can accelerate specific bottlenecks without changing your entire codebase!
/// </para>
/// </remarks>
public static class GpuTensorExtensions
{
    #region Tensor Extensions

    /// <summary>
    /// Transfers a CPU tensor to GPU memory.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="cpuTensor">The CPU tensor to transfer.</param>
    /// <param name="backend">The GPU backend to use.</param>
    /// <returns>A GPU tensor containing the same data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This uploads your tensor data to the GPU.
    ///
    /// When to use:
    /// - Before performing GPU-accelerated operations
    /// - When you have data on CPU but want GPU speed
    ///
    /// Performance tip:
    /// - Transfer is slow (memory bandwidth limited)
    /// - Do as many operations on GPU as possible before transferring back
    /// - Transfer once, compute many times!
    /// </para>
    /// </remarks>
    public static GpuTensor<T> ToGpu<T>(this Tensor<T> cpuTensor, IGpuBackend<T> backend)
        where T : unmanaged
    {
        if (cpuTensor == null)
        {
            throw new ArgumentNullException(nameof(cpuTensor));
        }

        if (backend == null)
        {
            throw new ArgumentNullException(nameof(backend));
        }

        return backend.ToGpu(cpuTensor);
    }

    /// <summary>
    /// Transfers a GPU tensor to CPU memory, converting to Tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gpuTensor">The GPU tensor to transfer.</param>
    /// <param name="backend">The GPU backend to use.</param>
    /// <returns>A CPU Tensor containing the same data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This downloads GPU data back to regular memory.
    ///
    /// When to use:
    /// - After GPU computations are complete
    /// - When you need to access individual elements
    /// - When saving results or displaying to user
    ///
    /// Note: Always dispose GPU tensors after transferring to avoid memory leaks!
    /// </para>
    /// </remarks>
    public static Tensor<T> ToCpu<T>(this GpuTensor<T> gpuTensor, IGpuBackend<T> backend)
        where T : unmanaged
    {
        if (gpuTensor == null)
        {
            throw new ArgumentNullException(nameof(gpuTensor));
        }

        if (backend == null)
        {
            throw new ArgumentNullException(nameof(backend));
        }

        return backend.ToCpu(gpuTensor);
    }

    #endregion

    #region Matrix Extensions

    /// <summary>
    /// Transfers a CPU matrix to GPU memory as a 2D tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="cpuMatrix">The CPU matrix to transfer.</param>
    /// <param name="backend">The GPU backend to use.</param>
    /// <returns>A GPU tensor containing the matrix data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Uploads a matrix to GPU for accelerated linear algebra.
    ///
    /// This is especially useful for:
    /// - Matrix multiplication (matmul)
    /// - Neural network weight operations
    /// - Large matrix transformations
    ///
    /// GPU matmul can be 10-100x faster for large matrices!
    /// </para>
    /// </remarks>
    public static GpuTensor<T> ToGpu<T>(this Matrix<T> cpuMatrix, IGpuBackend<T> backend)
        where T : unmanaged
    {
        if (cpuMatrix == null)
        {
            throw new ArgumentNullException(nameof(cpuMatrix));
        }

        if (backend == null)
        {
            throw new ArgumentNullException(nameof(backend));
        }

        // Convert Matrix to Tensor first
        var shape = new[] { cpuMatrix.Rows, cpuMatrix.Cols };
        var tensor = new Tensor<T>(shape);

        for (int i = 0; i < cpuMatrix.Rows; i++)
        {
            for (int j = 0; j < cpuMatrix.Cols; j++)
            {
                tensor[new[] { i, j }] = cpuMatrix[i, j];
            }
        }

        return backend.ToGpu(tensor);
    }

    /// <summary>
    /// Transfers a GPU tensor to CPU memory as a Matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="gpuTensor">The GPU tensor to transfer (must be 2D).</param>
    /// <param name="backend">The GPU backend to use.</param>
    /// <returns>A CPU Matrix containing the same data.</returns>
    /// <exception cref="ArgumentException">Thrown if the GPU tensor is not 2D.</exception>
    public static Matrix<T> ToMatrix<T>(this GpuTensor<T> gpuTensor, IGpuBackend<T> backend)
        where T : unmanaged
    {
        if (gpuTensor == null)
        {
            throw new ArgumentNullException(nameof(gpuTensor));
        }

        if (backend == null)
        {
            throw new ArgumentNullException(nameof(backend));
        }

        if (gpuTensor.Rank != 2)
        {
            throw new ArgumentException(
                $"GPU tensor must be 2D to convert to Matrix. Got rank {gpuTensor.Rank}");
        }

        var cpuTensor = backend.ToCpu(gpuTensor);
        var matrix = new Matrix<T>(gpuTensor.Shape[0], gpuTensor.Shape[1]);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Cols; j++)
            {
                matrix[i, j] = cpuTensor[new[] { i, j }];
            }
        }

        return matrix;
    }

    #endregion

    #region Vector Extensions

    /// <summary>
    /// Transfers a CPU vector to GPU memory as a 1D tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of vector elements.</typeparam>
    /// <param name="cpuVector">The CPU vector to transfer.</param>
    /// <param name="backend">The GPU backend to use.</param>
    /// <returns>A GPU tensor containing the vector data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Uploads a vector to GPU.
    ///
    /// Useful for:
    /// - Bias terms in neural networks
    /// - Gradient vectors
    /// - Large vector operations
    /// </para>
    /// </remarks>
    public static GpuTensor<T> ToGpu<T>(this Vector<T> cpuVector, IGpuBackend<T> backend)
        where T : unmanaged
    {
        if (cpuVector == null)
        {
            throw new ArgumentNullException(nameof(cpuVector));
        }

        if (backend == null)
        {
            throw new ArgumentNullException(nameof(backend));
        }

        // Convert Vector to Tensor first
        var shape = new[] { cpuVector.Length };
        var tensor = new Tensor<T>(shape);

        for (int i = 0; i < cpuVector.Length; i++)
        {
            tensor[new[] { i }] = cpuVector[i];
        }

        return backend.ToGpu(tensor);
    }

    /// <summary>
    /// Transfers a GPU tensor to CPU memory as a Vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of vector elements.</typeparam>
    /// <param name="gpuTensor">The GPU tensor to transfer (must be 1D).</param>
    /// <param name="backend">The GPU backend to use.</param>
    /// <returns>A CPU Vector containing the same data.</returns>
    /// <exception cref="ArgumentException">Thrown if the GPU tensor is not 1D.</exception>
    public static Vector<T> ToVector<T>(this GpuTensor<T> gpuTensor, IGpuBackend<T> backend)
        where T : unmanaged
    {
        if (gpuTensor == null)
        {
            throw new ArgumentNullException(nameof(gpuTensor));
        }

        if (backend == null)
        {
            throw new ArgumentNullException(nameof(backend));
        }

        if (gpuTensor.Rank != 1)
        {
            throw new ArgumentException(
                $"GPU tensor must be 1D to convert to Vector. Got rank {gpuTensor.Rank}");
        }

        var cpuTensor = backend.ToCpu(gpuTensor);
        var vector = new Vector<T>(gpuTensor.Shape[0]);

        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = cpuTensor[new[] { i }];
        }

        return vector;
    }

    #endregion

    #region Batch Operations

    /// <summary>
    /// Executes a GPU operation and automatically transfers the result back to CPU.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="backend">The GPU backend to use.</param>
    /// <param name="operation">The GPU operation to perform.</param>
    /// <returns>The result as a CPU tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A convenience method for GPU operations.
    ///
    /// This automatically handles:
    /// 1. Transfer to GPU
    /// 2. Perform operation
    /// 3. Transfer back to CPU
    /// 4. Cleanup GPU memory
    ///
    /// Example:
    /// <code>
    /// var result = inputTensor.WithGpu(backend, gpu =>
    /// {
    ///     var temp = backend.ReLU(gpu);
    ///     return backend.Add(temp, temp);
    /// });
    /// </code>
    /// </para>
    /// </remarks>
    public static Tensor<T> WithGpu<T>(
        this Tensor<T> tensor,
        IGpuBackend<T> backend,
        Func<GpuTensor<T>, GpuTensor<T>> operation)
        where T : unmanaged
    {
        using var gpuInput = tensor.ToGpu(backend);
        using var gpuResult = operation(gpuInput);
        return gpuResult.ToCpu(backend);
    }

    /// <summary>
    /// Executes a GPU operation on two tensors and returns the result on CPU.
    /// </summary>
    public static Tensor<T> WithGpu<T>(
        this Tensor<T> tensor1,
        Tensor<T> tensor2,
        IGpuBackend<T> backend,
        Func<GpuTensor<T>, GpuTensor<T>, GpuTensor<T>> operation)
        where T : unmanaged
    {
        using var gpu1 = tensor1.ToGpu(backend);
        using var gpu2 = tensor2.ToGpu(backend);
        using var gpuResult = operation(gpu1, gpu2);
        return gpuResult.ToCpu(backend);
    }

    #endregion

    #region Performance Helpers

    /// <summary>
    /// Estimates whether GPU acceleration would be beneficial for this tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor to evaluate.</param>
    /// <param name="threshold">Minimum elements to benefit from GPU (default: 100,000).</param>
    /// <returns>True if GPU acceleration is likely beneficial.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Helps you decide when to use GPU.
    ///
    /// Rules of thumb:
    /// - Small tensors (<100K elements): CPU faster (transfer overhead)
    /// - Medium tensors (100K-1M): GPU ~2-5x faster
    /// - Large tensors (>1M): GPU 10-100x faster
    ///
    /// Use this to automatically choose CPU or GPU!
    /// </para>
    /// </remarks>
    public static bool ShouldUseGpu<T>(this Tensor<T> tensor, int threshold = 100_000)
    {
        return tensor.Length >= threshold;
    }

    /// <summary>
    /// Estimates the transfer cost (in milliseconds) for moving this tensor to/from GPU.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor to evaluate.</param>
    /// <returns>Estimated transfer time in milliseconds.</returns>
    public static double EstimateTransferCost<T>(this Tensor<T> tensor)
        where T : unmanaged
    {
        // PCIe 3.0 x16 bandwidth: ~16 GB/s (conservative estimate)
        const double BANDWIDTH_GB_PER_SEC = 12.0; // Conservative to account for overhead
        const double BYTES_TO_GB = 1_000_000_000.0;

        unsafe
        {
            var elementSize = sizeof(T);
            var totalBytes = tensor.Length * elementSize;
            var transferTimeSeconds = totalBytes / (BANDWIDTH_GB_PER_SEC * BYTES_TO_GB / 1000.0);

            // Round-trip cost (to GPU + from GPU)
            return transferTimeSeconds * 2.0 * 1000.0; // Convert to milliseconds
        }
    }

    #endregion
}
