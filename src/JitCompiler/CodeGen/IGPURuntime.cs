namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Interface for GPU runtime implementations.
/// </summary>
/// <remarks>
/// <para>
/// This interface defines the contract for GPU runtime implementations that can
/// compile and execute generated kernel code. Implementations would wrap CUDA Runtime,
/// OpenCL, Metal, or Vulkan APIs.
/// </para>
/// <para><b>For Beginners:</b> This is the bridge between generated code and actual GPU execution.
///
/// The code generator produces kernel source code, but to actually run it:
/// 1. The source must be compiled to GPU machine code
/// 2. Memory must be allocated on the GPU
/// 3. Data must be transferred to the GPU
/// 4. The kernel must be launched
/// 5. Results must be transferred back
///
/// This interface defines all those operations.
/// </para>
/// </remarks>
public interface IGPURuntime : IDisposable
{
    /// <summary>
    /// Gets information about the current GPU device.
    /// </summary>
    GPUCodeGenerator.GPUDeviceInfo DeviceInfo { get; }

    /// <summary>
    /// Compiles kernel source code into an executable module.
    /// </summary>
    /// <param name="sourceCode">The kernel source code.</param>
    /// <param name="kernelName">The name of the kernel function.</param>
    /// <returns>A handle to the compiled kernel.</returns>
    IGPUKernelHandle CompileKernel(string sourceCode, string kernelName);

    /// <summary>
    /// Allocates memory on the GPU.
    /// </summary>
    /// <param name="sizeBytes">Number of bytes to allocate.</param>
    /// <returns>A handle to the allocated memory.</returns>
    IGPUMemoryHandle Allocate(long sizeBytes);

    /// <summary>
    /// Copies data from host to GPU memory.
    /// </summary>
    /// <param name="destination">GPU memory handle.</param>
    /// <param name="source">Source data array.</param>
    void CopyToDevice<T>(IGPUMemoryHandle destination, T[] source);

    /// <summary>
    /// Copies data from GPU to host memory.
    /// </summary>
    /// <param name="destination">Destination array.</param>
    /// <param name="source">GPU memory handle.</param>
    void CopyFromDevice<T>(T[] destination, IGPUMemoryHandle source);

    /// <summary>
    /// Launches a kernel with the specified configuration.
    /// </summary>
    /// <param name="kernel">The kernel to launch.</param>
    /// <param name="gridSize">Number of blocks in each dimension.</param>
    /// <param name="blockSize">Number of threads per block in each dimension.</param>
    /// <param name="sharedMemorySize">Dynamic shared memory size in bytes.</param>
    /// <param name="arguments">Kernel arguments (GPU memory handles or scalars).</param>
    void LaunchKernel(IGPUKernelHandle kernel, int[] gridSize, int[] blockSize, int sharedMemorySize, params object[] arguments);

    /// <summary>
    /// Synchronizes with the GPU, waiting for all pending operations to complete.
    /// </summary>
    void Synchronize();

    /// <summary>
    /// Frees GPU memory.
    /// </summary>
    /// <param name="memory">The memory handle to free.</param>
    void Free(IGPUMemoryHandle memory);
}
