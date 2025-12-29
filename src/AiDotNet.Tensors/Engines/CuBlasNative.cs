#if !NET462
using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// cuBLAS status codes returned by cuBLAS functions.
/// </summary>
public enum CublasStatus
{
    Success = 0,
    NotInitialized = 1,
    AllocFailed = 3,
    InvalidValue = 7,
    ArchMismatch = 8,
    MappingError = 11,
    ExecutionFailed = 13,
    InternalError = 14,
    NotSupported = 15,
    LicenseError = 16
}

/// <summary>
/// cuBLAS operation type for matrix transpose.
/// </summary>
public enum CublasOperation
{
    None = 0,      // CUBLAS_OP_N - No transpose
    Transpose = 1,  // CUBLAS_OP_T - Transpose
    ConjugateTranspose = 2  // CUBLAS_OP_C - Conjugate transpose
}

/// <summary>
/// CUDA driver API result codes.
/// </summary>
public enum CudaResult
{
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidContext = 201,
    ContextAlreadyCurrent = 202,
    MapFailed = 205,
    UnmapFailed = 206,
    ArrayIsMapped = 207,
    AlreadyMapped = 208,
    NoBinaryForGpu = 209,
    AlreadyAcquired = 210,
    NotMapped = 211,
    NotMappedAsArray = 212,
    NotMappedAsPointer = 213,
    LaunchFailed = 719,
    Unknown = 999
}

/// <summary>
/// Direct P/Invoke bindings for CUDA Driver API and cuBLAS.
/// Bypasses ILGPU to achieve peak GPU performance (~30,000 GFLOPS).
/// </summary>
/// <remarks>
/// <para><b>Why Direct P/Invoke?</b></para>
/// <para>
/// ILGPU's auto-grouped kernels are limited to ~52-86 GFLOPS for GEMM.
/// cuBLAS achieves ~30,000 GFLOPS through:
/// - Hand-tuned PTX/SASS assembly for each GPU architecture
/// - Tensor core utilization (WMMA instructions)
/// - Multi-level tiling (register, shared memory, L2 cache)
/// - Software pipelining and double-buffering
/// </para>
/// <para><b>Usage Pattern:</b></para>
/// <code>
/// using var context = new CudaContext();
/// using var blas = new CuBlasHandle(context);
///
/// var d_A = context.Allocate&lt;float&gt;(m * k);
/// var d_B = context.Allocate&lt;float&gt;(k * n);
/// var d_C = context.Allocate&lt;float&gt;(m * n);
///
/// context.CopyToDevice(d_A, hostA);
/// context.CopyToDevice(d_B, hostB);
///
/// blas.Sgemm(CublasOperation.None, CublasOperation.None,
///            m, n, k, 1.0f, d_A, m, d_B, k, 0.0f, d_C, m);
///
/// context.CopyToHost(hostC, d_C);
/// </code>
/// </remarks>
public static class CuBlasNative
{
    // cuBLAS library name varies by platform and CUDA version
    // Windows: cublas64_12.dll (CUDA 12.x)
    // Linux: libcublas.so.12
    private const string CublasLibrary = "cublas64_12";
    private const string CudaLibrary = "nvcuda";

    #region CUDA Driver API

    /// <summary>
    /// Initialize the CUDA driver API. Must be called before any other CUDA function.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuInit")]
    public static extern CudaResult cuInit(uint flags);

    /// <summary>
    /// Returns the number of CUDA-capable devices.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuDeviceGetCount")]
    public static extern CudaResult cuDeviceGetCount(out int count);

    /// <summary>
    /// Returns a handle to the specified device.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuDeviceGet")]
    public static extern CudaResult cuDeviceGet(out int device, int ordinal);

    /// <summary>
    /// Returns the name of the device.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuDeviceGetName")]
    public static extern CudaResult cuDeviceGetName(
        [MarshalAs(UnmanagedType.LPStr)] System.Text.StringBuilder name,
        int len,
        int device);

    /// <summary>
    /// Returns the compute capability of the device.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuDeviceGetAttribute")]
    public static extern CudaResult cuDeviceGetAttribute(
        out int value,
        int attribute,
        int device);

    /// <summary>
    /// Creates a CUDA context on the specified device.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuCtxCreate_v2")]
    public static extern CudaResult cuCtxCreate(out IntPtr context, uint flags, int device);

    /// <summary>
    /// Destroys a CUDA context.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuCtxDestroy_v2")]
    public static extern CudaResult cuCtxDestroy(IntPtr context);

    /// <summary>
    /// Pushes a context on the current CPU thread's stack.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuCtxPushCurrent_v2")]
    public static extern CudaResult cuCtxPushCurrent(IntPtr context);

    /// <summary>
    /// Pops the current context from the CPU thread's stack.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuCtxPopCurrent_v2")]
    public static extern CudaResult cuCtxPopCurrent(out IntPtr context);

    /// <summary>
    /// Synchronizes the current context.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuCtxSynchronize")]
    public static extern CudaResult cuCtxSynchronize();

    /// <summary>
    /// Allocates device memory.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuMemAlloc_v2")]
    public static extern CudaResult cuMemAlloc(out IntPtr devicePtr, ulong byteSize);

    /// <summary>
    /// Frees device memory.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuMemFree_v2")]
    public static extern CudaResult cuMemFree(IntPtr devicePtr);

    /// <summary>
    /// Copies memory from host to device.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuMemcpyHtoD_v2")]
    public static extern CudaResult cuMemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, ulong byteCount);

    /// <summary>
    /// Copies memory from device to host.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuMemcpyDtoH_v2")]
    public static extern CudaResult cuMemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, ulong byteCount);

    /// <summary>
    /// Copies memory from device to device.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuMemcpyDtoD_v2")]
    public static extern CudaResult cuMemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, ulong byteCount);

    /// <summary>
    /// Sets device memory to a value.
    /// </summary>
    [DllImport(CudaLibrary, EntryPoint = "cuMemsetD32_v2")]
    public static extern CudaResult cuMemsetD32(IntPtr dstDevice, uint value, ulong count);

    #endregion

    #region cuBLAS Handle Management

    /// <summary>
    /// Creates a cuBLAS handle.
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasCreate_v2")]
    public static extern CublasStatus cublasCreate(out IntPtr handle);

    /// <summary>
    /// Destroys a cuBLAS handle.
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasDestroy_v2")]
    public static extern CublasStatus cublasDestroy(IntPtr handle);

    /// <summary>
    /// Sets the cuBLAS stream.
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasSetStream_v2")]
    public static extern CublasStatus cublasSetStream(IntPtr handle, IntPtr stream);

    /// <summary>
    /// Gets the cuBLAS version.
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasGetVersion_v2")]
    public static extern CublasStatus cublasGetVersion(IntPtr handle, out int version);

    #endregion

    #region cuBLAS GEMM Operations

    /// <summary>
    /// Single-precision General Matrix Multiply (SGEMM).
    /// C = alpha * op(A) * op(B) + beta * C
    /// </summary>
    /// <remarks>
    /// This is the core operation for DenseLayer forward pass.
    /// Peak performance: ~30,000 GFLOPS on modern GPUs.
    ///
    /// Note: cuBLAS uses column-major order. For row-major matrices:
    /// - Swap A and B
    /// - Swap m and n
    /// - Swap lda and ldb
    /// - Use transpose operations accordingly
    /// </remarks>
    [DllImport(CublasLibrary, EntryPoint = "cublasSgemm_v2")]
    public static extern CublasStatus cublasSgemm(
        IntPtr handle,
        CublasOperation transa,
        CublasOperation transb,
        int m,          // Rows of op(A) and C
        int n,          // Columns of op(B) and C
        int k,          // Columns of op(A) and rows of op(B)
        ref float alpha,
        IntPtr A,       // Device pointer to matrix A
        int lda,        // Leading dimension of A
        IntPtr B,       // Device pointer to matrix B
        int ldb,        // Leading dimension of B
        ref float beta,
        IntPtr C,       // Device pointer to matrix C
        int ldc);       // Leading dimension of C

    /// <summary>
    /// Double-precision General Matrix Multiply (DGEMM).
    /// C = alpha * op(A) * op(B) + beta * C
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasDgemm_v2")]
    public static extern CublasStatus cublasDgemm(
        IntPtr handle,
        CublasOperation transa,
        CublasOperation transb,
        int m,
        int n,
        int k,
        ref double alpha,
        IntPtr A,
        int lda,
        IntPtr B,
        int ldb,
        ref double beta,
        IntPtr C,
        int ldc);

    /// <summary>
    /// Batched single-precision GEMM for processing multiple matrices at once.
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasSgemmBatched")]
    public static extern CublasStatus cublasSgemmBatched(
        IntPtr handle,
        CublasOperation transa,
        CublasOperation transb,
        int m, int n, int k,
        ref float alpha,
        IntPtr[] Aarray, int lda,
        IntPtr[] Barray, int ldb,
        ref float beta,
        IntPtr[] Carray, int ldc,
        int batchCount);

    /// <summary>
    /// Strided batched single-precision GEMM (more efficient for contiguous batches).
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasSgemmStridedBatched")]
    public static extern CublasStatus cublasSgemmStridedBatched(
        IntPtr handle,
        CublasOperation transa,
        CublasOperation transb,
        int m, int n, int k,
        ref float alpha,
        IntPtr A, int lda, long strideA,
        IntPtr B, int ldb, long strideB,
        ref float beta,
        IntPtr C, int ldc, long strideC,
        int batchCount);

    #endregion

    #region cuBLAS Vector Operations (for bias add, etc.)

    /// <summary>
    /// Single-precision scalar-vector multiply and add: y = alpha * x + y
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasSaxpy_v2")]
    public static extern CublasStatus cublasSaxpy(
        IntPtr handle,
        int n,
        ref float alpha,
        IntPtr x, int incx,
        IntPtr y, int incy);

    /// <summary>
    /// Single-precision vector copy: y = x
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasScopy_v2")]
    public static extern CublasStatus cublasScopy(
        IntPtr handle,
        int n,
        IntPtr x, int incx,
        IntPtr y, int incy);

    /// <summary>
    /// Single-precision dot product: result = x · y
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasSdot_v2")]
    public static extern CublasStatus cublasSdot(
        IntPtr handle,
        int n,
        IntPtr x, int incx,
        IntPtr y, int incy,
        out float result);

    /// <summary>
    /// Single-precision vector scaling: x = alpha * x
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasSscal_v2")]
    public static extern CublasStatus cublasSscal(
        IntPtr handle,
        int n,
        ref float alpha,
        IntPtr x, int incx);

    #endregion

    #region Tensor Core Operations (Volta+)

    /// <summary>
    /// Sets the math mode for cuBLAS (enables tensor cores).
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasSetMathMode")]
    public static extern CublasStatus cublasSetMathMode(IntPtr handle, int mode);

    // Math modes
    public const int CUBLAS_DEFAULT_MATH = 0;
    public const int CUBLAS_TENSOR_OP_MATH = 1;  // Allow tensor core acceleration
    public const int CUBLAS_TF32_TENSOR_OP_MATH = 3;  // TF32 for Ampere+

    /// <summary>
    /// Half-precision GEMM (uses tensor cores on Volta+).
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasHgemm")]
    public static extern CublasStatus cublasHgemm(
        IntPtr handle,
        CublasOperation transa,
        CublasOperation transb,
        int m, int n, int k,
        ref ushort alpha,  // __half
        IntPtr A, int lda,
        IntPtr B, int ldb,
        ref ushort beta,
        IntPtr C, int ldc);

    /// <summary>
    /// Mixed-precision GEMM with tensor cores.
    /// Computes in FP16, accumulates in FP32.
    /// </summary>
    [DllImport(CublasLibrary, EntryPoint = "cublasGemmEx")]
    public static extern CublasStatus cublasGemmEx(
        IntPtr handle,
        CublasOperation transa,
        CublasOperation transb,
        int m, int n, int k,
        IntPtr alpha,
        IntPtr A, int Atype, int lda,
        IntPtr B, int Btype, int ldb,
        IntPtr beta,
        IntPtr C, int Ctype, int ldc,
        int computeType,
        int algo);

    // CUDA data types for cublasGemmEx
    public const int CUDA_R_16F = 2;   // __half
    public const int CUDA_R_32F = 0;   // float
    public const int CUDA_R_64F = 1;   // double
    public const int CUDA_R_16BF = 14; // __nv_bfloat16

    // Compute types
    public const int CUBLAS_COMPUTE_16F = 64;
    public const int CUBLAS_COMPUTE_32F = 68;
    public const int CUBLAS_COMPUTE_32F_FAST_16F = 74;
    public const int CUBLAS_COMPUTE_32F_FAST_TF32 = 77;

    #endregion

    #region Helper Methods

    /// <summary>
    /// Checks if CUDA is available on this system.
    /// </summary>
    public static bool IsCudaAvailable()
    {
        try
        {
            var result = cuInit(0);
            if (result != CudaResult.Success)
                return false;

            int deviceCount;
            result = cuDeviceGetCount(out deviceCount);
            return result == CudaResult.Success && deviceCount > 0;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    /// <summary>
    /// Gets a descriptive error message for a CUDA result.
    /// </summary>
    public static string GetCudaErrorString(CudaResult result)
    {
        return result switch
        {
            CudaResult.Success => "Success",
            CudaResult.InvalidValue => "Invalid value",
            CudaResult.OutOfMemory => "Out of memory",
            CudaResult.NotInitialized => "CUDA not initialized",
            CudaResult.NoDevice => "No CUDA device available",
            CudaResult.InvalidDevice => "Invalid device",
            CudaResult.InvalidContext => "Invalid context",
            CudaResult.LaunchFailed => "Kernel launch failed",
            _ => $"Unknown CUDA error ({(int)result})"
        };
    }

    /// <summary>
    /// Gets a descriptive error message for a cuBLAS status.
    /// </summary>
    public static string GetCublasErrorString(CublasStatus status)
    {
        return status switch
        {
            CublasStatus.Success => "Success",
            CublasStatus.NotInitialized => "cuBLAS not initialized",
            CublasStatus.AllocFailed => "Allocation failed",
            CublasStatus.InvalidValue => "Invalid value",
            CublasStatus.ArchMismatch => "Architecture mismatch",
            CublasStatus.MappingError => "Mapping error",
            CublasStatus.ExecutionFailed => "Execution failed",
            CublasStatus.InternalError => "Internal error",
            CublasStatus.NotSupported => "Not supported",
            _ => $"Unknown cuBLAS error ({(int)status})"
        };
    }

    /// <summary>
    /// Throws an exception if the CUDA result is not success.
    /// </summary>
    public static void CheckCudaResult(CudaResult result, string operation = "CUDA operation")
    {
        if (result != CudaResult.Success)
            throw new InvalidOperationException($"{operation} failed: {GetCudaErrorString(result)}");
    }

    /// <summary>
    /// Throws an exception if the cuBLAS status is not success.
    /// </summary>
    public static void CheckCublasStatus(CublasStatus status, string operation = "cuBLAS operation")
    {
        if (status != CublasStatus.Success)
            throw new InvalidOperationException($"{operation} failed: {GetCublasErrorString(status)}");
    }

    #endregion
}

/// <summary>
/// Managed wrapper for a CUDA device memory allocation.
/// </summary>
public sealed class CudaDeviceMemory<T> : IDisposable where T : unmanaged
{
    private IntPtr _devicePtr;
    private readonly long _count;
    private bool _disposed;

    /// <summary>
    /// Gets the device pointer.
    /// </summary>
    public IntPtr DevicePtr => _devicePtr;

    /// <summary>
    /// Gets the number of elements.
    /// </summary>
    public long Count => _count;

    /// <summary>
    /// Gets the size in bytes.
    /// </summary>
    public ulong ByteSize => (ulong)(_count * Marshal.SizeOf<T>());

    internal CudaDeviceMemory(IntPtr devicePtr, long count)
    {
        _devicePtr = devicePtr;
        _count = count;
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_devicePtr != IntPtr.Zero)
        {
            CuBlasNative.cuMemFree(_devicePtr);
            _devicePtr = IntPtr.Zero;
        }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CudaDeviceMemory()
    {
        Dispose();
    }
}

/// <summary>
/// Managed wrapper for a CUDA context with cuBLAS support.
/// </summary>
public sealed class CudaBlasContext : IDisposable
{
    private IntPtr _cudaContext;
    private IntPtr _cublasHandle;
    private readonly int _deviceId;
    private bool _disposed;

    /// <summary>
    /// Gets the cuBLAS handle for BLAS operations.
    /// </summary>
    public IntPtr CublasHandle => _cublasHandle;

    /// <summary>
    /// Gets whether this context is valid.
    /// </summary>
    public bool IsValid => !_disposed && _cudaContext != IntPtr.Zero && _cublasHandle != IntPtr.Zero;

    /// <summary>
    /// Creates a new CUDA context with cuBLAS on the specified device.
    /// </summary>
    public CudaBlasContext(int deviceId = 0)
    {
        _deviceId = deviceId;

        // Initialize CUDA
        CuBlasNative.CheckCudaResult(CuBlasNative.cuInit(0), "cuInit");

        // Get device
        int device;
        CuBlasNative.CheckCudaResult(CuBlasNative.cuDeviceGet(out device, deviceId), "cuDeviceGet");

        // Create context
        CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxCreate(out _cudaContext, 0, device), "cuCtxCreate");

        // Create cuBLAS handle
        CuBlasNative.CheckCublasStatus(CuBlasNative.cublasCreate(out _cublasHandle), "cublasCreate");

        // Enable tensor cores if available (Volta+)
        CuBlasNative.cublasSetMathMode(_cublasHandle, CuBlasNative.CUBLAS_TENSOR_OP_MATH);
    }

    /// <summary>
    /// Allocates device memory.
    /// </summary>
    public CudaDeviceMemory<T> Allocate<T>(long count) where T : unmanaged
    {
        ThrowIfDisposed();

        ulong byteSize = (ulong)(count * Marshal.SizeOf<T>());
        IntPtr devicePtr;
        CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAlloc(out devicePtr, byteSize), "cuMemAlloc");

        return new CudaDeviceMemory<T>(devicePtr, count);
    }

    /// <summary>
    /// Copies data from host to device.
    /// </summary>
    public unsafe void CopyToDevice<T>(CudaDeviceMemory<T> dst, T[] src) where T : unmanaged
    {
        ThrowIfDisposed();

        if (src.Length > dst.Count)
            throw new ArgumentException($"Source array ({src.Length}) larger than device memory ({dst.Count})");

        fixed (T* srcPtr = src)
        {
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuMemcpyHtoD(dst.DevicePtr, (IntPtr)srcPtr, (ulong)(src.Length * sizeof(T))),
                "cuMemcpyHtoD");
        }
    }

    /// <summary>
    /// Copies data from device to host.
    /// </summary>
    public unsafe void CopyToHost<T>(T[] dst, CudaDeviceMemory<T> src) where T : unmanaged
    {
        ThrowIfDisposed();

        if (dst.Length > src.Count)
            throw new ArgumentException($"Destination array ({dst.Length}) larger than device memory ({src.Count})");

        fixed (T* dstPtr = dst)
        {
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuMemcpyDtoH((IntPtr)dstPtr, src.DevicePtr, (ulong)(dst.Length * sizeof(T))),
                "cuMemcpyDtoH");
        }
    }

    /// <summary>
    /// Copies data from host span to device.
    /// </summary>
    public unsafe void CopyToDevice<T>(CudaDeviceMemory<T> dst, ReadOnlySpan<T> src) where T : unmanaged
    {
        ThrowIfDisposed();

        if (src.Length > dst.Count)
            throw new ArgumentException($"Source span ({src.Length}) larger than device memory ({dst.Count})");

        fixed (T* srcPtr = src)
        {
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuMemcpyHtoD(dst.DevicePtr, (IntPtr)srcPtr, (ulong)(src.Length * sizeof(T))),
                "cuMemcpyHtoD");
        }
    }

    /// <summary>
    /// Copies data from device to host span.
    /// </summary>
    public unsafe void CopyToHost<T>(Span<T> dst, CudaDeviceMemory<T> src) where T : unmanaged
    {
        ThrowIfDisposed();

        if (dst.Length > src.Count)
            throw new ArgumentException($"Destination span ({dst.Length}) larger than device memory ({src.Count})");

        fixed (T* dstPtr = dst)
        {
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuMemcpyDtoH((IntPtr)dstPtr, src.DevicePtr, (ulong)(dst.Length * sizeof(T))),
                "cuMemcpyDtoH");
        }
    }

    /// <summary>
    /// Performs SGEMM: C = alpha * A * B + beta * C
    /// </summary>
    /// <remarks>
    /// For row-major matrices (C# default), this computes C = alpha * B^T * A^T + beta * C^T
    /// which is equivalent to C^T = alpha * A * B + beta * C.
    ///
    /// To compute C = A * B in row-major:
    /// - Call Sgemm with A and B swapped, m and n swapped
    /// - Result is correctly in row-major format
    /// </remarks>
    public void Sgemm(
        CublasOperation transa, CublasOperation transb,
        int m, int n, int k,
        float alpha,
        CudaDeviceMemory<float> A, int lda,
        CudaDeviceMemory<float> B, int ldb,
        float beta,
        CudaDeviceMemory<float> C, int ldc)
    {
        ThrowIfDisposed();

        CuBlasNative.CheckCublasStatus(
            CuBlasNative.cublasSgemm(
                _cublasHandle,
                transa, transb,
                m, n, k,
                ref alpha,
                A.DevicePtr, lda,
                B.DevicePtr, ldb,
                ref beta,
                C.DevicePtr, ldc),
            "cublasSgemm");
    }

    /// <summary>
    /// Performs DGEMM: C = alpha * A * B + beta * C
    /// </summary>
    public void Dgemm(
        CublasOperation transa, CublasOperation transb,
        int m, int n, int k,
        double alpha,
        CudaDeviceMemory<double> A, int lda,
        CudaDeviceMemory<double> B, int ldb,
        double beta,
        CudaDeviceMemory<double> C, int ldc)
    {
        ThrowIfDisposed();

        CuBlasNative.CheckCublasStatus(
            CuBlasNative.cublasDgemm(
                _cublasHandle,
                transa, transb,
                m, n, k,
                ref alpha,
                A.DevicePtr, lda,
                B.DevicePtr, ldb,
                ref beta,
                C.DevicePtr, ldc),
            "cublasDgemm");
    }

    /// <summary>
    /// Synchronizes the CUDA context.
    /// </summary>
    public void Synchronize()
    {
        ThrowIfDisposed();
        CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxSynchronize(), "cuCtxSynchronize");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CudaBlasContext));
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_cublasHandle != IntPtr.Zero)
        {
            CuBlasNative.cublasDestroy(_cublasHandle);
            _cublasHandle = IntPtr.Zero;
        }

        if (_cudaContext != IntPtr.Zero)
        {
            CuBlasNative.cuCtxDestroy(_cudaContext);
            _cudaContext = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CudaBlasContext()
    {
        Dispose();
    }
}

/// <summary>
/// High-level matrix multiplication using cuBLAS.
/// Handles row-major to column-major conversion automatically.
/// </summary>
/// <remarks>
/// <para><b>Performance Target: ~30,000 GFLOPS</b></para>
/// <para>
/// cuBLAS uses column-major storage (Fortran-style), while C#/.NET uses row-major.
/// For row-major C = A × B, we compute via cuBLAS: C^T = B^T × A^T
/// Since row-major C is stored the same as column-major C^T, we get correct results.
/// </para>
/// <para><b>Usage for DenseLayer:</b>
/// <code>
/// // DenseLayer forward: output = input @ weights.T + bias
/// // input: [batch, in_features]
/// // weights: [out_features, in_features]
/// // weights.T: [in_features, out_features]
/// // output: [batch, out_features]
///
/// using var cuBlas = new CuBlasMatMul();
/// var output = cuBlas.MatMul(input, weightsTransposed);
/// </code>
/// </para>
/// </remarks>
public sealed class CuBlasMatMul : IDisposable
{
    private CudaBlasContext? _context;
    private bool _disposed;
    private bool _initialized;
    private readonly object _initLock = new object();

    /// <summary>
    /// Gets whether cuBLAS is available on this system.
    /// </summary>
    public static bool IsAvailable => CuBlasNative.IsCudaAvailable();

    /// <summary>
    /// Gets whether this instance has been successfully initialized.
    /// </summary>
    public bool IsInitialized => _initialized && _context != null && _context.IsValid;

    /// <summary>
    /// Creates a new CuBlasMatMul instance.
    /// Initialization is deferred until first use for lazy loading.
    /// </summary>
    public CuBlasMatMul()
    {
    }

    /// <summary>
    /// Ensures the cuBLAS context is initialized.
    /// Thread-safe lazy initialization.
    /// </summary>
    /// <returns>True if initialization succeeded, false otherwise.</returns>
    private bool EnsureInitialized()
    {
        if (_initialized) return _context != null && _context.IsValid;

        lock (_initLock)
        {
            if (_initialized) return _context != null && _context.IsValid;

            try
            {
                _context = new CudaBlasContext();
                _initialized = true;
                return true;
            }
            catch (Exception)
            {
                _initialized = true; // Mark as attempted even if failed
                return false;
            }
        }
    }

    /// <summary>
    /// Performs row-major matrix multiplication: C = A × B
    /// </summary>
    /// <param name="A">Left matrix [m, k]</param>
    /// <param name="B">Right matrix [k, n]</param>
    /// <returns>Result matrix [m, n], or null if cuBLAS is unavailable.</returns>
    /// <remarks>
    /// <para>
    /// For row-major storage, cuBLAS computes: C^T = B^T × A^T
    /// which is equivalent to: C = A × B in row-major format.
    /// </para>
    /// </remarks>
    public float[]? MatMulFloat(float[] A, int m, int k, float[] B, int kB, int n)
    {
        if (k != kB)
            throw new ArgumentException($"Inner dimensions must match: A is [{m},{k}], B is [{kB},{n}]");

        if (!EnsureInitialized() || _context == null)
            return null;

        try
        {
            // Allocate device memory
            using var d_A = _context.Allocate<float>(m * k);
            using var d_B = _context.Allocate<float>(k * n);
            using var d_C = _context.Allocate<float>(m * n);

            // Copy to device
            _context.CopyToDevice(d_A, A);
            _context.CopyToDevice(d_B, B);

            // For row-major C = A × B, call cuBLAS as C^T = B^T × A^T
            // In column-major terms with our row-major data:
            // - What we see as A[m,k] in row-major is A^T[k,m] in column-major
            // - What we see as B[k,n] in row-major is B^T[n,k] in column-major
            // - We want C[m,n] in row-major, which is C^T[n,m] in column-major
            //
            // So we compute: C^T[n,m] = B^T[n,k] × A^T[k,m]
            // Using cuBLAS: cublasSgemm(N, N, n, m, k, alpha, B, n, A, k, beta, C, n)
            //
            // This computes C = B × A in cuBLAS notation (column-major),
            // which gives us C = A × B in row-major when we read C back.

            float alpha = 1.0f;
            float beta = 0.0f;

            _context.Sgemm(
                CublasOperation.None, CublasOperation.None,
                n, m, k,  // Swap m and n for the transpose trick
                alpha,
                d_B, n,   // B is [k, n] row-major = [n, k] col-major, leading dim = n
                d_A, k,   // A is [m, k] row-major = [k, m] col-major, leading dim = k
                beta,
                d_C, n);  // C is [m, n] row-major = [n, m] col-major, leading dim = n

            // Synchronize and copy back
            _context.Synchronize();

            var result = new float[m * n];
            _context.CopyToHost(result, d_C);

            return result;
        }
        catch (Exception)
        {
            return null; // Return null to signal fallback to CPU
        }
    }

    /// <summary>
    /// Performs row-major matrix multiplication: C = A × B (double precision)
    /// </summary>
    /// <param name="A">Left matrix [m, k]</param>
    /// <param name="B">Right matrix [k, n]</param>
    /// <returns>Result matrix [m, n], or null if cuBLAS is unavailable.</returns>
    public double[]? MatMulDouble(double[] A, int m, int k, double[] B, int kB, int n)
    {
        if (k != kB)
            throw new ArgumentException($"Inner dimensions must match: A is [{m},{k}], B is [{kB},{n}]");

        if (!EnsureInitialized() || _context == null)
            return null;

        try
        {
            // Allocate device memory
            using var d_A = _context.Allocate<double>(m * k);
            using var d_B = _context.Allocate<double>(k * n);
            using var d_C = _context.Allocate<double>(m * n);

            // Copy to device
            _context.CopyToDevice(d_A, A);
            _context.CopyToDevice(d_B, B);

            // Same transpose trick as float version
            double alpha = 1.0;
            double beta = 0.0;

            _context.Dgemm(
                CublasOperation.None, CublasOperation.None,
                n, m, k,
                alpha,
                d_B, n,
                d_A, k,
                beta,
                d_C, n);

            _context.Synchronize();

            var result = new double[m * n];
            _context.CopyToHost(result, d_C);

            return result;
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    /// Performs matrix multiplication with persistent GPU weights.
    /// The weights matrix stays on GPU, only input is uploaded per call.
    /// </summary>
    /// <param name="input">Input matrix [batch, inputSize] - uploaded each call.</param>
    /// <param name="cachedWeights">Weights already on GPU [inputSize, outputSize].</param>
    /// <returns>Output matrix [batch, outputSize], or null if failed.</returns>
    /// <remarks>
    /// <para><b>This is the key optimization for DenseLayer:</b></para>
    /// <para>
    /// - Weights (typically large, e.g., 768×3072 = 9MB) stay on GPU
    /// - Only activations (much smaller per batch) transfer each forward pass
    /// - Eliminates 99%+ of memory transfer overhead
    /// </para>
    /// </remarks>
    public float[]? MatMulWithCachedWeightsFloat(
        float[] input, int batch, int inputSize,
        CudaDeviceMemory<float> cachedWeights, int outputSize)
    {
        if (!EnsureInitialized() || _context == null)
            return null;

        try
        {
            // Allocate device memory for input and output
            using var d_input = _context.Allocate<float>(batch * inputSize);
            using var d_output = _context.Allocate<float>(batch * outputSize);

            // Copy input to device (weights already there)
            _context.CopyToDevice(d_input, input);

            // GEMM: output[batch, out] = input[batch, in] × weights[in, out]
            // For row-major with column-major cuBLAS, use the transpose trick
            float alpha = 1.0f;
            float beta = 0.0f;

            // Call with swapped operand order for row-major
            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasSgemm(
                    _context.CublasHandle,
                    CublasOperation.None, CublasOperation.None,
                    outputSize, batch, inputSize,  // n, m, k
                    ref alpha,
                    cachedWeights.DevicePtr, outputSize,  // weights [in, out] row-major
                    d_input.DevicePtr, inputSize,          // input [batch, in] row-major
                    ref beta,
                    d_output.DevicePtr, outputSize),       // output [batch, out] row-major
                "cublasSgemm with cached weights");

            _context.Synchronize();

            var result = new float[batch * outputSize];
            _context.CopyToHost(result, d_output);

            return result;
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    /// Allocates and uploads weights to GPU for persistent caching.
    /// </summary>
    /// <param name="weights">Weight matrix [inputSize, outputSize] in row-major order.</param>
    /// <returns>GPU memory handle, or null if allocation failed.</returns>
    public CudaDeviceMemory<float>? AllocateWeightsFloat(float[] weights)
    {
        if (!EnsureInitialized() || _context == null)
            return null;

        try
        {
            var deviceMem = _context.Allocate<float>(weights.Length);
            _context.CopyToDevice(deviceMem, weights);
            return deviceMem;
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    /// Allocates and uploads weights to GPU for persistent caching (double precision).
    /// </summary>
    public CudaDeviceMemory<double>? AllocateWeightsDouble(double[] weights)
    {
        if (!EnsureInitialized() || _context == null)
            return null;

        try
        {
            var deviceMem = _context.Allocate<double>(weights.Length);
            _context.CopyToDevice(deviceMem, weights);
            return deviceMem;
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    /// Gets GPU memory usage for diagnostic purposes.
    /// </summary>
    public (long used, long total) GetMemoryInfo()
    {
        // Note: Would need cuMemGetInfo P/Invoke for accurate data
        return (0, 0);
    }

    public void Dispose()
    {
        if (_disposed) return;

        _context?.Dispose();
        _context = null;
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CuBlasMatMul()
    {
        Dispose();
    }
}
#endif
