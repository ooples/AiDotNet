// Copyright (c) AiDotNet. All rights reserved.
// Direct CUDA backend for NVIDIA GPUs (Driver API + NVRTC + cuBLAS fallback).
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed class CudaBackend : IDirectGpuBackend
{
    private const int DefaultBlockSize = 256;
    private readonly Dictionary<string, IntPtr> _kernelCache;
    private IntPtr _cudaContext;
    private IntPtr _stream;
    private IntPtr _cublasHandle;
    private IntPtr _activationModule;
    private bool _disposed;

    public bool IsAvailable { get; }
    public string BackendName => "CUDA";
    public string DeviceName { get; }
    public string DeviceVendor => "NVIDIA";
    public int ComputeUnits { get; }
    public long GlobalMemoryBytes { get; }
    public long LocalMemoryBytes { get; }

    public static bool IsCudaAvailable => CudaNativeBindings.IsAvailable && NvrtcNativeBindings.IsAvailable;

    public CudaBackend()
    {
        _kernelCache = new Dictionary<string, IntPtr>(StringComparer.Ordinal);

        if (!CudaNativeBindings.IsAvailable || !NvrtcNativeBindings.IsAvailable)
        {
            IsAvailable = false;
            DeviceName = "None";
            return;
        }

        try
        {
            CuBlasNative.CheckCudaResult(CuBlasNative.cuInit(0), "cuInit");
            CuBlasNative.CheckCudaResult(CuBlasNative.cuDeviceGet(out int device, 0), "cuDeviceGet");

            var nameBuilder = new StringBuilder(256);
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuDeviceGetName(nameBuilder, nameBuilder.Capacity, device),
                "cuDeviceGetName");
            DeviceName = nameBuilder.ToString();

            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuDeviceGetAttribute(out int multiprocessors, (int)CudaDeviceAttribute.MultiprocessorCount, device),
                "cuDeviceGetAttribute(MultiprocessorCount)");
            ComputeUnits = multiprocessors;

            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuDeviceGetAttribute(out int sharedMem, (int)CudaDeviceAttribute.MaxSharedMemoryPerBlock, device),
                "cuDeviceGetAttribute(MaxSharedMemoryPerBlock)");
            LocalMemoryBytes = sharedMem;

            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuDeviceTotalMem(out ulong totalMem, device), "cuDeviceTotalMem");
            GlobalMemoryBytes = (long)totalMem;

            CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxCreate(out _cudaContext, 0, device), "cuCtxCreate");
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamCreate(out _stream, 0), "cuStreamCreate");

            CuBlasNative.CheckCublasStatus(CuBlasNative.cublasCreate(out _cublasHandle), "cublasCreate");
            CuBlasNative.CheckCublasStatus(CuBlasNative.cublasSetStream(_cublasHandle, _stream), "cublasSetStream");
            CuBlasNative.cublasSetMathMode(_cublasHandle, CuBlasNative.CUBLAS_TENSOR_OP_MATH);

            CompileActivationKernels(device);

            IsAvailable = true;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"CudaBackend initialization failed: {ex.GetType().Name}: {ex.Message}");
            DeviceName = "None";
            IsAvailable = false;
            Dispose();
        }
    }

    private readonly struct CudaContextScope : IDisposable
    {
        private readonly bool _active;

        public CudaContextScope(IntPtr context)
        {
            _active = context != IntPtr.Zero;
            if (_active)
                CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxPushCurrent(context), "cuCtxPushCurrent");
        }

        public void Dispose()
        {
            if (_active)
                CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxPopCurrent(out _), "cuCtxPopCurrent");
        }
    }

    private CudaContextScope PushContext()
    {
        return new CudaContextScope(_cudaContext);
    }

    private static (int Major, int Minor) GetComputeCapability(int device)
    {
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuDeviceGetAttribute(out int major, (int)CudaDeviceAttribute.ComputeCapabilityMajor, device),
            "cuDeviceGetAttribute(ComputeCapabilityMajor)");
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuDeviceGetAttribute(out int minor, (int)CudaDeviceAttribute.ComputeCapabilityMinor, device),
            "cuDeviceGetAttribute(ComputeCapabilityMinor)");

        if (major <= 0)
            return (5, 2);

        return (major, minor);
    }

    private void CompileActivationKernels(int device)
    {
        using var _ = PushContext();

        string source = CudaActivationKernels.GetSource();
        string[] kernelNames = CudaActivationKernels.GetKernelNames();

        var (major, minor) = GetComputeCapability(device);
        string arch = $"--gpu-architecture=compute_{major}{minor}";
        string[] options = new[] { arch, "--use_fast_math" };

        IntPtr program = IntPtr.Zero;
        var result = NvrtcNativeBindings.nvrtcCreateProgram(
            ref program,
            source,
            "activation_kernels.cu",
            0,
            IntPtr.Zero,
            IntPtr.Zero);
        if (result != NvrtcResult.Success)
            throw new InvalidOperationException($"NVRTC program creation failed: {NvrtcNativeBindings.GetErrorString(result)}");

        result = NvrtcNativeBindings.nvrtcCompileProgram(program, options.Length, options);
        if (result != NvrtcResult.Success)
        {
            string log = GetNvrtcLog(program);
            NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
            throw new InvalidOperationException($"NVRTC compile failed: {NvrtcNativeBindings.GetErrorString(result)}\n{log}");
        }

        result = NvrtcNativeBindings.nvrtcGetPTXSize(program, out UIntPtr ptxSize);
        if (result != NvrtcResult.Success || ptxSize == UIntPtr.Zero)
        {
            NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
            throw new InvalidOperationException("NVRTC failed to return PTX size.");
        }

        IntPtr ptx = Marshal.AllocHGlobal((int)ptxSize);
        result = NvrtcNativeBindings.nvrtcGetPTX(program, ptx);
        NvrtcNativeBindings.nvrtcDestroyProgram(ref program);

        if (result != NvrtcResult.Success)
        {
            Marshal.FreeHGlobal(ptx);
            throw new InvalidOperationException($"NVRTC get PTX failed: {NvrtcNativeBindings.GetErrorString(result)}");
        }

        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuModuleLoadData(out _activationModule, ptx), "cuModuleLoadData");
        Marshal.FreeHGlobal(ptx);

        foreach (var kernelName in kernelNames)
        {
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuModuleGetFunction(out IntPtr kernel, _activationModule, kernelName),
                $"cuModuleGetFunction({kernelName})");
            _kernelCache[kernelName] = kernel;
        }
    }

    private static string GetNvrtcLog(IntPtr program)
    {
        var result = NvrtcNativeBindings.nvrtcGetProgramLogSize(program, out UIntPtr logSize);
        if (result != NvrtcResult.Success || logSize == UIntPtr.Zero)
            return string.Empty;

        IntPtr logPtr = Marshal.AllocHGlobal((int)logSize);
        NvrtcNativeBindings.nvrtcGetProgramLog(program, logPtr);
        string log = Marshal.PtrToStringAnsi(logPtr) ?? string.Empty;
        Marshal.FreeHGlobal(logPtr);
        return log;
    }

    public IGpuBuffer AllocateBuffer(float[] data)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        using var _ = PushContext();
        int size = data.Length;
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(data), "Buffer size must be positive.");
        ulong byteSize = (ulong)size * sizeof(float);

        CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAlloc(out IntPtr devicePtr, byteSize), "cuMemAlloc");

        try
        {
            unsafe
            {
                fixed (float* src = data)
                {
                    CuBlasNative.CheckCudaResult(
                        CuBlasNative.cuMemcpyHtoD(devicePtr, (IntPtr)src, byteSize),
                        "cuMemcpyHtoD");
                }
            }
        }
        catch
        {
            CuBlasNative.cuMemFree(devicePtr);
            throw;
        }

        return new CudaGpuBuffer(_cudaContext, devicePtr, size);
    }

    public IGpuBuffer AllocateBuffer(int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Buffer size must be positive.");

        using var _ = PushContext();
        ulong byteSize = (ulong)size * sizeof(float);
        CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAlloc(out IntPtr devicePtr, byteSize), "cuMemAlloc");
        CuBlasNative.CheckCudaResult(CuBlasNative.cuMemsetD32(devicePtr, 0, (ulong)size), "cuMemsetD32");
        return new CudaGpuBuffer(_cudaContext, devicePtr, size);
    }

    public float[] DownloadBuffer(IGpuBuffer buffer)
    {
        var result = new float[buffer.Size];
        DownloadBuffer(buffer, result);
        return result;
    }

    public void DownloadBuffer(IGpuBuffer buffer, float[] destination)
    {
        if (destination.Length < buffer.Size)
            throw new ArgumentException("Destination array is too small.", nameof(destination));

        using var _ = PushContext();
        ulong byteSize = (ulong)(buffer.Size * sizeof(float));

        unsafe
        {
            fixed (float* dst = destination)
            {
                CuBlasNative.CheckCudaResult(
                    CuBlasNative.cuMemcpyDtoH((IntPtr)dst, buffer.Handle, byteSize),
                    "cuMemcpyDtoH");
            }
        }
    }

    public void Gemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        ValidateGemmArgs(A, B, C, M, N, K);

        using var _ = PushContext();
        float alphaVal = alpha;
        float betaVal = beta;

        // Row-major C = A * B. Use cuBLAS column-major trick: C^T = B^T * A^T.
        CuBlasNative.CheckCublasStatus(
            CuBlasNative.cublasSgemm(
                _cublasHandle,
                CublasOperation.None,
                CublasOperation.None,
                N, M, K,
                ref alphaVal,
                B.Handle, N,
                A.Handle, K,
                ref betaVal,
                C.Handle, N),
            "cublasSgemm");
    }

    public IGpuBuffer MatMul(IGpuBuffer A, IGpuBuffer B, int M, int N, int K)
    {
        ValidateGemmArgs(A, B, null, M, N, K);
        var output = AllocateBuffer(M * N);
        Gemm(A, B, output, M, N, K, 1.0f, 0.0f);
        return output;
    }

    public void BatchedGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount, float alpha = 1.0f, float beta = 0.0f)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        ValidateBatchedGemmArgs(A, B, C, M, N, K, batchCount);

        using var _ = PushContext();
        float alphaVal = alpha;
        float betaVal = beta;

        long strideA = (long)M * K;
        long strideB = (long)K * N;
        long strideC = (long)M * N;

        CuBlasNative.CheckCublasStatus(
            CuBlasNative.cublasSgemmStridedBatched(
                _cublasHandle,
                CublasOperation.None,
                CublasOperation.None,
                N, M, K,
                ref alphaVal,
                B.Handle, N, strideB,
                A.Handle, K, strideA,
                ref betaVal,
                C.Handle, N, strideC,
                batchCount),
            "cublasSgemmStridedBatched");
    }

    public IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        var output = MatMul(A, B, M, N, K);
        ApplyBiasInPlace(output, bias, M, N);
        Relu(output, output, M * N);
        return output;
    }

    public IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        var output = MatMul(A, B, M, N, K);
        ApplyBiasInPlace(output, bias, M, N);
        Gelu(output, output, M * N);
        return output;
    }

    public IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        var output = MatMul(A, B, M, N, K);
        ApplyBiasInPlace(output, bias, M, N);
        Sigmoid(output, output, M * N);
        return output;
    }

    public IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        var output = MatMul(A, B, M, N, K);
        ApplyBiasInPlace(output, bias, M, N);
        Tanh(output, output, M * N);
        return output;
    }

    public void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchElementwiseKernel("add_vectors", A, B, C, size);
    }

    public void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchElementwiseKernel("subtract_vectors", A, B, C, size);
    }

    public void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)    
    {
        LaunchElementwiseKernel("multiply_vectors", A, B, C, size);
    }

    public void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchElementwiseKernel("divide_vectors", A, B, C, size);
    }

    public void Min(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchElementwiseKernel("min_vectors", A, B, C, size);
    }

    public void Max(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchElementwiseKernel("max_vectors", A, B, C, size);
    }

    public void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size)       
    {
        LaunchScaleKernel(A, B, scalar, size);
    }

    public void Power(IGpuBuffer A, IGpuBuffer B, float exponent, int size)
    {
        LaunchUnaryWithScalarKernel("power_scalar", A, B, exponent, size);
    }

    public void Abs(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("abs_vector", A, B, size);
    }

    public void Exp(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("exp_vector", A, B, size);
    }

    public void Exp2(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("exp2_vector", A, B, size);
    }

    public void Exp10(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("exp10_vector", A, B, size);
    }

    public void ExpM1(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("expm1_vector", A, B, size);
    }

    public void Log(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("log_vector", A, B, size);
    }

    public void Log2(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("log2_vector", A, B, size);
    }

    public void Log1P(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("log1p_vector", A, B, size);
    }

    public void Sqrt(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("sqrt_vector", A, B, size);
    }

    public void Sign(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("sign_vector", A, B, size);
    }

    public void Relu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("relu", A, B, size);
    }

    public void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("sigmoid", A, B, size);
    }

    public void Tanh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("tanh_activation", A, B, size);
    }

    public void Gelu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("gelu", A, B, size);
    }

    public void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features)
    {
        LaunchSoftmaxKernel(A, B, batchSize, features);
    }

    public void Enforce2x4Sparsity(IGpuBuffer denseInput, IGpuBuffer sparseValues, IGpuBuffer sparseIndices, int M, int K)
    {
        throw new NotSupportedException("CUDA sparse 2:4 kernels are not implemented yet.");
    }

    public void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
    {
        throw new NotSupportedException("CUDA sparse 2:4 kernels are not implemented yet.");
    }

    public void SparseGemm(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        throw new NotSupportedException("CUDA sparse GEMM is not implemented yet.");
    }

    public IGpuBuffer SparseGemmBiasRelu(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        throw new NotSupportedException("CUDA sparse GEMM + bias + ReLU is not implemented yet.");
    }

    public float Sum(IGpuBuffer A, int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (size <= 0)
            return 0.0f;

        using var _ = PushContext();
        int blockSize = DefaultBlockSize;
        int gridSize = (size + blockSize - 1) / blockSize;

        using var partialBuffer = AllocateBuffer(gridSize);
        LaunchReductionKernel("reduce_sum", A, partialBuffer, size, blockSize);
        Synchronize();

        var partials = DownloadBuffer(partialBuffer);
        float sum = 0.0f;
        for (int i = 0; i < partials.Length; i++)
            sum += partials[i];
        return sum;
    }

    public float Max(IGpuBuffer A, int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (size <= 0)
            return float.MinValue;

        using var _ = PushContext();
        int blockSize = DefaultBlockSize;
        int gridSize = (size + blockSize - 1) / blockSize;

        using var partialBuffer = AllocateBuffer(gridSize);
        LaunchReductionKernel("reduce_max", A, partialBuffer, size, blockSize);
        Synchronize();

        var partials = DownloadBuffer(partialBuffer);
        float max = float.MinValue;
        for (int i = 0; i < partials.Length; i++)
            if (partials[i] > max) max = partials[i];
        return max;
    }

    public void SumAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (outerSize <= 0 || reduceSize <= 0)
            throw new ArgumentException("outerSize and reduceSize must be positive.");

        long requiredInput = (long)outerSize * reduceSize;
        if (requiredInput > int.MaxValue || A.Size < requiredInput)
            throw new ArgumentException("Input buffer size is too small for the specified dimensions.");

        if (B.Size < outerSize)
            throw new ArgumentException("Output buffer size is too small for the specified dimensions.");

        if (!_kernelCache.TryGetValue("sum_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sum_axis");

        using var _ = PushContext();
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = A.Handle;
        IntPtr outputPtr = B.Handle;
        int outer = outerSize;
        int reduce = reduceSize;
        unsafe
        {
            void** args = stackalloc void*[4];
            args[0] = &inputPtr;
            args[1] = &outputPtr;
            args[2] = &outer;
            args[3] = &reduce;
            LaunchKernel(kernel, grid, DefaultBlockSize, args);
        }
    }

    public void Synchronize()
    {
        if (!IsAvailable)
            return;

        using var _ = PushContext();
        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamSynchronize(_stream), "cuStreamSynchronize");
    }

    private unsafe void LaunchUnaryKernel(string kernelName, IGpuBuffer input, IGpuBuffer output, int size)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int n = size;
        void** args = stackalloc void*[3];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchElementwiseKernel(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchScaleKernel(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        if (!_kernelCache.TryGetValue("scale_vector", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: scale_vector");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        float scalarVal = scalar;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &scalarVal;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchUnaryWithScalarKernel(string kernelName, IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        float scalarVal = scalar;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &scalarVal;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchSoftmaxKernel(IGpuBuffer input, IGpuBuffer output, int batchSize, int features)
    {
        if (!_kernelCache.TryGetValue("softmax", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: softmax");

        using var _ = PushContext();
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int batches = batchSize;
        int feats = features;
        void** args = stackalloc void*[4];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batches;
        args[3] = &feats;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchReductionKernel(string kernelName, IGpuBuffer input, IGpuBuffer output, int size, int blockSize)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");

        uint grid = (uint)((size + blockSize - 1) / blockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int n = size;
        void** args = stackalloc void*[3];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &n;

        uint sharedBytes = (uint)(blockSize * sizeof(float));
        LaunchKernelWithSharedMem(kernel, grid, (uint)blockSize, sharedBytes, args);
    }

    private unsafe void ApplyBiasInPlace(IGpuBuffer data, IGpuBuffer bias, int rows, int cols)
    {
        if (!_kernelCache.TryGetValue("bias_add", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bias_add");

        using var _ = PushContext();
        int size = rows * cols;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr dataPtr = data.Handle;
        IntPtr biasPtr = bias.Handle;
        int r = rows;
        int c = cols;
        void** args = stackalloc void*[4];
        args[0] = &dataPtr;
        args[1] = &biasPtr;
        args[2] = &r;
        args[3] = &c;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchKernel(IntPtr kernel, uint gridX, uint blockX, void** args)
    {
        LaunchKernelWithSharedMem(kernel, gridX, blockX, 0, args);
    }

    private unsafe void LaunchKernelWithSharedMem(IntPtr kernel, uint gridX, uint blockX, uint sharedMemBytes, void** args)
    {
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, 1, 1,
                blockX, 1, 1,
                sharedMemBytes,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel");
    }

    private static void ValidateGemmArgs(IGpuBuffer A, IGpuBuffer B, IGpuBuffer? C, int M, int N, int K)
    {
        if (M <= 0 || N <= 0 || K <= 0)
            throw new ArgumentException("Matrix dimensions M, N, K must be positive.");

        long requiredA = (long)M * K;
        long requiredB = (long)K * N;
        long requiredC = (long)M * N;

        if (requiredA > int.MaxValue || requiredB > int.MaxValue || requiredC > int.MaxValue)
            throw new ArgumentException("Matrix dimensions are too large.");

        if (A.Size < requiredA)
            throw new ArgumentException("Buffer A is too small for the specified dimensions.");
        if (B.Size < requiredB)
            throw new ArgumentException("Buffer B is too small for the specified dimensions.");
        if (C != null && C.Size < requiredC)
            throw new ArgumentException("Buffer C is too small for the specified dimensions.");
    }

    private static void ValidateBatchedGemmArgs(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount)
    {
        if (batchCount <= 0)
            throw new ArgumentException("batchCount must be positive.");

        ValidateGemmArgs(A, B, C, M, N, K);

        long requiredA = (long)batchCount * M * K;
        long requiredB = (long)batchCount * K * N;
        long requiredC = (long)batchCount * M * N;

        if (requiredA > int.MaxValue || requiredB > int.MaxValue || requiredC > int.MaxValue)
            throw new ArgumentException("Batched GEMM dimensions are too large.");

        if (A.Size < requiredA)
            throw new ArgumentException("Buffer A is too small for the specified batch dimensions.");
        if (B.Size < requiredB)
            throw new ArgumentException("Buffer B is too small for the specified batch dimensions.");
        if (C.Size < requiredC)
            throw new ArgumentException("Buffer C is too small for the specified batch dimensions.");
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;

        if (_cublasHandle != IntPtr.Zero)
        {
            CuBlasNative.cublasDestroy(_cublasHandle);
            _cublasHandle = IntPtr.Zero;
        }

        if (_stream != IntPtr.Zero)
        {
            CudaNativeBindings.cuStreamDestroy(_stream);
            _stream = IntPtr.Zero;
        }

        if (_activationModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_activationModule);
            _activationModule = IntPtr.Zero;
        }

        if (_cudaContext != IntPtr.Zero)
        {
            CuBlasNative.cuCtxDestroy(_cudaContext);
            _cudaContext = IntPtr.Zero;
        }

        GC.SuppressFinalize(this);
    }

    ~CudaBackend()
    {
        Dispose();
    }

    internal sealed class CudaGpuBuffer : IGpuBuffer
    {
        private IntPtr _context;
        private IntPtr _devicePtr;

        public int Size { get; }
        public long SizeInBytes { get; }
        public IntPtr Handle => _devicePtr;

        public CudaGpuBuffer(IntPtr context, IntPtr devicePtr, int size)
        {
            _context = context;
            _devicePtr = devicePtr;
            Size = size;
            SizeInBytes = (long)size * sizeof(float);
        }

        public void Dispose()
        {
            if (_devicePtr == IntPtr.Zero)
                return;

            try
            {
                if (_context != IntPtr.Zero)
                {
                    CuBlasNative.cuCtxPushCurrent(_context);
                    CuBlasNative.cuMemFree(_devicePtr);
                    CuBlasNative.cuCtxPopCurrent(out _);
                }
            }
            catch
            {
                // Suppress disposal errors to avoid crashing finalizers.
            }

            _devicePtr = IntPtr.Zero;
            _context = IntPtr.Zero;
            GC.SuppressFinalize(this);
        }

        ~CudaGpuBuffer()
        {
            Dispose();
        }
    }
}
