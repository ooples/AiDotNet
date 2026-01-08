// Copyright (c) AiDotNet. All rights reserved.
// Direct CUDA backend for NVIDIA GPUs (Driver API + NVRTC + cuBLAS fallback).
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed class CudaBackend : IAsyncGpuBackend
{
    private const int DefaultBlockSize = 256;
    private readonly Dictionary<string, IntPtr> _kernelCache;
    private IntPtr _cudaContext;
    private IntPtr _stream;
    private IntPtr _cublasHandle;
    private CudaStream? _defaultStream;
    private IntPtr _activationModule;
    private IntPtr _convolutionModule;
    private IntPtr _fusedConvolutionModule;
    private IntPtr _poolingModule;
    private IntPtr _normalizationModule;
    private IntPtr _neuralNetModule;
    private IntPtr _fusedModule;
    private IntPtr _attentionModule;
    private IntPtr _fftModule;
    private IntPtr _spatialTransformerModule;
    private IntPtr _sparseModule;
    private IntPtr _locallyConnectedModule;
    private IntPtr _deformableConvModule;
    private bool _disposed;

    public bool IsAvailable { get; }
    public string BackendName => "CUDA";
    public string DeviceName { get; }
    public string DeviceVendor => "NVIDIA";
    public int ComputeUnits { get; }
    public long GlobalMemoryBytes { get; }
    public long LocalMemoryBytes { get; }

    // IAsyncGpuBackend properties
    public bool SupportsMultiStream => true;
    public bool SupportsEvents => true;
    public bool SupportsAsyncTransfer => true;
    public bool SupportsGraphCapture => false; // CUDA graphs not yet implemented
    public int MaxConcurrentStreams => 16;
    public IGpuStream DefaultStream => _defaultStream ?? throw new InvalidOperationException("Backend not initialized");

    public static bool IsCudaAvailable => CudaNativeBindings.IsAvailable && NvrtcNativeBindings.IsAvailable;

    public CudaBackend() : this(0)
    {
    }

    public CudaBackend(int deviceIndex)
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
            CuBlasNative.CheckCudaResult(CuBlasNative.cuDeviceGet(out int device, deviceIndex), "cuDeviceGet");

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
            _defaultStream = new CudaStream(this, _stream, GpuStreamType.Default, ownsHandle: false);

            CuBlasNative.CheckCublasStatus(CuBlasNative.cublasCreate(out _cublasHandle), "cublasCreate");
            CuBlasNative.CheckCublasStatus(CuBlasNative.cublasSetStream(_cublasHandle, _stream), "cublasSetStream");
            CuBlasNative.cublasSetMathMode(_cublasHandle, CuBlasNative.CUBLAS_TENSOR_OP_MATH);

            CompileAllKernels(device);

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

    private IntPtr CompileKernelModule(int device, string source, string moduleName, string[] kernelNames)
    {
        var (major, minor) = GetComputeCapability(device);
        string arch = $"--gpu-architecture=compute_{major}{minor}";
        string[] options = new[] { arch, "--use_fast_math" };

        IntPtr program = IntPtr.Zero;
        var result = NvrtcNativeBindings.nvrtcCreateProgram(
            ref program,
            source,
            moduleName + ".cu",
            0,
            IntPtr.Zero,
            IntPtr.Zero);
        if (result != NvrtcResult.Success)
            throw new InvalidOperationException($"NVRTC program creation failed for {moduleName}: {NvrtcNativeBindings.GetErrorString(result)}");

        result = NvrtcNativeBindings.nvrtcCompileProgram(program, options.Length, options);
        if (result != NvrtcResult.Success)
        {
            string log = GetNvrtcLog(program);
            NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
            throw new InvalidOperationException($"NVRTC compile failed for {moduleName}: {NvrtcNativeBindings.GetErrorString(result)}\n{log}");
        }

        result = NvrtcNativeBindings.nvrtcGetPTXSize(program, out UIntPtr ptxSize);
        if (result != NvrtcResult.Success || ptxSize == UIntPtr.Zero)
        {
            NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
            throw new InvalidOperationException($"NVRTC failed to return PTX size for {moduleName}.");
        }

        IntPtr ptx = Marshal.AllocHGlobal((int)ptxSize);
        result = NvrtcNativeBindings.nvrtcGetPTX(program, ptx);
        NvrtcNativeBindings.nvrtcDestroyProgram(ref program);

        if (result != NvrtcResult.Success)
        {
            Marshal.FreeHGlobal(ptx);
            throw new InvalidOperationException($"NVRTC get PTX failed for {moduleName}: {NvrtcNativeBindings.GetErrorString(result)}");
        }

        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuModuleLoadData(out IntPtr module, ptx), $"cuModuleLoadData({moduleName})");
        Marshal.FreeHGlobal(ptx);

        foreach (var kernelName in kernelNames)
        {
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuModuleGetFunction(out IntPtr kernel, module, kernelName),
                $"cuModuleGetFunction({kernelName})");
            _kernelCache[kernelName] = kernel;
        }

        return module;
    }

    private void CompileAllKernels(int device)
    {
        using var _ = PushContext();

        _activationModule = CompileKernelModule(device, CudaActivationKernels.GetSource(), "activation_kernels", CudaActivationKernels.GetKernelNames());
        _convolutionModule = CompileKernelModule(device, CudaConvolutionKernels.GetSource(), "convolution_kernels", CudaConvolutionKernels.GetKernelNames());
        _fusedConvolutionModule = CompileKernelModule(device, CudaFusedConvolutionKernels.GetSource(), "fused_convolution_kernels", CudaFusedConvolutionKernels.GetKernelNames());
        _poolingModule = CompileKernelModule(device, CudaPoolingKernels.GetSource(), "pooling_kernels", CudaPoolingKernels.GetKernelNames());
        _normalizationModule = CompileKernelModule(device, CudaNormalizationKernels.GetSource(), "normalization_kernels", CudaNormalizationKernels.GetKernelNames());
        _neuralNetModule = CompileKernelModule(device, CudaNeuralNetKernels.GetSource(), "neuralnet_kernels", CudaNeuralNetKernels.GetKernelNames());
        _fusedModule = CompileKernelModule(device, CudaFusedKernels.GetSource(), "fused_kernels", CudaFusedKernels.GetKernelNames());
        _attentionModule = CompileKernelModule(device, CudaAttentionKernels.GetSource(), "attention_kernels", CudaAttentionKernels.GetKernelNames());
        _fftModule = CompileKernelModule(device, Kernels.CudaFFTKernels.GetSource(), "fft_kernels", Kernels.CudaFFTKernels.GetKernelNames());
        _sparseModule = CompileKernelModule(device, CudaSparseKernels.GetSource(), "sparse_kernels", CudaSparseKernels.GetKernelNames());
        _spatialTransformerModule = CompileKernelModule(device, CudaSpatialTransformerKernels.GetSource(), "spatial_transformer_kernels", CudaSpatialTransformerKernels.GetKernelNames());

        // Compile Locally Connected kernels (unique weights per spatial position)
        _locallyConnectedModule = CompileKernelModule(device, CudaLocallyConnectedKernels.GetSource(), "locally_connected_kernels", CudaLocallyConnectedKernels.GetKernelNames());

        // Compile Deformable Convolution kernels (DCNv2 with learnable offsets and masks)
        _deformableConvModule = CompileKernelModule(device, CudaDeformableConvolutionKernels.GetSource(), "deformable_conv_kernels", CudaDeformableConvolutionKernels.GetKernelNames());
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

    public IGpuBuffer AllocateByteBuffer(int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Buffer size must be positive.");

        using var _ = PushContext();
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuMemAlloc(out IntPtr devicePtr, (ulong)size),
            "cuMemAlloc(byte)");

        return new CudaGpuByteBuffer(_cudaContext, devicePtr, size);
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
        ValidateBiasBuffer(bias, N);
        var output = AllocateBuffer(M * N);
        ExecuteFusedGemm("gemm_bias_relu", A, B, bias, output, M, N, K);
        return output;
    }

    public IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ValidateBiasBuffer(bias, N);
        var output = AllocateBuffer(M * N);
        ExecuteFusedGemm("gemm_bias_gelu", A, B, bias, output, M, N, K);
        return output;
    }

    public IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ValidateBiasBuffer(bias, N);
        var output = AllocateBuffer(M * N);
        ExecuteFusedGemm("gemm_bias_sigmoid", A, B, bias, output, M, N, K);
        return output;
    }

    public IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ValidateBiasBuffer(bias, N);
        var output = AllocateBuffer(M * N);
        ExecuteFusedGemm("gemm_bias_tanh", A, B, bias, output, M, N, K);
        return output;
    }

    public IGpuBuffer GemmBias(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ValidateBiasBuffer(bias, N);
        var output = MatMul(A, B, M, N, K);
        ApplyBiasInPlace(output, bias, M, N);
        return output;
    }

    public unsafe void BiasAdd(IGpuBuffer A, IGpuBuffer bias, IGpuBuffer C, int M, int N)
    {
        if (!_kernelCache.TryGetValue("bias_add_out", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bias_add_out");

        using var _ = PushContext();
        int size = M * N;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr biasPtr = bias.Handle;
        IntPtr cPtr = C.Handle;
        int rows = M;
        int cols = N;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &biasPtr;
        args[2] = &cPtr;
        args[3] = &rows;
        args[4] = &cols;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Conv2DBiasAdd(IGpuBuffer output, IGpuBuffer bias, int batch, int channels, int spatialSize)
    {
        if (!_kernelCache.TryGetValue("conv2d_bias_add", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: conv2d_bias_add");

        using var _ = PushContext();
        int totalSize = batch * channels * spatialSize;
        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr outPtr = output.Handle;
        IntPtr biasPtr = bias.Handle;
        void** args = stackalloc void*[5];
        args[0] = &outPtr;
        args[1] = &biasPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &spatialSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
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

    public unsafe void Squash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon)
    {
        if (!_kernelCache.TryGetValue("squash", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: squash");

        using var _ = PushContext();
        uint grid = (uint)((numCapsules + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int numCaps = numCapsules;
        int capDim = capsuleDim;
        float eps = epsilon;
        void** args = stackalloc void*[5];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &numCaps;
        args[3] = &capDim;
        args[4] = &eps;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon)
    {
        if (!_kernelCache.TryGetValue("squash_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: squash_backward");

        using var _ = PushContext();
        uint grid = (uint)((numCapsules + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int numCaps = numCapsules;
        int capDim = capsuleDim;
        float eps = epsilon;
        void** args = stackalloc void*[6];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &numCaps;
        args[4] = &capDim;
        args[5] = &eps;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize)
    {
        if (!_kernelCache.TryGetValue("tile_batch", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: tile_batch");

        using var _ = PushContext();
        int totalSize = repeats * innerSize;
        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int reps = repeats;
        int inner = innerSize;
        void** args = stackalloc void*[4];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &reps;
        args[3] = &inner;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats)
    {
        if (!_kernelCache.TryGetValue("tile_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: tile_axis");

        using var _ = PushContext();
        int totalSize = outerSize * axisSize * repeats * innerSize;
        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int outer = outerSize;
        int axis = axisSize;
        int inner = innerSize;
        int reps = repeats;
        void** args = stackalloc void*[6];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &outer;
        args[3] = &axis;
        args[4] = &inner;
        args[5] = &reps;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #region Trigonometric Operations

    public void Sin(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("sin_vector", A, B, size);
    }

    public void Cos(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("cos_vector", A, B, size);
    }

    public void Tan(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("tan_vector", A, B, size);
    }

    public void Asin(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("asin_vector", A, B, size);
    }

    public void Acos(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("acos_vector", A, B, size);
    }

    public void Atan(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("atan_vector", A, B, size);
    }

    #endregion

    #region Hyperbolic Operations

    public void Sinh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("sinh_vector", A, B, size);
    }

    public void Cosh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("cosh_vector", A, B, size);
    }

    public void Asinh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("asinh_vector", A, B, size);
    }

    public void Acosh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("acosh_vector", A, B, size);
    }

    public void Atanh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("atanh_vector", A, B, size);
    }

    #endregion

    #region Additional Unary Operations

    public void Reciprocal(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("reciprocal_vector", A, B, size);
    }

    public void Cbrt(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("cbrt_vector", A, B, size);
    }

    public void Log10(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("log10_vector", A, B, size);
    }

    public void Negate(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("negate_vector", A, B, size);
    }

    public void Floor(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("floor_vector", A, B, size);
    }

    public void Ceiling(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("ceil_vector", A, B, size);
    }

    public void Round(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("round_vector", A, B, size);
    }

    public void Truncate(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("trunc_vector", A, B, size);
    }

    #endregion

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

    #region CSR Sparse Operations (General Sparsity)

    /// <inheritdoc/>
    public unsafe void CsrSpMM(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer output,
        int M, int K, int N, int nnz)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("csr_spmm", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_spmm");

        using var _ = PushContext();

        // Launch configuration: rows x ceil(N/blockSize) grid
        uint gridX = (uint)M;
        uint gridY = (uint)((N + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr valuesPtr = csrValues.Handle;
        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr denseBPtr = denseB.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[9];
        args[0] = &valuesPtr;
        args[1] = &colIndicesPtr;
        args[2] = &rowPointersPtr;
        args[3] = &denseBPtr;
        args[4] = &outputPtr;
        args[5] = &M;
        args[6] = &K;
        args[7] = &N;
        args[8] = &nnz;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
    }

    /// <inheritdoc/>
    public unsafe void CsrSpMMBias(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer bias,
        IGpuBuffer output,
        int M, int K, int N, int nnz)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("csr_spmm_bias", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_spmm_bias");

        using var _ = PushContext();

        uint gridX = (uint)M;
        uint gridY = (uint)((N + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr valuesPtr = csrValues.Handle;
        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr denseBPtr = denseB.Handle;
        IntPtr biasPtr = bias.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[10];
        args[0] = &valuesPtr;
        args[1] = &colIndicesPtr;
        args[2] = &rowPointersPtr;
        args[3] = &denseBPtr;
        args[4] = &biasPtr;
        args[5] = &outputPtr;
        args[6] = &M;
        args[7] = &K;
        args[8] = &N;
        args[9] = &nnz;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
    }

    /// <inheritdoc/>
    public unsafe void ScatterAddEdges(
        IGpuBuffer input,
        IGpuBuffer sourceIndices,
        IGpuBuffer targetIndices,
        IGpuBuffer? edgeValues,
        IGpuBuffer output,
        int numNodes, int numEdges, int features)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("scatter_add_edges", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: scatter_add_edges");

        using var _ = PushContext();

        // First zero the output buffer
        ZeroBuffer(output, numNodes * features);

        // Launch configuration: edges x ceil(features/blockSize) grid
        uint gridX = (uint)numEdges;
        uint gridY = (uint)((features + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr inputPtr = input.Handle;
        IntPtr sourcePtr = sourceIndices.Handle;
        IntPtr targetPtr = targetIndices.Handle;
        IntPtr edgeValuesPtr = edgeValues?.Handle ?? IntPtr.Zero;
        IntPtr outputPtr = output.Handle;
        int hasEdgeValues = edgeValues is not null ? 1 : 0;

        void** args = stackalloc void*[8];
        args[0] = &inputPtr;
        args[1] = &sourcePtr;
        args[2] = &targetPtr;
        args[3] = &edgeValuesPtr;
        args[4] = &outputPtr;
        args[5] = &numNodes;
        args[6] = &numEdges;
        args[7] = &features;

        // Note: hasEdgeValues is passed as part of the kernel argument structure
        void** args2 = stackalloc void*[9];
        args2[0] = &inputPtr;
        args2[1] = &sourcePtr;
        args2[2] = &targetPtr;
        args2[3] = &edgeValuesPtr;
        args2[4] = &outputPtr;
        args2[5] = &numNodes;
        args2[6] = &numEdges;
        args2[7] = &features;
        args2[8] = &hasEdgeValues;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args2);
    }

    private unsafe void ZeroBuffer(IGpuBuffer buffer, int size)
    {
        if (!_kernelCache.TryGetValue("zero_buffer", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: zero_buffer");

        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr bufferPtr = buffer.Handle;

        void** args = stackalloc void*[2];
        args[0] = &bufferPtr;
        args[1] = &size;

        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchKernel2D(IntPtr kernel, uint gridX, uint gridY, uint blockX, uint blockY, void** args)
    {
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, gridY, 1,
                blockX, blockY, 1,
                0,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel");
    }

    /// <inheritdoc/>
    public unsafe void CsrSegmentedMax(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("csr_segmented_max", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_segmented_max");

        using var _ = PushContext();

        // Launch configuration: rows x ceil(N/blockSize) grid
        uint gridX = (uint)M;
        uint gridY = (uint)((N + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[7];
        args[0] = &colIndicesPtr;
        args[1] = &rowPointersPtr;
        args[2] = &inputPtr;
        args[3] = &outputPtr;
        args[4] = &M;
        args[5] = &K;
        args[6] = &N;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
    }

    /// <inheritdoc/>
    public unsafe void CsrSegmentedMin(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("csr_segmented_min", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_segmented_min");

        using var _ = PushContext();

        // Launch configuration: rows x ceil(N/blockSize) grid
        uint gridX = (uint)M;
        uint gridY = (uint)((N + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[7];
        args[0] = &colIndicesPtr;
        args[1] = &rowPointersPtr;
        args[2] = &inputPtr;
        args[3] = &outputPtr;
        args[4] = &M;
        args[5] = &K;
        args[6] = &N;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
    }

    /// <inheritdoc/>
    public unsafe void CsrSegmentedStdDev(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N,
        float epsilon = 1e-8f)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("csr_segmented_stddev", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_segmented_stddev");

        using var _ = PushContext();

        // Launch configuration: rows x ceil(N/blockSize) grid
        uint gridX = (uint)M;
        uint gridY = (uint)((N + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[8];
        args[0] = &colIndicesPtr;
        args[1] = &rowPointersPtr;
        args[2] = &inputPtr;
        args[3] = &outputPtr;
        args[4] = &M;
        args[5] = &K;
        args[6] = &N;
        args[7] = &epsilon;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
    }

    #endregion

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

    #region IAsyncGpuBackend Implementation

    /// <inheritdoc/>
    public IGpuStream CreateStream(GpuStreamType streamType)
    {
        return new CudaStream(this, streamType, 0);
    }

    /// <inheritdoc/>
    public IGpuStream CreateStream(GpuStreamType streamType, int priority)
    {
        return new CudaStream(this, streamType, priority);
    }

    /// <inheritdoc/>
    public IGpuEvent CreateEvent()
    {
        return new CudaEvent(this, null, enableTiming: false);
    }

    /// <inheritdoc/>
    public IGpuEvent CreateEvent(bool enableTiming)
    {
        return new CudaEvent(this, null, enableTiming);
    }

    /// <inheritdoc/>
    public void RecordEvent(IGpuEvent gpuEvent, IGpuStream stream)
    {
        if (gpuEvent is not CudaEvent cudaEvent)
            throw new ArgumentException("Event must be a CudaEvent", nameof(gpuEvent));

        cudaEvent.Record(stream);
    }

    /// <inheritdoc/>
    public void StreamWaitEvent(IGpuStream stream, IGpuEvent gpuEvent)
    {
        if (stream is not CudaStream cudaStream)
            throw new ArgumentException("Stream must be a CudaStream", nameof(stream));

        cudaStream.WaitEvent(gpuEvent);
    }

    /// <inheritdoc/>
    public GpuSyncPoint CreateSyncPoint(IGpuStream stream)
    {
        if (stream is not CudaStream cudaStream)
            throw new ArgumentException("Stream must be a CudaStream", nameof(stream));

        return new CudaSyncPoint(this, cudaStream);
    }

    /// <inheritdoc/>
    public GpuSyncPoint CreateSyncPoint()
    {
        if (_defaultStream == null)
            throw new InvalidOperationException("Backend not initialized");

        return new CudaSyncPoint(this, _defaultStream);
    }

    /// <inheritdoc/>
    public unsafe void UploadBufferAsync(float[] data, IGpuBuffer buffer, IGpuStream stream)
    {
        if (data == null) throw new ArgumentNullException(nameof(data));
        if (buffer == null) throw new ArgumentNullException(nameof(buffer));
        if (stream == null) throw new ArgumentNullException(nameof(stream));

        using var _ = PushContext();
        fixed (float* dataPtr = data)
        {
            var result = CudaNativeBindings.cuMemcpyHtoDAsync(
                buffer.Handle,
                (IntPtr)dataPtr,
                (ulong)(data.Length * sizeof(float)),
                stream.Handle);
            CuBlasNative.CheckCudaResult(result, "cuMemcpyHtoDAsync");
            // Synchronize stream to ensure transfer completes before the fixed block exits
            var syncResult = CudaNativeBindings.cuStreamSynchronize(stream.Handle);
            CuBlasNative.CheckCudaResult(syncResult, "cuStreamSynchronize");
        }
    }

    /// <inheritdoc/>
    public unsafe void UploadBufferAsync(ReadOnlySpan<float> data, IGpuBuffer buffer, IGpuStream stream)
    {
        if (buffer == null) throw new ArgumentNullException(nameof(buffer));
        if (stream == null) throw new ArgumentNullException(nameof(stream));

        using var _ = PushContext();
        fixed (float* dataPtr = data)
        {
            var result = CudaNativeBindings.cuMemcpyHtoDAsync(
                buffer.Handle,
                (IntPtr)dataPtr,
                (ulong)(data.Length * sizeof(float)),
                stream.Handle);
            CuBlasNative.CheckCudaResult(result, "cuMemcpyHtoDAsync");
            // Synchronize stream to ensure transfer completes before the fixed block exits
            var syncResult = CudaNativeBindings.cuStreamSynchronize(stream.Handle);
            CuBlasNative.CheckCudaResult(syncResult, "cuStreamSynchronize");
        }
    }

    /// <inheritdoc/>
    public unsafe void DownloadBufferAsync(IGpuBuffer buffer, float[] destination, IGpuStream stream)
    {
        if (buffer == null) throw new ArgumentNullException(nameof(buffer));
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (stream == null) throw new ArgumentNullException(nameof(stream));

        using var _ = PushContext();
        fixed (float* destPtr = destination)
        {
            var result = CudaNativeBindings.cuMemcpyDtoHAsync(
                (IntPtr)destPtr,
                buffer.Handle,
                (ulong)(destination.Length * sizeof(float)),
                stream.Handle);
            CuBlasNative.CheckCudaResult(result, "cuMemcpyDtoHAsync");
            // Synchronize stream to ensure transfer completes before the fixed block exits
            var syncResult = CudaNativeBindings.cuStreamSynchronize(stream.Handle);
            CuBlasNative.CheckCudaResult(syncResult, "cuStreamSynchronize");
        }
    }

    /// <inheritdoc/>
    public IGpuBuffer AllocateBufferAsync(float[] data, IGpuStream stream)
    {
        var buffer = AllocateBuffer(data.Length);
        UploadBufferAsync(data, buffer, stream);
        return buffer;
    }

    /// <inheritdoc/>
    public void CopyBufferAsync(IGpuBuffer source, IGpuBuffer destination, int size, IGpuStream stream)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (stream == null) throw new ArgumentNullException(nameof(stream));

        using var _ = PushContext();
        ulong byteSize = (ulong)size * sizeof(float);
        var result = CudaNativeBindings.cuMemcpyDtoDAsync(
            destination.Handle,
            source.Handle,
            byteSize,
            stream.Handle);
        CuBlasNative.CheckCudaResult(result, "cuMemcpyDtoDAsync");
    }

    /// <inheritdoc/>
    public void GemmAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha, float beta, IGpuStream stream)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        ValidateGemmArgs(A, B, C, M, N, K);

        using var _ = PushContext();
        float alphaVal = alpha;
        float betaVal = beta;

        // Set cuBLAS to use the specified stream
        CuBlasNative.CheckCublasStatus(CuBlasNative.cublasSetStream(_cublasHandle, stream.Handle), "cublasSetStream");

        try
        {
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
        finally
        {
            // Restore the default stream
            CuBlasNative.cublasSetStream(_cublasHandle, _stream);
        }
    }

    /// <inheritdoc/>
    public void FusedGemmBiasActivationAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output,
        int M, int N, int K, FusedActivationType activation, IGpuStream stream)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        // Map activation to fused kernel name
        string kernelName = activation switch
        {
            FusedActivationType.ReLU => "fused_gemm_bias_relu",
            FusedActivationType.Sigmoid => "fused_gemm_bias_sigmoid",
            FusedActivationType.Tanh => "fused_gemm_bias_tanh",
            FusedActivationType.None => "fused_gemm_bias",
            _ => throw new NotSupportedException($"Activation type {activation} not supported for fused GEMM")
        };

        ExecuteFusedGemmOnStream(kernelName, A, B, bias, output, M, N, K, stream);
    }

    /// <inheritdoc/>
    public void SynchronizeStream(IGpuStream stream)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));

        using var _ = PushContext();
        var result = CudaNativeBindings.cuStreamSynchronize(stream.Handle);
        CuBlasNative.CheckCudaResult(result, "cuStreamSynchronize");
    }

    /// <inheritdoc/>
    public bool QueryStreamComplete(IGpuStream stream)
    {
        if (stream is not CudaStream cudaStream)
            throw new ArgumentException("Stream must be a CudaStream", nameof(stream));

        return cudaStream.Query();
    }

    /// <inheritdoc/>
    public bool QueryEventComplete(IGpuEvent gpuEvent)
    {
        if (gpuEvent is not CudaEvent cudaEvent)
            throw new ArgumentException("Event must be a CudaEvent", nameof(gpuEvent));

        return cudaEvent.Query();
    }

    /// <inheritdoc/>
    public float GetEventElapsedTime(IGpuEvent start, IGpuEvent end)
    {
        if (start is not CudaEvent cudaStart)
            throw new ArgumentException("Start event must be a CudaEvent", nameof(start));
        if (end is not CudaEvent cudaEnd)
            throw new ArgumentException("End event must be a CudaEvent", nameof(end));

        return cudaEnd.GetElapsedTime(cudaStart);
    }

    /// <summary>
    /// Launches a kernel on a specific stream.
    /// </summary>
    private unsafe void LaunchKernelOnStream(IntPtr kernel, uint gridX, uint blockX, void** args, IntPtr stream, uint sharedMem = 0)
    {
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, 1, 1,
                blockX, 1, 1,
                sharedMem,
                stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel");
    }

    /// <summary>
    /// Launches a 2D kernel on a specific stream.
    /// </summary>
    private unsafe void LaunchKernel2DOnStream(IntPtr kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, void** args, IntPtr stream, uint sharedMem = 0)
    {
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, gridY, gridZ,
                blockX, blockY, 1,
                sharedMem,
                stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel2D");
    }

    /// <summary>
    /// Executes a fused GEMM kernel on a specific stream.
    /// </summary>
    private unsafe void ExecuteFusedGemmOnStream(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output, int M, int N, int K, IGpuStream stream)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA fused kernel not found: {kernelName}");

        using var _ = PushContext();

        const int TILE_SIZE = 16;
        uint gridX = (uint)((N + TILE_SIZE - 1) / TILE_SIZE);
        uint gridY = (uint)((M + TILE_SIZE - 1) / TILE_SIZE);

        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr biasPtr = bias.Handle;
        IntPtr outPtr = output.Handle;
        int m = M, n = N, k = K;

        void** args = stackalloc void*[7];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &biasPtr;
        args[3] = &outPtr;
        args[4] = &m;
        args[5] = &n;
        args[6] = &k;

        LaunchKernel2DOnStream(kernel, gridX, gridY, 1, TILE_SIZE, TILE_SIZE, args, stream.Handle);
    }

    #endregion

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

    private unsafe void LaunchKernel2D(IntPtr kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, void** args)
    {
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, gridY, gridZ,
                blockX, blockY, 1,
                0,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel2D");
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

    private static void ValidateBiasBuffer(IGpuBuffer bias, int n)
    {
        if (bias == null)
            throw new ArgumentNullException(nameof(bias));
        if (n <= 0)
            throw new ArgumentOutOfRangeException(nameof(n), "N must be positive.");
        if (bias.Size < n)
            throw new ArgumentException("Bias buffer size must be at least N.", nameof(bias));
    }

    private unsafe void ExecuteFusedGemm(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output, int M, int N, int K)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA fused kernel not found: {kernelName}");

        using var _ = PushContext();

        const int TILE_SIZE = 16;
        uint gridX = (uint)((N + TILE_SIZE - 1) / TILE_SIZE);
        uint gridY = (uint)((M + TILE_SIZE - 1) / TILE_SIZE);

        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr biasPtr = bias.Handle;
        IntPtr outPtr = output.Handle;
        int m = M, n = N, k = K;

        void** args = stackalloc void*[7];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &biasPtr;
        args[3] = &outPtr;
        args[4] = &m;
        args[5] = &n;
        args[6] = &k;

        LaunchKernel2D(kernel, gridX, gridY, 1, TILE_SIZE, TILE_SIZE, args);
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

    #region Convolution Operations

    public unsafe void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv2d_direct", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: conv2d_direct");

        using var _ = PushContext();
        int totalOutput = batch * outChannels * outHeight * outWidth;
        uint gridX = (uint)((totalOutput + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr kernelPtr = kernel.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[17];
        args[0] = &inputPtr;
        args[1] = &kernelPtr;
        args[2] = &outputPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        args[14] = &padH;
        args[15] = &padW;
        args[16] = &totalOutput;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void Conv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv2d_backward_input", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: conv2d_backward_input");

        using var _ = PushContext();
        int totalInput = batch * inChannels * inHeight * inWidth;
        uint gridX = (uint)((totalInput + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr kernelPtr = kernel.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        void** args = stackalloc void*[17];
        args[0] = &gradOutputPtr;
        args[1] = &kernelPtr;
        args[2] = &gradInputPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        args[14] = &padH;
        args[15] = &padW;
        args[16] = &totalInput;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void Conv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv2d_backward_kernel", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: conv2d_backward_kernel");

        using var _ = PushContext();
        int totalKernel = outChannels * inChannels * kernelH * kernelW;
        uint gridX = (uint)((totalKernel + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr gradKernelPtr = gradKernel.Handle;
        void** args = stackalloc void*[17];
        args[0] = &inputPtr;
        args[1] = &gradOutputPtr;
        args[2] = &gradKernelPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        args[14] = &padH;
        args[15] = &padW;
        args[16] = &totalKernel;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void Conv3D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inDepth, int inHeight, int inWidth,
        int outChannels, int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv3d_direct", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: conv3d_direct");

        using var _ = PushContext();
        int totalOutput = batch * outChannels * outDepth * outHeight * outWidth;
        uint gridX = (uint)((totalOutput + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr kernelPtr = kernel.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[23];
        args[0] = &inputPtr;
        args[1] = &kernelPtr;
        args[2] = &outputPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inDepth;
        args[6] = &inHeight;
        args[7] = &inWidth;
        args[8] = &outChannels;
        args[9] = &outDepth;
        args[10] = &outHeight;
        args[11] = &outWidth;
        args[12] = &kernelD;
        args[13] = &kernelH;
        args[14] = &kernelW;
        args[15] = &strideD;
        args[16] = &strideH;
        args[17] = &strideW;
        args[18] = &padD;
        args[19] = &padH;
        args[20] = &padW;
        args[21] = &totalOutput;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (!_kernelCache.TryGetValue("depthwise_conv2d", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: depthwise_conv2d");

        using var _ = PushContext();
        int totalOutput = batch * channels * outHeight * outWidth;
        uint gridX = (uint)((totalOutput + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr kernelPtr = kernel.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[14];
        args[0] = &inputPtr;
        args[1] = &kernelPtr;
        args[2] = &outputPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outHeight;
        args[8] = &outWidth;
        args[9] = &kernelH;
        args[10] = &kernelW;
        args[11] = &strideH;
        args[12] = &strideW;
        args[13] = &totalOutput;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        if (!_kernelCache.TryGetValue("conv_transpose2d", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: conv_transpose2d");

        using var _ = PushContext();
        int totalOutput = batch * outChannels * outHeight * outWidth;
        uint gridX = (uint)((totalOutput + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr kernelPtr = kernel.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[17];
        args[0] = &inputPtr;
        args[1] = &kernelPtr;
        args[2] = &outputPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        args[14] = &padH;
        args[15] = &padW;
        args[16] = &totalOutput;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    #region Locally Connected Convolution Operations

    public unsafe void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: locally_connected_conv2d");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * outChannels);

        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr biasPtr = bias?.Handle ?? IntPtr.Zero;
        IntPtr outputPtr = output.Handle;
        int hasBias = bias is not null ? 1 : 0;

        void** args = stackalloc void*[16];
        args[0] = &inputPtr;
        args[1] = &weightsPtr;
        args[2] = &biasPtr;
        args[3] = &outputPtr;
        args[4] = &batch;
        args[5] = &inChannels;
        args[6] = &inHeight;
        args[7] = &inWidth;
        args[8] = &outChannels;
        args[9] = &outHeight;
        args[10] = &outWidth;
        args[11] = &kernelH;
        args[12] = &kernelW;
        args[13] = &strideH;
        args[14] = &strideW;
        args[15] = &hasBias;
        LaunchKernel2D(cudaKernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d_backward_input", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: locally_connected_conv2d_backward_input");

        using var _ = PushContext();
        int totalInputSize = batch * inChannels * inHeight * inWidth;
        uint gridX = (uint)((totalInputSize + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr gradInputPtr = gradInput.Handle;

        void** args = stackalloc void*[14];
        args[0] = &gradOutputPtr;
        args[1] = &weightsPtr;
        args[2] = &gradInputPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void LocallyConnectedConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d_backward_weights", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: locally_connected_conv2d_backward_weights");

        using var _ = PushContext();
        int totalWeights = outHeight * outWidth * outChannels * inChannels * kernelH * kernelW;
        uint gridX = (uint)((totalWeights + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradWeightsPtr = gradWeights.Handle;

        void** args = stackalloc void*[14];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gradWeightsPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void LocallyConnectedConv2DBackwardBias(IGpuBuffer gradOutput, IGpuBuffer gradBias,
        int batch, int outChannels, int outHeight, int outWidth)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d_backward_bias", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: locally_connected_conv2d_backward_bias");

        using var _ = PushContext();
        uint gridX = (uint)((outChannels + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr gradBiasPtr = gradBias.Handle;

        void** args = stackalloc void*[6];
        args[0] = &gradOutputPtr;
        args[1] = &gradBiasPtr;
        args[2] = &batch;
        args[3] = &outChannels;
        args[4] = &outHeight;
        args[5] = &outWidth;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    #endregion

    #region Deformable Convolution Operations

    public unsafe void DeformableConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: deformable_conv2d");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * outChannels);

        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr offsetsPtr = offsets.Handle;
        IntPtr maskPtr = mask?.Handle ?? IntPtr.Zero;
        IntPtr outputPtr = output.Handle;
        int hasMask = mask is not null ? 1 : 0;

        void** args = stackalloc void*[23];
        args[0] = &inputPtr;
        args[1] = &weightsPtr;
        args[2] = &offsetsPtr;
        args[3] = &maskPtr;
        args[4] = &outputPtr;
        args[5] = &batch;
        args[6] = &inChannels;
        args[7] = &inHeight;
        args[8] = &inWidth;
        args[9] = &outChannels;
        args[10] = &outHeight;
        args[11] = &outWidth;
        args[12] = &kernelH;
        args[13] = &kernelW;
        args[14] = &strideH;
        args[15] = &strideW;
        args[16] = &padH;
        args[17] = &padW;
        args[18] = &dilationH;
        args[19] = &dilationW;
        args[20] = &groups;
        args[21] = &deformGroups;
        args[22] = &hasMask;
        LaunchKernel2D(cudaKernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_input", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: deformable_conv2d_backward_input");

        using var _ = PushContext();
        int totalInputSize = batch * inChannels * inHeight * inWidth;
        uint gridX = (uint)((totalInputSize + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr offsetsPtr = offsets.Handle;
        IntPtr maskPtr = mask?.Handle ?? IntPtr.Zero;
        IntPtr gradInputPtr = gradInput.Handle;
        int hasMask = mask is not null ? 1 : 0;

        void** args = stackalloc void*[23];
        args[0] = &gradOutputPtr;
        args[1] = &weightsPtr;
        args[2] = &offsetsPtr;
        args[3] = &maskPtr;
        args[4] = &gradInputPtr;
        args[5] = &batch;
        args[6] = &inChannels;
        args[7] = &inHeight;
        args[8] = &inWidth;
        args[9] = &outChannels;
        args[10] = &outHeight;
        args[11] = &outWidth;
        args[12] = &kernelH;
        args[13] = &kernelW;
        args[14] = &strideH;
        args[15] = &strideW;
        args[16] = &padH;
        args[17] = &padW;
        args[18] = &dilationH;
        args[19] = &dilationW;
        args[20] = &groups;
        args[21] = &deformGroups;
        args[22] = &hasMask;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void DeformableConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_weights", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: deformable_conv2d_backward_weights");

        using var _ = PushContext();
        int inChannelsPerGroup = inChannels / groups;
        int totalWeights = outChannels * inChannelsPerGroup * kernelH * kernelW;
        uint gridX = (uint)((totalWeights + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr offsetsPtr = offsets.Handle;
        IntPtr maskPtr = mask?.Handle ?? IntPtr.Zero;
        IntPtr gradWeightsPtr = gradWeights.Handle;
        int hasMask = mask is not null ? 1 : 0;

        void** args = stackalloc void*[23];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &offsetsPtr;
        args[3] = &maskPtr;
        args[4] = &gradWeightsPtr;
        args[5] = &batch;
        args[6] = &inChannels;
        args[7] = &inHeight;
        args[8] = &inWidth;
        args[9] = &outChannels;
        args[10] = &outHeight;
        args[11] = &outWidth;
        args[12] = &kernelH;
        args[13] = &kernelW;
        args[14] = &strideH;
        args[15] = &strideW;
        args[16] = &padH;
        args[17] = &padW;
        args[18] = &dilationH;
        args[19] = &dilationW;
        args[20] = &groups;
        args[21] = &deformGroups;
        args[22] = &hasMask;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void DeformableConv2DBackwardOffset(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffsets,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_offset", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: deformable_conv2d_backward_offset");

        using var _ = PushContext();
        const int blockSize = 16;
        int offsetChannels = 2 * kernelH * kernelW * deformGroups;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * offsetChannels);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr offsetsPtr = offsets.Handle;
        IntPtr maskPtr = mask?.Handle ?? IntPtr.Zero;
        IntPtr gradOffsetsPtr = gradOffsets.Handle;
        int hasMask = mask is not null ? 1 : 0;

        void** args = stackalloc void*[24];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &weightsPtr;
        args[3] = &offsetsPtr;
        args[4] = &maskPtr;
        args[5] = &gradOffsetsPtr;
        args[6] = &batch;
        args[7] = &inChannels;
        args[8] = &inHeight;
        args[9] = &inWidth;
        args[10] = &outChannels;
        args[11] = &outHeight;
        args[12] = &outWidth;
        args[13] = &kernelH;
        args[14] = &kernelW;
        args[15] = &strideH;
        args[16] = &strideW;
        args[17] = &padH;
        args[18] = &padW;
        args[19] = &dilationH;
        args[20] = &dilationW;
        args[21] = &groups;
        args[22] = &deformGroups;
        args[23] = &hasMask;
        LaunchKernel2D(cudaKernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void DeformableConv2DBackwardMask(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer gradMask,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_mask", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: deformable_conv2d_backward_mask");

        using var _ = PushContext();
        const int blockSize = 16;
        int maskChannels = kernelH * kernelW * deformGroups;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * maskChannels);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr offsetsPtr = offsets.Handle;
        IntPtr gradMaskPtr = gradMask.Handle;

        void** args = stackalloc void*[22];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &weightsPtr;
        args[3] = &offsetsPtr;
        args[4] = &gradMaskPtr;
        args[5] = &batch;
        args[6] = &inChannels;
        args[7] = &inHeight;
        args[8] = &inWidth;
        args[9] = &outChannels;
        args[10] = &outHeight;
        args[11] = &outWidth;
        args[12] = &kernelH;
        args[13] = &kernelW;
        args[14] = &strideH;
        args[15] = &strideW;
        args[16] = &padH;
        args[17] = &padW;
        args[18] = &dilationH;
        args[19] = &dilationW;
        args[20] = &groups;
        args[21] = &deformGroups;
        LaunchKernel2D(cudaKernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    #endregion

    #endregion


    #region Pooling Operations

    public unsafe void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (!_kernelCache.TryGetValue("maxpool2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: maxpool2d");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels);

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr indicesPtr = indices?.Handle ?? IntPtr.Zero;
        int saveIndices = indices is not null ? 1 : 0;
        void** args = stackalloc void*[16];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &indicesPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outHeight;
        args[8] = &outWidth;
        args[9] = &kernelH;
        args[10] = &kernelW;
        args[11] = &strideH;
        args[12] = &strideW;
        args[13] = &padH;
        args[14] = &padW;
        args[15] = &saveIndices;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void MaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (!_kernelCache.TryGetValue("maxpool2d_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: maxpool2d_backward");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels);

        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr indicesPtr = indices.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int outW = outWidth;
        void** args = stackalloc void*[8];
        args[0] = &gradOutPtr;
        args[1] = &indicesPtr;
        args[2] = &gradInPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outW;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        if (!_kernelCache.TryGetValue("avgpool2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: avgpool2d");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels);

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int countPad = countIncludePad ? 1 : 0;
        void** args = stackalloc void*[14];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &inHeight;
        args[5] = &inWidth;
        args[6] = &outHeight;
        args[7] = &outWidth;
        args[8] = &kernelH;
        args[9] = &kernelW;
        args[10] = &strideH;
        args[11] = &strideW;
        args[12] = &padH;
        args[13] = &padW;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        if (!_kernelCache.TryGetValue("avgpool2d_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: avgpool2d_backward");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((inWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((inHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels);

        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int countPad = countIncludePad ? 1 : 0;
        void** args = stackalloc void*[14];
        args[0] = &gradOutPtr;
        args[1] = &gradInPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &inHeight;
        args[5] = &inWidth;
        args[6] = &outHeight;
        args[7] = &outWidth;
        args[8] = &kernelH;
        args[9] = &kernelW;
        args[10] = &strideH;
        args[11] = &strideW;
        args[12] = &padH;
        args[13] = &padW;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_avgpool2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: global_avgpool2d");

        using var _ = PushContext();
        uint grid = (uint)((batch * channels + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &height;
        args[5] = &width;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_maxpool2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: global_maxpool2d");

        using var _ = PushContext();
        uint grid = (uint)((batch * channels + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &height;
        args[5] = &width;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        if (!_kernelCache.TryGetValue("adaptive_avgpool2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adaptive_avgpool2d");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels);

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[8];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &inHeight;
        args[5] = &inWidth;
        args[6] = &outHeight;
        args[7] = &outWidth;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void MaxPool3D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("maxpool3d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: maxpool3d");

        using var _ = PushContext();
        const int blockSize = 8;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels * outDepth);

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr indicesPtr = indices?.Handle ?? IntPtr.Zero;
        int saveIndices = indices is not null ? 1 : 0;

        void** args = stackalloc void*[18];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &indicesPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &inDepth;
        args[6] = &inHeight;
        args[7] = &inWidth;
        args[8] = &outDepth;
        args[9] = &outHeight;
        args[10] = &outWidth;
        args[11] = &kernelD;
        args[12] = &kernelH;
        args[13] = &kernelW;
        args[14] = &strideD;
        args[15] = &strideH;
        args[16] = &strideW;
        args[17] = &saveIndices;

        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth)
    {
        if (!_kernelCache.TryGetValue("maxpool3d_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: maxpool3d_backward");

        using var _ = PushContext();
        const int blockSize = 8;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels * outDepth);

        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr indicesPtr = indices.Handle;
        IntPtr gradInPtr = gradInput.Handle;

        void** args = stackalloc void*[11];
        args[0] = &gradOutPtr;
        args[1] = &indicesPtr;
        args[2] = &gradInPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &inDepth;
        args[6] = &inHeight;
        args[7] = &inWidth;
        args[8] = &outDepth;
        args[9] = &outHeight;
        args[10] = &outWidth;

        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void NearestNeighborUpsample3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        if (!_kernelCache.TryGetValue("nearest_upsample3d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: nearest_upsample3d");

        using var _ = PushContext();
        int outDepth = inDepth * scaleD;
        int outHeight = inHeight * scaleH;
        int outWidth = inWidth * scaleW;

        const int blockSize = 8;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels * outDepth);

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[10];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &inDepth;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &scaleD;
        args[8] = &scaleH;
        args[9] = &scaleW;

        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void NearestNeighborUpsample3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        if (!_kernelCache.TryGetValue("nearest_upsample3d_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: nearest_upsample3d_backward");

        using var _ = PushContext();
        int outDepth = inDepth * scaleD;
        int outHeight = inHeight * scaleH;
        int outWidth = inWidth * scaleW;

        const int blockSize = 8;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels * outDepth);

        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr gradInPtr = gradInput.Handle;

        void** args = stackalloc void*[10];
        args[0] = &gradOutPtr;
        args[1] = &gradInPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &inDepth;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &scaleD;
        args[8] = &scaleH;
        args[9] = &scaleW;

        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    #endregion

    #region Spatial Transformer Operations

    public unsafe void AffineGrid(IGpuBuffer theta, IGpuBuffer grid, int batch, int outputHeight, int outputWidth)
    {
        if (!_kernelCache.TryGetValue("affine_grid", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: affine_grid");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outputWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outputHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)batch;

        IntPtr thetaPtr = theta.Handle;
        IntPtr gridPtr = grid.Handle;
        void** args = stackalloc void*[5];
        args[0] = &thetaPtr;
        args[1] = &gridPtr;
        args[2] = &batch;
        args[3] = &outputHeight;
        args[4] = &outputWidth;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void GridSample(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        if (!_kernelCache.TryGetValue("grid_sample", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: grid_sample");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels);

        IntPtr inputPtr = input.Handle;
        IntPtr gridPtr = grid.Handle;
        IntPtr outputPtr = output.Handle;
        int alignCornersInt = alignCorners ? 1 : 0;

        void** args = stackalloc void*[12];
        args[0] = &inputPtr;
        args[1] = &gridPtr;
        args[2] = &outputPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outHeight;
        args[8] = &outWidth;
        args[9] = &paddingMode;
        args[10] = &alignCornersInt;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void GridSampleBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
        IGpuBuffer gradInput, IGpuBuffer gradGrid,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        if (!_kernelCache.TryGetValue("grid_sample_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: grid_sample_backward");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)batch;

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gridPtr = grid.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        IntPtr gradGridPtr = gradGrid.Handle;
        int alignCornersInt = alignCorners ? 1 : 0;

        void** args = stackalloc void*[14];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gridPtr;
        args[3] = &gradInputPtr;
        args[4] = &gradGridPtr;
        args[5] = &batch;
        args[6] = &channels;
        args[7] = &inHeight;
        args[8] = &inWidth;
        args[9] = &outHeight;
        args[10] = &outWidth;
        args[11] = &paddingMode;
        args[12] = &alignCornersInt;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    #endregion

    #region Normalization Operations

    public unsafe void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training)
    {
        if (!_kernelCache.TryGetValue("batchnorm_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: batchnorm_forward");

        using var _ = PushContext();
        uint gridX = (uint)channels;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr betaPtr = beta.Handle;
        IntPtr runMeanPtr = runningMean.Handle;
        IntPtr runVarPtr = runningVar.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        int trainingInt = training ? 1 : 0;
        void** args = stackalloc void*[14];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &gammaPtr;
        args[3] = &betaPtr;
        args[4] = &runMeanPtr;
        args[5] = &runVarPtr;
        args[6] = &saveMeanPtr;
        args[7] = &saveInvVarPtr;
        args[8] = &batch;
        args[9] = &channels;
        args[10] = &spatialSize;
        args[11] = &epsilon;
        args[12] = &momentum;
        args[13] = &trainingInt;
        LaunchKernel(kernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("batchnorm_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: batchnorm_backward");

        using var _ = PushContext();
        uint gridX = (uint)channels;
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        IntPtr gradGammaPtr = gradGamma.Handle;
        IntPtr gradBetaPtr = gradBeta.Handle;
        void** args = stackalloc void*[12];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gammaPtr;
        args[3] = &saveMeanPtr;
        args[4] = &saveInvVarPtr;
        args[5] = &gradInputPtr;
        args[6] = &gradGammaPtr;
        args[7] = &gradBetaPtr;
        args[8] = &batch;
        args[9] = &channels;
        args[10] = &spatialSize;
        args[11] = &epsilon;
        LaunchKernel(kernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("layernorm_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: layernorm_forward");

        using var _ = PushContext();
        uint gridX = (uint)batchSize;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr betaPtr = beta.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        void** args = stackalloc void*[9];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &gammaPtr;
        args[3] = &betaPtr;
        args[4] = &saveMeanPtr;
        args[5] = &saveInvVarPtr;
        args[6] = &batchSize;
        args[7] = &normalizedSize;
        args[8] = &epsilon;
        LaunchKernel(kernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("layernorm_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: layernorm_backward");

        using var _ = PushContext();
        uint gridX = (uint)batchSize;
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        IntPtr gradGammaPtr = gradGamma.Handle;
        IntPtr gradBetaPtr = gradBeta.Handle;
        void** args = stackalloc void*[11];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gammaPtr;
        args[3] = &saveMeanPtr;
        args[4] = &saveInvVarPtr;
        args[5] = &gradInputPtr;
        args[6] = &gradGammaPtr;
        args[7] = &gradBetaPtr;
        args[8] = &batchSize;
        args[9] = &normalizedSize;
        args[10] = &epsilon;
        LaunchKernel(kernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("groupnorm_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: groupnorm_forward");

        using var _ = PushContext();
        uint gridX = (uint)(batch * numGroups);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr betaPtr = beta.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        void** args = stackalloc void*[11];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &gammaPtr;
        args[3] = &betaPtr;
        args[4] = &saveMeanPtr;
        args[5] = &saveInvVarPtr;
        args[6] = &batch;
        args[7] = &numGroups;
        args[8] = &channels;
        args[9] = &spatialSize;
        args[10] = &epsilon;
        LaunchKernel(kernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("instancenorm_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: instancenorm_forward");

        using var _ = PushContext();
        uint gridX = (uint)(batch * channels);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr betaPtr = beta.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        void** args = stackalloc void*[10];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &gammaPtr;
        args[3] = &betaPtr;
        args[4] = &saveMeanPtr;
        args[5] = &saveInvVarPtr;
        args[6] = &batch;
        args[7] = &channels;
        args[8] = &spatialSize;
        args[9] = &epsilon;
        LaunchKernel(kernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void RmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("rmsnorm_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: rmsnorm_forward");

        using var _ = PushContext();
        uint gridX = (uint)batchSize;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr saveRmsPtr = saveRms.Handle;
        void** args = stackalloc void*[7];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &gammaPtr;
        args[3] = &saveRmsPtr;
        args[4] = &batchSize;
        args[5] = &normalizedSize;
        args[6] = &epsilon;
        LaunchKernel(kernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void RmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize, float epsilon)
    {
        // Compute gradInput using rmsnorm_backward kernel
        if (!_kernelCache.TryGetValue("rmsnorm_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: rmsnorm_backward");

        using var _ = PushContext();
        uint gridX = (uint)batchSize;
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr saveRmsPtr = saveRms.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        IntPtr gradGammaPtr = gradGamma.Handle;
        void** args = stackalloc void*[9];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gammaPtr;
        args[3] = &saveRmsPtr;
        args[4] = &gradInputPtr;
        args[5] = &gradGammaPtr;
        args[6] = &batchSize;
        args[7] = &normalizedSize;
        args[8] = &epsilon;
        LaunchKernel(kernel, gridX, DefaultBlockSize, args);

        // Compute gradGamma using rmsnorm_grad_gamma kernel
        if (!_kernelCache.TryGetValue("rmsnorm_grad_gamma", out var kernel2))
            throw new InvalidOperationException("CUDA kernel not found: rmsnorm_grad_gamma");

        uint gridGamma = (uint)((normalizedSize + DefaultBlockSize - 1) / DefaultBlockSize);
        void** args2 = stackalloc void*[6];
        args2[0] = &gradOutputPtr;
        args2[1] = &inputPtr;
        args2[2] = &saveRmsPtr;
        args2[3] = &gradGammaPtr;
        args2[4] = &batchSize;
        args2[5] = &normalizedSize;
        LaunchKernel(kernel2, gridGamma, DefaultBlockSize, args2);
    }

    #endregion


    #region Dropout Operations

    public unsafe void Dropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask, int size, float dropoutRate, ulong seed, bool training)
    {
        if (!_kernelCache.TryGetValue("dropout_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: dropout_forward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr maskPtr = mask.Handle;
        int trainingInt = training ? 1 : 0;
        void** args = stackalloc void*[6];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &maskPtr;
        args[3] = &size;
        args[4] = &dropoutRate;
        args[5] = &seed;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void DropoutBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size, float dropoutRate)
    {
        if (!_kernelCache.TryGetValue("dropout_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: dropout_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr maskPtr = mask.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        void** args = stackalloc void*[5];
        args[0] = &gradOutputPtr;
        args[1] = &maskPtr;
        args[2] = &gradInputPtr;
        args[3] = &size;
        args[4] = &dropoutRate;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion


    #region Embedding Operations

    public unsafe void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim)
    {
        if (!_kernelCache.TryGetValue("embedding_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: embedding_forward");

        using var _ = PushContext();
        uint grid = (uint)((numIndices + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr idxPtr = indices.Handle;
        IntPtr tablePtr = embeddingTable.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &idxPtr;
        args[1] = &tablePtr;
        args[2] = &outputPtr;
        args[3] = &numIndices;
        args[4] = &embeddingDim;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void EmbeddingBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize)
    {
        if (!_kernelCache.TryGetValue("embedding_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: embedding_backward");

        using var _ = PushContext();
        uint grid = (uint)((numIndices + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr idxPtr = indices.Handle;
        IntPtr gradEmbPtr = gradEmbedding.Handle;
        void** args = stackalloc void*[5];
        args[0] = &gradOutPtr;
        args[1] = &idxPtr;
        args[2] = &gradEmbPtr;
        args[3] = &numIndices;
        args[4] = &embeddingDim;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public IGpuBuffer AllocateIntBuffer(int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Buffer size must be positive.");

        using var _ = PushContext();
        ulong byteSize = (ulong)size * sizeof(int);
        CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAlloc(out IntPtr devicePtr, byteSize), "cuMemAlloc(int)");
        CuBlasNative.CheckCudaResult(CuBlasNative.cuMemsetD32(devicePtr, 0, (ulong)size), "cuMemsetD32(int)");
        return new CudaGpuBuffer(_cudaContext, devicePtr, size);
    }

    public IGpuBuffer AllocateIntBuffer(int[] data)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        using var _ = PushContext();
        int size = data.Length;
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(data), "Buffer size must be positive.");
        ulong byteSize = (ulong)size * sizeof(int);

        CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAlloc(out IntPtr devicePtr, byteSize), "cuMemAlloc(int)");

        try
        {
            unsafe
            {
                fixed (int* src = data)
                {
                    CuBlasNative.CheckCudaResult(
                        CuBlasNative.cuMemcpyHtoD(devicePtr, (IntPtr)src, byteSize),
                        "cuMemcpyHtoD(int)");
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

    #endregion


    #region Attention Operations

    public void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        // Attention: softmax(Q * K^T / sqrt(d_k)) * V
        using var _ = PushContext();
        int batchHeads = batch * numHeads;
        int qkSize = seqLen * seqLen;

        // Allocate temporary buffers
        using var scores = AllocateBuffer(batchHeads * qkSize);
        using var keyTransposed = AllocateBuffer(batchHeads * seqLen * headDim);

        // Transpose K: [batch*heads, seqLen, headDim] -> [batch*heads, headDim, seqLen]
        BatchedTranspose(key, keyTransposed, batchHeads, seqLen, headDim);

        // Q * K^T: [batch*heads, seqLen, headDim] x [batch*heads, headDim, seqLen] -> [batch*heads, seqLen, seqLen]
        BatchedGemm(query, keyTransposed, scores, seqLen, seqLen, headDim, batchHeads);

        // Scale by 1/sqrt(d_k)
        Scale(scores, scores, scale, batchHeads * qkSize);

        // Apply causal mask if needed
        if (isCausal && mask is not null)
        {
            Add(scores, mask, scores, batchHeads * qkSize);
        }

        // Softmax along the last dimension
        Softmax(scores, scores, batchHeads * seqLen, seqLen);

        // Copy attention weights if requested
        if (attentionWeights is not null)
        {
            Copy(scores, attentionWeights, batchHeads * qkSize);
        }

        // Multiply by V: [batch*heads, seqLen, seqLen] x [batch*heads, seqLen, headDim] -> [batch*heads, seqLen, headDim]
        BatchedGemm(scores, value, output, seqLen, headDim, seqLen, batchHeads);
    }

    public void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        using var _ = PushContext();
        int batchHeads = batch * numHeads;
        int qkSize = seqLen * seqLen;

        // Allocate temporary buffers
        using var gradScores = AllocateBuffer(batchHeads * qkSize);
        using var tempScores = AllocateBuffer(batchHeads * qkSize);
        using var attnTransposed = AllocateBuffer(batchHeads * qkSize);
        using var valueTransposed = AllocateBuffer(batchHeads * seqLen * headDim);
        using var gradScoresTransposed = AllocateBuffer(batchHeads * qkSize);

        // Transpose attention weights: [batch*heads, seqLen, seqLen]
        BatchedTranspose(attentionWeights, attnTransposed, batchHeads, seqLen, seqLen);

        // gradValue = attention_weights^T * gradOutput
        BatchedGemm(attnTransposed, gradOutput, gradValue, seqLen, headDim, seqLen, batchHeads);

        // Transpose V: [batch*heads, seqLen, headDim] -> [batch*heads, headDim, seqLen]
        BatchedTranspose(value, valueTransposed, batchHeads, seqLen, headDim);

        // gradScores = gradOutput * V^T
        BatchedGemm(gradOutput, valueTransposed, gradScores, seqLen, seqLen, headDim, batchHeads);

        // Softmax backward
        SoftmaxBackward(gradScores, attentionWeights, tempScores, batchHeads * seqLen, seqLen);

        // Scale
        Scale(tempScores, gradScores, scale, batchHeads * qkSize);

        // gradQuery = gradScores * K
        BatchedGemm(gradScores, key, gradQuery, seqLen, headDim, seqLen, batchHeads);

        // Transpose gradScores for gradKey computation
        BatchedTranspose(gradScores, gradScoresTransposed, batchHeads, seqLen, seqLen);

        // gradKey = gradScores^T * Q
        BatchedGemm(gradScoresTransposed, query, gradKey, seqLen, headDim, seqLen, batchHeads);
    }

    public void FlashAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? mask, int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        // Allocate temporary buffer for softmax stats (not returned but required by FlashAttentionV2)
        using var softmaxStats = AllocateBuffer(batch * numHeads * seqLen);

        // Use FlashAttentionV2 which is the proper GPU-accelerated implementation
        FlashAttentionV2(query, key, value, output, softmaxStats, batch, numHeads, seqLen, seqLen, headDim, scale, isCausal);
    }

    public unsafe void FlashAttentionV2(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        using var _ = PushContext();
        var kernel = _kernelCache["flash_attention_v2"];
        uint gridX = (uint)((seqQ + 31) / 32);
        uint gridY = (uint)(batch * numHeads);
        int causalFlag = isCausal ? 1 : 0;

        void** args = stackalloc void*[12];
        IntPtr qPtr = query.Handle;
        IntPtr kPtr = key.Handle;
        IntPtr vPtr = value.Handle;
        IntPtr oPtr = output.Handle;
        IntPtr sPtr = softmaxStats.Handle;
        args[0] = &qPtr;
        args[1] = &kPtr;
        args[2] = &vPtr;
        args[3] = &oPtr;
        args[4] = &sPtr;
        args[5] = &batch;
        args[6] = &numHeads;
        args[7] = &seqQ;
        args[8] = &seqK;
        args[9] = &headDim;
        args[10] = &scale;
        args[11] = &causalFlag;

        LaunchKernel2D(kernel, gridX, gridY, 1, 32, 1, args);
    }

    public unsafe void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        using var _ = PushContext();
        var kernel = _kernelCache["flash_attention_backward"];
        uint gridX = (uint)((seqQ + 63) / 64);
        uint gridY = (uint)(batch * numHeads);
        int causalFlag = isCausal ? 1 : 0;

        void** args = stackalloc void*[15];
        IntPtr goPtr = gradOutput.Handle;
        IntPtr qPtr = query.Handle;
        IntPtr kPtr = key.Handle;
        IntPtr vPtr = value.Handle;
        IntPtr oPtr = output.Handle;
        IntPtr sPtr = softmaxStats.Handle;
        IntPtr gqPtr = gradQuery.Handle;
        IntPtr gkPtr = gradKey.Handle;
        IntPtr gvPtr = gradValue.Handle;
        args[0] = &goPtr;
        args[1] = &qPtr;
        args[2] = &kPtr;
        args[3] = &vPtr;
        args[4] = &oPtr;
        args[5] = &sPtr;
        args[6] = &gqPtr;
        args[7] = &gkPtr;
        args[8] = &gvPtr;
        args[9] = &batch;
        args[10] = &numHeads;
        args[11] = &seqQ;
        args[12] = &seqK;
        args[13] = &headDim;
        args[14] = &scale;

        // Note: args[15] would be causalFlag but signature in kernel expects 15 args
        // Need to adjust - let me fix kernel arg count
        void** args2 = stackalloc void*[16];
        for (int i = 0; i < 15; i++) args2[i] = args[i];
        args2[15] = &causalFlag;

        LaunchKernel2D(kernel, gridX, gridY, 1, 64, 1, args2);
    }

    public unsafe void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        using var _ = PushContext();
        var kernel = _kernelCache["grouped_query_attention"];
        uint gridX = (uint)((seqQ + 31) / 32);
        uint gridY = (uint)(batch * numQHeads);
        int queriesPerKV = numQHeads / numKVHeads;
        int causalFlag = isCausal ? 1 : 0;
        int storeWeights = attentionWeights != null ? 1 : 0;
        IntPtr wPtr = attentionWeights?.Handle ?? IntPtr.Zero;

        void** args = stackalloc void*[14];
        IntPtr qPtr = query.Handle;
        IntPtr kPtr = key.Handle;
        IntPtr vPtr = value.Handle;
        IntPtr oPtr = output.Handle;
        args[0] = &qPtr;
        args[1] = &kPtr;
        args[2] = &vPtr;
        args[3] = &oPtr;
        args[4] = &wPtr;
        args[5] = &batch;
        args[6] = &numQHeads;
        args[7] = &numKVHeads;
        args[8] = &queriesPerKV;
        args[9] = &seqQ;
        args[10] = &seqK;
        args[11] = &headDim;
        args[12] = &scale;
        args[13] = &causalFlag;

        void** args2 = stackalloc void*[15];
        for (int i = 0; i < 14; i++) args2[i] = args[i];
        args2[14] = &storeWeights;

        LaunchKernel2D(kernel, gridX, gridY, 1, 32, 1, args2);
    }

    public unsafe void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale)
    {
        using var _ = PushContext();
        var kernel = _kernelCache["grouped_query_attention_backward"];
        uint gridX = (uint)((seqQ + 31) / 32);
        uint gridY = (uint)(batch * numQHeads);
        int queriesPerKV = numQHeads / numKVHeads;

        void** args = stackalloc void*[16];
        IntPtr goPtr = gradOutput.Handle;
        IntPtr qPtr = query.Handle;
        IntPtr kPtr = key.Handle;
        IntPtr vPtr = value.Handle;
        IntPtr wPtr = attentionWeights.Handle;
        IntPtr gqPtr = gradQuery.Handle;
        IntPtr gkPtr = gradKey.Handle;
        IntPtr gvPtr = gradValue.Handle;
        args[0] = &goPtr;
        args[1] = &qPtr;
        args[2] = &kPtr;
        args[3] = &vPtr;
        args[4] = &wPtr;
        args[5] = &gqPtr;
        args[6] = &gkPtr;
        args[7] = &gvPtr;
        args[8] = &batch;
        args[9] = &numQHeads;
        args[10] = &numKVHeads;
        args[11] = &queriesPerKV;
        args[12] = &seqQ;
        args[13] = &seqK;
        args[14] = &headDim;
        args[15] = &scale;

        LaunchKernel2D(kernel, gridX, gridY, 1, 32, 1, args);
    }

    #endregion


    #region Transpose and Reshape Operations

    public unsafe void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols)
    {
        if (!_kernelCache.TryGetValue("transpose_2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: transpose_2d");

        using var _ = PushContext();
        int total = rows * cols;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &rows;
        args[3] = &cols;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols)
    {
        if (!_kernelCache.TryGetValue("batched_transpose", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: batched_transpose");

        using var _ = PushContext();
        int total = batch * rows * cols;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &batch;
        args[3] = &rows;
        args[4] = &cols;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation)
    {
        // Handle common fast paths first
        if (shape.Length == 2 && permutation.Length == 2 && permutation[0] == 1 && permutation[1] == 0)
        {
            Transpose(input, output, shape[0], shape[1]);
            return;
        }

        if (shape.Length == 3 && permutation.Length == 3)
        {
            int batch = shape[0];
            int rows = shape[1];
            int cols = shape[2];

            // Handle (0, 2, 1) - transpose last two dimensions
            if (permutation[0] == 0 && permutation[1] == 2 && permutation[2] == 1)
            {
                BatchedTranspose(input, output, batch, rows, cols);
                return;
            }
        }

        // General permute using dedicated kernel
        if (!_kernelCache.TryGetValue("permute_general", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: permute_general");

        using var _ = PushContext();

        int ndims = shape.Length;
        int totalSize = 1;
        for (int i = 0; i < ndims; i++)
            totalSize *= shape[i];

        // Compute input strides (row-major)
        int[] inputStrides = new int[ndims];
        inputStrides[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
            inputStrides[i] = inputStrides[i + 1] * shape[i + 1];

        // Compute output shape and strides after permutation
        int[] outputShape = new int[ndims];
        for (int i = 0; i < ndims; i++)
            outputShape[i] = shape[permutation[i]];

        int[] outputStrides = new int[ndims];
        outputStrides[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1];

        // Allocate device memory for strides and permutation
        using var inputStridesBuffer = AllocateIntBuffer(inputStrides);
        using var outputStridesBuffer = AllocateIntBuffer(outputStrides);
        using var permutationBuffer = AllocateIntBuffer(permutation);

        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr inputStridesPtr = inputStridesBuffer.Handle;
        IntPtr outputStridesPtr = outputStridesBuffer.Handle;
        IntPtr permutationPtr = permutationBuffer.Handle;

        void** args = stackalloc void*[7];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &inputStridesPtr;
        args[3] = &outputStridesPtr;
        args[4] = &permutationPtr;
        args[5] = &ndims;
        args[6] = &totalSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public void Copy(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        using var _ = PushContext();
        ulong byteSize = (ulong)size * sizeof(float);
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuMemcpyDtoD(destination.Handle, source.Handle, byteSize),
            "cuMemcpyDtoD");
    }

    public void Fill(IGpuBuffer buffer, float value, int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        using var _ = PushContext();
        // cuMemsetD32 sets 32-bit values (net471 compatible conversion)
        byte[] bytes = BitConverter.GetBytes(value);
        uint bits = BitConverter.ToUInt32(bytes, 0);
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuMemsetD32(buffer.Handle, bits, (ulong)size),
            "cuMemsetD32");
    }

    /// <inheritdoc/>
    public unsafe void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows, int srcCols, int destTotalCols, int destColOffset)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("copy_2d_strided", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: copy_2d_strided");

        using var _ = PushContext();

        // Launch configuration: srcCols x numRows grid
        uint gridX = (uint)((srcCols + DefaultBlockSize - 1) / DefaultBlockSize);
        uint gridY = (uint)numRows;

        IntPtr srcPtr = source.Handle;
        IntPtr dstPtr = destination.Handle;

        void** args = stackalloc void*[6];
        args[0] = &srcPtr;
        args[1] = &dstPtr;
        args[2] = &numRows;
        args[3] = &srcCols;
        args[4] = &destTotalCols;
        args[5] = &destColOffset;

        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, gridY, 1,
                (uint)DefaultBlockSize, 1, 1,
                0,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel (copy_2d_strided)");
    }

    /// <inheritdoc/>
    public unsafe void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        // Check for the kernel, if not available fall back to CPU implementation via memcpy pattern
        if (!_kernelCache.TryGetValue("nearest_neighbor_upsample", out var kernel))
        {
            // Fallback: Download, upsample on CPU, upload
            NearestNeighborUpsampleFallback(input, output, batchChannels, height, width, scaleFactor);
            return;
        }

        using var _ = PushContext();

        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        int outputSize = batchChannels * outHeight * outWidth;

        // Launch configuration: output elements / block size
        uint grid = (uint)((outputSize + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr srcPtr = input.Handle;
        IntPtr dstPtr = output.Handle;

        void** args = stackalloc void*[7];
        args[0] = &srcPtr;
        args[1] = &dstPtr;
        args[2] = &batchChannels;
        args[3] = &height;
        args[4] = &width;
        args[5] = &scaleFactor;
        args[6] = &outputSize;

        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                grid, 1, 1,
                (uint)DefaultBlockSize, 1, 1,
                0,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel (nearest_neighbor_upsample)");
    }

    /// <summary>
    /// CPU fallback for nearest-neighbor upsampling when kernel is not available.
    /// </summary>
    private unsafe void NearestNeighborUpsampleFallback(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        int inputSize = batchChannels * height * width;
        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        int outputSize = batchChannels * outHeight * outWidth;

        // Download input using existing method
        var inputData = new float[inputSize];
        DownloadBuffer(input, inputData);

        // Perform CPU upsampling
        var outputData = new float[outputSize];
        for (int bc = 0; bc < batchChannels; bc++)
        {
            for (int oh = 0; oh < outHeight; oh++)
            {
                for (int ow = 0; ow < outWidth; ow++)
                {
                    int ih = oh / scaleFactor;
                    int iw = ow / scaleFactor;
                    int inputIdx = bc * height * width + ih * width + iw;
                    int outputIdx = bc * outHeight * outWidth + oh * outWidth + ow;
                    outputData[outputIdx] = inputData[inputIdx];
                }
            }
        }

        // Upload output using CUDA memory copy
        using var _ = PushContext();
        ulong byteSize = (ulong)outputSize * sizeof(float);
        fixed (float* src = outputData)
        {
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuMemcpyHtoD(output.Handle, (IntPtr)src, byteSize),
                "cuMemcpyHtoD (upsample fallback)");
        }
    }

    #endregion


    #region Activation Gradient Operations

    public unsafe void ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("relu_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: relu_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("sigmoid_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sigmoid_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr outPtr = output.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &outPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void TanhBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("tanh_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: tanh_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr outPtr = output.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &outPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("gelu_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: gelu_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
    {
        if (!_kernelCache.TryGetValue("softmax_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: softmax_backward");

        using var _ = PushContext();
        uint grid = (uint)batchSize;
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr outPtr = output.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int batch = batchSize;
        int feat = features;
        void** args = stackalloc void*[5];
        args[0] = &gradOutPtr;
        args[1] = &outPtr;
        args[2] = &gradInPtr;
        args[3] = &batch;
        args[4] = &feat;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("leaky_relu", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: leaky_relu");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = A.Handle;
        IntPtr outputPtr = B.Handle;
        float alphaVal = alpha;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &alphaVal;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void LeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("leaky_relu_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: leaky_relu_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        float alphaVal = alpha;
        int n = size;
        void** args = stackalloc void*[5];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &alphaVal;
        args[4] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Elu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("elu", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: elu");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = A.Handle;
        IntPtr outputPtr = B.Handle;
        float alphaVal = alpha;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &alphaVal;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void EluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer output, IGpuBuffer gradInput, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("elu_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: elu_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr outPtr = output.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        float alphaVal = alpha;
        int n = size;
        void** args = stackalloc void*[6];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &outPtr;
        args[3] = &gradInPtr;
        args[4] = &alphaVal;
        args[5] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public void Swish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("swish", A, B, size);
    }

    public unsafe void SwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("swish_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: swish_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public void Silu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("silu", A, B, size);
    }

    public void Mish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("mish", A, B, size);
    }

    public void Softplus(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("softplus", A, B, size);
    }

    public void Hardswish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("hardswish", A, B, size);
    }

    public unsafe void Selu(IGpuBuffer A, IGpuBuffer B, float alpha, float scale, int size)
    {
        if (!_kernelCache.TryGetValue("selu", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: selu");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &alpha;
        args[3] = &scale;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public void Hardsigmoid(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("hardsigmoid", A, B, size);
    }

    public unsafe void Hardtanh(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size)
    {
        if (!_kernelCache.TryGetValue("hardtanh", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hardtanh");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &minVal;
        args[3] = &maxVal;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion



    #region Loss Function Operations

    public unsafe float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses)
    {
        if (!_kernelCache.TryGetValue("cross_entropy_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: cross_entropy_loss");

        using var _ = PushContext();
        using var temp = AllocateBuffer(batchSize);
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr outputPtr = temp.Handle;
        void** args = stackalloc void*[5];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &outputPtr;
        args[3] = &batchSize;
        args[4] = &numClasses;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(temp, batchSize) / batchSize;
    }

    public unsafe void CrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int batchSize, int numClasses)
    {
        if (!_kernelCache.TryGetValue("cross_entropy_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: cross_entropy_backward");

        using var _ = PushContext();
        uint grid = (uint)((batchSize * numClasses + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        int totalSize = batchSize * numClasses;
        void** args = stackalloc void*[5];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &gradPtr;
        args[3] = &batchSize;
        args[4] = &numClasses;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float BinaryCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (!_kernelCache.TryGetValue("bce_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bce_loss");

        using var _ = PushContext();
        using var temp = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr outputPtr = temp.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &outputPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(temp, size) / size;
    }

    public unsafe void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("bce_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bce_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &gradPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (!_kernelCache.TryGetValue("mse_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mse_loss");

        using var _ = PushContext();
        using var temp = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr outputPtr = temp.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &outputPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(temp, size) / size;
    }

    public unsafe void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("mse_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mse_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &gradPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta)
    {
        if (!_kernelCache.TryGetValue("smooth_l1_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: smooth_l1_loss");

        using var _ = PushContext();
        using var temp = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr outputPtr = temp.Handle;
        void** args = stackalloc void*[5];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &outputPtr;
        args[3] = &size;
        args[4] = &beta;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(temp, size) / size;
    }

    public unsafe void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta)
    {
        if (!_kernelCache.TryGetValue("smooth_l1_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: smooth_l1_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[5];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &gradPtr;
        args[3] = &size;
        args[4] = &beta;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion


    #region Utility Operations

    public unsafe void Clamp(IGpuBuffer A, IGpuBuffer B, float min, float max, int size)
    {
        if (!_kernelCache.TryGetValue("clamp", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: clamp");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = A.Handle;
        IntPtr outputPtr = B.Handle;
        void** args = stackalloc void*[5];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &min;
        args[3] = &max;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float L2Norm(IGpuBuffer A, int size)
    {
        if (!_kernelCache.TryGetValue("l2_norm_squared", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: l2_norm_squared");

        using var _ = PushContext();
        using var temp = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = A.Handle;
        IntPtr outputPtr = temp.Handle;
        void** args = stackalloc void*[3];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        float sumSquared = Sum(temp, size);
        return (float)Math.Sqrt(sumSquared);
    }

    public void ClipByValue(IGpuBuffer A, IGpuBuffer B, float clipValue, int size)
    {
        Clamp(A, B, -clipValue, clipValue, size);
    }

    public void ClipByNorm(IGpuBuffer A, IGpuBuffer B, float maxNorm, int size)
    {
        float norm = L2Norm(A, size);
        if (norm > maxNorm)
        {
            float scale = maxNorm / norm;
            Scale(A, B, scale, size);
        }
        else
        {
            Copy(A, B, size);
        }
    }

    public unsafe void Fma(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int size)
    {
        // D = A * B + C
        Multiply(A, B, D, size);
        Add(D, C, D, size);
    }

    public unsafe void ScatterAdd(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer destination, int sourceSize, int destSize)
    {
        if (!_kernelCache.TryGetValue("embedding_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: embedding_backward");

        // ScatterAdd is essentially the embedding backward operation
        using var _ = PushContext();
        uint grid = (uint)((sourceSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr srcPtr = source.Handle;
        IntPtr idxPtr = indices.Handle;
        IntPtr dstPtr = destination.Handle;
        int embDim = 1;
        void** args = stackalloc void*[5];
        args[0] = &srcPtr;
        args[1] = &idxPtr;
        args[2] = &dstPtr;
        args[3] = &sourceSize;
        args[4] = &embDim;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource,
        int numIndices, int featureSize)
    {
        // ScatterAddBackward is essentially a Gather operation
        if (!_kernelCache.TryGetValue("embedding_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: embedding_forward");

        using var _ = PushContext();
        uint grid = (uint)((numIndices + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr idxPtr = indices.Handle;
        IntPtr gradDstPtr = gradDestination.Handle;
        IntPtr gradSrcPtr = gradSource.Handle;
        void** args = stackalloc void*[5];
        args[0] = &idxPtr;
        args[1] = &gradDstPtr;
        args[2] = &gradSrcPtr;
        args[3] = &numIndices;
        args[4] = &featureSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize)
    {
        if (!_kernelCache.TryGetValue("embedding_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: embedding_forward");

        using var _ = PushContext();
        uint grid = (uint)((numIndices + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr idxPtr = indices.Handle;
        IntPtr srcPtr = source.Handle;
        IntPtr outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &idxPtr;
        args[1] = &srcPtr;
        args[2] = &outPtr;
        args[3] = &numIndices;
        args[4] = &featureSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion


    #region Comparison Operations

    public unsafe void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("greater_than", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: greater_than");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("less_than", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: less_than");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("equal", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: equal");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("where_select", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: where_select");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr condPtr = condition.Handle;
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[5];
        args[0] = &condPtr;
        args[1] = &aPtr;
        args[2] = &bPtr;
        args[3] = &cPtr;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size)
    {
        if (!_kernelCache.TryGetValue("not_equal_scalar", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: not_equal_scalar");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &cPtr;
        args[2] = &scalar;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion


    #region Statistics Operations

    public unsafe void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("mean_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mean_axis");

        using var _ = PushContext();
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &outerSize;
        args[3] = &reduceSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("var_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: var_axis");

        using var _ = PushContext();
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr meanPtr = mean.Handle;
        IntPtr varPtr = variance.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &meanPtr;
        args[2] = &varPtr;
        args[3] = &outerSize;
        args[4] = &reduceSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("argmax", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: argmax");

        using var _ = PushContext();
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr idxPtr = indices.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &idxPtr;
        args[2] = &outerSize;
        args[3] = &reduceSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("argmin", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: argmin");

        using var _ = PushContext();
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr idxPtr = indices.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &idxPtr;
        args[2] = &outerSize;
        args[3] = &reduceSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void MaxAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("max_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: max_axis");

        using var _ = PushContext();
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &outerSize;
        args[3] = &reduceSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void TopK(IGpuBuffer A, IGpuBuffer values, IGpuBuffer indices, int outerSize, int reduceSize, int k, bool sorted = true)
    {
        // Use optimized small-k kernel for k <= 8
        string kernelName = k <= 8 ? "topk_small" : "topk";
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");

        using var _ = PushContext();
        // One block per row
        uint gridX = (uint)outerSize;
        IntPtr aPtr = A.Handle;
        IntPtr valPtr = values.Handle;
        IntPtr idxPtr = indices.Handle;

        if (k <= 8)
        {
            // topk_small kernel: input, values, indices, outerSize, reduceSize, k
            int paramK = k;
            void** args = stackalloc void*[6];
            args[0] = &aPtr;
            args[1] = &valPtr;
            args[2] = &idxPtr;
            args[3] = &outerSize;
            args[4] = &reduceSize;
            args[5] = &paramK;
            LaunchKernel(kernel, gridX, (uint)Math.Min(256, reduceSize), args);
        }
        else
        {
            // topk kernel needs shared memory for top-k values and indices
            int sortedInt = sorted ? 1 : 0;
            void** args = stackalloc void*[7];
            args[0] = &aPtr;
            args[1] = &valPtr;
            args[2] = &idxPtr;
            args[3] = &outerSize;
            args[4] = &reduceSize;
            args[5] = &k;
            args[6] = &sortedInt;
            uint sharedMem = (uint)(k * (sizeof(float) + sizeof(int)));
            LaunchKernelWithSharedMem(kernel, gridX, (uint)Math.Min(256, reduceSize), sharedMem, args);
        }
    }

    public unsafe void BroadcastMultiplyLastAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        if (!_kernelCache.TryGetValue("broadcast_multiply_last_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: broadcast_multiply_last_axis");

        using var _ = PushContext();
        int totalSize = outerSize * innerSize;
        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &outerSize;
        args[4] = &innerSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void BroadcastMultiplyFirstAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        if (!_kernelCache.TryGetValue("broadcast_multiply_first_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: broadcast_multiply_first_axis");

        using var _ = PushContext();
        int totalSize = outerSize * innerSize;
        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &outerSize;
        args[4] = &innerSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion


    #region Optimizer Operations

    public unsafe void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("sgd_momentum_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sgd_momentum_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr velPtr = velocity.Handle;
        void** args = stackalloc void*[7];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &velPtr;
        args[3] = &learningRate;
        args[4] = &momentum;
        args[5] = &weightDecay;
        args[6] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("adam_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adam_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        IntPtr vPtr = v.Handle;
        void** args = stackalloc void*[11];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &vPtr;
        args[4] = &learningRate;
        args[5] = &beta1;
        args[6] = &beta2;
        args[7] = &epsilon;
        args[8] = &weightDecay;
        args[9] = &step;
        args[10] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("adamw_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adamw_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        IntPtr vPtr = v.Handle;
        void** args = stackalloc void*[11];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &vPtr;
        args[4] = &learningRate;
        args[5] = &beta1;
        args[6] = &beta2;
        args[7] = &epsilon;
        args[8] = &weightDecay;
        args[9] = &step;
        args[10] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void RmspropUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer squaredAvg,
        float learningRate, float rho, float epsilon, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("rmsprop_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: rmsprop_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr sqAvgPtr = squaredAvg.Handle;
        void** args = stackalloc void*[8];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &sqAvgPtr;
        args[3] = &learningRate;
        args[4] = &rho;
        args[5] = &epsilon;
        args[6] = &weightDecay;
        args[7] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void AdagradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumulatedGrad,
        float learningRate, float epsilon, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("adagrad_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adagrad_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr accumPtr = accumulatedGrad.Handle;
        void** args = stackalloc void*[7];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &accumPtr;
        args[3] = &learningRate;
        args[4] = &epsilon;
        args[5] = &weightDecay;
        args[6] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void NagUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("nag_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: nag_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr velPtr = velocity.Handle;
        void** args = stackalloc void*[7];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &velPtr;
        args[3] = &learningRate;
        args[4] = &momentum;
        args[5] = &weightDecay;
        args[6] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void LarsUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
    {
        if (!_kernelCache.TryGetValue("lars_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: lars_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr velPtr = velocity.Handle;
        void** args = stackalloc void*[8];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &velPtr;
        args[3] = &learningRate;
        args[4] = &momentum;
        args[5] = &weightDecay;
        args[6] = &trustCoeff;
        args[7] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("lamb_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: lamb_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        IntPtr vPtr = v.Handle;
        void** args = stackalloc void*[11];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &vPtr;
        args[4] = &learningRate;
        args[5] = &beta1;
        args[6] = &beta2;
        args[7] = &epsilon;
        args[8] = &weightDecay;
        args[9] = &step;
        args[10] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SgdUpdate(IGpuBuffer param, IGpuBuffer gradient,
        float learningRate, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("sgd_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sgd_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        void** args = stackalloc void*[5];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &learningRate;
        args[3] = &weightDecay;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void AdadeltaUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        float rho, float epsilon, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("adadelta_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adadelta_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr accumGradPtr = accumGrad.Handle;
        IntPtr accumUpdatePtr = accumUpdate.Handle;
        void** args = stackalloc void*[8];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &accumGradPtr;
        args[3] = &accumUpdatePtr;
        args[4] = &rho;
        args[5] = &epsilon;
        args[6] = &weightDecay;
        args[7] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void AmsgradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("amsgrad_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: amsgrad_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        IntPtr vPtr = v.Handle;
        IntPtr vMaxPtr = vMax.Handle;
        void** args = stackalloc void*[12];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &vPtr;
        args[4] = &vMaxPtr;
        args[5] = &learningRate;
        args[6] = &beta1;
        args[7] = &beta2;
        args[8] = &epsilon;
        args[9] = &weightDecay;
        args[10] = &step;
        args[11] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void AdamaxUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer u,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("adamax_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adamax_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        IntPtr uPtr = u.Handle;
        void** args = stackalloc void*[11];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &uPtr;
        args[4] = &learningRate;
        args[5] = &beta1;
        args[6] = &beta2;
        args[7] = &epsilon;
        args[8] = &weightDecay;
        args[9] = &step;
        args[10] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void LionUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m,
        float learningRate, float beta1, float beta2, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("lion_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: lion_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        void** args = stackalloc void*[8];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &learningRate;
        args[4] = &beta1;
        args[5] = &beta2;
        args[6] = &weightDecay;
        args[7] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void NadamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("nadam_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: nadam_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        IntPtr vPtr = v.Handle;
        void** args = stackalloc void*[11];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &vPtr;
        args[4] = &learningRate;
        args[5] = &beta1;
        args[6] = &beta2;
        args[7] = &epsilon;
        args[8] = &weightDecay;
        args[9] = &step;
        args[10] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void FtrlUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer z, IGpuBuffer n,
        float learningRate, float l1Reg, float l2Reg, float beta, int size)
    {
        if (!_kernelCache.TryGetValue("ftrl_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: ftrl_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr zPtr = z.Handle;
        IntPtr nPtr = n.Handle;
        void** args = stackalloc void*[9];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &zPtr;
        args[3] = &nPtr;
        args[4] = &learningRate;
        args[5] = &l1Reg;
        args[6] = &l2Reg;
        args[7] = &beta;
        args[8] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
    {
        // CUDA doesn't have a built-in conversion kernel in our current set
        // For now, do a simple copy. In production, this would use a proper FP16 conversion kernel.
        if (!_kernelCache.TryGetValue("convert_fp32_to_fp16", out var kernel))
        {
            // Fallback: just copy the data as-is
            Copy(input, 0, output, 0, size);
            return;
        }

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle;
        IntPtr outPtr = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &inPtr;
        args[1] = &outPtr;
        args[2] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
    {
        // CUDA doesn't have a built-in conversion kernel in our current set
        // For now, do a simple copy. In production, this would use a proper FP32 conversion kernel.
        if (!_kernelCache.TryGetValue("convert_fp16_to_fp32", out var kernel))
        {
            // Fallback: just copy the data as-is
            Copy(input, 0, output, 0, size);
            return;
        }

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle;
        IntPtr outPtr = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &inPtr;
        args[1] = &outPtr;
        args[2] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion

    #region FFT and Signal Processing

    /// <inheritdoc/>
    public unsafe void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        using var _ = PushContext();

        // Copy input to output for in-place FFT
        CudaCopyBuffer(inputReal, outputReal, n);
        CudaCopyBuffer(inputImag, outputImag, n);

        int log2n = (int)MathHelper.Log2(n);
        IntPtr outRealPtr = outputReal.Handle;
        IntPtr outImagPtr = outputImag.Handle;
        int inv = inverse ? 1 : 0;

        // Bit-reversal permutation
        if (_kernelCache.TryGetValue("bit_reverse_permutation", out var bitRevKernel))
        {
            uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
            void** args = stackalloc void*[4];
            args[0] = &outRealPtr;
            args[1] = &outImagPtr;
            args[2] = &n;
            args[3] = &log2n;
            LaunchKernel(bitRevKernel, grid, (uint)DefaultBlockSize, args);
        }

        // FFT butterfly stages
        if (_kernelCache.TryGetValue("fft_butterfly", out var butterflyKernel))
        {
            // Pre-allocate args outside the loop to avoid CA2014
            void** butterflyArgs = stackalloc void*[5];
            butterflyArgs[0] = &outRealPtr;
            butterflyArgs[1] = &outImagPtr;
            butterflyArgs[2] = &n;
            butterflyArgs[4] = &inv;

            for (int stride = 2; stride <= n; stride *= 2)
            {
                uint grid = (uint)((n / 2 + DefaultBlockSize - 1) / DefaultBlockSize);
                butterflyArgs[3] = &stride;
                LaunchKernel(butterflyKernel, grid, (uint)DefaultBlockSize, butterflyArgs);
            }
        }

        // Scale for inverse FFT
        if (inverse && _kernelCache.TryGetValue("scale_inverse", out var scaleKernel))
        {
            uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
            void** args = stackalloc void*[3];
            args[0] = &outRealPtr;
            args[1] = &outImagPtr;
            args[2] = &n;
            LaunchKernel(scaleKernel, grid, (uint)DefaultBlockSize, args);
        }
    }

    /// <inheritdoc/>
    public unsafe void RFFT(IGpuBuffer input, IGpuBuffer outputReal, IGpuBuffer outputImag, int n)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        using var _ = PushContext();

        // Allocate temporary buffers
        var tempReal = AllocateBuffer(n);
        var tempImag = AllocateBuffer(n);

        try
        {
            // Copy input to tempReal, zero tempImag
            CudaCopyBuffer(input, tempReal, n);
            CuBlasNative.CheckCudaResult(CuBlasNative.cuMemsetD32(tempImag.Handle, 0, (ulong)n), "cuMemsetD32");

            // Perform full complex FFT
            FFT(tempReal, tempImag, tempReal, tempImag, n, false);

            // Extract positive frequencies
            if (_kernelCache.TryGetValue("rfft_postprocess", out var rfftKernel))
            {
                int outLen = n / 2 + 1;
                uint grid = (uint)((outLen + DefaultBlockSize - 1) / DefaultBlockSize);
                IntPtr tempRealPtr = tempReal.Handle;
                IntPtr tempImagPtr = tempImag.Handle;
                IntPtr outRealPtr = outputReal.Handle;
                IntPtr outImagPtr = outputImag.Handle;
                void** args = stackalloc void*[5];
                args[0] = &tempRealPtr;
                args[1] = &tempImagPtr;
                args[2] = &outRealPtr;
                args[3] = &outImagPtr;
                args[4] = &n;
                LaunchKernel(rfftKernel, grid, (uint)DefaultBlockSize, args);
            }
        }
        finally
        {
            tempReal.Dispose();
            tempImag.Dispose();
        }
    }

    /// <inheritdoc/>
    public unsafe void IRFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer output, int n)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        using var _ = PushContext();

        var tempReal = AllocateBuffer(n);
        var tempImag = AllocateBuffer(n);

        try
        {
            // Reconstruct negative frequencies
            if (_kernelCache.TryGetValue("irfft_preprocess", out var irfftKernel))
            {
                uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
                IntPtr inRealPtr = inputReal.Handle;
                IntPtr inImagPtr = inputImag.Handle;
                IntPtr tempRealPtr = tempReal.Handle;
                IntPtr tempImagPtr = tempImag.Handle;
                void** args = stackalloc void*[5];
                args[0] = &inRealPtr;
                args[1] = &inImagPtr;
                args[2] = &tempRealPtr;
                args[3] = &tempImagPtr;
                args[4] = &n;
                LaunchKernel(irfftKernel, grid, (uint)DefaultBlockSize, args);
            }

            // Perform inverse FFT
            FFT(tempReal, tempImag, tempReal, tempImag, n, true);

            // Copy real part to output
            CudaCopyBuffer(tempReal, output, n);
        }
        finally
        {
            tempReal.Dispose();
            tempImag.Dispose();
        }
    }

    /// <inheritdoc/>
    public unsafe void BatchedFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int batch, int n, bool inverse)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        using var _ = PushContext();

        // Copy input to output for in-place FFT
        CudaCopyBuffer(inputReal, outputReal, batch * n);
        CudaCopyBuffer(inputImag, outputImag, batch * n);

        int log2n = (int)MathHelper.Log2(n);
        IntPtr outRealPtr = outputReal.Handle;
        IntPtr outImagPtr = outputImag.Handle;
        int inv = inverse ? 1 : 0;

        // Batched bit-reversal
        if (_kernelCache.TryGetValue("batched_bit_reverse", out var bitRevKernel))
        {
            void** args = stackalloc void*[5];
            args[0] = &outRealPtr;
            args[1] = &outImagPtr;
            args[2] = &batch;
            args[3] = &n;
            args[4] = &log2n;
            LaunchKernel2D(bitRevKernel, (uint)((n + 15) / 16), (uint)batch, 1, 16, 1, args);
        }

        // Batched FFT butterfly stages
        if (_kernelCache.TryGetValue("batched_fft_butterfly", out var butterflyKernel))
        {
            // Pre-allocate args outside the loop to avoid CA2014
            void** butterflyArgs = stackalloc void*[6];
            butterflyArgs[0] = &outRealPtr;
            butterflyArgs[1] = &outImagPtr;
            butterflyArgs[2] = &batch;
            butterflyArgs[3] = &n;
            butterflyArgs[5] = &inv;

            for (int stride = 2; stride <= n; stride *= 2)
            {
                butterflyArgs[4] = &stride;
                LaunchKernel2D(butterflyKernel, (uint)((n / 2 + 15) / 16), (uint)batch, 1, 16, 1, butterflyArgs);
            }
        }

        // Scale for inverse FFT
        if (inverse && _kernelCache.TryGetValue("scale_inverse", out var scaleKernel))
        {
            int total = batch * n;
            uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
            void** args = stackalloc void*[3];
            args[0] = &outRealPtr;
            args[1] = &outImagPtr;
            args[2] = &total;
            LaunchKernel(scaleKernel, grid, (uint)DefaultBlockSize, args);
        }
    }

    /// <inheritdoc/>
    public unsafe void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int height, int width, bool inverse)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        using var _ = PushContext();

        // Copy input to output for in-place FFT
        CudaCopyBuffer(inputReal, outputReal, height * width);
        CudaCopyBuffer(inputImag, outputImag, height * width);

        int log2Width = (int)MathHelper.Log2(width);
        int log2Height = (int)MathHelper.Log2(height);
        IntPtr outRealPtr = outputReal.Handle;
        IntPtr outImagPtr = outputImag.Handle;
        int inv = inverse ? 1 : 0;

        // Row-wise FFT (process each row as a separate FFT)
        if (_kernelCache.TryGetValue("fft_rows_butterfly", out var rowButterfly))
        {
            // Pre-allocate args outside the loop to avoid CA2014
            void** rowArgs = stackalloc void*[6];
            rowArgs[0] = &outRealPtr;
            rowArgs[1] = &outImagPtr;
            rowArgs[2] = &height;
            rowArgs[3] = &width;
            rowArgs[5] = &inv;

            for (int stride = 2; stride <= width; stride *= 2)
            {
                rowArgs[4] = &stride;
                LaunchKernel2D(rowButterfly, (uint)((width / 2 + 15) / 16), (uint)((height + 15) / 16), 1, 16, 16, rowArgs);
            }
        }

        // Column-wise FFT
        if (_kernelCache.TryGetValue("fft_cols_butterfly", out var colButterfly))
        {
            // Pre-allocate args outside the loop to avoid CA2014
            void** colArgs = stackalloc void*[6];
            colArgs[0] = &outRealPtr;
            colArgs[1] = &outImagPtr;
            colArgs[2] = &height;
            colArgs[3] = &width;
            colArgs[5] = &inv;

            for (int stride = 2; stride <= height; stride *= 2)
            {
                colArgs[4] = &stride;
                LaunchKernel2D(colButterfly, (uint)((height / 2 + 15) / 16), (uint)((width + 15) / 16), 1, 16, 16, colArgs);
            }
        }

        // Scale for inverse FFT
        if (inverse && _kernelCache.TryGetValue("scale_inverse", out var scaleKernel))
        {
            int total = height * width;
            uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
            void** args = stackalloc void*[3];
            args[0] = &outRealPtr;
            args[1] = &outImagPtr;
            args[2] = &total;
            LaunchKernel(scaleKernel, grid, (uint)DefaultBlockSize, args);
        }
    }

    /// <inheritdoc/>
    public unsafe void ApplyWindow(IGpuBuffer input, IGpuBuffer window, IGpuBuffer output, int n)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("apply_window", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: apply_window");

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle;
        IntPtr winPtr = window.Handle;
        IntPtr outPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &inPtr;
        args[1] = &winPtr;
        args[2] = &outPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("complex_magnitude", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: complex_magnitude");

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr realPtr = real.Handle;
        IntPtr imagPtr = imag.Handle;
        IntPtr magPtr = magnitude.Handle;
        void** args = stackalloc void*[4];
        args[0] = &realPtr;
        args[1] = &imagPtr;
        args[2] = &magPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("complex_phase", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: complex_phase");

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr realPtr = real.Handle;
        IntPtr imagPtr = imag.Handle;
        IntPtr phasePtr = phase.Handle;
        void** args = stackalloc void*[4];
        args[0] = &realPtr;
        args[1] = &imagPtr;
        args[2] = &phasePtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("polar_to_complex", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: polar_to_complex");

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr magPtr = magnitude.Handle;
        IntPtr phasePtr = phase.Handle;
        IntPtr realPtr = real.Handle;
        IntPtr imagPtr = imag.Handle;
        void** args = stackalloc void*[5];
        args[0] = &magPtr;
        args[1] = &phasePtr;
        args[2] = &realPtr;
        args[3] = &imagPtr;
        args[4] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec, int numFrames, int numFreqs, int nMels)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("apply_mel_filterbank", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: apply_mel_filterbank");

        using var _ = PushContext();
        IntPtr powerPtr = powerSpec.Handle;
        IntPtr fbPtr = filterbank.Handle;
        IntPtr melPtr = melSpec.Handle;
        void** args = stackalloc void*[6];
        args[0] = &powerPtr;
        args[1] = &fbPtr;
        args[2] = &melPtr;
        args[3] = &numFrames;
        args[4] = &numFreqs;
        args[5] = &nMels;
        LaunchKernel2D(kernel, (uint)((nMels + 31) / 32), (uint)numFrames, 1, 32, 1, args);
    }

    /// <inheritdoc/>
    public unsafe void PowerToDb(IGpuBuffer power, IGpuBuffer db, int n, float refValue, float minDb)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("power_to_db", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: power_to_db");

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr powerPtr = power.Handle;
        IntPtr dbPtr = db.Handle;
        void** args = stackalloc void*[5];
        args[0] = &powerPtr;
        args[1] = &dbPtr;
        args[2] = &n;
        args[3] = &refValue;
        args[4] = &minDb;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("db_to_power", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: db_to_power");

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr dbPtr = db.Handle;
        IntPtr powerPtr = power.Handle;
        void** args = stackalloc void*[4];
        args[0] = &dbPtr;
        args[1] = &powerPtr;
        args[2] = &n;
        args[3] = &refValue;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    private void CudaCopyBuffer(IGpuBuffer src, IGpuBuffer dst, int size)
    {
        ulong byteSize = (ulong)size * sizeof(float);
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuMemcpyDtoD(dst.Handle, src.Handle, byteSize),
            "cuMemcpyDtoD");
    }

    public void Copy(IGpuBuffer source, int sourceOffset, IGpuBuffer destination, int destinationOffset, int length)
    {
        throw new NotImplementedException("Strided copy not implemented for CUDA backend yet.");
    }

    public void ArgMaxAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        throw new NotImplementedException("ArgMaxAxis not implemented for CUDA backend yet.");
    }

    public void GenerateRandomUniform(IGpuBuffer output, int size, float min, float max, ulong seed)
    {
        throw new NotImplementedException("GenerateRandomUniform not implemented for CUDA backend yet.");
    }

    public void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
    {
        throw new NotImplementedException("GenerateRandomNormal not implemented for CUDA backend yet.");
    }

    public void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output, int batchSize, int numCenters, int inputDim)
    {
        throw new NotImplementedException("RbfForward not implemented for CUDA backend yet.");
    }

    public void StdpUpdate(IGpuBuffer weights, IGpuBuffer preTrace, IGpuBuffer postTrace, IGpuBuffer preSpike, IGpuBuffer postSpike,
        float ltpRate, float ltdRate, float homeostasisRate, float minWeight, float maxWeight, int numPre, int numPost)
    {
        throw new NotImplementedException("StdpUpdate not implemented for CUDA backend yet.");
    }

    public void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input, float decay, float threshold, int size)
    {
        throw new NotImplementedException("UpdateTraces not implemented for CUDA backend yet.");
    }

    #endregion


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

        if (_convolutionModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_convolutionModule);
            _convolutionModule = IntPtr.Zero;
        }

        if (_poolingModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_poolingModule);
            _poolingModule = IntPtr.Zero;
        }

        if (_normalizationModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_normalizationModule);
            _normalizationModule = IntPtr.Zero;
        }

        if (_neuralNetModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_neuralNetModule);
            _neuralNetModule = IntPtr.Zero;
        }

        if (_activationModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_activationModule);
            _activationModule = IntPtr.Zero;
        }

        if (_fftModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_fftModule);
            _fftModule = IntPtr.Zero;
        }

        if (_spatialTransformerModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_spatialTransformerModule);
            _spatialTransformerModule = IntPtr.Zero;
        }

        if (_locallyConnectedModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_locallyConnectedModule);
            _locallyConnectedModule = IntPtr.Zero;
        }

        if (_deformableConvModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_deformableConvModule);
            _deformableConvModule = IntPtr.Zero;
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

    internal sealed class CudaGpuByteBuffer : IGpuBuffer
    {
        private IntPtr _context;
        private IntPtr _devicePtr;

        public int Size { get; }
        public long SizeInBytes { get; }
        public IntPtr Handle => _devicePtr;

        public CudaGpuByteBuffer(IntPtr context, IntPtr devicePtr, int size)
        {
            _context = context;
            _devicePtr = devicePtr;
            Size = size;
            SizeInBytes = size;
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

        ~CudaGpuByteBuffer()
        {
            Dispose();
        }
    }
}
