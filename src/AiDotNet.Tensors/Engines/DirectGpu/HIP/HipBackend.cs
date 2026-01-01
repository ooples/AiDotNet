// Copyright (c) AiDotNet. All rights reserved.
// HIP backend for AMD GPU with real MFMA (Matrix Fused Multiply-Add) support.
// Target: 25,000+ GFLOPS on MI200, 15,000+ GFLOPS on RX 7900.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.Sparsity;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// HIP backend for AMD GPUs with real MFMA (Matrix Core) acceleration.
/// Automatically selects the optimal kernel based on GPU architecture:
/// - CDNA (MI100/200/300): Full MFMA with wave64
/// - RDNA3 (RX 7000): WMMA with wave32
/// - RDNA2 (RX 6000): Optimized scalar with wave32
/// </summary>
/// <remarks>
/// <para><b>Performance Targets:</b></para>
/// <list type="bullet">
/// <item>MI200: 25,000+ GFLOPS (full MFMA)</item>
/// <item>MI100: 20,000+ GFLOPS (MFMA)</item>
/// <item>RX 7900 XTX: 15,000+ GFLOPS (WMMA)</item>
/// <item>RX 6800 XT: 8,000+ GFLOPS (optimized scalar)</item>
/// </list>
/// </remarks>
public sealed class HipBackend : IDirectGpuBackend
{
    private IntPtr _stream;
    private IntPtr _mfmaModule;
    private IntPtr _mfmaGemmF32;
    private IntPtr _mfmaGemmF16;
    private IntPtr _scalarGemmF32;
    private IntPtr _rdnaGemmWave32;
    private readonly Dictionary<string, IntPtr> _kernelCache;
    private AmdGpuArchitecture _architecture;
    private bool _disposed;
    private HipDeviceProperties _deviceProps;

    public bool IsAvailable { get; }
    public string BackendName => $"HIP ({GetKernelTypeName()})";
    public string DeviceName { get; }

    private string GetKernelTypeName()
    {
        return _architecture switch
        {
            AmdGpuArchitecture.MI100 or AmdGpuArchitecture.MI200 or AmdGpuArchitecture.MI300 => "MFMA",
            AmdGpuArchitecture.RDNA3 => "RDNA3",
            AmdGpuArchitecture.RDNA2 => "RDNA2",
            AmdGpuArchitecture.RDNA => "RDNA",
            _ => "Scalar"
        };
    }
    public string DeviceVendor => "AMD";
    public int ComputeUnits { get; }
    public long GlobalMemoryBytes { get; }
    public long LocalMemoryBytes { get; }

    /// <summary>
    /// Gets the detected AMD GPU architecture.
    /// </summary>
    public AmdGpuArchitecture Architecture => _architecture;

    /// <summary>
    /// Gets whether HIP is available on this system.
    /// </summary>
    public static bool IsHipAvailable => HipNativeBindings.IsAvailable;

    public HipBackend()
    {
        _kernelCache = new Dictionary<string, IntPtr>();

        if (!HipNativeBindings.IsAvailable)
        {
            IsAvailable = false;
            DeviceName = "None";
            return;
        }

        try
        {
            // Initialize HIP and get device info
            var result = HipNativeBindings.hipSetDevice(0);
            if (result != HipError.Success)
            {
                IsAvailable = false;
                DeviceName = "None";
                return;
            }

            // Get device properties
            _deviceProps = new HipDeviceProperties();
            result = HipNativeBindings.hipGetDeviceProperties(ref _deviceProps, 0);
            if (result != HipError.Success)
            {
                IsAvailable = false;
                DeviceName = "None";
                return;
            }

            DeviceName = _deviceProps.Name?.Trim() ?? "Unknown AMD GPU";
            ComputeUnits = _deviceProps.MultiProcessorCount;
            GlobalMemoryBytes = (long)(ulong)_deviceProps.TotalGlobalMem;
            LocalMemoryBytes = (long)(ulong)_deviceProps.SharedMemPerBlock;

            // Detect architecture from GCN arch name
            _architecture = DetectArchitecture(_deviceProps.GcnArchName, 0);

            // Create compute stream
            result = HipNativeBindings.hipStreamCreate(ref _stream);
            if (result != HipError.Success)
            {
                IsAvailable = false;
                return;
            }

            // Compile MFMA kernels
            CompileKernels();

            IsAvailable = true;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"HipBackend initialization failed: {ex.Message}");
            IsAvailable = false;
            DeviceName = "None";
        }
    }

    private AmdGpuArchitecture DetectArchitecture(string gcnArchName, int gcnArch)
    {
        // Parse GCN architecture name (e.g., "gfx90a", "gfx1100", "gfx1030")
        if (string.IsNullOrEmpty(gcnArchName))
        {
            return gcnArch switch
            {
                >= 1100 => AmdGpuArchitecture.RDNA3,
                >= 1030 => AmdGpuArchitecture.RDNA2,
                >= 1010 => AmdGpuArchitecture.RDNA,
                >= 940 => AmdGpuArchitecture.MI300,
                >= 908 => AmdGpuArchitecture.MI100,
                _ => AmdGpuArchitecture.GCN
            };
        }

        string archLower = gcnArchName.ToLowerInvariant();
        if (archLower.Contains("gfx942") || archLower.Contains("gfx941") || archLower.Contains("gfx940"))
            return AmdGpuArchitecture.MI300;
        if (archLower.Contains("gfx90a"))
            return AmdGpuArchitecture.MI200;
        if (archLower.Contains("gfx908"))
            return AmdGpuArchitecture.MI100;
        if (archLower.Contains("gfx1100") || archLower.Contains("gfx1101") || archLower.Contains("gfx1102"))
            return AmdGpuArchitecture.RDNA3;
        if (archLower.Contains("gfx1030") || archLower.Contains("gfx1031") || archLower.Contains("gfx1032"))
            return AmdGpuArchitecture.RDNA2;
        if (archLower.Contains("gfx1010") || archLower.Contains("gfx1011") || archLower.Contains("gfx1012"))
            return AmdGpuArchitecture.RDNA;

        return AmdGpuArchitecture.GCN;
    }

    private void CompileKernels()
    {
        // Get kernel source
        string source = HipMfmaKernel.GetSource();
        string compileFlags = HipMfmaKernel.GetCompileFlags(_architecture);

        Console.WriteLine($"[HipBackend] Compiling kernels for {_architecture} with flags: {compileFlags}");

        try
        {
            // Compile using HIP RTC
            IntPtr prog = IntPtr.Zero;
            var rtcResult = HipNativeBindings.hiprtcCreateProgram(
                ref prog,
                source,
                "mfma_gemm",
                0,
                IntPtr.Zero,
                IntPtr.Zero);

            if (rtcResult != HipRtcResult.Success)
            {
                Console.WriteLine($"[HipBackend] Failed to create HIP RTC program: {rtcResult}");
                System.Diagnostics.Debug.WriteLine($"Failed to create HIP RTC program: {rtcResult}");
                // Fall back to OpenCL backend via flag
                return;
            }

            // Compile with architecture-specific flags
            var options = new List<string>(compileFlags.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                "-O3",
                "-ffast-math"
            };
            rtcResult = HipNativeBindings.hiprtcCompileProgram(prog, options.Count, options.ToArray());

            if (rtcResult != HipRtcResult.Success)
            {
                // Get compile log for debugging
                UIntPtr logSize = UIntPtr.Zero;
                HipNativeBindings.hiprtcGetProgramLogSize(prog, ref logSize);
                if ((ulong)logSize > 0)
                {
                    IntPtr logPtr = Marshal.AllocHGlobal((int)(ulong)logSize);
                    HipNativeBindings.hiprtcGetProgramLog(prog, logPtr);
                    string log = Marshal.PtrToStringAnsi(logPtr) ?? "";
                    Marshal.FreeHGlobal(logPtr);
                    Console.WriteLine($"[HipBackend] HIP RTC compile FAILED: {rtcResult}");
                    Console.WriteLine($"[HipBackend] Compile log: {log}");
                    System.Diagnostics.Debug.WriteLine($"HIP RTC compile log: {log}");
                }
                else
                {
                    Console.WriteLine($"[HipBackend] HIP RTC compile FAILED: {rtcResult} (no log)");
                }
                HipNativeBindings.hiprtcDestroyProgram(ref prog);
                return;
            }

            Console.WriteLine("[HipBackend] HIP RTC compile succeeded, getting code...");

            // Get compiled code
            UIntPtr codeSize = UIntPtr.Zero;
            rtcResult = HipNativeBindings.hiprtcGetCodeSize(prog, ref codeSize);
            if (rtcResult != HipRtcResult.Success || (ulong)codeSize == 0)
            {
                HipNativeBindings.hiprtcDestroyProgram(ref prog);
                return;
            }

            IntPtr code = Marshal.AllocHGlobal((int)(ulong)codeSize);
            rtcResult = HipNativeBindings.hiprtcGetCode(prog, code);
            HipNativeBindings.hiprtcDestroyProgram(ref prog);

            if (rtcResult != HipRtcResult.Success)
            {
                Marshal.FreeHGlobal(code);
                return;
            }

            // Load module from compiled code
            var hipResult = HipNativeBindings.hipModuleLoadData(ref _mfmaModule, code);
            Marshal.FreeHGlobal(code);

            if (hipResult != HipError.Success)
            {
                Console.WriteLine($"[HipBackend] Failed to load HIP module: {hipResult}");
                System.Diagnostics.Debug.WriteLine($"Failed to load HIP module: {hipResult}");
                return;
            }

            Console.WriteLine("[HipBackend] Module loaded, getting kernel functions...");

            // Get kernel functions - try to load all available kernels
            // MFMA kernels (CDNA only)
            hipResult = HipNativeBindings.hipModuleGetFunction(ref _mfmaGemmF32, _mfmaModule, "mfma_gemm_f32");
            if (hipResult == HipError.Success)
                _kernelCache["mfma_gemm_f32"] = _mfmaGemmF32;

            hipResult = HipNativeBindings.hipModuleGetFunction(ref _mfmaGemmF16, _mfmaModule, "mfma_gemm_f16");
            if (hipResult == HipError.Success)
                _kernelCache["mfma_gemm_f16"] = _mfmaGemmF16;

            // Scalar kernel (works on all GPUs including RDNA1, RDNA2, GCN)
            hipResult = HipNativeBindings.hipModuleGetFunction(ref _scalarGemmF32, _mfmaModule, "scalar_gemm_f32");
            if (hipResult == HipError.Success)
                _kernelCache["scalar_gemm_f32"] = _scalarGemmF32;

            // RDNA optimized wave32 kernel
            hipResult = HipNativeBindings.hipModuleGetFunction(ref _rdnaGemmWave32, _mfmaModule, "rdna_gemm_wave32");
            if (hipResult == HipError.Success)
                _kernelCache["rdna_gemm_wave32"] = _rdnaGemmWave32;

            Console.WriteLine($"[HipBackend] Kernel compilation complete. Available kernels: {string.Join(", ", _kernelCache.Keys)}");
            System.Diagnostics.Debug.WriteLine($"HIP kernels compiled successfully for {_architecture}. Available kernels: {string.Join(", ", _kernelCache.Keys)}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HipBackend] Kernel compilation EXCEPTION: {ex.GetType().Name}: {ex.Message}");
            System.Diagnostics.Debug.WriteLine($"HIP kernel compilation failed: {ex.Message}");
        }
    }

    #region Memory Management

    public IGpuBuffer AllocateBuffer(float[] data)
    {
        IntPtr devicePtr = IntPtr.Zero;
        var size = (UIntPtr)(data.Length * sizeof(float));

        var result = HipNativeBindings.hipMalloc(ref devicePtr, size);
        HipNativeBindings.CheckError(result, "hipMalloc");

        // Copy data to device
        GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            result = HipNativeBindings.hipMemcpy(
                devicePtr,
                handle.AddrOfPinnedObject(),
                size,
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D");
        }
        finally
        {
            handle.Free();
        }

        return new HipGpuBuffer(devicePtr, data.Length);
    }

    public IGpuBuffer AllocateBuffer(int size)
    {
        IntPtr devicePtr = IntPtr.Zero;
        var sizeBytes = (UIntPtr)(size * sizeof(float));

        var result = HipNativeBindings.hipMalloc(ref devicePtr, sizeBytes);
        HipNativeBindings.CheckError(result, "hipMalloc");

        // Zero-initialize
        result = HipNativeBindings.hipMemset(devicePtr, 0, sizeBytes);
        HipNativeBindings.CheckError(result, "hipMemset");

        return new HipGpuBuffer(devicePtr, size);
    }

    public float[] DownloadBuffer(IGpuBuffer buffer)
    {
        var hipBuffer = (HipGpuBuffer)buffer;
        float[] result = new float[hipBuffer.Size];
        DownloadBuffer(buffer, result);
        return result;
    }

    public void DownloadBuffer(IGpuBuffer buffer, float[] destination)
    {
        var hipBuffer = (HipGpuBuffer)buffer;
        var size = (UIntPtr)(hipBuffer.Size * sizeof(float));

        GCHandle handle = GCHandle.Alloc(destination, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                handle.AddrOfPinnedObject(),
                hipBuffer.Handle,
                size,
                HipMemcpyKind.DeviceToHost);
            HipNativeBindings.CheckError(result, "hipMemcpy D2H");
        }
        finally
        {
            handle.Free();
        }
    }

    #endregion

    #region GEMM Operations

    public void Gemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        var bufferA = (HipGpuBuffer)A;
        var bufferB = (HipGpuBuffer)B;
        var bufferC = (HipGpuBuffer)C;

        // Select kernel based on architecture and matrix size
        IntPtr kernel = SelectGemmKernel(M, N, K);

        if (kernel == IntPtr.Zero)
        {
            // Fallback to CPU (should not happen if initialized properly)
            throw new InvalidOperationException("No suitable GEMM kernel available");
        }

        // Calculate grid/block dimensions based on kernel type
        GemmKernelType kernelType = GetKernelType(kernel);

        int tileM, tileN, blockSize;
        switch (kernelType)
        {
            case GemmKernelType.Mfma:
                // MFMA: 128x128 workgroup tiles, 256 threads (4 warps of 64)
                tileM = 128;
                tileN = 128;
                blockSize = 256;
                break;
            case GemmKernelType.RdnaWave32:
                // RDNA wave32: 32x32 tiles, 256 threads
                tileM = 32;
                tileN = 32;
                blockSize = 256;
                break;
            case GemmKernelType.Scalar:
            default:
                // Scalar: 16x16 tiles, 256 threads
                tileM = 16;
                tileN = 16;
                blockSize = 256;
                break;
        }

        uint gridDimX = (uint)((M + tileM - 1) / tileM);
        uint gridDimY = (uint)((N + tileN - 1) / tileN);

        // Prepare kernel arguments
        IntPtr[] kernelArgs = new IntPtr[8];
        GCHandle[] handles = new GCHandle[8];

        try
        {
            // Marshal arguments
            var argA = bufferA.Handle;
            var argB = bufferB.Handle;
            var argC = bufferC.Handle;

            handles[0] = GCHandle.Alloc(argA, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(argB, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(argC, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(M, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(N, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(K, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(alpha, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(beta, GCHandleType.Pinned);

            for (int i = 0; i < 8; i++)
            {
                kernelArgs[i] = handles[i].AddrOfPinnedObject();
            }

            // Allocate and populate kernel params array
            GCHandle kernelParamsHandle = GCHandle.Alloc(kernelArgs, GCHandleType.Pinned);
            try
            {
                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    gridDimX, gridDimY, 1,
                    (uint)blockSize, 1, 1,
                    0, // shared memory
                    _stream,
                    kernelParamsHandle.AddrOfPinnedObject(),
                    IntPtr.Zero);

                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");
            }
            finally
            {
                kernelParamsHandle.Free();
            }
        }
        finally
        {
            foreach (var handle in handles)
            {
                if (handle.IsAllocated)
                    handle.Free();
            }
        }

        // Synchronize
        var syncResult = HipNativeBindings.hipStreamSynchronize(_stream);
        HipNativeBindings.CheckError(syncResult, "hipStreamSynchronize");
    }

    private IntPtr SelectGemmKernel(int M, int N, int K)
    {
        // Select kernel based on architecture - use the recommended kernel from HipMfmaKernel
        string recommendedKernel = HipMfmaKernel.GetRecommendedKernel(_architecture);

        if (_kernelCache.TryGetValue(recommendedKernel, out var kernel))
            return kernel;

        // Fallback order: try scalar first (works everywhere), then RDNA wave32, then MFMA
        if (_kernelCache.TryGetValue("scalar_gemm_f32", out var scalarKernel))
            return scalarKernel;

        if (_kernelCache.TryGetValue("rdna_gemm_wave32", out var rdnaKernel))
            return rdnaKernel;

        if (_kernelCache.TryGetValue("mfma_gemm_f32", out var mfmaKernel))
            return mfmaKernel;

        return IntPtr.Zero;
    }

    /// <summary>
    /// Gets the kernel type for the currently selected kernel.
    /// </summary>
    private GemmKernelType GetKernelType(IntPtr kernel)
    {
        if (_kernelCache.TryGetValue("mfma_gemm_f32", out var mfmaKernel) && kernel == mfmaKernel)
            return GemmKernelType.Mfma;
        if (_kernelCache.TryGetValue("mfma_gemm_f16", out var mfmaF16Kernel) && kernel == mfmaF16Kernel)
            return GemmKernelType.Mfma;
        if (_kernelCache.TryGetValue("rdna_gemm_wave32", out var rdnaKernel) && kernel == rdnaKernel)
            return GemmKernelType.RdnaWave32;
        if (_kernelCache.TryGetValue("scalar_gemm_f32", out var scalarKernel) && kernel == scalarKernel)
            return GemmKernelType.Scalar;

        return GemmKernelType.Scalar;  // Default to scalar dimensions
    }

    private enum GemmKernelType
    {
        Mfma,       // 128x128 workgroup tiles, 256 threads
        RdnaWave32, // 32x32 tiles, 256 threads
        Scalar      // 16x16 tiles, 256 threads
    }

    public IGpuBuffer MatMul(IGpuBuffer A, IGpuBuffer B, int M, int N, int K)
    {
        var C = AllocateBuffer(M * N);
        Gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        return C;
    }

    public void BatchedGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount, float alpha = 1.0f, float beta = 0.0f)
    {
        if (batchCount <= 0)
            throw new ArgumentException("Batch count must be positive", nameof(batchCount));

        System.Diagnostics.Debug.WriteLine("HipBackend BatchedGemm is executing on CPU fallback; TODO: implement GPU batched GEMM.");

        // Download all data
        var aData = DownloadBuffer(A);
        var bData = DownloadBuffer(B);
        var cData = DownloadBuffer(C);

        int aStride = M * K;
        int bStride = K * N;
        int cStride = M * N;

        // Process each batch
        for (int batch = 0; batch < batchCount; batch++)
        {
            int aOffset = batch * aStride;
            int bOffset = batch * bStride;
            int cOffset = batch * cStride;

            // Perform matrix multiplication for this batch
            for (int row = 0; row < M; row++)
            {
                for (int col = 0; col < N; col++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++)
                    {
                        sum += aData[aOffset + row * K + k] * bData[bOffset + k * N + col];
                    }
                    int cIdx = cOffset + row * N + col;
                    cData[cIdx] = alpha * sum + beta * cData[cIdx];
                }
            }
        }

        // Upload result
        UploadToBuffer(C, cData);
    }

    #endregion

    #region Fused Operations

    public IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        // For now, execute as separate operations
        // TODO: Implement fused HIP kernel for GEMM+Bias+ReLU
        var C = MatMul(A, B, M, N, K);

        // Apply bias + ReLU on GPU
        // This is a placeholder - in production, use a fused kernel
        ApplyBiasAndActivation(C, bias, M, N, ActivationType.ReLU);

        return C;
    }

    public IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        var C = MatMul(A, B, M, N, K);
        ApplyBiasAndActivation(C, bias, M, N, ActivationType.GELU);
        return C;
    }

    public IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        var C = MatMul(A, B, M, N, K);
        ApplyBiasAndActivation(C, bias, M, N, ActivationType.Sigmoid);
        return C;
    }

    public IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        var C = MatMul(A, B, M, N, K);
        ApplyBiasAndActivation(C, bias, M, N, ActivationType.Tanh);
        return C;
    }

    private void ApplyBiasAndActivation(IGpuBuffer C, IGpuBuffer bias, int M, int N, ActivationType activation)
    {
        // Download, apply on CPU, upload (temporary until fused kernel is ready)
        var cData = DownloadBuffer(C);
        var biasData = DownloadBuffer(bias);

        for (int row = 0; row < M; row++)
        {
            for (int col = 0; col < N; col++)
            {
                int idx = row * N + col;
                float val = cData[idx] + biasData[col];

                cData[idx] = activation switch
                {
                    ActivationType.ReLU => Math.Max(0, val),
                    ActivationType.Sigmoid => 1.0f / (1.0f + MathF.Exp(-val)),
                    ActivationType.Tanh => MathF.Tanh(val),
                    ActivationType.GELU => val * 0.5f * (1.0f + MathF.Tanh(0.7978845608f * (val + 0.044715f * val * val * val))),
                    _ => val
                };
            }
        }

        // Upload back
        var hipBuffer = (HipGpuBuffer)C;
        var size = (UIntPtr)(cData.Length * sizeof(float));
        GCHandle handle = GCHandle.Alloc(cData, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                hipBuffer.Handle,
                handle.AddrOfPinnedObject(),
                size,
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (bias+activation)");
        }
        finally
        {
            handle.Free();
        }
    }

    #endregion

    #region Element-wise Operations

    public void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        // Simple CPU fallback for now - TODO: Add HIP elementwise kernels
        var aData = DownloadBuffer(A);
        var bData = DownloadBuffer(B);
        var cData = new float[size];

        for (int i = 0; i < size; i++)
            cData[i] = aData[i] + bData[i];

        UploadToBuffer(C, cData);
    }

    public void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = DownloadBuffer(B);
        var cData = new float[size];

        for (int i = 0; i < size; i++)
            cData[i] = aData[i] - bData[i];

        UploadToBuffer(C, cData);
    }

    public void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)    
    {
        var aData = DownloadBuffer(A);
        var bData = DownloadBuffer(B);
        var cData = new float[size];

        for (int i = 0; i < size; i++)
            cData[i] = aData[i] * bData[i];

        UploadToBuffer(C, cData);
    }

    public void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = DownloadBuffer(B);
        var cData = new float[size];

        for (int i = 0; i < size; i++)
            cData[i] = aData[i] / bData[i];

        UploadToBuffer(C, cData);
    }

    public void Min(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = DownloadBuffer(B);
        var cData = new float[size];

        for (int i = 0; i < size; i++)
            cData[i] = MathF.Min(aData[i], bData[i]);

        UploadToBuffer(C, cData);
    }

    public void Max(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = DownloadBuffer(B);
        var cData = new float[size];

        for (int i = 0; i < size; i++)
            cData[i] = MathF.Max(aData[i], bData[i]);

        UploadToBuffer(C, cData);
    }

    public void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size)       
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = aData[i] * scalar;

        UploadToBuffer(B, bData);
    }

    public void Power(IGpuBuffer A, IGpuBuffer B, float exponent, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = MathF.Pow(aData[i], exponent);

        UploadToBuffer(B, bData);
    }

    public void Abs(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = MathF.Abs(aData[i]);

        UploadToBuffer(B, bData);
    }

    public void Exp(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = MathF.Exp(aData[i]);

        UploadToBuffer(B, bData);
    }

    public void Exp2(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = MathF.Pow(2.0f, aData[i]);

        UploadToBuffer(B, bData);
    }

    public void Exp10(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = MathF.Pow(10.0f, aData[i]);

        UploadToBuffer(B, bData);
    }

    public void ExpM1(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = MathF.Exp(aData[i]) - 1.0f;

        UploadToBuffer(B, bData);
    }

    public void Log(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = MathF.Log(aData[i]);

        UploadToBuffer(B, bData);
    }

    public void Log2(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = MathF.Log2(aData[i]);

        UploadToBuffer(B, bData);
    }

    public void Log1P(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = MathF.Log(1.0f + aData[i]);

        UploadToBuffer(B, bData);
    }

    public void Sqrt(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = MathF.Sqrt(aData[i]);

        UploadToBuffer(B, bData);
    }

    public void Sign(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = aData[i] > 0.0f ? 1.0f : (aData[i] < 0.0f ? -1.0f : 0.0f);

        UploadToBuffer(B, bData);
    }

    public void Relu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = Math.Max(0, aData[i]);

        UploadToBuffer(B, bData);
    }

    public void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = 1.0f / (1.0f + MathF.Exp(-aData[i]));

        UploadToBuffer(B, bData);
    }

    public void Tanh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
            bData[i] = MathF.Tanh(aData[i]);

        UploadToBuffer(B, bData);
    }

    public void Gelu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[size];

        const float SQRT_2_OVER_PI = 0.7978845608f;
        const float COEFF = 0.044715f;

        for (int i = 0; i < size; i++)
        {
            float x = aData[i];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            bData[i] = 0.5f * x * (1.0f + MathF.Tanh(inner));
        }

        UploadToBuffer(B, bData);
    }

    public void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[batchSize * features];

        for (int b = 0; b < batchSize; b++)
        {
            int offset = b * features;

            // Find max for numerical stability
            float max = float.MinValue;
            for (int f = 0; f < features; f++)
                max = Math.Max(max, aData[offset + f]);

            // Compute exp and sum
            float sum = 0;
            for (int f = 0; f < features; f++)
            {
                bData[offset + f] = MathF.Exp(aData[offset + f] - max);
                sum += bData[offset + f];
            }

            // Normalize
            for (int f = 0; f < features; f++)
                bData[offset + f] /= sum;
        }

        UploadToBuffer(B, bData);
    }

    private void UploadToBuffer(IGpuBuffer buffer, float[] data)
    {
        var hipBuffer = (HipGpuBuffer)buffer;
        var size = (UIntPtr)(data.Length * sizeof(float));

        GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                hipBuffer.Handle,
                handle.AddrOfPinnedObject(),
                size,
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D");
        }
        finally
        {
            handle.Free();
        }
    }

    #endregion

    #region Sparse Operations (2:4 Structured Sparsity)

    public IGpuBuffer AllocateByteBuffer(int size)
    {
        IntPtr devicePtr = IntPtr.Zero;
        var sizeBytes = (UIntPtr)size;

        var result = HipNativeBindings.hipMalloc(ref devicePtr, sizeBytes);
        HipNativeBindings.CheckError(result, "hipMalloc (byte buffer)");

        // Zero-initialize
        result = HipNativeBindings.hipMemset(devicePtr, 0, sizeBytes);
        HipNativeBindings.CheckError(result, "hipMemset (byte buffer)");

        return new HipGpuByteBuffer(devicePtr, size);
    }

    public void Enforce2x4Sparsity(IGpuBuffer denseInput, IGpuBuffer sparseValues, IGpuBuffer sparseIndices, int M, int K)
    {
        if (K % 4 != 0)
            throw new ArgumentException("K must be divisible by 4 for 2:4 sparsity", nameof(K));

        // Download dense data
        var denseData = DownloadBuffer(denseInput);

        // Use SparsityUtils for CPU-side sparsity enforcement
        var compressed = SparsityUtils.CompressTo2x4(denseData, M, K);

        // Upload compressed values
        UploadToBuffer(sparseValues, compressed.Values);

        // Upload indices
        var byteBuffer = (HipGpuByteBuffer)sparseIndices;
        UploadBytesToBuffer(byteBuffer, compressed.Indices);
    }

    public void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
    {
        if (K % 4 != 0)
            throw new ArgumentException("K must be divisible by 4 for 2:4 sparsity", nameof(K));

        // Download compressed data
        var values = DownloadBuffer(sparseValues);
        var indices = DownloadBytesFromBuffer((HipGpuByteBuffer)sparseIndices);

        // Create compressed representation
        var compressed = new Compressed2x4Sparse(values, indices, M, K);

        // Decompress
        var denseData = SparsityUtils.DecompressFrom2x4(compressed);

        // Upload result
        UploadToBuffer(denseOutput, denseData);
    }

    public void SparseGemm(
        IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices,
        IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K,
        float alpha = 1.0f, float beta = 0.0f)
    {
        if (K % 4 != 0)
            throw new ArgumentException("K must be divisible by 4 for 2:4 sparsity", nameof(K));

        // Download sparse A data
        var aValues = DownloadBuffer(sparseAValues);
        var aIndices = DownloadBytesFromBuffer((HipGpuByteBuffer)sparseAIndices);

        // Create compressed representation
        var sparseA = new Compressed2x4Sparse(aValues, aIndices, M, K);

        // Download B and existing C
        var bData = DownloadBuffer(B);
        var cData = DownloadBuffer(C);

        // Execute sparse GEMM on CPU (later can be HIP-accelerated)
        SparsityUtils.SparseGemmCpu(sparseA, bData, cData, N, alpha, beta);

        // Upload result
        UploadToBuffer(C, cData);
    }

    public IGpuBuffer SparseGemmBiasRelu(
        IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices,
        IGpuBuffer B, IGpuBuffer bias,
        int M, int N, int K)
    {
        if (K % 4 != 0)
            throw new ArgumentException("K must be divisible by 4 for 2:4 sparsity", nameof(K));

        // Create output buffer
        var C = AllocateBuffer(M * N);

        // Execute sparse GEMM
        SparseGemm(sparseAValues, sparseAIndices, B, C, M, N, K, 1.0f, 0.0f);

        // Download, apply bias + ReLU
        var cData = DownloadBuffer(C);
        var biasData = DownloadBuffer(bias);

        for (int row = 0; row < M; row++)
        {
            for (int col = 0; col < N; col++)
            {
                int idx = row * N + col;
                float val = cData[idx] + biasData[col];
                cData[idx] = Math.Max(0, val);
            }
        }

        // Upload back
        UploadToBuffer(C, cData);
        return C;
    }

    private void UploadBytesToBuffer(HipGpuByteBuffer buffer, byte[] data)
    {
        var size = (UIntPtr)data.Length;

        GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                buffer.Handle,
                handle.AddrOfPinnedObject(),
                size,
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (bytes)");
        }
        finally
        {
            handle.Free();
        }
    }

    private byte[] DownloadBytesFromBuffer(HipGpuByteBuffer buffer)
    {
        byte[] result = new byte[buffer.Size];
        var size = (UIntPtr)buffer.Size;

        GCHandle handle = GCHandle.Alloc(result, GCHandleType.Pinned);
        try
        {
            var hipResult = HipNativeBindings.hipMemcpy(
                handle.AddrOfPinnedObject(),
                buffer.Handle,
                size,
                HipMemcpyKind.DeviceToHost);
            HipNativeBindings.CheckError(hipResult, "hipMemcpy D2H (bytes)");
        }
        finally
        {
            handle.Free();
        }

        return result;
    }

    #endregion

    #region Reduction Operations

    public float Sum(IGpuBuffer A, int size)
    {
        var data = DownloadBuffer(A);
        float sum = 0;
        for (int i = 0; i < size; i++)
            sum += data[i];
        return sum;
    }

    public float Max(IGpuBuffer A, int size)
    {
        var data = DownloadBuffer(A);
        float max = float.MinValue;
        for (int i = 0; i < size; i++)
            if (data[i] > max) max = data[i];
        return max;
    }

    public void SumAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        var aData = DownloadBuffer(A);
        var bData = new float[outerSize];

        for (int o = 0; o < outerSize; o++)
        {
            float sum = 0;
            for (int r = 0; r < reduceSize; r++)
                sum += aData[o * reduceSize + r];
            bData[o] = sum;
        }

        UploadToBuffer(B, bData);
    }

    #endregion

    public void Synchronize()
    {
        if (_stream != IntPtr.Zero)
        {
            var result = HipNativeBindings.hipStreamSynchronize(_stream);
            // Don't throw on sync errors, just log
            if (result != HipError.Success)
            {
                System.Diagnostics.Debug.WriteLine($"hipStreamSynchronize warning: {result}");
            }
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_mfmaModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_mfmaModule);
            _mfmaModule = IntPtr.Zero;
        }

        if (_stream != IntPtr.Zero)
        {
            HipNativeBindings.hipStreamDestroy(_stream);
            _stream = IntPtr.Zero;
        }

        _kernelCache.Clear();
        _disposed = true;
    }
}

/// <summary>
/// HIP GPU buffer wrapper implementing IGpuBuffer.
/// </summary>
internal sealed class HipGpuBuffer : IGpuBuffer
{
    public IntPtr Handle { get; }
    public int Size { get; }
    public long SizeInBytes => Size * sizeof(float);
    private bool _disposed;

    public HipGpuBuffer(IntPtr handle, int size)
    {
        Handle = handle;
        Size = size;
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (Handle != IntPtr.Zero)
        {
            var result = HipNativeBindings.hipFree(Handle);
            if (result != HipError.Success)
            {
                System.Diagnostics.Debug.WriteLine($"hipFree warning: {result}");
            }
        }

        _disposed = true;
    }
}

/// <summary>
/// HIP GPU byte buffer wrapper implementing IGpuBuffer.
/// Used for sparse matrix indices (1 byte per group of 4 elements).
/// </summary>
internal sealed class HipGpuByteBuffer : IGpuBuffer
{
    public IntPtr Handle { get; }
    public int Size { get; }
    public long SizeInBytes => Size;
    private bool _disposed;

    public HipGpuByteBuffer(IntPtr handle, int size)
    {
        Handle = handle;
        Size = size;
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (Handle != IntPtr.Zero)
        {
            var result = HipNativeBindings.hipFree(Handle);
            if (result != HipError.Success)
            {
                System.Diagnostics.Debug.WriteLine($"hipFree (byte buffer) warning: {result}");
            }
        }

        _disposed = true;
    }
}
