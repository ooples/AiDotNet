// Copyright (c) AiDotNet. All rights reserved.
// HIP backend for AMD GPU with real MFMA (Matrix Fused Multiply-Add) support.
// Target: 25,000+ GFLOPS on MI200, 15,000+ GFLOPS on RX 7900.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.Sparsity;
using AiDotNet.Tensors.Engines.Gpu;
using FusedActivationType = AiDotNet.Tensors.Engines.FusedActivationType;

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
public sealed class HipBackend : IAsyncGpuBackend
{
    private IntPtr _stream;
    private HipStream? _defaultStream;
    private IntPtr _mfmaModule;
    private IntPtr _mfmaGemmF32;
    private IntPtr _mfmaGemmF16;
    private IntPtr _scalarGemmF32;
    private IntPtr _rdnaGemmWave32;
    private readonly Dictionary<string, IntPtr> _kernelCache;
    private AmdGpuArchitecture _architecture;
    private bool _disposed;
    private HipDeviceProperties _deviceProps;

    // Additional kernel modules
    private IntPtr _activationModule;
    private IntPtr _neuralNetModule;
    private IntPtr _convolutionModule;
    private IntPtr _fusedConvolutionModule;
    private IntPtr _poolingModule;
    private IntPtr _normalizationModule;
    private IntPtr _fusedModule;
    private IntPtr _attentionModule;
    private IntPtr _fftModule;
    private IntPtr _sparseModule;
    private IntPtr _locallyConnectedModule;
    private IntPtr _deformableConvModule;
    private IntPtr _optimizerModule;
    private IntPtr _specializedModule;

    private const int DefaultBlockSize = 256;

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

    /// <summary>
    /// Enables or disables diagnostic logging for HIP operations.
    /// </summary>
    public static bool EnableDiagnostics
    {
        get => HipNativeBindings.EnableDiagnostics;
        set => HipNativeBindings.EnableDiagnostics = value;
    }

    /// <summary>
    /// Gets whether elementwise operations are GPU-accelerated.
    /// Returns true for HipBackend as all elementwise operations use GPU kernels.
    /// </summary>
    public bool SupportsElementwiseGpu => true;

    #region IAsyncGpuBackend Properties

    /// <inheritdoc/>
    public bool SupportsMultiStream => true;

    /// <inheritdoc/>
    public bool SupportsEvents => true;

    /// <inheritdoc/>
    public bool SupportsAsyncTransfer => true;

    /// <inheritdoc/>
    public bool SupportsGraphCapture => false; // HIP graph capture is more limited

    /// <inheritdoc/>
    public int MaxConcurrentStreams => 16;

    /// <inheritdoc/>
    public IGpuStream DefaultStream
    {
        get
        {
            if (_defaultStream == null && IsAvailable)
            {
                var newStream = new HipStream(this, _stream, GpuStreamType.Default, true);
                System.Threading.Interlocked.CompareExchange(ref _defaultStream, newStream, null);
                // If another thread won the race, dispose our unused stream
                if (_defaultStream != newStream)
                {
                    newStream.Dispose();
                }
            }
            return _defaultStream ?? throw new InvalidOperationException("Backend not available");
        }
    }

    #endregion

    public HipBackend() : this(0)
    {
    }

    public HipBackend(int deviceIndex)
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
            var result = HipNativeBindings.hipSetDevice(deviceIndex);
            if (result != HipError.Success)
            {
                IsAvailable = false;
                DeviceName = "None";
                return;
            }

            // Get device properties
            _deviceProps = new HipDeviceProperties();
            result = HipNativeBindings.hipGetDeviceProperties(ref _deviceProps, deviceIndex);
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

            // Verify at least one GEMM kernel compiled successfully
            bool hasGemmKernel = _kernelCache.ContainsKey("scalar_gemm_f32") ||
                                 _kernelCache.ContainsKey("mfma_gemm_f32") ||
                                 _kernelCache.ContainsKey("rdna_gemm_wave32");

            if (!hasGemmKernel)
            {
                Console.WriteLine("[HipBackend] No GEMM kernels compiled - backend not available");
                IsAvailable = false;
                return;
            }

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
            // Compile MFMA/GEMM module
            CompileKernelModule(source, "mfma_gemm", ref _mfmaModule, new[]
            {
                "mfma_gemm_f32", "mfma_gemm_f16", "scalar_gemm_f32", "rdna_gemm_wave32"
            });

            // Store GEMM kernel handles for quick lookup
            if (_kernelCache.TryGetValue("mfma_gemm_f32", out var mfmaF32))
                _mfmaGemmF32 = mfmaF32;
            if (_kernelCache.TryGetValue("mfma_gemm_f16", out var mfmaF16))
                _mfmaGemmF16 = mfmaF16;
            if (_kernelCache.TryGetValue("scalar_gemm_f32", out var scalarF32))
                _scalarGemmF32 = scalarF32;
            if (_kernelCache.TryGetValue("rdna_gemm_wave32", out var rdnaWave32))
                _rdnaGemmWave32 = rdnaWave32;

            // Compile Activation kernels
            CompileKernelModule(HipActivationKernels.GetSource(), "activation", ref _activationModule,
                HipActivationKernels.GetKernelNames());

            // Compile Neural Net kernels
            CompileKernelModule(HipNeuralNetKernels.GetSource(), "neural_net", ref _neuralNetModule,
                HipNeuralNetKernels.GetKernelNames());

            // Compile Convolution kernels
            CompileKernelModule(HipConvolutionKernels.GetSource(), "convolution", ref _convolutionModule,
                HipConvolutionKernels.GetKernelNames());

            // Compile Fused Convolution kernels (Conv2D + Bias/BatchNorm + Activation)
            CompileKernelModule(HipFusedConvolutionKernels.GetSource(), "fused_convolution", ref _fusedConvolutionModule,
                HipFusedConvolutionKernels.GetKernelNames());

            // Compile Pooling kernels
            CompileKernelModule(HipPoolingKernels.GetSource(), "pooling", ref _poolingModule,
                HipPoolingKernels.GetKernelNames());

            // Compile Normalization kernels
            CompileKernelModule(HipNormalizationKernels.GetSource(), "normalization", ref _normalizationModule,
                HipNormalizationKernels.GetKernelNames());

            // Compile Fused kernels (GEMM + Bias + Activation in single pass)
            CompileKernelModule(HipFusedKernels.GetSource(), "fused", ref _fusedModule,
                HipFusedKernels.GetKernelNames());

            // Compile Attention kernels (FlashAttention, GQA, ScaledDotProduct)
            CompileKernelModule(HipAttentionKernels.GetSource(), "attention", ref _attentionModule,
                HipAttentionKernels.GetKernelNames());

            // Compile FFT kernels (Cooley-Tukey radix-2 FFT, STFT, Mel spectrogram)
            CompileKernelModule(HipFFTKernels.GetSource(), "fft", ref _fftModule,
                HipFFTKernels.GetKernelNames());

            // Compile Sparse kernels (CSR SpMM, GNN message passing)
            CompileKernelModule(HipSparseKernels.GetSource(), "sparse", ref _sparseModule,
                HipSparseKernels.GetKernelNames());

            // Compile Locally Connected kernels (unique weights per spatial position)
            CompileKernelModule(HipLocallyConnectedKernels.GetSource(), "locally_connected", ref _locallyConnectedModule,
                HipLocallyConnectedKernels.GetKernelNames());

            // Compile Deformable Convolution kernels (DCNv2 with learnable offsets and masks)
            CompileKernelModule(HipDeformableConvolutionKernels.GetSource(), "deformable_conv", ref _deformableConvModule,
                HipDeformableConvolutionKernels.GetKernelNames());

            // Compile Optimizer kernels (SGD, Adam, AdamW, RMSprop, etc.)
            CompileKernelModule(HipOptimizerKernels.GetSource(), "optimizer", ref _optimizerModule,
                HipOptimizerKernels.GetKernelNames());

            // Compile Specialized kernels (hyperbolic geometry, octonion algebra, quantum computing)
            CompileKernelModule(Kernels.HipSpecializedKernels.GetSource(), "specialized", ref _specializedModule,
                Kernels.HipSpecializedKernels.GetKernelNames());

            Console.WriteLine($"[HipBackend] Kernel compilation complete. Available kernels: {_kernelCache.Count}");
            System.Diagnostics.Debug.WriteLine($"HIP kernels compiled successfully for {_architecture}. Total: {_kernelCache.Count}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HipBackend] Kernel compilation EXCEPTION: {ex.GetType().Name}: {ex.Message}");
            System.Diagnostics.Debug.WriteLine($"HIP kernel compilation failed: {ex.Message}");
        }
    }

    private void CompileKernelModule(string source, string moduleName, ref IntPtr module, string[] kernelNames)
    {
        string compileFlags = HipMfmaKernel.GetCompileFlags(_architecture);

        IntPtr prog = IntPtr.Zero;
        var rtcResult = HipNativeBindings.hiprtcCreateProgram(
            ref prog, source, moduleName, 0, IntPtr.Zero, IntPtr.Zero);

        if (rtcResult != HipRtcResult.Success)
        {
            Console.WriteLine($"[HipBackend] Failed to create program for {moduleName}: {rtcResult}");
            return;
        }

        var options = new List<string>(compileFlags.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries))
        {
            "-O3",
            "-ffast-math"
        };
        rtcResult = HipNativeBindings.hiprtcCompileProgram(prog, options.Count, options.ToArray());

        if (rtcResult != HipRtcResult.Success)
        {
            UIntPtr logSize = UIntPtr.Zero;
            HipNativeBindings.hiprtcGetProgramLogSize(prog, ref logSize);
            if ((ulong)logSize > 0)
            {
                IntPtr logPtr = Marshal.AllocHGlobal((int)(ulong)logSize);
                HipNativeBindings.hiprtcGetProgramLog(prog, logPtr);
                string log = Marshal.PtrToStringAnsi(logPtr) ?? "";
                Marshal.FreeHGlobal(logPtr);
                Console.WriteLine($"[HipBackend] Compile failed for {moduleName}: {log}");
            }
            HipNativeBindings.hiprtcDestroyProgram(ref prog);
            return;
        }

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

        var hipResult = HipNativeBindings.hipModuleLoadData(ref module, code);
        Marshal.FreeHGlobal(code);

        if (hipResult != HipError.Success)
        {
            Console.WriteLine($"[HipBackend] Failed to load module {moduleName}: {hipResult}");
            return;
        }

        // Load all kernel functions
        foreach (var kernelName in kernelNames)
        {
            IntPtr func = IntPtr.Zero;
            hipResult = HipNativeBindings.hipModuleGetFunction(ref func, module, kernelName);
            if (hipResult == HipError.Success)
                _kernelCache[kernelName] = func;
        }
    }

    private unsafe void LaunchKernel(IntPtr kernel, uint gridX, uint blockSize, IntPtr[] args, uint sharedMem = 0)
    {
        LaunchKernelOnStream(kernel, gridX, blockSize, args, _stream, sharedMem);
    }

    private unsafe void LaunchKernelOnStream(IntPtr kernel, uint gridX, uint blockSize, IntPtr[] args, IntPtr stream, uint sharedMem = 0)
    {
        GCHandle argsHandle = GCHandle.Alloc(args, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipModuleLaunchKernel(
                kernel, gridX, 1, 1, blockSize, 1, 1,
                sharedMem, stream, argsHandle.AddrOfPinnedObject(), IntPtr.Zero);
            HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");
        }
        finally
        {
            argsHandle.Free();
        }
    }

    private unsafe void LaunchKernel2D(IntPtr kernel, uint gridX, uint gridY, uint blockX, uint blockY, IntPtr[] args, uint sharedMem = 0)
    {
        LaunchKernel2DOnStream(kernel, gridX, gridY, blockX, blockY, args, _stream, sharedMem);
    }

    private unsafe void LaunchKernel2DOnStream(IntPtr kernel, uint gridX, uint gridY, uint blockX, uint blockY, IntPtr[] args, IntPtr stream, uint sharedMem = 0)
    {
        GCHandle argsHandle = GCHandle.Alloc(args, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipModuleLaunchKernel(
                kernel, gridX, gridY, 1, blockX, blockY, 1,
                sharedMem, stream, argsHandle.AddrOfPinnedObject(), IntPtr.Zero);
            HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");
        }
        finally
        {
            argsHandle.Free();
        }
    }

    private unsafe void LaunchKernel3D(IntPtr kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, IntPtr[] args, uint sharedMem = 0)
    {
        LaunchKernel3DOnStream(kernel, gridX, gridY, gridZ, blockX, blockY, blockZ, args, _stream, sharedMem);
    }

    private unsafe void LaunchKernel3DOnStream(IntPtr kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, IntPtr[] args, IntPtr stream, uint sharedMem = 0)
    {
        GCHandle argsHandle = GCHandle.Alloc(args, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipModuleLaunchKernel(
                kernel, gridX, gridY, gridZ, blockX, blockY, blockZ,
                sharedMem, stream, argsHandle.AddrOfPinnedObject(), IntPtr.Zero);
            HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");
        }
        finally
        {
            argsHandle.Free();
        }
    }

    private unsafe void ExecuteFusedGemm(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output, int M, int N, int K)
    {
        ExecuteFusedGemmOnStream(kernelName, A, B, bias, output, M, N, K, _stream, synchronize: true);
    }

    private unsafe void ExecuteFusedGemmOnStream(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output, int M, int N, int K, IntPtr stream, bool synchronize)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"HIP fused kernel not found: {kernelName}");

        const int TILE_SIZE = 16;
        uint gridX = (uint)((N + TILE_SIZE - 1) / TILE_SIZE);
        uint gridY = (uint)((M + TILE_SIZE - 1) / TILE_SIZE);

        var handles = new GCHandle[7];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(bias.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(M, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(N, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(K, GCHandleType.Pinned);

            var args = new IntPtr[]
            {
                handles[0].AddrOfPinnedObject(),
                handles[1].AddrOfPinnedObject(),
                handles[2].AddrOfPinnedObject(),
                handles[3].AddrOfPinnedObject(),
                handles[4].AddrOfPinnedObject(),
                handles[5].AddrOfPinnedObject(),
                handles[6].AddrOfPinnedObject()
            };

            LaunchKernel2DOnStream(kernel, gridX, gridY, TILE_SIZE, TILE_SIZE, args, stream);
            if (synchronize)
            {
                var syncResult = HipNativeBindings.hipStreamSynchronize(stream);
                HipNativeBindings.CheckError(syncResult, "hipStreamSynchronize");
            }
        }
        finally
        {
            foreach (var h in handles)
                if (h.IsAllocated)
                    h.Free();
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
        var output = AllocateBuffer(M * N);
        ExecuteFusedGemm("gemm_bias_relu", A, B, bias, output, M, N, K);
        return output;
    }

    public IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        var output = AllocateBuffer(M * N);
        ExecuteFusedGemm("gemm_bias_gelu", A, B, bias, output, M, N, K);
        return output;
    }

    public IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        var output = AllocateBuffer(M * N);
        ExecuteFusedGemm("gemm_bias_sigmoid", A, B, bias, output, M, N, K);
        return output;
    }

    public IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        var output = AllocateBuffer(M * N);
        ExecuteFusedGemm("gemm_bias_tanh", A, B, bias, output, M, N, K);
        return output;
    }

    public IGpuBuffer GemmBias(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        // GEMM + bias without activation
        var output = AllocateBuffer(M * N);
        ExecuteFusedGemm("gemm_bias", A, B, bias, output, M, N, K);
        return output;
    }

    public unsafe void BiasAdd(IGpuBuffer A, IGpuBuffer bias, IGpuBuffer C, int M, int N)
    {
        if (!_kernelCache.TryGetValue("bias_add", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: bias_add");

        // First copy A to C (bias_add is in-place)
        var size = (UIntPtr)(M * N * sizeof(float));
        var result = HipNativeBindings.hipMemcpy(C.Handle, A.Handle, size, HipMemcpyKind.DeviceToDevice);
        HipNativeBindings.CheckError(result, "hipMemcpy D2D");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(C.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(bias.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(M, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(N, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((M * N + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <summary>
    /// Adds bias to Conv2D output in NCHW format.
    /// Operation: output[b, c, h, w] += bias[c]
    /// </summary>
    public unsafe void Conv2DBiasAdd(IGpuBuffer output, IGpuBuffer bias, int batch, int channels, int spatialSize)
    {
        if (!_kernelCache.TryGetValue("conv2d_bias_add", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv2d_bias_add");

        int totalSize = batch * channels * spatialSize;
        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(bias.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(spatialSize, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region Element-wise Operations

    public void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("add_vectors", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: add_vectors");

        LaunchBinaryOp(krnl, A, B, C, size);
    }

    public void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("subtract_vectors", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: subtract_vectors");

        LaunchBinaryOp(krnl, A, B, C, size);
    }

    public void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("multiply_vectors", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: multiply_vectors");

        LaunchBinaryOp(krnl, A, B, C, size);
    }

    public void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("divide_vectors", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: divide_vectors");

        LaunchBinaryOp(krnl, A, B, C, size);
    }

    public void Min(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("min_vectors", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: min_vectors");

        LaunchBinaryOp(krnl, A, B, C, size);
    }

    public void Max(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("max_vectors", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: max_vectors");

        LaunchBinaryOp(krnl, A, B, C, size);
    }

    public unsafe void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        if (!_kernelCache.TryGetValue("scale_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: scale_vector");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(scalar, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void Power(IGpuBuffer A, IGpuBuffer B, float exponent, int size)
    {
        // Power uses same pattern as Scale
        if (!_kernelCache.TryGetValue("power_scalar", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: power_scalar");

        LaunchScalarOp(krnl, A, B, exponent, size);
    }

    private unsafe void LaunchScalarOp(IntPtr krnl, IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(scalar, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void Abs(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("abs_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: abs_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Exp(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("exp_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: exp_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Exp2(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("exp2_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: exp2_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Exp10(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("exp10_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: exp10_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void ExpM1(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("expm1_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: expm1_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Log(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("log_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: log_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Log2(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("log2_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: log2_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Log1P(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("log1p_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: log1p_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Sqrt(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("sqrt_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sqrt_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Sign(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("sign_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sign_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Relu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("relu", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: relu");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("sigmoid", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sigmoid");

        LaunchUnaryOp(krnl, A, B, size);
    }

    private unsafe void LaunchUnaryOp(IntPtr krnl, IGpuBuffer A, IGpuBuffer B, int size)
    {
        var handles = new GCHandle[3];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[3];
            for (int i = 0; i < 3; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void Tanh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("tanh_activation", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: tanh_activation");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Gelu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("gelu", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: gelu");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public unsafe void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("leaky_relu", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: leaky_relu");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(alpha, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void LeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("leaky_relu_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: leaky_relu_backward");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(alpha, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void Elu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("elu", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: elu");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(alpha, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void EluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer output, IGpuBuffer gradInput, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("elu_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: elu_backward");

        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(alpha, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void Swish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("swish", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: swish");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public unsafe void SwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("swish_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: swish_backward");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void Silu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        // SiLU is the same as Swish
        Swish(A, B, size);
    }

    public void Mish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("mish_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: mish_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Softplus(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("softplus_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: softplus_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Hardswish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("hardswish_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hardswish_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public unsafe void Selu(IGpuBuffer A, IGpuBuffer B, float alpha, float scale, int size)
    {
        if (!_kernelCache.TryGetValue("selu_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: selu_vector");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(alpha, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(scale, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void Hardsigmoid(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("hardsigmoid_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hardsigmoid_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public unsafe void Hardtanh(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size)
    {
        if (!_kernelCache.TryGetValue("hardtanh_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hardtanh_vector");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(minVal, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(maxVal, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    // SiLU backward uses SwishBackward since they're mathematically equivalent
    public void SiluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        SwishBackward(gradOutput, input, gradInput, size);
    }

    public unsafe void MishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("mish_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: mish_backward");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void SoftplusBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("softplus_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: softplus_backward");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void HardswishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("hardswish_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hardswish_backward");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void SeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, float scale, int size)
    {
        if (!_kernelCache.TryGetValue("selu_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: selu_backward");

        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(alpha, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(scale, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void HardsigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("hardsigmoid_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hardsigmoid_backward");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void HardtanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float minVal, float maxVal, int size)
    {
        if (!_kernelCache.TryGetValue("hardtanh_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hardtanh_backward");

        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(minVal, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(maxVal, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #region Trigonometric Operations

    public void Sin(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("sin_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sin_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Cos(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("cos_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: cos_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Tan(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("tan_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: tan_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Asin(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("asin_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: asin_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Acos(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("acos_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: acos_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Atan(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("atan_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: atan_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    #endregion

    #region Hyperbolic Operations

    public void Sinh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("sinh_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sinh_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Cosh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("cosh_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: cosh_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Asinh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("asinh_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: asinh_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Acosh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("acosh_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: acosh_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Atanh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("atanh_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: atanh_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    #endregion

    #region Additional Unary Operations

    public void Reciprocal(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("reciprocal_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: reciprocal_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Cbrt(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("cbrt_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: cbrt_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Log10(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("log10_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: log10_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Negate(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("negate_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: negate_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Floor(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("floor_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: floor_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Ceiling(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("ceil_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: ceil_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Round(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("round_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: round_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Truncate(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("trunc_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: trunc_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    #endregion

    public unsafe void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features)
    {
        if (!_kernelCache.TryGetValue("softmax", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: softmax");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(features, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void Squash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon)
    {
        if (!_kernelCache.TryGetValue("squash", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: squash");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(numCapsules, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(capsuleDim, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((numCapsules + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon)
    {
        if (!_kernelCache.TryGetValue("squash_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: squash_backward");

        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(numCapsules, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(capsuleDim, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((numCapsules + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void CapsulePredictions(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int outputCapsules, int outputDim)
    {
        throw new NotImplementedException("CapsulePredictions kernel not yet implemented for HIP backend");
    }

    public void CapsuleTransform(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int numCapsules, int capsuleDim)
    {
        throw new NotImplementedException("CapsuleTransform kernel not yet implemented for HIP backend");
    }

    public void CapsuleWeightedSum(IGpuBuffer coupling, IGpuBuffer predictions, IGpuBuffer output,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        throw new NotImplementedException("CapsuleWeightedSum kernel not yet implemented for HIP backend");
    }

    public void CapsuleAgreement(IGpuBuffer predictions, IGpuBuffer output, IGpuBuffer agreement,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        throw new NotImplementedException("CapsuleAgreement kernel not yet implemented for HIP backend");
    }

    public void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize)
    {
        if (!_kernelCache.TryGetValue("tile_batch", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: tile_batch");

        int totalSize = repeats * innerSize;
        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(repeats, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(innerSize, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats)
    {
        if (!_kernelCache.TryGetValue("tile_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: tile_axis");

        int totalSize = outerSize * axisSize * repeats * innerSize;
        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(outerSize, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(axisSize, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(innerSize, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(repeats, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
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

    #region CSR Sparse Operations (General Sparsity)

    /// <inheritdoc/>
    public void CsrSpMM(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer output,
        int M, int K, int N, int nnz)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("csr_spmm", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: csr_spmm");

        // Launch configuration: rows x ceil(N/blockSize) grid
        int gridX = M;
        int gridY = (N + DefaultBlockSize - 1) / DefaultBlockSize;

        var valuesHandle = ((HipGpuBuffer)csrValues).Handle;
        var colIndicesHandle = ((HipGpuBuffer)csrColIndices).Handle;
        var rowPointersHandle = ((HipGpuBuffer)csrRowPointers).Handle;
        var denseBHandle = ((HipGpuBuffer)denseB).Handle;
        var outputHandle = ((HipGpuBuffer)output).Handle;

        LaunchKernel2D(kernel, gridX, gridY, DefaultBlockSize, 1,
            valuesHandle, colIndicesHandle, rowPointersHandle, denseBHandle, outputHandle,
            M, K, N, nnz);
    }

    /// <inheritdoc/>
    public void CsrSpMMBias(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer bias,
        IGpuBuffer output,
        int M, int K, int N, int nnz)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("csr_spmm_bias", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: csr_spmm_bias");

        int gridX = M;
        int gridY = (N + DefaultBlockSize - 1) / DefaultBlockSize;

        var valuesHandle = ((HipGpuBuffer)csrValues).Handle;
        var colIndicesHandle = ((HipGpuBuffer)csrColIndices).Handle;
        var rowPointersHandle = ((HipGpuBuffer)csrRowPointers).Handle;
        var denseBHandle = ((HipGpuBuffer)denseB).Handle;
        var biasHandle = ((HipGpuBuffer)bias).Handle;
        var outputHandle = ((HipGpuBuffer)output).Handle;

        LaunchKernel2DBias(kernel, gridX, gridY, DefaultBlockSize, 1,
            valuesHandle, colIndicesHandle, rowPointersHandle, denseBHandle, biasHandle, outputHandle,
            M, K, N, nnz);
    }

    /// <inheritdoc/>
    public void ScatterAddEdges(
        IGpuBuffer input,
        IGpuBuffer sourceIndices,
        IGpuBuffer targetIndices,
        IGpuBuffer? edgeValues,
        IGpuBuffer output,
        int numNodes, int numEdges, int features)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("scatter_add_edges", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: scatter_add_edges");

        if (!_kernelCache.TryGetValue("zero_buffer", out var zeroKernel))
            throw new InvalidOperationException("HIP kernel not found: zero_buffer");

        // First zero the output buffer
        int outputSize = numNodes * features;
        int zeroGrid = (outputSize + DefaultBlockSize - 1) / DefaultBlockSize;
        var outputHandle = ((HipGpuBuffer)output).Handle;
        LaunchKernel1D(zeroKernel, zeroGrid, DefaultBlockSize, outputHandle, outputSize);
        Synchronize();

        // Launch scatter-add kernel
        int gridX = numEdges;
        int gridY = (features + DefaultBlockSize - 1) / DefaultBlockSize;

        var inputHandle = ((HipGpuBuffer)input).Handle;
        var sourceHandle = ((HipGpuBuffer)sourceIndices).Handle;
        var targetHandle = ((HipGpuBuffer)targetIndices).Handle;
        var edgeValuesHandle = edgeValues is not null ? ((HipGpuBuffer)edgeValues).Handle : IntPtr.Zero;
        int hasEdgeValues = edgeValues is not null ? 1 : 0;

        LaunchScatterAddKernel(kernel, gridX, gridY, DefaultBlockSize, 1,
            inputHandle, sourceHandle, targetHandle, edgeValuesHandle, outputHandle,
            numNodes, numEdges, features, hasEdgeValues);
    }

    /// <inheritdoc/>
    public void CsrSegmentedMax(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("csr_segmented_max", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: csr_segmented_max");

        int gridX = M;
        int gridY = (N + DefaultBlockSize - 1) / DefaultBlockSize;

        var colIndicesHandle = ((HipGpuBuffer)csrColIndices).Handle;
        var rowPointersHandle = ((HipGpuBuffer)csrRowPointers).Handle;
        var inputHandle = ((HipGpuBuffer)input).Handle;
        var outputHandle = ((HipGpuBuffer)output).Handle;

        LaunchCsrSegmentedKernel(kernel, gridX, gridY, DefaultBlockSize, 1,
            colIndicesHandle, rowPointersHandle, inputHandle, outputHandle, M, K, N);
    }

    /// <inheritdoc/>
    public void CsrSegmentedMin(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("csr_segmented_min", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: csr_segmented_min");

        int gridX = M;
        int gridY = (N + DefaultBlockSize - 1) / DefaultBlockSize;

        var colIndicesHandle = ((HipGpuBuffer)csrColIndices).Handle;
        var rowPointersHandle = ((HipGpuBuffer)csrRowPointers).Handle;
        var inputHandle = ((HipGpuBuffer)input).Handle;
        var outputHandle = ((HipGpuBuffer)output).Handle;

        LaunchCsrSegmentedKernel(kernel, gridX, gridY, DefaultBlockSize, 1,
            colIndicesHandle, rowPointersHandle, inputHandle, outputHandle, M, K, N);
    }

    /// <inheritdoc/>
    public void CsrSegmentedStdDev(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N,
        float epsilon = 1e-8f)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("csr_segmented_stddev", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: csr_segmented_stddev");

        int gridX = M;
        int gridY = (N + DefaultBlockSize - 1) / DefaultBlockSize;

        var colIndicesHandle = ((HipGpuBuffer)csrColIndices).Handle;
        var rowPointersHandle = ((HipGpuBuffer)csrRowPointers).Handle;
        var inputHandle = ((HipGpuBuffer)input).Handle;
        var outputHandle = ((HipGpuBuffer)output).Handle;

        LaunchCsrSegmentedStdDevKernel(kernel, gridX, gridY, DefaultBlockSize, 1,
            colIndicesHandle, rowPointersHandle, inputHandle, outputHandle, M, K, N, epsilon);
    }

    private unsafe void LaunchCsrSegmentedKernel(IntPtr kernel, int gridX, int gridY, int blockX, int blockY,
        IntPtr colIndices, IntPtr rowPointers, IntPtr input, IntPtr output, int M, int K, int N)
    {
        void*[] args = new void*[7];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [colIndices, rowPointers, input, output];
            int[] ints = [M, K, N];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &h[2];
                argsPtr[3] = &h[3];
                argsPtr[4] = &i[0];
                argsPtr[5] = &i[1];
                argsPtr[6] = &i[2];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)gridX, (uint)gridY, 1,
                    (uint)blockX, (uint)blockY, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (csr_segmented)");
            }
        }
    }

    private unsafe void LaunchCsrSegmentedStdDevKernel(IntPtr kernel, int gridX, int gridY, int blockX, int blockY,
        IntPtr colIndices, IntPtr rowPointers, IntPtr input, IntPtr output, int M, int K, int N, float epsilon)
    {
        void*[] args = new void*[8];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [colIndices, rowPointers, input, output];
            int[] ints = [M, K, N];
            float epsCopy = epsilon;

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &h[2];
                argsPtr[3] = &h[3];
                argsPtr[4] = &i[0];
                argsPtr[5] = &i[1];
                argsPtr[6] = &i[2];
                argsPtr[7] = &epsCopy;

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)gridX, (uint)gridY, 1,
                    (uint)blockX, (uint)blockY, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (csr_segmented_stddev)");
            }
        }
    }

    private unsafe void LaunchKernel2D(IntPtr kernel, int gridX, int gridY, int blockX, int blockY,
        IntPtr values, IntPtr colIndices, IntPtr rowPointers, IntPtr denseB, IntPtr output,
        int M, int K, int N, int nnz)
    {
        void*[] args = new void*[9];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [values, colIndices, rowPointers, denseB, output];
            int[] ints = [M, K, N, nnz];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &h[2];
                argsPtr[3] = &h[3];
                argsPtr[4] = &h[4];
                argsPtr[5] = &i[0];
                argsPtr[6] = &i[1];
                argsPtr[7] = &i[2];
                argsPtr[8] = &i[3];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)gridX, (uint)gridY, 1,
                    (uint)blockX, (uint)blockY, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (csr_spmm)");
            }
        }
    }

    private unsafe void LaunchKernel2DBias(IntPtr kernel, int gridX, int gridY, int blockX, int blockY,
        IntPtr values, IntPtr colIndices, IntPtr rowPointers, IntPtr denseB, IntPtr bias, IntPtr output,
        int M, int K, int N, int nnz)
    {
        void*[] args = new void*[10];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [values, colIndices, rowPointers, denseB, bias, output];
            int[] ints = [M, K, N, nnz];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &h[2];
                argsPtr[3] = &h[3];
                argsPtr[4] = &h[4];
                argsPtr[5] = &h[5];
                argsPtr[6] = &i[0];
                argsPtr[7] = &i[1];
                argsPtr[8] = &i[2];
                argsPtr[9] = &i[3];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)gridX, (uint)gridY, 1,
                    (uint)blockX, (uint)blockY, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (csr_spmm_bias)");
            }
        }
    }

    private unsafe void LaunchKernel1D(IntPtr kernel, int grid, int block, IntPtr buffer, int size)
    {
        void*[] args = new void*[2];
        fixed (void** argsPtr = args)
        {
            IntPtr bufferCopy = buffer;
            int sizeCopy = size;
            argsPtr[0] = &bufferCopy;
            argsPtr[1] = &sizeCopy;

            var result = HipNativeBindings.hipModuleLaunchKernel(
                kernel,
                (uint)grid, 1, 1,
                (uint)block, 1, 1,
                0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
            HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (zero_buffer)");
        }
    }

    private unsafe void LaunchScatterAddKernel(IntPtr kernel, int gridX, int gridY, int blockX, int blockY,
        IntPtr input, IntPtr source, IntPtr target, IntPtr edgeValues, IntPtr output,
        int numNodes, int numEdges, int features, int hasEdgeValues)
    {
        void*[] args = new void*[9];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [input, source, target, edgeValues, output];
            int[] ints = [numNodes, numEdges, features, hasEdgeValues];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &h[2];
                argsPtr[3] = &h[3];
                argsPtr[4] = &h[4];
                argsPtr[5] = &i[0];
                argsPtr[6] = &i[1];
                argsPtr[7] = &i[2];
                argsPtr[8] = &i[3];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)gridX, (uint)gridY, 1,
                    (uint)blockX, (uint)blockY, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (scatter_add_edges)");
            }
        }
    }

    #endregion

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
        if (!_kernelCache.TryGetValue("reduce_sum", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: reduce_sum");

        return ReduceToScalar(krnl, A, size);
    }

    public float Max(IGpuBuffer A, int size)
    {
        if (!_kernelCache.TryGetValue("reduce_max", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: reduce_max");

        return ReduceToScalar(krnl, A, size);
    }

    private unsafe float ReduceToScalar(IntPtr kernel, IGpuBuffer input, int size)
    {
        const int blockSize = 256;
        int currentSize = size;
        IGpuBuffer currentInput = input;
        IGpuBuffer? tempBuffer1 = null;
        IGpuBuffer? tempBuffer2 = null;

        try
        {
            while (currentSize > 1)
            {
                int gridSize = (currentSize + blockSize - 1) / blockSize;
                int sharedMemSize = blockSize * sizeof(float);

                // Allocate output buffer for partial results
                IGpuBuffer output;
                if (tempBuffer1 is null)
                {
                    tempBuffer1 = AllocateBuffer(gridSize);
                    output = tempBuffer1;
                }
                else if (currentInput == tempBuffer1)
                {
                    if (tempBuffer2 is null || ((HipGpuBuffer)tempBuffer2).Size < gridSize)
                    {
                        tempBuffer2?.Dispose();
                        tempBuffer2 = AllocateBuffer(gridSize);
                    }
                    output = tempBuffer2;
                }
                else
                {
                    output = tempBuffer1;
                }

                var handles = new GCHandle[3];
                try
                {
                    handles[0] = GCHandle.Alloc(currentInput.Handle, GCHandleType.Pinned);
                    handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
                    handles[2] = GCHandle.Alloc(currentSize, GCHandleType.Pinned);

                    var args = new IntPtr[3];
                    for (int i = 0; i < 3; i++) args[i] = handles[i].AddrOfPinnedObject();

                    LaunchKernel(kernel, (uint)gridSize, (uint)blockSize, args, (uint)sharedMemSize);
                    Synchronize();
                }
                finally
                {
                    foreach (var h in handles) if (h.IsAllocated) h.Free();
                }

                currentInput = output;
                currentSize = gridSize;
            }

            // Download the single result
            var result = new float[1];
            var hipResult = (HipGpuBuffer)currentInput;
            GCHandle handle = GCHandle.Alloc(result, GCHandleType.Pinned);
            try
            {
                var res = HipNativeBindings.hipMemcpy(
                    handle.AddrOfPinnedObject(),
                    hipResult.Handle,
                    (UIntPtr)sizeof(float),
                    HipMemcpyKind.DeviceToHost);
                HipNativeBindings.CheckError(res, "hipMemcpy D2H");
            }
            finally
            {
                handle.Free();
            }

            return result[0];
        }
        finally
        {
            tempBuffer1?.Dispose();
            tempBuffer2?.Dispose();
        }
    }

    public unsafe void SumAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("sum_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sum_axis");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(outerSize, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(reduceSize, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region Convolution Operations

    public unsafe void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv2d_direct", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv2d_direct");

        var handles = new GCHandle[14];
        try
        {
            var pInput = input.Handle;
            var pKernel = kernel.Handle;
            var pOutput = output.Handle;

            handles[0] = GCHandle.Alloc(pInput, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pKernel, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pOutput, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(strideW, GCHandleType.Pinned);

            var args = new IntPtr[18];
            for (int i = 0; i < 14; i++) args[i] = handles[i].AddrOfPinnedObject();

            // Additional params
            var h14 = GCHandle.Alloc(padH, GCHandleType.Pinned); args[14] = h14.AddrOfPinnedObject();
            var h15 = GCHandle.Alloc(padW, GCHandleType.Pinned); args[15] = h15.AddrOfPinnedObject();
            var h16 = GCHandle.Alloc(dilationH, GCHandleType.Pinned); args[16] = h16.AddrOfPinnedObject();
            var h17 = GCHandle.Alloc(dilationW, GCHandleType.Pinned); args[17] = h17.AddrOfPinnedObject();

            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * outChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            h14.Free(); h15.Free(); h16.Free(); h17.Free();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void Conv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv2d_backward_input", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv2d_backward_input");

        var handles = new GCHandle[18];
        try
        {
            var pGradOut = gradOutput.Handle;
            var pKernel = kernel.Handle;
            var pGradIn = gradInput.Handle;

            handles[0] = GCHandle.Alloc(pGradOut, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pKernel, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pGradIn, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(padH, GCHandleType.Pinned);
            handles[15] = GCHandle.Alloc(padW, GCHandleType.Pinned);
            handles[16] = GCHandle.Alloc(dilationH, GCHandleType.Pinned);
            handles[17] = GCHandle.Alloc(dilationW, GCHandleType.Pinned);

            var args = new IntPtr[18];
            for (int i = 0; i < 18; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((inWidth + 15) / 16);
            uint gridY = (uint)((inHeight + 15) / 16);
            uint gridZ = (uint)(batch * inChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void Conv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv2d_backward_kernel", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv2d_backward_kernel");

        var handles = new GCHandle[18];
        try
        {
            var pInput = input.Handle;
            var pGradOut = gradOutput.Handle;
            var pGradKernel = gradKernel.Handle;

            handles[0] = GCHandle.Alloc(pInput, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pGradOut, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pGradKernel, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(padH, GCHandleType.Pinned);
            handles[15] = GCHandle.Alloc(padW, GCHandleType.Pinned);
            handles[16] = GCHandle.Alloc(dilationH, GCHandleType.Pinned);
            handles[17] = GCHandle.Alloc(dilationW, GCHandleType.Pinned);

            var args = new IntPtr[18];
            for (int i = 0; i < 18; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((kernelW + 15) / 16);
            uint gridY = (uint)((kernelH + 15) / 16);
            uint gridZ = (uint)(outChannels * inChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void Conv3D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inDepth, int inHeight, int inWidth,
        int outChannels, int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv3d_direct", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv3d_direct");

        var handles = new GCHandle[22];
        try
        {
            var pInput = input.Handle;
            var pKernel = kernel.Handle;
            var pOutput = output.Handle;

            handles[0] = GCHandle.Alloc(pInput, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pKernel, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pOutput, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inDepth, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outDepth, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(kernelD, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[15] = GCHandle.Alloc(strideD, GCHandleType.Pinned);
            handles[16] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[17] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[18] = GCHandle.Alloc(padD, GCHandleType.Pinned);
            handles[19] = GCHandle.Alloc(padH, GCHandleType.Pinned);
            handles[20] = GCHandle.Alloc(padW, GCHandleType.Pinned);
            handles[21] = GCHandle.Alloc(dilationD, GCHandleType.Pinned);

            var args = new IntPtr[25];
            for (int i = 0; i < 22; i++) args[i] = handles[i].AddrOfPinnedObject();

            var h22 = GCHandle.Alloc(dilationH, GCHandleType.Pinned); args[22] = h22.AddrOfPinnedObject();
            var h23 = GCHandle.Alloc(dilationW, GCHandleType.Pinned); args[23] = h23.AddrOfPinnedObject();

            uint gridX = (uint)((outWidth + 7) / 8);
            uint gridY = (uint)((outHeight + 7) / 8);
            uint gridZ = (uint)(batch * outChannels * outDepth);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 8, 8, 1, args);
            Synchronize();

            h22.Free(); h23.Free();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (!_kernelCache.TryGetValue("depthwise_conv2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: depthwise_conv2d");

        var handles = new GCHandle[14];
        try
        {
            var pInput = input.Handle;
            var pKernel = kernel.Handle;
            var pOutput = output.Handle;

            handles[0] = GCHandle.Alloc(pInput, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pKernel, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pOutput, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(padH, GCHandleType.Pinned);

            var args = new IntPtr[15];
            for (int i = 0; i < 14; i++) args[i] = handles[i].AddrOfPinnedObject();
            var h14 = GCHandle.Alloc(padW, GCHandleType.Pinned); args[14] = h14.AddrOfPinnedObject();

            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * channels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            h14.Free();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        if (!_kernelCache.TryGetValue("conv_transpose2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv_transpose2d");

        var handles = new GCHandle[15];
        try
        {
            var pInput = input.Handle;
            var pKernel = kernel.Handle;
            var pOutput = output.Handle;

            handles[0] = GCHandle.Alloc(pInput, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pKernel, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pOutput, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(padH, GCHandleType.Pinned);

            var args = new IntPtr[16];
            for (int i = 0; i < 15; i++) args[i] = handles[i].AddrOfPinnedObject();
            var h15 = GCHandle.Alloc(padW, GCHandleType.Pinned); args[15] = h15.AddrOfPinnedObject();

            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * outChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            h15.Free();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void ConvTranspose2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        // Fallback: CPU implementation (HIP kernel can be added later)
        var gradOutData = DownloadBuffer(gradOutput);
        var kernelData = DownloadBuffer(kernel);
        var gradInputData = new float[batch * inChannels * inHeight * inWidth];

        // Backward pass: dL/dX = conv(dL/dY, W) with padding adjustment
        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int ih = 0; ih < inHeight; ih++)
                {
                    for (int iw = 0; iw < inWidth; iw++)
                    {
                        float sum = 0;
                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int oh = ih * strideH - padH + kh;
                                    int ow = iw * strideW - padW + kw;
                                    if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth)
                                    {
                                        int goIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
                                        int kIdx = ((ic * outChannels + oc) * kernelH + kh) * kernelW + kw;
                                        sum += gradOutData[goIdx] * kernelData[kIdx];
                                    }
                                }
                            }
                        }
                        gradInputData[((b * inChannels + ic) * inHeight + ih) * inWidth + iw] = sum;
                    }
                }
            }
        }

        // Upload to GPU
        var handle = GCHandle.Alloc(gradInputData, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                gradInput.Handle,
                handle.AddrOfPinnedObject(),
                (UIntPtr)(gradInputData.Length * sizeof(float)),
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (ConvTranspose2DBackwardInput)");
        }
        finally
        {
            handle.Free();
        }
    }

    public void ConvTranspose2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        // Fallback: CPU implementation (HIP kernel can be added later)
        var inputData = DownloadBuffer(input);
        var gradOutData = DownloadBuffer(gradOutput);
        var gradKernelData = new float[inChannels * outChannels * kernelH * kernelW];

        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    for (int kh = 0; kh < kernelH; kh++)
                    {
                        for (int kw = 0; kw < kernelW; kw++)
                        {
                            float sum = 0;
                            for (int ih = 0; ih < inHeight; ih++)
                            {
                                for (int iw = 0; iw < inWidth; iw++)
                                {
                                    int oh = ih * strideH - padH + kh;
                                    int ow = iw * strideW - padW + kw;
                                    if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth)
                                    {
                                        int inIdx = ((b * inChannels + ic) * inHeight + ih) * inWidth + iw;
                                        int goIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
                                        sum += inputData[inIdx] * gradOutData[goIdx];
                                    }
                                }
                            }
                            int kIdx = ((ic * outChannels + oc) * kernelH + kh) * kernelW + kw;
                            gradKernelData[kIdx] += sum;
                        }
                    }
                }
            }
        }

        // Upload to GPU
        var handle = GCHandle.Alloc(gradKernelData, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                gradKernel.Handle,
                handle.AddrOfPinnedObject(),
                (UIntPtr)(gradKernelData.Length * sizeof(float)),
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (ConvTranspose2DBackwardKernel)");
        }
        finally
        {
            handle.Free();
        }
    }

    #endregion

    #region Pooling Operations

    public unsafe void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (!_kernelCache.TryGetValue("maxpool2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: maxpool2d");

        var handles = new GCHandle[15];
        try
        {
            var pInput = input.Handle;
            var pOutput = output.Handle;
            var pIndices = indices?.Handle ?? IntPtr.Zero;
            int saveIndices = indices is not null ? 1 : 0;

            handles[0] = GCHandle.Alloc(pInput, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pOutput, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pIndices, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(padH, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(padW, GCHandleType.Pinned);

            var args = new IntPtr[16];
            for (int i = 0; i < 15; i++) args[i] = handles[i].AddrOfPinnedObject();
            var h15 = GCHandle.Alloc(saveIndices, GCHandleType.Pinned); args[15] = h15.AddrOfPinnedObject();

            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * channels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            h15.Free();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void MaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (!_kernelCache.TryGetValue("maxpool2d_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: maxpool2d_backward");

        var handles = new GCHandle[11];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(indices.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);

            var args = new IntPtr[9];
            for (int i = 0; i < 9; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * channels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        if (!_kernelCache.TryGetValue("avgpool2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: avgpool2d");

        var handles = new GCHandle[14];
        try
        {
            int countPadInt = countIncludePad ? 1 : 0;

            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(padH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(padW, GCHandleType.Pinned);

            var args = new IntPtr[15];
            for (int i = 0; i < 14; i++) args[i] = handles[i].AddrOfPinnedObject();
            var h14 = GCHandle.Alloc(countPadInt, GCHandleType.Pinned); args[14] = h14.AddrOfPinnedObject();

            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * channels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            h14.Free();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        if (!_kernelCache.TryGetValue("avgpool2d_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: avgpool2d_backward");

        var handles = new GCHandle[14];
        try
        {
            int countPadInt = countIncludePad ? 1 : 0;

            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(padH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(padW, GCHandleType.Pinned);

            var args = new IntPtr[15];
            for (int i = 0; i < 14; i++) args[i] = handles[i].AddrOfPinnedObject();
            var h14 = GCHandle.Alloc(countPadInt, GCHandleType.Pinned); args[14] = h14.AddrOfPinnedObject();

            uint gridX = (uint)((inWidth + 15) / 16);
            uint gridY = (uint)((inHeight + 15) / 16);
            uint gridZ = (uint)(batch * channels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            h14.Free();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_avgpool2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: global_avgpool2d");

        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(height, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(width, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((batch * channels + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_maxpool2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: global_maxpool2d");

        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(height, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(width, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((batch * channels + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void GlobalAvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_avgpool2d_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: global_avgpool2d_backward");

        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(height, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(width, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            int totalElements = batch * channels * height * width;
            uint grid = (uint)((totalElements + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void GlobalMaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_maxpool2d_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: global_maxpool2d_backward");

        // First zero out the gradient input
        Fill(gradInput, 0f, batch * channels * height * width);

        var handles = new GCHandle[7];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(indices.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(height, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(width, GCHandleType.Pinned);

            var args = new IntPtr[7];
            for (int i = 0; i < 7; i++) args[i] = handles[i].AddrOfPinnedObject();

            int totalOutputs = batch * channels;
            uint grid = (uint)((totalOutputs + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        if (!_kernelCache.TryGetValue("adaptive_avgpool2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: adaptive_avgpool2d");

        var handles = new GCHandle[8];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);

            var args = new IntPtr[8];
            for (int i = 0; i < 8; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * channels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void MaxPool3D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("maxpool3d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: maxpool3d");

        int saveIndices = indices is not null ? 1 : 0;
        IntPtr indicesPtr = indices?.Handle ?? IntPtr.Zero;

        var handles = new GCHandle[18];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(indicesPtr, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inDepth, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outDepth, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(kernelD, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(strideD, GCHandleType.Pinned);
            handles[15] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[16] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[17] = GCHandle.Alloc(saveIndices, GCHandleType.Pinned);

            var args = new IntPtr[18];
            for (int i = 0; i < 18; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((outWidth + 7) / 8);
            uint gridY = (uint)((outHeight + 7) / 8);
            uint gridZ = (uint)(batch * channels * outDepth);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 8, 8, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth)
    {
        if (!_kernelCache.TryGetValue("maxpool3d_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: maxpool3d_backward");

        var handles = new GCHandle[11];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(indices.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inDepth, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outDepth, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);

            var args = new IntPtr[11];
            for (int i = 0; i < 11; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((outWidth + 7) / 8);
            uint gridY = (uint)((outHeight + 7) / 8);
            uint gridZ = (uint)(batch * channels * outDepth);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 8, 8, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void NearestNeighborUpsample3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        if (!_kernelCache.TryGetValue("nearest_upsample3d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: nearest_upsample3d");

        int outDepth = inDepth * scaleD;
        int outHeight = inHeight * scaleH;
        int outWidth = inWidth * scaleW;

        var handles = new GCHandle[10];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(inDepth, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(scaleD, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(scaleH, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(scaleW, GCHandleType.Pinned);

            var args = new IntPtr[10];
            for (int i = 0; i < 10; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((outWidth + 7) / 8);
            uint gridY = (uint)((outHeight + 7) / 8);
            uint gridZ = (uint)(batch * channels * outDepth);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 8, 8, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void NearestNeighborUpsample3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        if (!_kernelCache.TryGetValue("nearest_upsample3d_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: nearest_upsample3d_backward");

        int outDepth = inDepth * scaleD;
        int outHeight = inHeight * scaleH;
        int outWidth = inWidth * scaleW;

        var handles = new GCHandle[10];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(inDepth, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(scaleD, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(scaleH, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(scaleW, GCHandleType.Pinned);

            var args = new IntPtr[10];
            for (int i = 0; i < 10; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((outWidth + 7) / 8);
            uint gridY = (uint)((outHeight + 7) / 8);
            uint gridZ = (uint)(batch * channels * outDepth);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 8, 8, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region Normalization Operations

    public unsafe void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training)
    {
        if (!_kernelCache.TryGetValue("batchnorm_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: batchnorm_forward");

        var handles = new GCHandle[14];
        try
        {
            int trainingInt = training ? 1 : 0;
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gamma.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(beta.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(runningMean.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(runningVar.Handle, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(saveMean.Handle, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(saveInvVar.Handle, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(spatialSize, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(momentum, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(trainingInt, GCHandleType.Pinned);

            var args = new IntPtr[14];
            for (int i = 0; i < 14; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((channels + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("batchnorm_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: batchnorm_backward");

        var handles = new GCHandle[12];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gamma.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(saveMean.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(saveInvVar.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(gradGamma.Handle, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(gradBeta.Handle, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(spatialSize, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);

            var args = new IntPtr[12];
            for (int i = 0; i < 12; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((channels + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("layernorm_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: layernorm_forward");

        var handles = new GCHandle[9];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gamma.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(beta.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(saveMean.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(saveInvVar.Handle, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(normalizedSize, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);

            var args = new IntPtr[9];
            for (int i = 0; i < 9; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize, float epsilon)
    {
        // First pass: compute gradInput
        if (!_kernelCache.TryGetValue("layernorm_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: layernorm_backward");

        var handles = new GCHandle[11];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gamma.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(saveMean.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(saveInvVar.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(gradGamma.Handle, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(gradBeta.Handle, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(normalizedSize, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);

            var args = new IntPtr[11];
            for (int i = 0; i < 11; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }

        // Second pass: compute gradGamma and gradBeta
        if (_kernelCache.TryGetValue("layernorm_grad_params", out var gradKrnl))
        {
            var handles2 = new GCHandle[8];
            try
            {
                handles2[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
                handles2[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
                handles2[2] = GCHandle.Alloc(saveMean.Handle, GCHandleType.Pinned);
                handles2[3] = GCHandle.Alloc(saveInvVar.Handle, GCHandleType.Pinned);
                handles2[4] = GCHandle.Alloc(gradGamma.Handle, GCHandleType.Pinned);
                handles2[5] = GCHandle.Alloc(gradBeta.Handle, GCHandleType.Pinned);
                handles2[6] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
                handles2[7] = GCHandle.Alloc(normalizedSize, GCHandleType.Pinned);

                var args2 = new IntPtr[8];
                for (int i = 0; i < 8; i++) args2[i] = handles2[i].AddrOfPinnedObject();

                uint grid2 = (uint)((normalizedSize + DefaultBlockSize - 1) / DefaultBlockSize);
                LaunchKernel(gradKrnl, grid2, DefaultBlockSize, args2);
                Synchronize();
            }
            finally
            {
                foreach (var h in handles2) if (h.IsAllocated) h.Free();
            }
        }
    }

    public unsafe void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("groupnorm_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: groupnorm_forward");

        var handles = new GCHandle[11];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gamma.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(beta.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(saveMean.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(saveInvVar.Handle, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(numGroups, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(spatialSize, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);

            var args = new IntPtr[11];
            for (int i = 0; i < 11; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((batch * numGroups + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("instancenorm_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: instancenorm_forward");

        var handles = new GCHandle[10];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gamma.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(beta.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(saveMean.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(saveInvVar.Handle, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(channels, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(spatialSize, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);

            var args = new IntPtr[10];
            for (int i = 0; i < 10; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((batch * channels + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void InstanceNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        // Fallback: implement using CPU operations (same as CudaBackend fallback)
        var gradOutData = DownloadBuffer(gradOutput);
        var inputData = DownloadBuffer(input);
        var gammaData = DownloadBuffer(gamma);
        var meanData = DownloadBuffer(saveMean);
        var invVarData = DownloadBuffer(saveInvVar);
        var gradInputData = new float[gradOutData.Length];
        var gradGammaData = new float[channels];
        var gradBetaData = new float[channels];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int offset = (b * channels + c) * spatialSize;
                float meanVal = meanData[b * channels + c];
                float invStd = invVarData[b * channels + c];
                float g = gammaData[c];

                // First pass: compute sums for gradient correction
                float sumDelta = 0.0f;
                float sumDeltaXNorm = 0.0f;
                for (int s = 0; s < spatialSize; s++)
                {
                    float go = gradOutData[offset + s];
                    float x = inputData[offset + s];
                    float xNorm = (x - meanVal) * invStd;
                    float delta = go * g;

                    gradGammaData[c] += go * xNorm;
                    gradBetaData[c] += go;

                    sumDelta += delta;
                    sumDeltaXNorm += delta * xNorm;
                }

                // Second pass: compute gradInput with proper correction terms
                float invN = 1.0f / spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    float go = gradOutData[offset + s];
                    float x = inputData[offset + s];
                    float xNorm = (x - meanVal) * invStd;
                    float delta = go * g;

                    // dx = invStd * invN * (N * delta - sum(delta) - xNorm * sum(delta * xNorm))
                    gradInputData[offset + s] = invStd * invN * (spatialSize * delta - sumDelta - xNorm * sumDeltaXNorm);
                }
            }
        }

        // Upload results to GPU buffers using hipMemcpy
        var handle1 = GCHandle.Alloc(gradInputData, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                gradInput.Handle,
                handle1.AddrOfPinnedObject(),
                (UIntPtr)(gradInputData.Length * sizeof(float)),
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (InstanceNormBackward gradInput)");
        }
        finally
        {
            handle1.Free();
        }

        var handle2 = GCHandle.Alloc(gradGammaData, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                gradGamma.Handle,
                handle2.AddrOfPinnedObject(),
                (UIntPtr)(gradGammaData.Length * sizeof(float)),
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (InstanceNormBackward gradGamma)");
        }
        finally
        {
            handle2.Free();
        }

        var handle3 = GCHandle.Alloc(gradBetaData, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                gradBeta.Handle,
                handle3.AddrOfPinnedObject(),
                (UIntPtr)(gradBetaData.Length * sizeof(float)),
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (InstanceNormBackward gradBeta)");
        }
        finally
        {
            handle3.Free();
        }
    }

    public unsafe void RmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("rmsnorm_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: rmsnorm_forward");

        var handles = new GCHandle[7];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gamma.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(saveRms.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(normalizedSize, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);

            var args = new IntPtr[7];
            for (int i = 0; i < 7; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void RmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("rmsnorm_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: rmsnorm_backward");

        var handles = new GCHandle[9];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gamma.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(saveRms.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(gradGamma.Handle, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(normalizedSize, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);

            var args = new IntPtr[9];
            for (int i = 0; i < 9; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)batchSize;
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }

        // Compute gradGamma
        if (!_kernelCache.TryGetValue("rmsnorm_grad_gamma", out var krnl2))
            throw new InvalidOperationException("HIP kernel not found: rmsnorm_grad_gamma");

        var handles2 = new GCHandle[6];
        try
        {
            handles2[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles2[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles2[2] = GCHandle.Alloc(saveRms.Handle, GCHandleType.Pinned);
            handles2[3] = GCHandle.Alloc(gradGamma.Handle, GCHandleType.Pinned);
            handles2[4] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles2[5] = GCHandle.Alloc(normalizedSize, GCHandleType.Pinned);

            var args2 = new IntPtr[6];
            for (int i = 0; i < 6; i++) args2[i] = handles2[i].AddrOfPinnedObject();

            uint grid2 = (uint)((normalizedSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl2, grid2, DefaultBlockSize, args2);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles2) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region Dropout Operations

    public unsafe void Dropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask, int size, float dropoutRate, ulong seed, bool training)
    {
        if (!training)
        {
            // During inference, just copy input to output
            Copy(input, output, size);
            return;
        }

        // Generate mask using CPU and upload (GPU RNG would need additional kernel)
        float scale = 1.0f / (1.0f - dropoutRate);
        var rand = new Random((int)(seed % int.MaxValue));
        var maskData = new float[size];
        for (int i = 0; i < size; i++)
            maskData[i] = rand.NextDouble() > dropoutRate ? 1.0f : 0.0f;

        // Upload mask to GPU
        fixed (float* ptr = maskData)
        {
            var result = HipNativeBindings.hipMemcpy(
                mask.Handle, (IntPtr)ptr,
                (UIntPtr)(size * sizeof(float)),
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (mask)");
        }

        if (!_kernelCache.TryGetValue("dropout_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: dropout_forward");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(mask.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(scale, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void DropoutBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size, float dropoutRate)
    {
        if (!_kernelCache.TryGetValue("dropout_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: dropout_backward");

        float scale = 1.0f / (1.0f - dropoutRate);
        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(mask.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(scale, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region Embedding Operations

    public unsafe void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim)
    {
        if (!_kernelCache.TryGetValue("embedding_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: embedding_forward");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(indices.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(embeddingTable.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(numIndices, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(embeddingDim, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((numIndices + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void EmbeddingBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize)
    {
        if (!_kernelCache.TryGetValue("embedding_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: embedding_backward");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(indices.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradEmbedding.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(numIndices, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(embeddingDim, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((numIndices + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public IGpuBuffer AllocateIntBuffer(int size)
    {
        IntPtr devicePtr = IntPtr.Zero;
        var sizeBytes = (UIntPtr)(size * sizeof(int));

        var result = HipNativeBindings.hipMalloc(ref devicePtr, sizeBytes);
        HipNativeBindings.CheckError(result, "hipMalloc(int)");

        result = HipNativeBindings.hipMemset(devicePtr, 0, sizeBytes);
        HipNativeBindings.CheckError(result, "hipMemset(int)");

        return new HipGpuBuffer(devicePtr, size);
    }

    public IGpuBuffer AllocateIntBuffer(int[] data)
    {
        IntPtr devicePtr = IntPtr.Zero;
        var size = data.Length;
        var sizeBytes = (UIntPtr)(size * sizeof(int));

        var result = HipNativeBindings.hipMalloc(ref devicePtr, sizeBytes);
        HipNativeBindings.CheckError(result, "hipMalloc(int)");

        GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            result = HipNativeBindings.hipMemcpy(
                devicePtr,
                handle.AddrOfPinnedObject(),
                sizeBytes,
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (int)");
        }
        finally
        {
            handle.Free();
        }

        return new HipGpuBuffer(devicePtr, size);
    }

    #region Locally Connected Convolution Operations

    public unsafe void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: locally_connected_conv2d");

        var handles = new GCHandle[14];
        try
        {
            var pInput = input.Handle;
            var pWeights = weights.Handle;
            var pBias = bias?.Handle ?? IntPtr.Zero;
            var pOutput = output.Handle;
            int hasBias = bias != null ? 1 : 0;

            handles[0] = GCHandle.Alloc(pInput, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pWeights, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pBias, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(pOutput, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(strideH, GCHandleType.Pinned);

            var extraHandles = new GCHandle[2];
            extraHandles[0] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            extraHandles[1] = GCHandle.Alloc(hasBias, GCHandleType.Pinned);

            var args = new IntPtr[16];
            for (int i = 0; i < 14; i++) args[i] = handles[i].AddrOfPinnedObject();
            args[14] = extraHandles[0].AddrOfPinnedObject();
            args[15] = extraHandles[1].AddrOfPinnedObject();

            // Grid dimensions: outWidth x outHeight x (batch * outChannels)
            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * outChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            foreach (var h in extraHandles) if (h.IsAllocated) h.Free();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d_backward_input", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: locally_connected_conv2d_backward_input");

        var handles = new GCHandle[15];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(weights.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(strideW, GCHandleType.Pinned);

            var args = new IntPtr[14];
            for (int i = 0; i < 14; i++) args[i] = handles[i].AddrOfPinnedObject();

            int totalSize = batch * inChannels * inHeight * inWidth;
            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void LocallyConnectedConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d_backward_weights", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: locally_connected_conv2d_backward_weights");

        var handles = new GCHandle[15];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradWeights.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(strideW, GCHandleType.Pinned);

            var args = new IntPtr[14];
            for (int i = 0; i < 14; i++) args[i] = handles[i].AddrOfPinnedObject();

            int totalWeights = outHeight * outWidth * outChannels * inChannels * kernelH * kernelW;
            uint grid = (uint)((totalWeights + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void LocallyConnectedConv2DBackwardBias(IGpuBuffer gradOutput, IGpuBuffer gradBias,
        int batch, int outChannels, int outHeight, int outWidth)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d_backward_bias", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: locally_connected_conv2d_backward_bias");

        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(gradBias.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((outChannels + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
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
        if (!_kernelCache.TryGetValue("deformable_conv2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: deformable_conv2d");

        var handles = new GCHandle[22];
        try
        {
            var pInput = input.Handle;
            var pWeights = weights.Handle;
            var pOffsets = offsets.Handle;
            var pMask = mask?.Handle ?? IntPtr.Zero;
            var pOutput = output.Handle;
            int hasMask = mask != null ? 1 : 0;

            handles[0] = GCHandle.Alloc(pInput, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pWeights, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pOffsets, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(pMask, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(pOutput, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[15] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[16] = GCHandle.Alloc(padH, GCHandleType.Pinned);
            handles[17] = GCHandle.Alloc(padW, GCHandleType.Pinned);
            handles[18] = GCHandle.Alloc(dilationH, GCHandleType.Pinned);
            handles[19] = GCHandle.Alloc(dilationW, GCHandleType.Pinned);
            handles[20] = GCHandle.Alloc(groups, GCHandleType.Pinned);
            handles[21] = GCHandle.Alloc(deformGroups, GCHandleType.Pinned);

            var extraHandle = GCHandle.Alloc(hasMask, GCHandleType.Pinned);

            var args = new IntPtr[23];
            for (int i = 0; i < 22; i++) args[i] = handles[i].AddrOfPinnedObject();
            args[22] = extraHandle.AddrOfPinnedObject();

            // Grid dimensions: outWidth x outHeight x (batch * outChannels)
            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * outChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            extraHandle.Free();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_input", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: deformable_conv2d_backward_input");

        var handles = new GCHandle[22];
        try
        {
            var pGradOutput = gradOutput.Handle;
            var pWeights = weights.Handle;
            var pOffsets = offsets.Handle;
            var pMask = mask?.Handle ?? IntPtr.Zero;
            var pGradInput = gradInput.Handle;
            int hasMask = mask != null ? 1 : 0;

            handles[0] = GCHandle.Alloc(pGradOutput, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pWeights, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pOffsets, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(pMask, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(pGradInput, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[15] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[16] = GCHandle.Alloc(padH, GCHandleType.Pinned);
            handles[17] = GCHandle.Alloc(padW, GCHandleType.Pinned);
            handles[18] = GCHandle.Alloc(dilationH, GCHandleType.Pinned);
            handles[19] = GCHandle.Alloc(dilationW, GCHandleType.Pinned);
            handles[20] = GCHandle.Alloc(groups, GCHandleType.Pinned);
            handles[21] = GCHandle.Alloc(deformGroups, GCHandleType.Pinned);

            var extraHandle = GCHandle.Alloc(hasMask, GCHandleType.Pinned);

            var args = new IntPtr[23];
            for (int i = 0; i < 22; i++) args[i] = handles[i].AddrOfPinnedObject();
            args[22] = extraHandle.AddrOfPinnedObject();

            int totalInputSize = batch * inChannels * inHeight * inWidth;
            uint grid = (uint)((totalInputSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();

            extraHandle.Free();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void DeformableConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_weights", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: deformable_conv2d_backward_weights");

        var handles = new GCHandle[22];
        try
        {
            var pGradOutput = gradOutput.Handle;
            var pInput = input.Handle;
            var pOffsets = offsets.Handle;
            var pMask = mask?.Handle ?? IntPtr.Zero;
            var pGradWeights = gradWeights.Handle;
            int hasMask = mask != null ? 1 : 0;

            handles[0] = GCHandle.Alloc(pGradOutput, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pInput, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pOffsets, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(pMask, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(pGradWeights, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[15] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[16] = GCHandle.Alloc(padH, GCHandleType.Pinned);
            handles[17] = GCHandle.Alloc(padW, GCHandleType.Pinned);
            handles[18] = GCHandle.Alloc(dilationH, GCHandleType.Pinned);
            handles[19] = GCHandle.Alloc(dilationW, GCHandleType.Pinned);
            handles[20] = GCHandle.Alloc(groups, GCHandleType.Pinned);
            handles[21] = GCHandle.Alloc(deformGroups, GCHandleType.Pinned);

            var extraHandle = GCHandle.Alloc(hasMask, GCHandleType.Pinned);

            var args = new IntPtr[23];
            for (int i = 0; i < 22; i++) args[i] = handles[i].AddrOfPinnedObject();
            args[22] = extraHandle.AddrOfPinnedObject();

            int inChannelsPerGroup = inChannels / groups;
            int totalWeights = outChannels * inChannelsPerGroup * kernelH * kernelW;
            uint grid = (uint)((totalWeights + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();

            extraHandle.Free();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void DeformableConv2DBackwardOffset(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffsets,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_offset", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: deformable_conv2d_backward_offset");

        var handles = new GCHandle[24];
        try
        {
            var pGradOutput = gradOutput.Handle;
            var pInput = input.Handle;
            var pWeights = weights.Handle;
            var pOffsets = offsets.Handle;
            var pMask = mask?.Handle ?? IntPtr.Zero;
            var pGradOffsets = gradOffsets.Handle;
            int hasMask = mask != null ? 1 : 0;

            handles[0] = GCHandle.Alloc(pGradOutput, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pInput, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pWeights, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(pOffsets, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(pMask, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(pGradOffsets, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[15] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[16] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[17] = GCHandle.Alloc(padH, GCHandleType.Pinned);
            handles[18] = GCHandle.Alloc(padW, GCHandleType.Pinned);
            handles[19] = GCHandle.Alloc(dilationH, GCHandleType.Pinned);
            handles[20] = GCHandle.Alloc(dilationW, GCHandleType.Pinned);
            handles[21] = GCHandle.Alloc(groups, GCHandleType.Pinned);
            handles[22] = GCHandle.Alloc(deformGroups, GCHandleType.Pinned);
            handles[23] = GCHandle.Alloc(hasMask, GCHandleType.Pinned);

            var args = new IntPtr[24];
            for (int i = 0; i < 24; i++) args[i] = handles[i].AddrOfPinnedObject();

            // Grid for offset gradients: outWidth x outHeight x (batch * 2*kH*kW*deformGroups)
            int offsetChannels = 2 * kernelH * kernelW * deformGroups;
            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * offsetChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void DeformableConv2DBackwardMask(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer gradMask,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_mask", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: deformable_conv2d_backward_mask");

        var handles = new GCHandle[23];
        try
        {
            var pGradOutput = gradOutput.Handle;
            var pInput = input.Handle;
            var pWeights = weights.Handle;
            var pOffsets = offsets.Handle;
            var pGradMask = gradMask.Handle;

            handles[0] = GCHandle.Alloc(pGradOutput, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(pInput, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(pWeights, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(pOffsets, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(pGradMask, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(inChannels, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(inHeight, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(inWidth, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(outChannels, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(outHeight, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(outWidth, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(kernelH, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(kernelW, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(strideH, GCHandleType.Pinned);
            handles[15] = GCHandle.Alloc(strideW, GCHandleType.Pinned);
            handles[16] = GCHandle.Alloc(padH, GCHandleType.Pinned);
            handles[17] = GCHandle.Alloc(padW, GCHandleType.Pinned);
            handles[18] = GCHandle.Alloc(dilationH, GCHandleType.Pinned);
            handles[19] = GCHandle.Alloc(dilationW, GCHandleType.Pinned);
            handles[20] = GCHandle.Alloc(groups, GCHandleType.Pinned);
            handles[21] = GCHandle.Alloc(deformGroups, GCHandleType.Pinned);

            var args = new IntPtr[22];
            for (int i = 0; i < 22; i++) args[i] = handles[i].AddrOfPinnedObject();

            // Grid for mask gradients: outWidth x outHeight x (batch * kH*kW*deformGroups)
            int maskChannels = kernelH * kernelW * deformGroups;
            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * maskChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #endregion

    #region Attention Operations

    public unsafe void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        // Compute attention: softmax((Q @ K^T) * scale) @ V
        int totalBatches = batch * numHeads;
        int matrixSize = seqLen * headDim;

        // Allocate temporary buffers for Q @ K^T result (scores)
        using var scores = AllocateBuffer(new float[totalBatches * seqLen * seqLen]);
        using var keyT = AllocateBuffer(new float[totalBatches * headDim * seqLen]);

        // Transpose K for each batch
        BatchedTranspose(key, keyT, totalBatches, seqLen, headDim);

        // Q @ K^T for each batch
        for (int b = 0; b < totalBatches; b++)
        {
            int qOffset = b * matrixSize;
            int kOffset = b * matrixSize;
            int sOffset = b * seqLen * seqLen;

            // Use offset-based gemm or slice buffers
            using var qSlice = AllocateBuffer(new float[seqLen * headDim]);
            using var kSlice = AllocateBuffer(new float[headDim * seqLen]);
            using var sSlice = AllocateBuffer(new float[seqLen * seqLen]);

            Copy(query, qSlice, seqLen * headDim);
            Copy(keyT, kSlice, headDim * seqLen);
            Gemm(qSlice, kSlice, sSlice, seqLen, seqLen, headDim);
            Scale(sSlice, sSlice, scale, seqLen * seqLen);
            Softmax(sSlice, sSlice, seqLen, seqLen);

            // Store attention weights if requested
            if (attentionWeights is not null)
                Copy(sSlice, attentionWeights, seqLen * seqLen);

            // scores @ V
            using var outSlice = AllocateBuffer(new float[seqLen * headDim]);
            using var vSlice = AllocateBuffer(new float[seqLen * headDim]);
            Copy(value, vSlice, seqLen * headDim);
            Gemm(sSlice, vSlice, outSlice, seqLen, headDim, seqLen);
            Copy(outSlice, output, seqLen * headDim);
        }
    }

    public void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        // Simplified backward pass using stored attention weights
        int totalBatches = batch * numHeads;

        for (int b = 0; b < totalBatches; b++)
        {
            // gradValue = attentionWeights^T @ gradOutput
            using var attnT = AllocateBuffer(new float[seqLen * seqLen]);
            using var attnSlice = AllocateBuffer(new float[seqLen * seqLen]);
            using var gradOutSlice = AllocateBuffer(new float[seqLen * headDim]);
            using var gradVSlice = AllocateBuffer(new float[seqLen * headDim]);

            Copy(attentionWeights, attnSlice, seqLen * seqLen);
            Transpose(attnSlice, attnT, seqLen, seqLen);
            Copy(gradOutput, gradOutSlice, seqLen * headDim);
            Gemm(attnT, gradOutSlice, gradVSlice, seqLen, headDim, seqLen);
            Copy(gradVSlice, gradValue, seqLen * headDim);

            // gradQuery = (gradScores @ K) where gradScores = gradAttn * scale
            using var gradScores = AllocateBuffer(new float[seqLen * seqLen]);
            using var vSlice = AllocateBuffer(new float[seqLen * headDim]);
            Copy(value, vSlice, seqLen * headDim);
            using var vT = AllocateBuffer(new float[headDim * seqLen]);
            Transpose(vSlice, vT, seqLen, headDim);
            Gemm(gradOutSlice, vT, gradScores, seqLen, seqLen, headDim);

            // Apply softmax backward
            using var softmaxGrad = AllocateBuffer(new float[seqLen * seqLen]);
            SoftmaxBackward(gradScores, attnSlice, softmaxGrad, seqLen, seqLen);
            Scale(softmaxGrad, softmaxGrad, scale, seqLen * seqLen);

            // gradQuery = gradScores @ K
            using var kSlice = AllocateBuffer(new float[seqLen * headDim]);
            using var gradQSlice = AllocateBuffer(new float[seqLen * headDim]);
            Copy(key, kSlice, seqLen * headDim);
            Gemm(softmaxGrad, kSlice, gradQSlice, seqLen, headDim, seqLen);
            Copy(gradQSlice, gradQuery, seqLen * headDim);

            // gradKey = gradScores^T @ Q
            using var gradScoresT = AllocateBuffer(new float[seqLen * seqLen]);
            Transpose(softmaxGrad, gradScoresT, seqLen, seqLen);
            using var qSlice = AllocateBuffer(new float[seqLen * headDim]);
            using var gradKSlice = AllocateBuffer(new float[seqLen * headDim]);
            Copy(query, qSlice, seqLen * headDim);
            Gemm(gradScoresT, qSlice, gradKSlice, seqLen, headDim, seqLen);
            Copy(gradKSlice, gradKey, seqLen * headDim);
        }
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
        if (!_kernelCache.TryGetValue("flash_attention_v2", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: flash_attention_v2");

        int causalFlag = isCausal ? 1 : 0;
        var handles = new GCHandle[12];
        try
        {
            handles[0] = GCHandle.Alloc(query.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(key.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(value.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(softmaxStats.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(numHeads, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(seqQ, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(seqK, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(headDim, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(scale, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(causalFlag, GCHandleType.Pinned);

            var args = new IntPtr[12];
            for (int i = 0; i < 12; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((seqQ + 31) / 32);
            uint gridY = (uint)(batch * numHeads);
            LaunchKernel2D(krnl, gridX, gridY, 32, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        if (!_kernelCache.TryGetValue("flash_attention_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: flash_attention_backward");

        int causalFlag = isCausal ? 1 : 0;
        var handles = new GCHandle[16];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(query.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(key.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(value.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(softmaxStats.Handle, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(gradQuery.Handle, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(gradKey.Handle, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(gradValue.Handle, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(numHeads, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(seqQ, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(seqK, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(headDim, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(scale, GCHandleType.Pinned);
            handles[15] = GCHandle.Alloc(causalFlag, GCHandleType.Pinned);

            var args = new IntPtr[16];
            for (int i = 0; i < 16; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((seqQ + 63) / 64);
            uint gridY = (uint)(batch * numHeads);
            LaunchKernel2D(krnl, gridX, gridY, 64, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        if (!_kernelCache.TryGetValue("grouped_query_attention", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: grouped_query_attention");

        int queriesPerKV = numQHeads / numKVHeads;
        int causalFlag = isCausal ? 1 : 0;
        int storeWeights = attentionWeights != null ? 1 : 0;
        IntPtr wPtr = attentionWeights?.Handle ?? IntPtr.Zero;

        var handles = new GCHandle[15];
        try
        {
            handles[0] = GCHandle.Alloc(query.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(key.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(value.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(wPtr, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(numQHeads, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(numKVHeads, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(queriesPerKV, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(seqQ, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(seqK, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(headDim, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(scale, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(causalFlag, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(storeWeights, GCHandleType.Pinned);

            var args = new IntPtr[15];
            for (int i = 0; i < 15; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((seqQ + 31) / 32);
            uint gridY = (uint)(batch * numQHeads);
            LaunchKernel2D(krnl, gridX, gridY, 32, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale)
    {
        if (!_kernelCache.TryGetValue("grouped_query_attention_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: grouped_query_attention_backward");

        int queriesPerKV = numQHeads / numKVHeads;

        var handles = new GCHandle[16];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(query.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(key.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(value.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(attentionWeights.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(gradQuery.Handle, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(gradKey.Handle, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(gradValue.Handle, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(numQHeads, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(numKVHeads, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(queriesPerKV, GCHandleType.Pinned);
            handles[12] = GCHandle.Alloc(seqQ, GCHandleType.Pinned);
            handles[13] = GCHandle.Alloc(seqK, GCHandleType.Pinned);
            handles[14] = GCHandle.Alloc(headDim, GCHandleType.Pinned);
            handles[15] = GCHandle.Alloc(scale, GCHandleType.Pinned);

            var args = new IntPtr[16];
            for (int i = 0; i < 16; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((seqQ + 31) / 32);
            uint gridY = (uint)(batch * numQHeads);
            LaunchKernel2D(krnl, gridX, gridY, 32, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region Transpose and Reshape Operations

    public unsafe void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols)
    {
        if (!_kernelCache.TryGetValue("transpose_2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: transpose_2d");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(rows, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(cols, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((cols + 15) / 16);
            uint gridY = (uint)((rows + 15) / 16);
            LaunchKernel2D(krnl, gridX, gridY, 16, 16, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols)
    {
        if (!_kernelCache.TryGetValue("batched_transpose", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: batched_transpose");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(rows, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(cols, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint gridX = (uint)((cols + 15) / 16);
            uint gridY = (uint)((rows + 15) / 16);
            uint gridZ = (uint)batch;
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation)
    {
        int ndims = shape.Length;
        if (ndims == 2 && permutation[0] == 1 && permutation[1] == 0)
        {
            Transpose(input, output, shape[0], shape[1]);
            return;
        }

        if (!_kernelCache.TryGetValue("permute_general", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: permute_general");

        int totalSize = 1;
        for (int i = 0; i < ndims; i++) totalSize *= shape[i];

        int[] inputStrides = new int[ndims];
        inputStrides[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
            inputStrides[i] = inputStrides[i + 1] * shape[i + 1];

        int[] outputShape = new int[ndims];
        for (int i = 0; i < ndims; i++)
            outputShape[i] = shape[permutation[i]];

        int[] outputStrides = new int[ndims];
        outputStrides[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1];

        using var inputStridesBuffer = AllocateIntBuffer(inputStrides);
        using var outputStridesBuffer = AllocateIntBuffer(outputStrides);
        using var permutationBuffer = AllocateIntBuffer(permutation);

        var handles = new GCHandle[7];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(inputStridesBuffer.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(outputStridesBuffer.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(permutationBuffer.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(ndims, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(totalSize, GCHandleType.Pinned);

            var args = new IntPtr[7];
            for (int i = 0; i < 7; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void Copy(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        var sizeBytes = (UIntPtr)(size * sizeof(float));
        var result = HipNativeBindings.hipMemcpy(
            destination.Handle,
            source.Handle,
            sizeBytes,
            HipMemcpyKind.DeviceToDevice);
        HipNativeBindings.CheckError(result, "hipMemcpy D2D");
    }

    public void Fill(IGpuBuffer buffer, float value, int size)
    {
        // Download, fill, upload (HIP doesn't have direct memset for float patterns)
        var data = new float[size];
        for (int i = 0; i < size; i++)
            data[i] = value;
        UploadToBuffer(buffer, data);
    }

    /// <inheritdoc/>
    public unsafe void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows, int srcCols, int destTotalCols, int destColOffset)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("copy_2d_strided", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: copy_2d_strided");

        int gridX = (srcCols + DefaultBlockSize - 1) / DefaultBlockSize;
        int gridY = numRows;

        var srcHandle = ((HipGpuBuffer)source).Handle;
        var dstHandle = ((HipGpuBuffer)destination).Handle;

        void*[] args = new void*[6];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [srcHandle, dstHandle];
            int[] ints = [numRows, srcCols, destTotalCols, destColOffset];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &i[0];
                argsPtr[3] = &i[1];
                argsPtr[4] = &i[2];
                argsPtr[5] = &i[3];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)gridX, (uint)gridY, 1,
                    (uint)DefaultBlockSize, 1, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (copy_2d_strided)");
            }
        }
    }

    /// <inheritdoc/>
    public unsafe void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        // Check for the kernel, if not available fall back to CPU implementation
        if (!_kernelCache.TryGetValue("nearest_neighbor_upsample", out var kernel))
        {
            NearestNeighborUpsampleFallback(input, output, batchChannels, height, width, scaleFactor);
            return;
        }

        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        int outputSize = batchChannels * outHeight * outWidth;

        int grid = (outputSize + DefaultBlockSize - 1) / DefaultBlockSize;

        var srcHandle = ((HipGpuBuffer)input).Handle;
        var dstHandle = ((HipGpuBuffer)output).Handle;

        void*[] args = new void*[7];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [srcHandle, dstHandle];
            int[] ints = [batchChannels, height, width, scaleFactor, outputSize];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &i[0];
                argsPtr[3] = &i[1];
                argsPtr[4] = &i[2];
                argsPtr[5] = &i[3];
                argsPtr[6] = &i[4];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)grid, 1, 1,
                    (uint)DefaultBlockSize, 1, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (nearest_neighbor_upsample)");
            }
        }
    }

    /// <summary>
    /// CPU fallback for nearest-neighbor upsampling when kernel is not available.
    /// </summary>
    private void NearestNeighborUpsampleFallback(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        int outputSize = batchChannels * outHeight * outWidth;

        // Download input using existing method
        var inputData = DownloadBuffer(input);

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

        // Upload output using existing method
        UploadToBuffer(output, outputData);
    }

    /// <inheritdoc/>
    public unsafe void NearestNeighborUpsampleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        // First zero out the gradient input
        int inputSize = batchChannels * height * width;
        Fill(gradInput, 0f, inputSize);

        if (!_kernelCache.TryGetValue("nearest_neighbor_upsample_backward", out var kernel))
        {
            NearestNeighborUpsampleBackwardFallback(gradOutput, gradInput, batchChannels, height, width, scaleFactor);
            return;
        }

        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        int outputSize = batchChannels * outHeight * outWidth;

        int grid = (outputSize + DefaultBlockSize - 1) / DefaultBlockSize;

        var gradOutHandle = ((HipGpuBuffer)gradOutput).Handle;
        var gradInHandle = ((HipGpuBuffer)gradInput).Handle;

        void*[] args = new void*[7];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [gradOutHandle, gradInHandle];
            int[] ints = [batchChannels, height, width, scaleFactor, outputSize];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &i[0];
                argsPtr[3] = &i[1];
                argsPtr[4] = &i[2];
                argsPtr[5] = &i[3];
                argsPtr[6] = &i[4];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)grid, 1, 1,
                    (uint)DefaultBlockSize, 1, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (nearest_neighbor_upsample_backward)");
            }
        }
    }

    /// <summary>
    /// CPU fallback for nearest-neighbor upsampling backward when kernel is not available.
    /// </summary>
    private void NearestNeighborUpsampleBackwardFallback(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
    {
        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        int outputSize = batchChannels * outHeight * outWidth;
        int inputSize = batchChannels * height * width;

        // Download gradOutput
        var gradOutData = DownloadBuffer(gradOutput);

        // Accumulate gradients on CPU
        var gradInData = new float[inputSize];
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
                    gradInData[inputIdx] += gradOutData[outputIdx];
                }
            }
        }

        // Upload gradInput
        UploadToBuffer(gradInput, gradInData);
    }

    #endregion

    #region Activation Gradient Operations

    public unsafe void ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("relu_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: relu_backward");

        LaunchUnaryOp(krnl, gradOutput.Handle, input.Handle, gradInput.Handle, size);
    }

    public unsafe void SigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("sigmoid_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sigmoid_backward");

        LaunchUnaryOp(krnl, gradOutput.Handle, output.Handle, gradInput.Handle, size);
    }

    public unsafe void TanhBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("tanh_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: tanh_backward");

        LaunchUnaryOp(krnl, gradOutput.Handle, output.Handle, gradInput.Handle, size);
    }

    public unsafe void GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("gelu_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: gelu_backward");

        LaunchUnaryOp(krnl, gradOutput.Handle, input.Handle, gradInput.Handle, size);
    }

    public unsafe void SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
    {
        if (!_kernelCache.TryGetValue("softmax_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: softmax_backward");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(gradOutput.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(features, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region Helper Methods

    private unsafe void LaunchElementwiseOp(IntPtr krnl, IntPtr input, IntPtr output, int size)
    {
        var handles = new GCHandle[3];
        try
        {
            handles[0] = GCHandle.Alloc(input, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[3];
            for (int i = 0; i < 3; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    private unsafe void LaunchUnaryOp(IntPtr krnl, IntPtr a, IntPtr b, IntPtr c, int size)
    {
        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(a, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(b, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(c, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region Loss Function Operations

    public unsafe float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses)
    {
        if (!_kernelCache.TryGetValue("cross_entropy_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: cross_entropy_loss");

        using var lossBuffer = AllocateBuffer(new float[batchSize]);

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(predictions.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(targets.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(lossBuffer.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(numClasses, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }

        var lossData = new float[batchSize];
        DownloadBuffer(lossBuffer, lossData);
        float sum = 0;
        for (int i = 0; i < batchSize; i++) sum += lossData[i];
        return sum / batchSize;
    }

    public unsafe void CrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int batchSize, int numClasses)
    {
        if (!_kernelCache.TryGetValue("cross_entropy_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: cross_entropy_backward");

        int total = batchSize * numClasses;
        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(predictions.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(targets.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(numClasses, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe float BinaryCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (!_kernelCache.TryGetValue("bce_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: bce_loss");

        using var lossBuffer = AllocateBuffer(new float[size]);

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(predictions.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(targets.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(lossBuffer.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }

        var lossData = new float[size];
        DownloadBuffer(lossBuffer, lossData);
        float sum = 0;
        for (int i = 0; i < size; i++) sum += lossData[i];
        return sum / size;
    }

    public unsafe void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("bce_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: bce_backward");

        LaunchUnaryOp(krnl, predictions.Handle, targets.Handle, gradInput.Handle, size);
    }

    public unsafe float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (!_kernelCache.TryGetValue("mse_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: mse_loss");

        using var lossBuffer = AllocateBuffer(new float[size]);

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(predictions.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(targets.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(lossBuffer.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }

        var lossData = new float[size];
        DownloadBuffer(lossBuffer, lossData);
        float sum = 0;
        for (int i = 0; i < size; i++) sum += lossData[i];
        return sum / size;
    }

    public unsafe void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("mse_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: mse_backward");

        LaunchUnaryOp(krnl, predictions.Handle, targets.Handle, gradInput.Handle, size);
    }

    public unsafe float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta)
    {
        if (!_kernelCache.TryGetValue("smooth_l1_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: smooth_l1_loss");

        using var lossBuffer = AllocateBuffer(new float[size]);

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(predictions.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(targets.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(lossBuffer.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(beta, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }

        var lossData = new float[size];
        DownloadBuffer(lossBuffer, lossData);
        float sum = 0;
        for (int i = 0; i < size; i++) sum += lossData[i];
        return sum / size;
    }

    public unsafe void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta)
    {
        if (!_kernelCache.TryGetValue("smooth_l1_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: smooth_l1_backward");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(predictions.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(targets.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(gradInput.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(beta, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region Utility Operations

    public unsafe void Clamp(IGpuBuffer A, IGpuBuffer B, float min, float max, int size)
    {
        if (!_kernelCache.TryGetValue("clamp", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: clamp");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(min, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(max, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe float L2Norm(IGpuBuffer A, int size)
    {
        if (!_kernelCache.TryGetValue("l2_norm_squared", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: l2_norm_squared");

        using var squaredBuffer = AllocateBuffer(new float[size]);
        LaunchElementwiseOp(krnl, A.Handle, squaredBuffer.Handle, size);

        var data = new float[size];
        DownloadBuffer(squaredBuffer, data);
        float sum = 0;
        for (int i = 0; i < size; i++) sum += data[i];
        return (float)Math.Sqrt(sum);
    }

    public void ClipByValue(IGpuBuffer A, IGpuBuffer B, float clipValue, int size)
    {
        Clamp(A, B, -clipValue, clipValue, size);
    }

    public unsafe void ClipByNorm(IGpuBuffer A, IGpuBuffer B, float maxNorm, int size)
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
        // CPU fallback for scatter - no dedicated kernel
        var srcData = new float[sourceSize];
        var idxData = new float[sourceSize];
        var dstData = new float[destSize];

        DownloadBuffer(source, srcData);
        DownloadBuffer(indices, idxData);
        DownloadBuffer(destination, dstData);

        for (int i = 0; i < sourceSize; i++)
        {
            int idx = (int)idxData[i];
            if (idx >= 0 && idx < destSize)
                dstData[idx] += srcData[i];
        }

        UploadToBuffer(destination, dstData);
    }

    public unsafe void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource,
        int numIndices, int featureSize)
    {
        // ScatterAddBackward is essentially a Gather operation
        // Uses the embedding forward kernel since it's equivalent
        Embedding(indices, gradDestination, gradSource, numIndices, featureSize);
    }

    public unsafe void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize)
    {
        // Uses the embedding forward kernel since it's equivalent
        Embedding(indices, source, output, numIndices, featureSize);
    }

    #endregion

    #region Comparison Operations

    public unsafe void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("greater_than", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: greater_than");

        LaunchBinaryOp(krnl, A, B, C, size);
    }

    public unsafe void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("less_than", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: less_than");

        LaunchBinaryOp(krnl, A, B, C, size);
    }

    public unsafe void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("equals", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: equals");

        LaunchBinaryOp(krnl, A, B, C, size);
    }

    public unsafe void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("where_cond", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: where_cond");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(condition.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(C.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size)
    {
        if (!_kernelCache.TryGetValue("not_equal_scalar", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: not_equal_scalar");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(C.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(scalar, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    private unsafe void LaunchBinaryOp(IntPtr krnl, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(C.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region Statistics Operations

    public unsafe void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("sum_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sum_axis");

        // First sum, then divide
        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(outerSize, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(reduceSize, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }

        // Divide by reduceSize to get mean
        Scale(B, B, 1.0f / reduceSize, outerSize);
    }

    public unsafe void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("compute_mean_var", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: compute_mean_var");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(mean.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(variance.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(outerSize, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(reduceSize, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((reduceSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("argmax_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: argmax_axis");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(indices.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(outerSize, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(reduceSize, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("argmin_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: argmin_axis");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(indices.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(outerSize, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(reduceSize, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void MaxAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("max_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: max_axis");

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(outerSize, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(reduceSize, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void TopK(IGpuBuffer A, IGpuBuffer values, IGpuBuffer indices, int outerSize, int reduceSize, int k, bool sorted = true)
    {
        // TODO: Implement HIP TopK kernel
        throw new NotImplementedException("TopK kernel not yet implemented for HIP backend.");
    }

    public void AffineGrid(IGpuBuffer theta, IGpuBuffer grid, int batch, int outputHeight, int outputWidth)
    {
        // TODO: Implement HIP AffineGrid kernel
        throw new NotImplementedException("AffineGrid kernel not yet implemented for HIP backend.");
    }

    public void GridSample(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        // TODO: Implement HIP GridSample kernel
        throw new NotImplementedException("GridSample kernel not yet implemented for HIP backend.");
    }

    public void GridSampleBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
        IGpuBuffer gradInput, IGpuBuffer gradGrid,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        // TODO: Implement HIP GridSampleBackward kernel
        throw new NotImplementedException("GridSampleBackward kernel not yet implemented for HIP backend.");
    }

    public unsafe void BroadcastMultiplyLastAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        if (!_kernelCache.TryGetValue("broadcast_multiply_last_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: broadcast_multiply_last_axis");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(C.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(outerSize, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(innerSize, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            int totalSize = outerSize * innerSize;
            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public unsafe void BroadcastMultiplyFirstAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        if (!_kernelCache.TryGetValue("broadcast_multiply_first_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: broadcast_multiply_first_axis");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(A.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(B.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(C.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(outerSize, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(innerSize, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            int totalSize = outerSize * innerSize;
            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region FFT and Signal Processing

    /// <summary>
    /// Performs in-place FFT or IFFT using the Cooley-Tukey radix-2 algorithm.
    /// </summary>
    public unsafe void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        // Copy input to output buffers for in-place operation
        HipCopyBuffer(inputReal, outputReal, n);
        HipCopyBuffer(inputImag, outputImag, n);

        int log2n = (int)Math.Log(n, 2);

        // Bit-reversal permutation
        if (_kernelCache.TryGetValue("bit_reverse_permutation", out var bitRevKernel))
        {
            uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
            var handles = new GCHandle[4];
            try
            {
                handles[0] = GCHandle.Alloc(outputReal.Handle, GCHandleType.Pinned);
                handles[1] = GCHandle.Alloc(outputImag.Handle, GCHandleType.Pinned);
                handles[2] = GCHandle.Alloc(n, GCHandleType.Pinned);
                handles[3] = GCHandle.Alloc(log2n, GCHandleType.Pinned);

                var args = new IntPtr[]
                {
                    handles[0].AddrOfPinnedObject(),
                    handles[1].AddrOfPinnedObject(),
                    handles[2].AddrOfPinnedObject(),
                    handles[3].AddrOfPinnedObject()
                };

                LaunchKernel(bitRevKernel, gridSize, (uint)DefaultBlockSize, args);
            }
            finally
            {
                foreach (var h in handles) if (h.IsAllocated) h.Free();
            }
        }

        // Butterfly stages
        if (_kernelCache.TryGetValue("fft_butterfly", out var butterflyKernel))
        {
            int inverseFlag = inverse ? 1 : 0;
            for (int stride = 2; stride <= n; stride *= 2)
            {
                int numButterflies = n / 2;
                uint gridSize = (uint)((numButterflies + DefaultBlockSize - 1) / DefaultBlockSize);

                var handles = new GCHandle[5];
                try
                {
                    handles[0] = GCHandle.Alloc(outputReal.Handle, GCHandleType.Pinned);
                    handles[1] = GCHandle.Alloc(outputImag.Handle, GCHandleType.Pinned);
                    handles[2] = GCHandle.Alloc(n, GCHandleType.Pinned);
                    handles[3] = GCHandle.Alloc(stride, GCHandleType.Pinned);
                    handles[4] = GCHandle.Alloc(inverseFlag, GCHandleType.Pinned);

                    var args = new IntPtr[]
                    {
                        handles[0].AddrOfPinnedObject(),
                        handles[1].AddrOfPinnedObject(),
                        handles[2].AddrOfPinnedObject(),
                        handles[3].AddrOfPinnedObject(),
                        handles[4].AddrOfPinnedObject()
                    };

                    LaunchKernel(butterflyKernel, gridSize, (uint)DefaultBlockSize, args);
                }
                finally
                {
                    foreach (var h in handles) if (h.IsAllocated) h.Free();
                }
            }
        }

        // Scale by 1/N for inverse FFT
        if (inverse && _kernelCache.TryGetValue("scale_inverse", out var scaleKernel))
        {
            uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
            var handles = new GCHandle[3];
            try
            {
                handles[0] = GCHandle.Alloc(outputReal.Handle, GCHandleType.Pinned);
                handles[1] = GCHandle.Alloc(outputImag.Handle, GCHandleType.Pinned);
                handles[2] = GCHandle.Alloc(n, GCHandleType.Pinned);

                var args = new IntPtr[]
                {
                    handles[0].AddrOfPinnedObject(),
                    handles[1].AddrOfPinnedObject(),
                    handles[2].AddrOfPinnedObject()
                };

                LaunchKernel(scaleKernel, gridSize, (uint)DefaultBlockSize, args);
            }
            finally
            {
                foreach (var h in handles) if (h.IsAllocated) h.Free();
            }
        }

        Synchronize();
    }

    /// <summary>
    /// Real-to-complex FFT exploiting conjugate symmetry.
    /// </summary>
    public unsafe void RFFT(IGpuBuffer input, IGpuBuffer outputReal, IGpuBuffer outputImag, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        // Allocate temporary buffers for full complex FFT
        using var tempReal = AllocateBuffer(n);
        using var tempImag = AllocateBuffer(n);

        // Copy real input to tempReal, zero tempImag
        HipCopyBuffer(input, tempReal, n);
        HipZeroBuffer(tempImag, n);

        // Perform full FFT
        FFT(tempReal, tempImag, tempReal, tempImag, n, inverse: false);

        // Post-process to extract positive frequencies (n/2 + 1 elements)
        if (_kernelCache.TryGetValue("rfft_postprocess", out var postprocessKernel))
        {
            int outLen = n / 2 + 1;
            uint gridSize = (uint)((outLen + DefaultBlockSize - 1) / DefaultBlockSize);

            var handles = new GCHandle[5];
            try
            {
                handles[0] = GCHandle.Alloc(tempReal.Handle, GCHandleType.Pinned);
                handles[1] = GCHandle.Alloc(tempImag.Handle, GCHandleType.Pinned);
                handles[2] = GCHandle.Alloc(outputReal.Handle, GCHandleType.Pinned);
                handles[3] = GCHandle.Alloc(outputImag.Handle, GCHandleType.Pinned);
                handles[4] = GCHandle.Alloc(n, GCHandleType.Pinned);

                var args = new IntPtr[]
                {
                    handles[0].AddrOfPinnedObject(),
                    handles[1].AddrOfPinnedObject(),
                    handles[2].AddrOfPinnedObject(),
                    handles[3].AddrOfPinnedObject(),
                    handles[4].AddrOfPinnedObject()
                };

                LaunchKernel(postprocessKernel, gridSize, (uint)DefaultBlockSize, args);
            }
            finally
            {
                foreach (var h in handles) if (h.IsAllocated) h.Free();
            }
        }

        Synchronize();
    }

    /// <summary>
    /// Complex-to-real IFFT exploiting conjugate symmetry.
    /// </summary>
    public unsafe void IRFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer output, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        // Allocate temporary buffers for full complex FFT
        using var tempReal = AllocateBuffer(n);
        using var tempImag = AllocateBuffer(n);

        // Pre-process to reconstruct negative frequencies
        if (_kernelCache.TryGetValue("irfft_preprocess", out var preprocessKernel))
        {
            uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

            var handles = new GCHandle[5];
            try
            {
                handles[0] = GCHandle.Alloc(inputReal.Handle, GCHandleType.Pinned);
                handles[1] = GCHandle.Alloc(inputImag.Handle, GCHandleType.Pinned);
                handles[2] = GCHandle.Alloc(tempReal.Handle, GCHandleType.Pinned);
                handles[3] = GCHandle.Alloc(tempImag.Handle, GCHandleType.Pinned);
                handles[4] = GCHandle.Alloc(n, GCHandleType.Pinned);

                var args = new IntPtr[]
                {
                    handles[0].AddrOfPinnedObject(),
                    handles[1].AddrOfPinnedObject(),
                    handles[2].AddrOfPinnedObject(),
                    handles[3].AddrOfPinnedObject(),
                    handles[4].AddrOfPinnedObject()
                };

                LaunchKernel(preprocessKernel, gridSize, (uint)DefaultBlockSize, args);
            }
            finally
            {
                foreach (var h in handles) if (h.IsAllocated) h.Free();
            }
        }

        // Perform full inverse FFT
        using var outputImag = AllocateBuffer(n);
        FFT(tempReal, tempImag, output, outputImag, n, inverse: true);
    }

    /// <summary>
    /// Batched FFT for processing multiple signals in parallel.
    /// </summary>
    public unsafe void BatchedFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int batch, int n, bool inverse)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        // Copy input to output buffers for in-place operation
        HipCopyBuffer(inputReal, outputReal, batch * n);
        HipCopyBuffer(inputImag, outputImag, batch * n);

        int log2n = (int)Math.Log(n, 2);

        // Batched bit-reversal permutation
        if (_kernelCache.TryGetValue("batched_bit_reverse", out var bitRevKernel))
        {
            uint gridX = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
            uint gridY = (uint)batch;

            var handles = new GCHandle[5];
            try
            {
                handles[0] = GCHandle.Alloc(outputReal.Handle, GCHandleType.Pinned);
                handles[1] = GCHandle.Alloc(outputImag.Handle, GCHandleType.Pinned);
                handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
                handles[3] = GCHandle.Alloc(n, GCHandleType.Pinned);
                handles[4] = GCHandle.Alloc(log2n, GCHandleType.Pinned);

                var args = new IntPtr[]
                {
                    handles[0].AddrOfPinnedObject(),
                    handles[1].AddrOfPinnedObject(),
                    handles[2].AddrOfPinnedObject(),
                    handles[3].AddrOfPinnedObject(),
                    handles[4].AddrOfPinnedObject()
                };

                LaunchKernel2D(bitRevKernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
            }
            finally
            {
                foreach (var h in handles) if (h.IsAllocated) h.Free();
            }
        }

        // Batched butterfly stages
        if (_kernelCache.TryGetValue("batched_fft_butterfly", out var butterflyKernel))
        {
            int inverseFlag = inverse ? 1 : 0;
            for (int stride = 2; stride <= n; stride *= 2)
            {
                int numButterflies = n / 2;
                uint gridX = (uint)((numButterflies + DefaultBlockSize - 1) / DefaultBlockSize);
                uint gridZ = (uint)batch;

                var handles = new GCHandle[6];
                try
                {
                    handles[0] = GCHandle.Alloc(outputReal.Handle, GCHandleType.Pinned);
                    handles[1] = GCHandle.Alloc(outputImag.Handle, GCHandleType.Pinned);
                    handles[2] = GCHandle.Alloc(batch, GCHandleType.Pinned);
                    handles[3] = GCHandle.Alloc(n, GCHandleType.Pinned);
                    handles[4] = GCHandle.Alloc(stride, GCHandleType.Pinned);
                    handles[5] = GCHandle.Alloc(inverseFlag, GCHandleType.Pinned);

                    var args = new IntPtr[]
                    {
                        handles[0].AddrOfPinnedObject(),
                        handles[1].AddrOfPinnedObject(),
                        handles[2].AddrOfPinnedObject(),
                        handles[3].AddrOfPinnedObject(),
                        handles[4].AddrOfPinnedObject(),
                        handles[5].AddrOfPinnedObject()
                    };

                    LaunchKernel3D(butterflyKernel, gridX, 1, gridZ, (uint)DefaultBlockSize, 1, 1, args);
                }
                finally
                {
                    foreach (var h in handles) if (h.IsAllocated) h.Free();
                }
            }
        }

        // Scale by 1/N for inverse FFT (batched)
        if (inverse && _kernelCache.TryGetValue("scale_inverse", out var scaleKernel))
        {
            int total = batch * n;
            uint gridSize = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
            var handles = new GCHandle[3];
            try
            {
                handles[0] = GCHandle.Alloc(outputReal.Handle, GCHandleType.Pinned);
                handles[1] = GCHandle.Alloc(outputImag.Handle, GCHandleType.Pinned);
                handles[2] = GCHandle.Alloc(total, GCHandleType.Pinned);

                var args = new IntPtr[]
                {
                    handles[0].AddrOfPinnedObject(),
                    handles[1].AddrOfPinnedObject(),
                    handles[2].AddrOfPinnedObject()
                };

                LaunchKernel(scaleKernel, gridSize, (uint)DefaultBlockSize, args);
            }
            finally
            {
                foreach (var h in handles) if (h.IsAllocated) h.Free();
            }
        }

        Synchronize();
    }

    /// <summary>
    /// 2D FFT using row-column decomposition.
    /// </summary>
    public unsafe void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int height, int width, bool inverse)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        int total = height * width;
        int log2Width = (int)Math.Log(width, 2);
        int log2Height = (int)Math.Log(height, 2);

        // Copy input to output buffers for in-place operation
        HipCopyBuffer(inputReal, outputReal, total);
        HipCopyBuffer(inputImag, outputImag, total);

        // Row-wise bit reversal and FFT
        if (_kernelCache.TryGetValue("bit_reverse_permutation", out var bitRevKernel) &&
            _kernelCache.TryGetValue("fft_rows_butterfly", out var rowsButterflyKernel))
        {
            // Bit reversal for each row (using batched approach)
            for (int row = 0; row < height; row++)
            {
                int offset = row * width;
                // Note: For production, we should use offset buffers or batched kernel
                // This is a simplified version that operates row by row
            }

            int inverseFlag = inverse ? 1 : 0;
            for (int stride = 2; stride <= width; stride *= 2)
            {
                int numButterflies = width / 2;
                uint gridX = (uint)((numButterflies + DefaultBlockSize - 1) / DefaultBlockSize);
                uint gridY = (uint)height;

                var handles = new GCHandle[6];
                try
                {
                    handles[0] = GCHandle.Alloc(outputReal.Handle, GCHandleType.Pinned);
                    handles[1] = GCHandle.Alloc(outputImag.Handle, GCHandleType.Pinned);
                    handles[2] = GCHandle.Alloc(height, GCHandleType.Pinned);
                    handles[3] = GCHandle.Alloc(width, GCHandleType.Pinned);
                    handles[4] = GCHandle.Alloc(stride, GCHandleType.Pinned);
                    handles[5] = GCHandle.Alloc(inverseFlag, GCHandleType.Pinned);

                    var args = new IntPtr[]
                    {
                        handles[0].AddrOfPinnedObject(),
                        handles[1].AddrOfPinnedObject(),
                        handles[2].AddrOfPinnedObject(),
                        handles[3].AddrOfPinnedObject(),
                        handles[4].AddrOfPinnedObject(),
                        handles[5].AddrOfPinnedObject()
                    };

                    LaunchKernel2D(rowsButterflyKernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
                }
                finally
                {
                    foreach (var h in handles) if (h.IsAllocated) h.Free();
                }
            }
        }

        // Column-wise FFT
        if (_kernelCache.TryGetValue("fft_cols_butterfly", out var colsButterflyKernel))
        {
            int inverseFlag = inverse ? 1 : 0;
            for (int stride = 2; stride <= height; stride *= 2)
            {
                int numButterflies = height / 2;
                uint gridX = (uint)((numButterflies + DefaultBlockSize - 1) / DefaultBlockSize);
                uint gridY = (uint)width;

                var handles = new GCHandle[6];
                try
                {
                    handles[0] = GCHandle.Alloc(outputReal.Handle, GCHandleType.Pinned);
                    handles[1] = GCHandle.Alloc(outputImag.Handle, GCHandleType.Pinned);
                    handles[2] = GCHandle.Alloc(height, GCHandleType.Pinned);
                    handles[3] = GCHandle.Alloc(width, GCHandleType.Pinned);
                    handles[4] = GCHandle.Alloc(stride, GCHandleType.Pinned);
                    handles[5] = GCHandle.Alloc(inverseFlag, GCHandleType.Pinned);

                    var args = new IntPtr[]
                    {
                        handles[0].AddrOfPinnedObject(),
                        handles[1].AddrOfPinnedObject(),
                        handles[2].AddrOfPinnedObject(),
                        handles[3].AddrOfPinnedObject(),
                        handles[4].AddrOfPinnedObject(),
                        handles[5].AddrOfPinnedObject()
                    };

                    LaunchKernel2D(colsButterflyKernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
                }
                finally
                {
                    foreach (var h in handles) if (h.IsAllocated) h.Free();
                }
            }
        }

        // Scale by 1/(height*width) for inverse FFT
        if (inverse && _kernelCache.TryGetValue("scale_inverse", out var scaleKernel))
        {
            uint gridSize = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
            var handles = new GCHandle[3];
            try
            {
                handles[0] = GCHandle.Alloc(outputReal.Handle, GCHandleType.Pinned);
                handles[1] = GCHandle.Alloc(outputImag.Handle, GCHandleType.Pinned);
                handles[2] = GCHandle.Alloc(total, GCHandleType.Pinned);

                var args = new IntPtr[]
                {
                    handles[0].AddrOfPinnedObject(),
                    handles[1].AddrOfPinnedObject(),
                    handles[2].AddrOfPinnedObject()
                };

                LaunchKernel(scaleKernel, gridSize, (uint)DefaultBlockSize, args);
            }
            finally
            {
                foreach (var h in handles) if (h.IsAllocated) h.Free();
            }
        }

        Synchronize();
    }

    /// <summary>
    /// Applies a window function element-wise.
    /// </summary>
    public unsafe void ApplyWindow(IGpuBuffer input, IGpuBuffer window, IGpuBuffer output, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("apply_window", out var kernel))
            throw new InvalidOperationException("apply_window kernel not found");

        uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(window.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(n, GCHandleType.Pinned);

            var args = new IntPtr[]
            {
                handles[0].AddrOfPinnedObject(),
                handles[1].AddrOfPinnedObject(),
                handles[2].AddrOfPinnedObject(),
                handles[3].AddrOfPinnedObject()
            };

            LaunchKernel(kernel, gridSize, (uint)DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <summary>
    /// Computes magnitude from complex numbers: magnitude = sqrt(real^2 + imag^2).
    /// </summary>
    public unsafe void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("complex_magnitude", out var kernel))
            throw new InvalidOperationException("complex_magnitude kernel not found");

        uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(real.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(imag.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(magnitude.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(n, GCHandleType.Pinned);

            var args = new IntPtr[]
            {
                handles[0].AddrOfPinnedObject(),
                handles[1].AddrOfPinnedObject(),
                handles[2].AddrOfPinnedObject(),
                handles[3].AddrOfPinnedObject()
            };

            LaunchKernel(kernel, gridSize, (uint)DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <summary>
    /// Computes phase from complex numbers: phase = atan2(imag, real).
    /// </summary>
    public unsafe void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("complex_phase", out var kernel))
            throw new InvalidOperationException("complex_phase kernel not found");

        uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(real.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(imag.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(phase.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(n, GCHandleType.Pinned);

            var args = new IntPtr[]
            {
                handles[0].AddrOfPinnedObject(),
                handles[1].AddrOfPinnedObject(),
                handles[2].AddrOfPinnedObject(),
                handles[3].AddrOfPinnedObject()
            };

            LaunchKernel(kernel, gridSize, (uint)DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <summary>
    /// Converts polar coordinates to complex: real = mag*cos(phase), imag = mag*sin(phase).
    /// </summary>
    public unsafe void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("polar_to_complex", out var kernel))
            throw new InvalidOperationException("polar_to_complex kernel not found");

        uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(magnitude.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(phase.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(real.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(imag.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(n, GCHandleType.Pinned);

            var args = new IntPtr[]
            {
                handles[0].AddrOfPinnedObject(),
                handles[1].AddrOfPinnedObject(),
                handles[2].AddrOfPinnedObject(),
                handles[3].AddrOfPinnedObject(),
                handles[4].AddrOfPinnedObject()
            };

            LaunchKernel(kernel, gridSize, (uint)DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <summary>
    /// Applies Mel filterbank to power spectrum.
    /// </summary>
    public unsafe void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec, int numFrames, int numFreqs, int nMels)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("apply_mel_filterbank", out var kernel))
            throw new InvalidOperationException("apply_mel_filterbank kernel not found");

        uint gridX = (uint)numFrames;
        uint gridY = (uint)((nMels + DefaultBlockSize - 1) / DefaultBlockSize);

        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(powerSpec.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(filterbank.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(melSpec.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(numFrames, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(numFreqs, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(nMels, GCHandleType.Pinned);

            var args = new IntPtr[]
            {
                handles[0].AddrOfPinnedObject(),
                handles[1].AddrOfPinnedObject(),
                handles[2].AddrOfPinnedObject(),
                handles[3].AddrOfPinnedObject(),
                handles[4].AddrOfPinnedObject(),
                handles[5].AddrOfPinnedObject()
            };

            LaunchKernel2D(kernel, gridX, gridY, 1, (uint)DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <summary>
    /// Converts power spectrum to decibels.
    /// </summary>
    public unsafe void PowerToDb(IGpuBuffer power, IGpuBuffer db, int n, float refValue, float minDb)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("power_to_db", out var kernel))
            throw new InvalidOperationException("power_to_db kernel not found");

        uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(power.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(db.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(n, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(refValue, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(minDb, GCHandleType.Pinned);

            var args = new IntPtr[]
            {
                handles[0].AddrOfPinnedObject(),
                handles[1].AddrOfPinnedObject(),
                handles[2].AddrOfPinnedObject(),
                handles[3].AddrOfPinnedObject(),
                handles[4].AddrOfPinnedObject()
            };

            LaunchKernel(kernel, gridSize, (uint)DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <summary>
    /// Converts decibels to power spectrum.
    /// </summary>
    public unsafe void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("db_to_power", out var kernel))
            throw new InvalidOperationException("db_to_power kernel not found");

        uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(db.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(power.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(n, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(refValue, GCHandleType.Pinned);

            var args = new IntPtr[]
            {
                handles[0].AddrOfPinnedObject(),
                handles[1].AddrOfPinnedObject(),
                handles[2].AddrOfPinnedObject(),
                handles[3].AddrOfPinnedObject()
            };

            LaunchKernel(kernel, gridSize, (uint)DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <summary>
    /// Helper method to copy GPU buffer contents.
    /// </summary>
    private void HipCopyBuffer(IGpuBuffer source, IGpuBuffer destination, int count)
    {
        var size = (UIntPtr)(count * sizeof(float));
        var result = HipNativeBindings.hipMemcpy(
            destination.Handle,
            source.Handle,
            size,
            HipMemcpyKind.DeviceToDevice);
        HipNativeBindings.CheckError(result, "hipMemcpy D2D");
    }

    /// <summary>
    /// Helper method to zero GPU buffer contents.
    /// </summary>
    private void HipZeroBuffer(IGpuBuffer buffer, int count)
    {
        var size = (UIntPtr)(count * sizeof(float));
        var result = HipNativeBindings.hipMemset(buffer.Handle, 0, size);
        HipNativeBindings.CheckError(result, "hipMemset");
    }

    #endregion

    #region IAsyncGpuBackend Methods

    /// <inheritdoc/>
    public IGpuStream CreateStream(GpuStreamType streamType)
    {
        return new HipStream(this, streamType);
    }

    /// <inheritdoc/>
    public IGpuStream CreateStream(GpuStreamType streamType, int priority)
    {
        return new HipStream(this, streamType, priority);
    }

    /// <inheritdoc/>
    public IGpuEvent CreateEvent()
    {
        return new HipEvent(this, null, true);
    }

    /// <inheritdoc/>
    public IGpuEvent CreateEvent(bool enableTiming)
    {
        return new HipEvent(this, null, enableTiming);
    }

    /// <inheritdoc/>
    public void RecordEvent(IGpuEvent gpuEvent, IGpuStream stream)
    {
        if (gpuEvent is not HipEvent hipEvent)
        {
            throw new ArgumentException("Event must be a HipEvent", nameof(gpuEvent));
        }

        if (stream is not HipStream)
        {
            throw new ArgumentException("Stream must be a HipStream", nameof(stream));
        }

        hipEvent.Record(stream);
    }

    /// <inheritdoc/>
    public void StreamWaitEvent(IGpuStream stream, IGpuEvent gpuEvent)
    {
        if (stream is not HipStream hipStream)
        {
            throw new ArgumentException("Stream must be a HipStream", nameof(stream));
        }

        hipStream.WaitEvent(gpuEvent);
    }

    /// <inheritdoc/>
    public GpuSyncPoint CreateSyncPoint(IGpuStream stream)
    {
        if (stream is not HipStream hipStream)
        {
            throw new ArgumentException("Stream must be a HipStream", nameof(stream));
        }

        return new HipSyncPoint(this, hipStream);
    }

    /// <inheritdoc/>
    public GpuSyncPoint CreateSyncPoint()
    {
        return CreateSyncPoint(DefaultStream);
    }

    /// <inheritdoc/>
    public void UploadBufferAsync(float[] data, IGpuBuffer buffer, IGpuStream stream)
    {
        var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            var size = (UIntPtr)(data.Length * sizeof(float));
            var result = HipNativeBindings.hipMemcpyAsync(
                buffer.Handle,
                handle.AddrOfPinnedObject(),
                size,
                HipMemcpyKind.HostToDevice,
                stream.Handle);
            HipNativeBindings.CheckError(result, "hipMemcpyAsync H2D");
            // Synchronize stream to ensure transfer completes before freeing the pinned handle
            var syncResult = HipNativeBindings.hipStreamSynchronize(stream.Handle);
            HipNativeBindings.CheckError(syncResult, "hipStreamSynchronize");
        }
        finally
        {
            handle.Free();
        }
    }

    /// <inheritdoc/>
    public void UploadBufferAsync(ReadOnlySpan<float> data, IGpuBuffer buffer, IGpuStream stream)
    {
        // Copy to array for pinning (ReadOnlySpan can't be pinned directly)
        var array = data.ToArray();
        UploadBufferAsync(array, buffer, stream);
    }

    /// <inheritdoc/>
    public void DownloadBufferAsync(IGpuBuffer buffer, float[] destination, IGpuStream stream)
    {
        var handle = GCHandle.Alloc(destination, GCHandleType.Pinned);
        try
        {
            var size = (UIntPtr)(destination.Length * sizeof(float));
            var result = HipNativeBindings.hipMemcpyAsync(
                handle.AddrOfPinnedObject(),
                buffer.Handle,
                size,
                HipMemcpyKind.DeviceToHost,
                stream.Handle);
            HipNativeBindings.CheckError(result, "hipMemcpyAsync D2H");
            // Synchronize stream to ensure transfer completes before freeing the pinned handle
            var syncResult = HipNativeBindings.hipStreamSynchronize(stream.Handle);
            HipNativeBindings.CheckError(syncResult, "hipStreamSynchronize");
        }
        finally
        {
            handle.Free();
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
        var byteSize = (UIntPtr)(size * sizeof(float));
        var result = HipNativeBindings.hipMemcpyAsync(
            destination.Handle,
            source.Handle,
            byteSize,
            HipMemcpyKind.DeviceToDevice,
            stream.Handle);
        HipNativeBindings.CheckError(result, "hipMemcpyAsync D2D");
    }

    /// <inheritdoc/>
    public void GemmAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha, float beta, IGpuStream stream)
    {
        GemmOnStream(A, B, C, M, N, K, alpha, beta, stream.Handle, synchronize: false);
    }

    /// <inheritdoc/>
    public void FusedGemmBiasActivationAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output,
        int M, int N, int K, FusedActivationType activation, IGpuStream stream)
    {
        // Map activation type to appropriate kernel
        string kernelName = activation switch
        {
            FusedActivationType.ReLU => "gemm_bias_relu",
            FusedActivationType.Sigmoid => "gemm_bias_sigmoid",
            FusedActivationType.Tanh => "gemm_bias_tanh",
            FusedActivationType.None => "gemm_bias",
            _ => "gemm_bias_relu" // Default to ReLU
        };

        ExecuteFusedGemmOnStream(kernelName, A, B, bias, output, M, N, K, stream.Handle, synchronize: false);
    }

    /// <summary>
    /// Executes GEMM on a specific stream with optional synchronization.
    /// </summary>
    private void GemmOnStream(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha, float beta, IntPtr stream, bool synchronize)
    {
        var bufferA = (HipGpuBuffer)A;
        var bufferB = (HipGpuBuffer)B;
        var bufferC = (HipGpuBuffer)C;

        IntPtr kernel = SelectGemmKernel(M, N, K);
        if (kernel == IntPtr.Zero)
        {
            throw new InvalidOperationException("No suitable GEMM kernel available");
        }

        GemmKernelType kernelType = GetKernelType(kernel);

        int tileM, tileN, blockSize;
        switch (kernelType)
        {
            case GemmKernelType.Mfma:
                tileM = 128;
                tileN = 128;
                blockSize = 256;
                break;
            case GemmKernelType.RdnaWave32:
                tileM = 32;
                tileN = 32;
                blockSize = 256;
                break;
            case GemmKernelType.Scalar:
            default:
                tileM = 16;
                tileN = 16;
                blockSize = 256;
                break;
        }

        uint gridDimX = (uint)((M + tileM - 1) / tileM);
        uint gridDimY = (uint)((N + tileN - 1) / tileN);

        IntPtr[] kernelArgs = new IntPtr[8];
        GCHandle[] handles = new GCHandle[8];

        try
        {
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

            GCHandle kernelParamsHandle = GCHandle.Alloc(kernelArgs, GCHandleType.Pinned);
            try
            {
                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    gridDimX, gridDimY, 1,
                    (uint)blockSize, 1, 1,
                    0,
                    stream,
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

        if (synchronize)
        {
            var syncResult = HipNativeBindings.hipStreamSynchronize(stream);
            HipNativeBindings.CheckError(syncResult, "hipStreamSynchronize");
        }
    }

    /// <inheritdoc/>
    public void SynchronizeStream(IGpuStream stream)
    {
        if (stream is not HipStream)
        {
            throw new ArgumentException("Stream must be a HipStream", nameof(stream));
        }

        stream.Synchronize();
    }

    /// <inheritdoc/>
    public bool QueryStreamComplete(IGpuStream stream)
    {
        if (stream is not HipStream)
        {
            throw new ArgumentException("Stream must be a HipStream", nameof(stream));
        }

        return stream.Query();
    }

    /// <inheritdoc/>
    public bool QueryEventComplete(IGpuEvent gpuEvent)
    {
        if (gpuEvent is not HipEvent)
        {
            throw new ArgumentException("Event must be a HipEvent", nameof(gpuEvent));
        }

        return gpuEvent.Query();
    }

    /// <inheritdoc/>
    public float GetEventElapsedTime(IGpuEvent start, IGpuEvent end)
    {
        if (end is not HipEvent hipEnd)
        {
            throw new ArgumentException("Event must be a HipEvent", nameof(end));
        }

        return hipEnd.GetElapsedTime(start);
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

    public void Copy(IGpuBuffer source, int sourceOffset, IGpuBuffer destination, int destinationOffset, int length)
    {
        throw new NotImplementedException("Strided copy not implemented for HIP backend yet.");
    }

    public void ArgMaxAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        throw new NotImplementedException("ArgMaxAxis not implemented for HIP backend yet.");
    }

    public void GenerateRandomUniform(IGpuBuffer output, int size, float min, float max, ulong seed)
    {
        throw new NotImplementedException("GenerateRandomUniform not implemented for HIP backend yet.");
    }

    public void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
    {
        throw new NotImplementedException("GenerateRandomNormal not implemented for HIP backend yet.");
    }

    public void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output, int batchSize, int numCenters, int inputDim)
    {
        throw new NotImplementedException("RbfForward not implemented for HIP backend yet.");
    }

    public void StdpUpdate(IGpuBuffer weights, IGpuBuffer preTrace, IGpuBuffer postTrace, IGpuBuffer preSpike, IGpuBuffer postSpike,
        float ltpRate, float ltdRate, float homeostasisRate, float minWeight, float maxWeight, int numPre, int numPost)
    {
        throw new NotImplementedException("StdpUpdate not implemented for HIP backend yet.");
    }

    public void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input, float decay, float threshold, int size)
    {
        throw new NotImplementedException("UpdateTraces not implemented for HIP backend yet.");
    }

    #region Hyperbolic Geometry Operations

    public void PoincareProject(IGpuBuffer input, IGpuBuffer output, int batchSize, int dim, float curvature, float epsilon = 1e-5f)
    {
        if (!_kernelCache.TryGetValue("poincare_project", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: poincare_project");

        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(dim, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(curvature, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void MobiusAdd(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        if (!_kernelCache.TryGetValue("mobius_add", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: mobius_add");

        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(x.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(y.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(dim, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(curvature, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void PoincareExpMap(IGpuBuffer basePoint, IGpuBuffer tangentVec, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        if (!_kernelCache.TryGetValue("poincare_exp_map", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: poincare_exp_map");

        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(basePoint.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(tangentVec.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(dim, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(curvature, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void PoincareDistance(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        if (!_kernelCache.TryGetValue("poincare_distance", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: poincare_distance");

        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        var handles = new GCHandle[6];
        try
        {
            handles[0] = GCHandle.Alloc(x.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(y.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(dim, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(curvature, GCHandleType.Pinned);

            var args = new IntPtr[6];
            for (int i = 0; i < 6; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void HyperbolicLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
    {
        if (!_kernelCache.TryGetValue("hyperbolic_linear_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: hyperbolic_linear_forward");

        int totalThreads = batchSize * outputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
        var handles = new GCHandle[9];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(weights.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(biases.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inputFeatures, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(outputFeatures, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(curvature, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);

            var args = new IntPtr[9];
            for (int i = 0; i < 9; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region Octonion Algebra Operations

    public void OctonionMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        if (!_kernelCache.TryGetValue("octonion_multiply", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: octonion_multiply");

        uint grid = (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize);
        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(a.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(b.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(count, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void OctonionAdd(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        if (!_kernelCache.TryGetValue("octonion_add", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: octonion_add");

        int totalElements = count * 8;
        uint grid = (uint)((totalElements + DefaultBlockSize - 1) / DefaultBlockSize);
        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(a.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(b.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(count, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void OctonionLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        if (!_kernelCache.TryGetValue("octonion_linear_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: octonion_linear_forward");

        int totalThreads = batchSize * outputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
        var handles = new GCHandle[7];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(weights.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(biases.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(inputFeatures, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(outputFeatures, GCHandleType.Pinned);

            var args = new IntPtr[7];
            for (int i = 0; i < 7; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    #region Quantum Computing Operations

    public void QuantumMeasurement(IGpuBuffer realPart, IGpuBuffer imagPart, IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        if (!_kernelCache.TryGetValue("quantum_measurement", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: quantum_measurement");

        int totalElements = batchSize * stateSize;
        uint grid = (uint)((totalElements + DefaultBlockSize - 1) / DefaultBlockSize);
        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(realPart.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(imagPart.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(probabilities.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(stateSize, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void NormalizeProbabilities(IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        if (!_kernelCache.TryGetValue("normalize_probabilities", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: normalize_probabilities");

        uint sharedMemSize = (uint)(DefaultBlockSize * sizeof(float));
        var handles = new GCHandle[3];
        try
        {
            handles[0] = GCHandle.Alloc(probabilities.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(stateSize, GCHandleType.Pinned);

            var args = new IntPtr[3];
            for (int i = 0; i < 3; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, (uint)batchSize, DefaultBlockSize, args, sharedMemSize);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void ComplexMatVec(IGpuBuffer matReal, IGpuBuffer matImag, IGpuBuffer vecReal, IGpuBuffer vecImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batchSize, int dim)
    {
        if (!_kernelCache.TryGetValue("complex_matvec", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: complex_matvec");

        int totalElements = batchSize * dim;
        uint grid = (uint)((totalElements + DefaultBlockSize - 1) / DefaultBlockSize);
        var handles = new GCHandle[8];
        try
        {
            handles[0] = GCHandle.Alloc(matReal.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(matImag.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(vecReal.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(vecImag.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(outReal.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(outImag.Handle, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(dim, GCHandleType.Pinned);

            var args = new IntPtr[8];
            for (int i = 0; i < 8; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void QuantumRotation(IGpuBuffer stateReal, IGpuBuffer stateImag, IGpuBuffer outReal, IGpuBuffer outImag,
        IGpuBuffer angles, int numQubits, int batchSize)
    {
        if (!_kernelCache.TryGetValue("quantum_rotation", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: quantum_rotation");

        var handles = new GCHandle[7];
        try
        {
            handles[0] = GCHandle.Alloc(stateReal.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(stateImag.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(outReal.Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(outImag.Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(angles.Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(numQubits, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);

            var args = new IntPtr[7];
            for (int i = 0; i < 7; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, (uint)batchSize, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    public void MeasurementForward(IGpuBuffer input, IGpuBuffer output, int batchSize, int stateSize)
    {
        if (!_kernelCache.TryGetValue("measurement_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: measurement_forward");

        uint sharedMemSize = (uint)(DefaultBlockSize * sizeof(float));
        var handles = new GCHandle[4];
        try
        {
            handles[0] = GCHandle.Alloc(input.Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(output.Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(batchSize, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(stateSize, GCHandleType.Pinned);

            var args = new IntPtr[4];
            for (int i = 0; i < 4; i++) args[i] = handles[i].AddrOfPinnedObject();

            LaunchKernel(kernel, (uint)batchSize, DefaultBlockSize, args, sharedMemSize);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;

        // Dispose the default stream wrapper (does not destroy underlying stream)
        _defaultStream?.Dispose();
        _defaultStream = null;

        // Unload all kernel modules
        if (_mfmaModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_mfmaModule);
            _mfmaModule = IntPtr.Zero;
        }
        if (_activationModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_activationModule);
            _activationModule = IntPtr.Zero;
        }
        if (_neuralNetModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_neuralNetModule);
            _neuralNetModule = IntPtr.Zero;
        }
        if (_convolutionModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_convolutionModule);
            _convolutionModule = IntPtr.Zero;
        }
        if (_poolingModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_poolingModule);
            _poolingModule = IntPtr.Zero;
        }
        if (_normalizationModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_normalizationModule);
            _normalizationModule = IntPtr.Zero;
        }
        if (_fusedModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_fusedModule);
            _fusedModule = IntPtr.Zero;
        }
        if (_attentionModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_attentionModule);
            _attentionModule = IntPtr.Zero;
        }
        if (_fftModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_fftModule);
            _fftModule = IntPtr.Zero;
        }
        if (_locallyConnectedModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_locallyConnectedModule);
            _locallyConnectedModule = IntPtr.Zero;
        }
        if (_deformableConvModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_deformableConvModule);
            _deformableConvModule = IntPtr.Zero;
        }
        if (_optimizerModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_optimizerModule);
            _optimizerModule = IntPtr.Zero;
        }

        if (_stream != IntPtr.Zero)
        {
            HipNativeBindings.hipStreamDestroy(_stream);
            _stream = IntPtr.Zero;
        }

        _kernelCache.Clear();
        _disposed = true;
    }

    #region Optimizer Operations

    /// <inheritdoc/>
    public unsafe void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("sgd_momentum_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sgd_momentum_update");

        var handles = new GCHandle[7];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)velocity).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(momentum, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[7];
            for (int i = 0; i < 7; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void SgdUpdate(IGpuBuffer param, IGpuBuffer gradient,
        float learningRate, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("sgd_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sgd_update");

        var handles = new GCHandle[5];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[5];
            for (int i = 0; i < 5; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("adam_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: adam_update");

        var handles = new GCHandle[11];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)m).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(((HipGpuBuffer)v).Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(beta1, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(beta2, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(step, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[11];
            for (int i = 0; i < 11; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("adamw_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: adamw_update");

        var handles = new GCHandle[11];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)m).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(((HipGpuBuffer)v).Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(beta1, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(beta2, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(step, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[11];
            for (int i = 0; i < 11; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void RmspropUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer squaredAvg,
        float learningRate, float rho, float epsilon, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("rmsprop_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: rmsprop_update");

        var handles = new GCHandle[8];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)squaredAvg).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(rho, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[8];
            for (int i = 0; i < 8; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void AdagradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumulatedGrad,
        float learningRate, float epsilon, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("adagrad_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: adagrad_update");

        var handles = new GCHandle[7];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)accumulatedGrad).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[7];
            for (int i = 0; i < 7; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void NagUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("nag_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: nag_update");

        var handles = new GCHandle[7];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)velocity).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(momentum, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[7];
            for (int i = 0; i < 7; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void LarsUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
    {
        if (!_kernelCache.TryGetValue("lars_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: lars_update");

        var handles = new GCHandle[8];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)velocity).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(momentum, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(trustCoeff, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[8];
            for (int i = 0; i < 8; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("lamb_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: lamb_update");

        var handles = new GCHandle[11];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)m).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(((HipGpuBuffer)v).Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(beta1, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(beta2, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(step, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[11];
            for (int i = 0; i < 11; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void AdadeltaUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        float rho, float epsilon, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("adadelta_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: adadelta_update");

        var handles = new GCHandle[8];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)accumGrad).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(((HipGpuBuffer)accumUpdate).Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(rho, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[8];
            for (int i = 0; i < 8; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void AmsgradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("amsgrad_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: amsgrad_update");

        var handles = new GCHandle[12];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)m).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(((HipGpuBuffer)v).Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(((HipGpuBuffer)vMax).Handle, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(beta1, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(beta2, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(step, GCHandleType.Pinned);
            handles[11] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[12];
            for (int i = 0; i < 12; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void AdamaxUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer u,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("adamax_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: adamax_update");

        var handles = new GCHandle[11];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)m).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(((HipGpuBuffer)u).Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(beta1, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(beta2, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(step, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[11];
            for (int i = 0; i < 11; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void LionUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m,
        float learningRate, float beta1, float beta2, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("lion_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: lion_update");

        var handles = new GCHandle[8];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)m).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(beta1, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(beta2, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[8];
            for (int i = 0; i < 8; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void NadamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("nadam_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: nadam_update");

        var handles = new GCHandle[11];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)m).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(((HipGpuBuffer)v).Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(beta1, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(beta2, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(epsilon, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(weightDecay, GCHandleType.Pinned);
            handles[9] = GCHandle.Alloc(step, GCHandleType.Pinned);
            handles[10] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[11];
            for (int i = 0; i < 11; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void FtrlUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer z, IGpuBuffer n,
        float learningRate, float l1Reg, float l2Reg, float beta, int size)
    {
        if (!_kernelCache.TryGetValue("ftrl_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: ftrl_update");

        var handles = new GCHandle[9];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)param).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)gradient).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(((HipGpuBuffer)z).Handle, GCHandleType.Pinned);
            handles[3] = GCHandle.Alloc(((HipGpuBuffer)n).Handle, GCHandleType.Pinned);
            handles[4] = GCHandle.Alloc(learningRate, GCHandleType.Pinned);
            handles[5] = GCHandle.Alloc(l1Reg, GCHandleType.Pinned);
            handles[6] = GCHandle.Alloc(l2Reg, GCHandleType.Pinned);
            handles[7] = GCHandle.Alloc(beta, GCHandleType.Pinned);
            handles[8] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[9];
            for (int i = 0; i < 9; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
    {
        // HIP doesn't have a built-in conversion kernel in our current set
        // For now, do a simple copy. In production, this would use a proper FP16 conversion kernel.
        // The mixed precision workflow typically handles this at a higher level.
        if (!_kernelCache.TryGetValue("convert_fp32_to_fp16", out var krnl))
        {
            // Fallback: just copy the data as-is (both buffers are float, this is a no-op placeholder)
            Copy(input, output, size);
            return;
        }

        var handles = new GCHandle[3];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)input).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)output).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[3];
            for (int i = 0; i < 3; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <inheritdoc/>
    public unsafe void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
    {
        // HIP doesn't have a built-in conversion kernel in our current set
        // For now, do a simple copy. In production, this would use a proper FP32 conversion kernel.
        if (!_kernelCache.TryGetValue("convert_fp16_to_fp32", out var krnl))
        {
            // Fallback: just copy the data as-is (both buffers are float, this is a no-op placeholder)
            Copy(input, output, size);
            return;
        }

        var handles = new GCHandle[3];
        try
        {
            handles[0] = GCHandle.Alloc(((HipGpuBuffer)input).Handle, GCHandleType.Pinned);
            handles[1] = GCHandle.Alloc(((HipGpuBuffer)output).Handle, GCHandleType.Pinned);
            handles[2] = GCHandle.Alloc(size, GCHandleType.Pinned);

            var args = new IntPtr[3];
            for (int i = 0; i < 3; i++) args[i] = handles[i].AddrOfPinnedObject();

            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
        }
        finally
        {
            foreach (var h in handles) if (h.IsAllocated) h.Free();
        }
    }

    #endregion
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
