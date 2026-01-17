// Copyright (c) AiDotNet. All rights reserved.
// OpenCL backend using pure P/Invoke - no managed GPU runtime dependency.
// Works on ALL .NET versions including .NET Framework 4.6.2.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// OpenCL backend for direct GPU access on AMD, Intel, and NVIDIA GPUs.
    /// Uses pure P/Invoke with no managed GPU runtime dependency.
    /// </summary>
    /// <remarks>
    /// <para><b>Key Features:</b></para>
    /// <list type="bullet">
    /// <item>Works on ALL .NET versions (4.6.2, 4.7.1, net8.0, etc.)</item>
    /// <item>No managed GPU runtime dependency - pure P/Invoke</item>
    /// <item>Double-buffered GEMM for compute/memory overlap</item>
    /// <item>Fused operations (GEMM+Bias+Activation)</item>
    /// <item>Bank-conflict-free shared memory</item>
    /// </list>
    /// </remarks>
    public sealed class OpenClBackend : IAsyncGpuBackend
    {
        private DirectOpenClContext? _context;
        private readonly Dictionary<string, DirectOpenClKernel> _kernelCache;
        private readonly List<DirectOpenClProgram> _programs;
        private DynamicGemmKernel? _dynamicGemm;
        private bool _disposed;
        private OpenClCommandQueue? _defaultStream;
        private const string OfflineTuningEnvVar = "AIDOTNET_GPU_TUNE";
        private const string OfflineTuningTrialsEnvVar = "AIDOTNET_GPU_TUNE_TRIALS";
        private const string OfflineTuningDiagEnvVar = "AIDOTNET_GPU_TUNE_DIAG";
        private const string OfflineTuningLogEnvVar = "AIDOTNET_GPU_TUNE_LOG";
        private const string OfflineTuningCsvEnvVar = "AIDOTNET_GPU_TUNE_CSV";
        private const string OfflineTuningProgressEnvVar = "AIDOTNET_GPU_TUNE_PROGRESS";
        private const string OfflineTuningResetEnvVar = "AIDOTNET_GPU_TUNE_RESET";
        private const string OfflineTuningHeartbeatEnvVar = "AIDOTNET_GPU_TUNE_HEARTBEAT";
        private const int DefaultBayesianTrials = 500;
        private readonly Dictionary<(int M, int N, int K), GemmConfig> _tunedConfigCache = new();
        private readonly object _tunedConfigLock = new();
        private bool _tuningDbResetDone;
        private readonly ILogger? _logger;
        private readonly object _clblastLock = new();
        private bool _clblastBaselineInitialized;
        private GemmConfig? _clblastBaselineConfig;
        private bool _clblastPackingKernelsReady;
        private bool _clblastDirectKernelsReady;
        private ClBlastCopyParameters _clblastCopyParams;
        private ClBlastPadParameters _clblastPadParams;
        private ClBlastPadTransposeParameters _clblastPadTransposeParams;
        private ClBlastTransposeParameters _clblastTransposeParams;
        private ClBlastXgemmDirectParameters _clblastDirectParams;
        private int _clblastMinIndirectSize;

        public bool IsAvailable { get; }
        public string BackendName => "OpenCL";
        public string DeviceName { get; }
        public string DeviceVendor { get; }
        public int ComputeUnits { get; }
        public long GlobalMemoryBytes { get; }
        public long LocalMemoryBytes { get; }

        // IAsyncGpuBackend properties
        public bool SupportsMultiStream => true;
        public bool SupportsEvents => true;
        public bool SupportsAsyncTransfer => true;
        public bool SupportsGraphCapture => false;
        public int MaxConcurrentStreams => 8;
        public IGpuStream DefaultStream => _defaultStream ?? throw new InvalidOperationException("Backend not initialized");

        // Dynamic GPU capabilities - initialized from device queries
        private readonly ulong _maxWorkGroupSize;
        private readonly ulong[] _maxWorkItemSizes;
        private readonly bool _supportsFp16;
        private readonly bool _supportsSubgroups;
        private bool _mixedPrecisionKernelsAvailable;

        /// <summary>
        /// Gets whether OpenCL is available on this system.
        /// </summary>
        public static bool IsOpenClAvailable => DirectOpenClContext.IsAvailable;

        public OpenClBackend(ILogger? logger = null) : this(0, logger)
        {
        }

        public OpenClBackend(int deviceIndex, ILogger? logger = null)
        {
            _logger = logger;
            _kernelCache = new Dictionary<string, DirectOpenClKernel>();
            _programs = new List<DirectOpenClProgram>();
            _maxWorkItemSizes = Array.Empty<ulong>();

            if (!DirectOpenClContext.IsAvailable)
            {
                IsAvailable = false;
                DeviceName = "None";
                DeviceVendor = "None";
                return;
            }

            try
            {
                Console.WriteLine($"[OpenClBackend] Creating DirectOpenClContext for device {deviceIndex}...");
                _context = new DirectOpenClContext(deviceIndex);
                Console.WriteLine($"[OpenClBackend] Context created: Device={_context.DeviceName}, Vendor={_context.DeviceVendor}");

                IsAvailable = true;
                DeviceName = _context.DeviceName;
                DeviceVendor = _context.DeviceVendor;

                // Get CU count from OpenCL, with environment variable override
                int detectedCUs = (int)_context.MaxComputeUnits;
                string? envCUs = Environment.GetEnvironmentVariable("AIDOTNET_GPU_COMPUTE_UNITS");
                if (int.TryParse(envCUs, out int overrideCUs) && overrideCUs > 0 && overrideCUs <= 256)
                {
                    Console.WriteLine($"[OpenClBackend] CU override: {detectedCUs} -> {overrideCUs} (via AIDOTNET_GPU_COMPUTE_UNITS)");
                    ComputeUnits = overrideCUs;
                }
                else
                {
                    ComputeUnits = detectedCUs;
                }

                GlobalMemoryBytes = (long)_context.GlobalMemSize;
                LocalMemoryBytes = (long)_context.LocalMemSize;

                // Store GPU capabilities for dynamic work group sizing
                _maxWorkGroupSize = _context.MaxWorkGroupSize;
                _maxWorkItemSizes = _context.MaxWorkItemSizes;
                _supportsFp16 = _context.SupportsFp16;
                _supportsSubgroups = _context.SupportsSubgroups;

                // Print GPU capabilities for diagnostics
                Console.WriteLine($"[OpenClBackend] GPU Capabilities:");
                Console.WriteLine($"[OpenClBackend]   Compute Units: {ComputeUnits}");
                Console.WriteLine($"[OpenClBackend]   Max Work Group Size: {_maxWorkGroupSize}");
                if (_maxWorkItemSizes.Length >= 2)
                {
                    Console.WriteLine($"[OpenClBackend]   Max Work Item Sizes: [{string.Join(", ", _maxWorkItemSizes)}]");
                }
                Console.WriteLine($"[OpenClBackend]   Local Memory: {LocalMemoryBytes / 1024} KB");
                Console.WriteLine($"[OpenClBackend]   Supports FP16: {_supportsFp16}");
                Console.WriteLine($"[OpenClBackend]   Supports Subgroups: {_supportsSubgroups}");

                // Initialize default stream wrapper
                _defaultStream = new OpenClCommandQueue(this, _context.CommandQueue, _context.Context, _context.Device,
                    GpuStreamType.Default, _context.IsProfilingEnabled, ownsHandle: false);
                Console.WriteLine("[OpenClBackend] Default command queue wrapper initialized.");

                Console.WriteLine("[OpenClBackend] Compiling kernels...");
                CompileKernels();
                Console.WriteLine($"[OpenClBackend] Kernels compiled successfully. Total: {_kernelCache.Count}");

                // Initialize dynamic kernel generator for Bayesian-optimized GEMM
                _dynamicGemm = new DynamicGemmKernel(_context);
                Console.WriteLine("[OpenClBackend] Dynamic GEMM kernel generator initialized.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[OpenClBackend] Initialization FAILED: {ex.GetType().Name}: {ex.Message}");
                if (ex.InnerException != null)
                    Console.WriteLine($"[OpenClBackend] Inner: {ex.InnerException.Message}");
                System.Diagnostics.Debug.WriteLine($"OpenClBackend initialization failed: {ex.Message}");
                IsAvailable = false;
                DeviceName = "None";
                DeviceVendor = "None";
                _context?.Dispose();
                _context = null;
            }
        }

        private void CompileKernels()
        {
            if (_context == null) return;

            // Aggressive optimization flags for maximum performance
            // -cl-fast-relaxed-math: Allows aggressive math optimizations (FMA, reordering)
            // -cl-mad-enable: Enables fused multiply-add instructions (critical for GEMM)
            // -cl-unsafe-math-optimizations: More aggressive optimizations
            // -cl-finite-math-only: Assume no NaN/Inf (safe for GEMM)
            // -cl-no-signed-zeros: Ignore sign of zeros for faster math
            const string optimizationFlags = OpenClBuildOptions.OptimizationFlags;

            try
            {
                // Compile GEMM kernels with aggressive optimizations
                Console.WriteLine("[OpenClBackend] Compiling GEMM kernels...");
                var gemmProgram = new DirectOpenClProgram(_context, GemmKernel.GetSource());
                gemmProgram.Build(optimizationFlags);
                _programs.Add(gemmProgram);
                foreach (var name in GemmKernel.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, gemmProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] GEMM kernels: {string.Join(", ", GemmKernel.GetKernelNames())}");

                // Compile activation kernels
                Console.WriteLine("[OpenClBackend] Compiling activation kernels...");
                var activationProgram = new DirectOpenClProgram(_context, ActivationKernels.GetSource());
                activationProgram.Build(optimizationFlags);
                _programs.Add(activationProgram);
                foreach (var name in ActivationKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, activationProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Activation kernels: {string.Join(", ", ActivationKernels.GetKernelNames())}");

                // Compile fused kernels
                Console.WriteLine("[OpenClBackend] Compiling fused kernels...");
                var fusedProgram = new DirectOpenClProgram(_context, FusedKernels.GetSource());
                fusedProgram.Build(optimizationFlags);
                _programs.Add(fusedProgram);
                foreach (var name in FusedKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, fusedProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Fused kernels: {string.Join(", ", FusedKernels.GetKernelNames())}");

                // Compile reduction kernels
                Console.WriteLine("[OpenClBackend] Compiling reduction kernels...");
                var reductionProgram = new DirectOpenClProgram(_context, ReductionKernels.GetSource());
                reductionProgram.Build(optimizationFlags);
                _programs.Add(reductionProgram);
                foreach (var name in ReductionKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, reductionProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Reduction kernels: {string.Join(", ", ReductionKernels.GetKernelNames())}");

                // Compile packing/pad-copy kernels
                Console.WriteLine("[OpenClBackend] Compiling packing kernels...");
                var packingProgram = new DirectOpenClProgram(_context, PackingKernels.GetSource());
                packingProgram.Build(optimizationFlags);
                _programs.Add(packingProgram);
                foreach (var name in PackingKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, packingProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Packing kernels: {string.Join(", ", PackingKernels.GetKernelNames())}");

                // Compile sparse GEMM kernels (2:4 structured sparsity)
                Console.WriteLine("[OpenClBackend] Compiling sparse GEMM kernels...");
                var sparseProgram = new DirectOpenClProgram(_context, SparseGemmKernels.GetSource());
                sparseProgram.Build(optimizationFlags);
                _programs.Add(sparseProgram);
                foreach (var name in SparseGemmKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, sparseProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Sparse GEMM kernels: {string.Join(", ", SparseGemmKernels.GetKernelNames())}");

                // Compile CSR sparse kernels (general sparsity for GNN)
                Console.WriteLine("[OpenClBackend] Compiling CSR sparse kernels...");
                var csrSparseProgram = new DirectOpenClProgram(_context, CsrSparseKernels.GetSource());
                csrSparseProgram.Build(optimizationFlags);
                _programs.Add(csrSparseProgram);
                foreach (var name in CsrSparseKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, csrSparseProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] CSR sparse kernels: {string.Join(", ", CsrSparseKernels.GetKernelNames())}");

                // Compile convolution kernels
                Console.WriteLine("[OpenClBackend] Compiling convolution kernels...");
                var convProgram = new DirectOpenClProgram(_context, ConvolutionKernels.GetSource());
                convProgram.Build(optimizationFlags);
                _programs.Add(convProgram);
                foreach (var name in ConvolutionKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, convProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Convolution kernels: {string.Join(", ", ConvolutionKernels.GetKernelNames())}");

                // Compile fused convolution kernels
                Console.WriteLine("[OpenClBackend] Compiling fused convolution kernels...");
                var fusedConvProgram = new DirectOpenClProgram(_context, FusedConvolutionKernels.GetSource());
                fusedConvProgram.Build(optimizationFlags);
                _programs.Add(fusedConvProgram);
                foreach (var name in FusedConvolutionKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, fusedConvProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Fused convolution kernels: {string.Join(", ", FusedConvolutionKernels.GetKernelNames())}");

                // Compile pooling kernels
                Console.WriteLine("[OpenClBackend] Compiling pooling kernels...");
                var poolProgram = new DirectOpenClProgram(_context, PoolingKernels.GetSource());
                poolProgram.Build(optimizationFlags);
                _programs.Add(poolProgram);
                foreach (var name in PoolingKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, poolProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Pooling kernels: {string.Join(", ", PoolingKernels.GetKernelNames())}");

                // Compile normalization kernels
                Console.WriteLine("[OpenClBackend] Compiling normalization kernels...");
                var normProgram = new DirectOpenClProgram(_context, NormalizationKernels.GetSource());
                normProgram.Build(optimizationFlags);
                _programs.Add(normProgram);
                foreach (var name in NormalizationKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, normProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Normalization kernels: {string.Join(", ", NormalizationKernels.GetKernelNames())}");

                // Compile neural network kernels (activation gradients, loss, optimizers)
                Console.WriteLine("[OpenClBackend] Compiling neural network kernels...");
                var nnProgram = new DirectOpenClProgram(_context, NeuralNetKernels.GetSource());
                nnProgram.Build(optimizationFlags);
                _programs.Add(nnProgram);
                foreach (var name in NeuralNetKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, nnProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Neural network kernels compiled: {NeuralNetKernels.GetKernelNames().Length} kernels");

                // Compile mixed precision kernels only if device actually supports FP16
                // This is non-fatal - if compilation fails, we fall back to no FP16 support
                _mixedPrecisionKernelsAvailable = false;
                if (_supportsFp16)
                {
                    try
                    {
                        Console.WriteLine("[OpenClBackend] Compiling mixed precision kernels...");
                        var mixedPrecisionSource = string.Join("\n\n",
                            MixedPrecisionKernels.ConvertFp32ToFp16,
                            MixedPrecisionKernels.ConvertFp16ToFp32,
                            MixedPrecisionKernels.MixedPrecisionForward,
                            MixedPrecisionKernels.MixedPrecisionBackward,
                            MixedPrecisionKernels.AccumulateGradientFp32);
                        var mpProgram = new DirectOpenClProgram(_context, mixedPrecisionSource);
                        mpProgram.Build(optimizationFlags);
                        _programs.Add(mpProgram);
                        _kernelCache["convert_fp32_to_fp16"] = new DirectOpenClKernel(_context, mpProgram, "convert_fp32_to_fp16");
                        _kernelCache["convert_fp16_to_fp32"] = new DirectOpenClKernel(_context, mpProgram, "convert_fp16_to_fp32");
                        _kernelCache["mixed_precision_forward"] = new DirectOpenClKernel(_context, mpProgram, "mixed_precision_forward");
                        _kernelCache["mixed_precision_backward"] = new DirectOpenClKernel(_context, mpProgram, "mixed_precision_backward");
                        _kernelCache["accumulate_gradient_fp32"] = new DirectOpenClKernel(_context, mpProgram, "accumulate_gradient_fp32");
                        _mixedPrecisionKernelsAvailable = true;
                        Console.WriteLine("[OpenClBackend] Mixed precision kernels compiled: 5 kernels");
                    }
                    catch (Exception ex)
                    {
                        // Mixed precision compilation failed - this is non-fatal
                        // Device may report FP16 support but have driver issues with these kernels
                        Console.WriteLine($"[OpenClBackend] Warning: Mixed precision kernel compilation failed (non-fatal): {ex.Message}");
                        Console.WriteLine("[OpenClBackend] Continuing without mixed precision support.");
                    }
                }
                else
                {
                    Console.WriteLine("[OpenClBackend] Skipping mixed precision kernels (FP16 not supported on this device).");
                }

                // Compile attention kernels (FlashAttention, GQA, ScaledDotProduct)
                Console.WriteLine("[OpenClBackend] Compiling attention kernels...");
                var attnProgram = new DirectOpenClProgram(_context, AttentionKernels.GetSource());
                attnProgram.Build(optimizationFlags);
                _programs.Add(attnProgram);
                foreach (var name in AttentionKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, attnProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Attention kernels compiled: {AttentionKernels.GetKernelNames().Length} kernels");

                // Compile FFT kernels
                Console.WriteLine("[OpenClBackend] Compiling FFT kernels...");
                var fftProgram = new DirectOpenClProgram(_context, FFTKernels.GetSource());
                fftProgram.Build(optimizationFlags);
                _programs.Add(fftProgram);
                foreach (var name in FFTKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, fftProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] FFT kernels compiled: {FFTKernels.GetKernelNames().Length} kernels");

                // Compile spatial transformer kernels (TopK, AffineGrid, GridSample)
                Console.WriteLine("[OpenClBackend] Compiling spatial transformer kernels...");
                var stProgram = new DirectOpenClProgram(_context, SpatialTransformerKernels.GetSource());
                stProgram.Build(optimizationFlags);
                _programs.Add(stProgram);
                foreach (var name in new[] { "topk", "affine_grid", "grid_sample", "grid_sample_backward" })
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, stProgram, name);
                }
                Console.WriteLine("[OpenClBackend] Spatial transformer kernels compiled: 4 kernels");

                // Compile locally connected convolution kernels
                Console.WriteLine("[OpenClBackend] Compiling locally connected kernels...");
                var locallyConnectedProgram = new DirectOpenClProgram(_context, LocallyConnectedKernels.GetSource());
                locallyConnectedProgram.Build(optimizationFlags);
                _programs.Add(locallyConnectedProgram);
                foreach (var name in LocallyConnectedKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, locallyConnectedProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Locally connected kernels compiled: {LocallyConnectedKernels.GetKernelNames().Length} kernels");

                // Compile deformable convolution kernels
                Console.WriteLine("[OpenClBackend] Compiling deformable convolution kernels...");
                var deformableProgram = new DirectOpenClProgram(_context, DeformableConvolutionKernels.GetSource());
                deformableProgram.Build(optimizationFlags);
                _programs.Add(deformableProgram);
                foreach (var name in DeformableConvolutionKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, deformableProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Deformable convolution kernels compiled: {DeformableConvolutionKernels.GetKernelNames().Length} kernels");

                // Compile random number generation kernels
                Console.WriteLine("[OpenClBackend] Compiling random number kernels...");
                var randomProgram = new DirectOpenClProgram(_context, RandomKernels.GetKernels());
                randomProgram.Build(optimizationFlags);
                _programs.Add(randomProgram);
                foreach (var name in new[] { "GenerateRandomUniform", "GenerateRandomNormal" })
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, randomProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Random number kernels compiled: 2 kernels");

                // Compile specialized kernels (hyperbolic geometry, octonion algebra, quantum computing)
                Console.WriteLine("[OpenClBackend] Compiling specialized kernels...");
                var specializedProgram = new DirectOpenClProgram(_context, SpecializedKernels.GetSource());
                specializedProgram.Build(optimizationFlags);
                _programs.Add(specializedProgram);
                foreach (var name in SpecializedKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, specializedProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Specialized kernels compiled: {SpecializedKernels.GetKernelNames().Length} kernels");

                // Compile FP16 conversion kernels (half-precision float conversion)
                Console.WriteLine("[OpenClBackend] Compiling FP16 conversion kernels...");
                var fp16Program = new DirectOpenClProgram(_context, Fp16Kernels.GetSource());
                fp16Program.Build(optimizationFlags);
                _programs.Add(fp16Program);
                foreach (var name in Fp16Kernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, fp16Program, name);
                }
                Console.WriteLine($"[OpenClBackend] FP16 conversion kernels compiled: {Fp16Kernels.GetKernelNames().Length} kernels");

                // Compile loss function kernels (MSE, BCE, CE, Huber, Focal, Triplet, etc.)
                Console.WriteLine("[OpenClBackend] Compiling loss function kernels...");
                var lossProgram = new DirectOpenClProgram(_context, LossKernels.GetSource());
                lossProgram.Build(optimizationFlags);
                _programs.Add(lossProgram);
                foreach (var name in LossKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, lossProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] Loss function kernels compiled: {LossKernels.GetKernelNames().Length} kernels");

                // Compile LSTM sequence kernels (forward/backward for BPTT training)
                Console.WriteLine("[OpenClBackend] Compiling LSTM sequence kernels...");
                var lstmProgram = new DirectOpenClProgram(_context, LstmKernels.GetSource());
                lstmProgram.Build(optimizationFlags);
                _programs.Add(lstmProgram);
                foreach (var name in LstmKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, lstmProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] LSTM sequence kernels compiled: {LstmKernels.GetKernelNames().Length} kernels");

                // Compile GRU sequence kernels (forward/backward for BPTT training)
                Console.WriteLine("[OpenClBackend] Compiling GRU sequence kernels...");
                var gruProgram = new DirectOpenClProgram(_context, GruKernels.GetSource());
                gruProgram.Build(optimizationFlags);
                _programs.Add(gruProgram);
                foreach (var name in GruKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, gruProgram, name);
                }
                Console.WriteLine($"[OpenClBackend] GRU sequence kernels compiled: {GruKernels.GetKernelNames().Length} kernels");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Kernel compilation failed: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Calculates optimal 2D work group sizes based on GPU capabilities.
        /// </summary>
        /// <param name="globalSizeX">The global size in X dimension</param>
        /// <param name="globalSizeY">The global size in Y dimension</param>
        /// <returns>Tuple of (localSizeX, localSizeY) that respects GPU limits</returns>
        private (int localSizeX, int localSizeY) CalculateOptimalWorkGroupSize(int globalSizeX, int globalSizeY)
        {
            // Start with ideal tile size (16x16 = 256 is a common sweet spot)
            int idealSize = 16;

            // Get per-dimension limits
            ulong maxX = _maxWorkItemSizes.Length > 0 ? _maxWorkItemSizes[0] : 256;
            ulong maxY = _maxWorkItemSizes.Length > 1 ? _maxWorkItemSizes[1] : 256;
            ulong maxTotal = _maxWorkGroupSize > 0 ? _maxWorkGroupSize : 256;

            // Calculate the largest square that fits within constraints
            int localSizeX = idealSize;
            int localSizeY = idealSize;

            // Ensure we don't exceed per-dimension limits
            localSizeX = Math.Min(localSizeX, (int)maxX);
            localSizeY = Math.Min(localSizeY, (int)maxY);

            // Ensure product doesn't exceed total work group size
            while ((ulong)(localSizeX * localSizeY) > maxTotal && localSizeX > 1 && localSizeY > 1)
            {
                // Reduce the larger dimension first
                if (localSizeX >= localSizeY)
                    localSizeX /= 2;
                else
                    localSizeY /= 2;
            }

            // Ensure local size doesn't exceed global size (avoids wasted work items)
            localSizeX = Math.Min(localSizeX, globalSizeX);
            localSizeY = Math.Min(localSizeY, globalSizeY);

            // Ensure we have at least 1
            localSizeX = Math.Max(localSizeX, 1);
            localSizeY = Math.Max(localSizeY, 1);

            return (localSizeX, localSizeY);
        }

        /// <summary>
        /// Calculates optimal 1D work group size based on GPU capabilities.
        /// </summary>
        /// <param name="globalSize">The global size</param>
        /// <returns>Optimal local size that respects GPU limits</returns>
        private int CalculateOptimalWorkGroupSize1D(int globalSize)
        {
            // Start with ideal size (256 is common)
            int idealSize = 256;

            ulong maxX = _maxWorkItemSizes.Length > 0 ? _maxWorkItemSizes[0] : 256;
            ulong maxTotal = _maxWorkGroupSize > 0 ? _maxWorkGroupSize : 256;

            int localSize = Math.Min(idealSize, (int)Math.Min(maxX, maxTotal));
            localSize = Math.Min(localSize, globalSize);
            localSize = Math.Max(localSize, 1);

            return localSize;
        }

        private string GetDeviceSignature()
        {
            if (_context == null)
                return string.Empty;

            return $"{_context.DeviceVendor}|{_context.DeviceName}|{_context.DriverVersion}|{_context.OpenClVersion}";
        }

        private static int GetEnvInt(string name, int defaultValue)
        {
            var value = Environment.GetEnvironmentVariable(name);
            return int.TryParse(value, out var parsed) && parsed > 0 ? parsed : defaultValue;
        }

        private static bool GetEnvBool(string name, bool defaultValue = false)
        {
            var value = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(value))
                return defaultValue;

            if (value.Equals("1", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("true", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("yes", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("on", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            if (value.Equals("0", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("false", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("no", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("off", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            return defaultValue;
        }

        private static bool TryGetOfflineTuningMode(out bool useExhaustive)
        {
            useExhaustive = false;
            var value = Environment.GetEnvironmentVariable(OfflineTuningEnvVar);
            if (string.IsNullOrWhiteSpace(value))
                return false;

            if (value.Equals("0", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("false", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("off", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (value.Equals("exhaustive", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("full", StringComparison.OrdinalIgnoreCase) ||
                value.Equals("all", StringComparison.OrdinalIgnoreCase))
            {
                useExhaustive = true;
            }

            return true;
        }

        private void ConfigureOfflineTuning()
        {
            bool enableDiag = GetEnvBool(OfflineTuningDiagEnvVar);
            if (enableDiag)
                EnableTuningDiagnostics = true;

            var logPath = Environment.GetEnvironmentVariable(OfflineTuningLogEnvVar);
            if (!string.IsNullOrWhiteSpace(logPath))
                GemmAutoTuner.LogFilePath = logPath;

            var csvPath = Environment.GetEnvironmentVariable(OfflineTuningCsvEnvVar);
            if (!string.IsNullOrWhiteSpace(csvPath))
                GemmAutoTuner.TrialLogFilePath = csvPath;

            var progressEnv = Environment.GetEnvironmentVariable(OfflineTuningProgressEnvVar);
            if (!string.IsNullOrWhiteSpace(progressEnv))
                GemmAutoTuner.ProgressInterval = GetEnvInt(OfflineTuningProgressEnvVar, GemmAutoTuner.ProgressInterval);

            GemmAutoTuner.TrialHeartbeatSeconds = GetEnvInt(
                OfflineTuningHeartbeatEnvVar,
                GemmAutoTuner.EnableProgress ? 10 : 0);

            if (_logger != null)
            {
                GemmAutoTuner.Logger = _logger;
            }
            else if (enableDiag && GemmAutoTuner.Logger == null)
            {
                GemmAutoTuner.Logger = new SimpleConsoleLogger();
            }
        }

        private sealed class SimpleConsoleLogger : ILogger
        {
            private sealed class NullScope : IDisposable
            {
                public static readonly NullScope Instance = new();
                public void Dispose() { }
            }

            public IDisposable BeginScope<TState>(TState state) where TState : notnull => NullScope.Instance;
            public bool IsEnabled(LogLevel logLevel) => logLevel >= LogLevel.Information;

            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
                Func<TState, Exception?, string> formatter)
            {
                if (!IsEnabled(logLevel))
                    return;

                string message = formatter(state, exception);
                if (exception != null)
                    message = $"{message} ({exception.GetType().Name}: {exception.Message})";

                var color = GetDiagColor(message);
                if (color.HasValue)
                {
                    var previous = Console.ForegroundColor;
                    Console.ForegroundColor = color.Value;
                    Console.WriteLine(message);
                    Console.ForegroundColor = previous;
                }
                else
                {
                    Console.WriteLine(message);
                }
            }

            private static ConsoleColor? GetDiagColor(string message)
            {
                if (message.Contains("Diag[HIGH]", StringComparison.OrdinalIgnoreCase))
                    return ConsoleColor.Red;
                if (message.Contains("Diag[MED]", StringComparison.OrdinalIgnoreCase))
                    return ConsoleColor.Yellow;
                if (message.Contains("Diag[LOW]", StringComparison.OrdinalIgnoreCase))
                    return ConsoleColor.Green;
                return null;
            }
        }

        #region Memory Management

        public IGpuBuffer AllocateBuffer(float[] data)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var buffer = new DirectOpenClBuffer(_context, data);
            return new DirectOpenClGpuBuffer(buffer);
        }

        public IGpuBuffer AllocateBuffer(int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var buffer = new DirectOpenClBuffer(_context, size);
            return new DirectOpenClGpuBuffer(buffer);
        }

        public float[] DownloadBuffer(IGpuBuffer buffer)
        {
            var openClBuffer = (DirectOpenClGpuBuffer)buffer;
            return openClBuffer.Download();
        }

        public void DownloadBuffer(IGpuBuffer buffer, float[] destination)
        {
            var openClBuffer = (DirectOpenClGpuBuffer)buffer;
            openClBuffer.Download(destination);
        }

        public void Copy(IGpuBuffer source, int srcOffset, IGpuBuffer destination, int destOffset, int size)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var srcHandle = ((DirectOpenClGpuBuffer)source).Buffer.Handle;
            var destHandle = ((DirectOpenClGpuBuffer)destination).Buffer.Handle;

            // Offsets and size in bytes
            var srcOffsetBytes = new UIntPtr((ulong)srcOffset * sizeof(float));
            var destOffsetBytes = new UIntPtr((ulong)destOffset * sizeof(float));
            var sizeBytes = new UIntPtr((ulong)size * sizeof(float));

            int err = OpenClNativeBindings.EnqueueCopyBuffer(
                _context.CommandQueue,
                srcHandle,
                destHandle,
                srcOffsetBytes,
                destOffsetBytes,
                sizeBytes,
                0, IntPtr.Zero, IntPtr.Zero);

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"OpenCL copy failed: {err}");
        }

        #endregion

        #region GEMM Operations

        private bool TryGetTunedConfig(int M, int N, int K, out GemmConfig config)
        {
            config = default;
            var key = (M, N, K);
            bool offlineEnabled = TryGetOfflineTuningMode(out bool useExhaustive);
            if (offlineEnabled)
                ConfigureOfflineTuning();

            GemmConfig? fallbackConfig = null;
            lock (_tunedConfigLock)
            {
                if (_tunedConfigCache.TryGetValue(key, out var cachedConfig))
                {
                    if (!offlineEnabled)
                    {
                        config = cachedConfig;
                        return true;
                    }

                    fallbackConfig = cachedConfig;
                }
            }

            try
            {
                string deviceSignature = GetDeviceSignature();
                using var database = new GemmTuningDatabase(deviceSignature: deviceSignature);
                if (offlineEnabled && GetEnvBool(OfflineTuningResetEnvVar) && !_tuningDbResetDone)
                {
                    database.Clear();
                    lock (_tunedConfigLock)
                    {
                        _tunedConfigCache.Clear();
                    }
                    fallbackConfig = null;
                    _tuningDbResetDone = true;
                }

                var cachedEntry = database.GetBestConfigWithGflops(M, N, K);
                if (cachedEntry.HasValue && cachedEntry.Value.GFlops > 0)
                {
                    var cachedConfig = cachedEntry.Value.Config;
                    var validationError = DynamicGemmKernel.ValidateConfig(cachedConfig);
                    if (validationError == null)
                    {
                        fallbackConfig ??= cachedConfig;

                        if (!offlineEnabled)
                        {
                            lock (_tunedConfigLock)
                            {
                                _tunedConfigCache[key] = cachedConfig;
                            }

                            config = cachedConfig;
                            return true;
                        }
                    }

                    if (EnableTuningDiagnostics)
                    {
                        Console.WriteLine($"[GEMM] Cached config invalid: {validationError}");
                    }
                }

                if (!offlineEnabled)
                    return false;

                if (useExhaustive)
                {
                    var results = RunExhaustiveGemmOptimization(M, N, K);
                    if (results.Length > 0 && results[0].IsValid)
                    {
                        lock (_tunedConfigLock)
                        {
                            _tunedConfigCache[key] = results[0].Config;
                        }

                        config = results[0].Config;
                        return true;
                    }

                    if (fallbackConfig.HasValue)
                    {
                        lock (_tunedConfigLock)
                        {
                            _tunedConfigCache[key] = fallbackConfig.Value;
                        }

                        config = fallbackConfig.Value;
                        return true;
                    }

                    return false;
                }

                int maxTrials = GetEnvInt(OfflineTuningTrialsEnvVar, DefaultBayesianTrials);
                var tunedResults = RunBayesianGemmOptimization(M, N, K, maxTrials);
                if (tunedResults.Length > 0 && tunedResults[0].IsValid)
                {
                    lock (_tunedConfigLock)
                    {
                        _tunedConfigCache[key] = tunedResults[0].Config;
                    }

                    config = tunedResults[0].Config;
                    return true;
                }

                if (fallbackConfig.HasValue)
                {
                    lock (_tunedConfigLock)
                    {
                        _tunedConfigCache[key] = fallbackConfig.Value;
                    }

                    config = fallbackConfig.Value;
                    return true;
                }

                return false;
            }
            catch (Exception ex)
            {
                if (EnableTuningDiagnostics)
                {
                    Console.WriteLine($"[GEMM] Tuning lookup failed: {ex.Message}");
                }

                return false;
            }
        }

        private bool TryGetClBlastBaselineConfig(out GemmConfig config)
        {
            config = default;
            if (_context == null)
                return false;

            lock (_clblastLock)
            {
                if (_clblastBaselineInitialized)
                {
                    if (_clblastBaselineConfig.HasValue)
                    {
                        config = _clblastBaselineConfig.Value;
                        return true;
                    }

                    return false;
                }

                _clblastBaselineInitialized = true;
                var deviceInfo = ClBlastDeviceInfo.FromContext(_context);
                _clblastCopyParams = ClBlastCopyDatabase.GetParameters(deviceInfo);
                _clblastPadParams = ClBlastPadDatabase.GetParameters(deviceInfo);
                _clblastPadTransposeParams = ClBlastPadTransposeDatabase.GetParameters(deviceInfo);
                _clblastTransposeParams = ClBlastTransposeDatabase.GetParameters(deviceInfo);
                _clblastDirectParams = ClBlastXgemmDirectDatabase.GetParameters(deviceInfo);
                _clblastMinIndirectSize = ClBlastGemmRoutineDatabase.GetXgemmMinIndirectSize(deviceInfo);
                Console.WriteLine($"[OpenClBackend] CLBlast MinIndirectSize threshold: {_clblastMinIndirectSize} (use INDIRECT for M/N >= {_clblastMinIndirectSize})");

                if (ClBlastXgemmDatabase.TryGetConfig(deviceInfo, out var baseline))
                {
                    _clblastBaselineConfig = baseline;
                    config = baseline;
                    return true;
                }

                // FALLBACK: Use default CLBlast baseline config for devices not in database
                // This is the proven CLBlast Xgemm kernel 0 configuration (2300+ GFLOPS)
                // GEMMK=0, MWG=64, NWG=64, KWG=16, VWM=2, VWN=2, SA=1, SB=1
                var defaultBaseline = new GemmConfig
                {
                    TileM = 64,
                    TileN = 64,
                    TileK = 16,
                    ThreadTileM = 8,
                    ThreadTileN = 8,
                    VectorWidthM = 2,
                    VectorWidthN = 2,
                    UseDoubleBuffering = false,
                    UseVectorizedLoads = true,
                    KReg = 1,
                    KUnroll = 2,
                    UseSubgroupOps = false,
                    StrideM = false,
                    StrideN = true,
                    CacheA = true,
                    CacheB = true,
                    MdimaSize = 16,
                    NdimbSize = 8,
                    UseTrueVectorLDS = true,
                    UseColumnMajorA = true,
                    KernelName = "clblast_baseline_k0"
                };
                _clblastBaselineConfig = defaultBaseline;
                config = defaultBaseline;
                return true;
            }
        }

        private bool EnsureClBlastPackingKernels()
        {
            if (_context == null)
                return false;

            if (_clblastPackingKernelsReady)
                return true;

            lock (_clblastLock)
            {
                if (_clblastPackingKernelsReady)
                    return true;

                if (_clblastPadParams.DimX <= 0 ||
                    _clblastPadTransposeParams.Tile <= 0 ||
                    _clblastCopyParams.DimX <= 0 ||
                    _clblastTransposeParams.Dim <= 0)
                {
                    var deviceInfo = ClBlastDeviceInfo.FromContext(_context);
                    _clblastCopyParams = ClBlastCopyDatabase.GetParameters(deviceInfo);
                    _clblastPadParams = ClBlastPadDatabase.GetParameters(deviceInfo);
                    _clblastPadTransposeParams = ClBlastPadTransposeDatabase.GetParameters(deviceInfo);
                    _clblastTransposeParams = ClBlastTransposeDatabase.GetParameters(deviceInfo);
                }

                var source = ClBlastPackingKernels.BuildSource(
                    _clblastPadParams,
                    _clblastPadTransposeParams,
                    _clblastCopyParams,
                    _clblastTransposeParams);
                var program = new DirectOpenClProgram(_context, source);
                program.Build(OpenClBuildOptions.OptimizationFlags);
                _programs.Add(program);

                foreach (var name in ClBlastPackingKernels.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, program, name);
                }

                _clblastPackingKernelsReady = true;
                return true;
            }
        }

        private bool EnsureClBlastDirectKernels()
        {
            if (_context == null)
                return false;

            if (_clblastDirectKernelsReady)
                return true;

            lock (_clblastLock)
            {
                if (_clblastDirectKernelsReady)
                    return true;

                if (_clblastDirectParams.Wgd <= 0 || _clblastDirectParams.MdimCd <= 0 || _clblastDirectParams.NdimCd <= 0)
                {
                    var deviceInfo = ClBlastDeviceInfo.FromContext(_context);
                    _clblastDirectParams = ClBlastXgemmDirectDatabase.GetParameters(deviceInfo);
                }

                var source = ClBlastXgemmDirectKernel.BuildSource(_clblastDirectParams);
                var program = new DirectOpenClProgram(_context, source);
                program.Build(OpenClBuildOptions.OptimizationFlags);
                _programs.Add(program);

                foreach (var name in ClBlastXgemmDirectKernel.GetKernelNames())
                {
                    _kernelCache[name] = new DirectOpenClKernel(_context, program, name);
                }

                _clblastDirectKernelsReady = true;
                return true;
            }
        }

        private (int globalX, int globalY) GetClBlastPadGlobal(int destOne, int destTwo)
        {
            int globalX = CeilToMultiple(CeilDiv(destOne, _clblastPadParams.WorkPerThreadX), _clblastPadParams.DimX);
            int globalY = CeilToMultiple(CeilDiv(destTwo, _clblastPadParams.WorkPerThreadY), _clblastPadParams.DimY);
            return (globalX, globalY);
        }

        private (int globalX, int globalY) GetClBlastPadTransposeGlobal(int destOne, int destTwo)
        {
            int globalX = CeilToMultiple(CeilDiv(destOne, _clblastPadTransposeParams.WorkPerThread), _clblastPadTransposeParams.Tile);
            int globalY = CeilToMultiple(CeilDiv(destTwo, _clblastPadTransposeParams.WorkPerThread), _clblastPadTransposeParams.Tile);
            return (globalX, globalY);
        }

        private static int CeilToMultiple(int value, int multiple)
        {
            if (multiple <= 0)
                return value;

            return ((value + multiple - 1) / multiple) * multiple;
        }

        private static bool IsMultiple(int value, int multiple)
        {
            return multiple > 0 && value % multiple == 0;
        }

        private void ClBlastCopyMatrix(
            IGpuBuffer src,
            IGpuBuffer dst,
            int srcOne,
            int srcTwo,
            int srcLd,
            int srcOffset,
            int destOne,
            int destTwo,
            int destLd,
            int destOffset,
            bool pad)
        {
            bool useFast = srcOffset == 0 &&
                destOffset == 0 &&
                srcOne == destOne &&
                srcTwo == destTwo &&
                srcLd == destLd;

            if (useFast)
            {
                int vectorWidth = _clblastCopyParams.VectorWidth;
                int workPerThread = _clblastCopyParams.WorkPerThread;
                int dimX = _clblastCopyParams.DimX;
                int dimY = _clblastCopyParams.DimY;
                if (IsMultiple(srcLd, vectorWidth) &&
                    IsMultiple(srcOne, vectorWidth * dimX) &&
                    IsMultiple(srcTwo, workPerThread * dimY))
                {
                    var kernel = _kernelCache["CopyMatrixFast"];
                    kernel.SetArg(0, srcLd);
                    kernel.SetArg(1, ((DirectOpenClGpuBuffer)src).Buffer.Handle);
                    kernel.SetArg(2, ((DirectOpenClGpuBuffer)dst).Buffer.Handle);
                    kernel.SetArg(3, 1.0f);

                    int fastGlobalX = destOne / vectorWidth;
                    int fastGlobalY = destTwo / workPerThread;
                    kernel.Execute2D(fastGlobalX, fastGlobalY, dimX, dimY);
                    return;
                }
            }

            string kernelName = pad ? "CopyPadMatrix" : "CopyMatrix";
            var padKernel = _kernelCache[kernelName];
            padKernel.SetArg(0, srcOne);
            padKernel.SetArg(1, srcTwo);
            padKernel.SetArg(2, srcLd);
            padKernel.SetArg(3, srcOffset);
            padKernel.SetArg(4, ((DirectOpenClGpuBuffer)src).Buffer.Handle);
            padKernel.SetArg(5, destOne);
            padKernel.SetArg(6, destTwo);
            padKernel.SetArg(7, destLd);
            padKernel.SetArg(8, destOffset);
            padKernel.SetArg(9, ((DirectOpenClGpuBuffer)dst).Buffer.Handle);
            padKernel.SetArg(10, 1.0f);
            if (pad)
            {
                padKernel.SetArg(11, 0);
            }
            else
            {
                padKernel.SetArg(11, 0);
                padKernel.SetArg(12, 0);
                padKernel.SetArg(13, 0);
            }

            var (globalX, globalY) = GetClBlastPadGlobal(destOne, destTwo);
            padKernel.Execute2D(globalX, globalY, _clblastPadParams.DimX, _clblastPadParams.DimY);
        }

        private void ClBlastTransposeMatrix(
            IGpuBuffer src,
            IGpuBuffer dst,
            int srcOne,
            int srcTwo,
            int srcLd,
            int srcOffset,
            int destOne,
            int destTwo,
            int destLd,
            int destOffset,
            bool pad)
        {
            bool useFast = srcOffset == 0 &&
                destOffset == 0 &&
                srcOne == destOne &&
                srcTwo == destTwo &&
                srcLd == destLd;

            if (useFast)
            {
                int workPerThread = _clblastTransposeParams.WorkPerThread;
                int dim = _clblastTransposeParams.Dim;
                if (IsMultiple(srcLd, workPerThread) &&
                    IsMultiple(srcOne, workPerThread * dim) &&
                    IsMultiple(srcTwo, workPerThread * dim))
                {
                    var kernel = _kernelCache["TransposeMatrixFast"];
                    kernel.SetArg(0, srcLd);
                    kernel.SetArg(1, ((DirectOpenClGpuBuffer)src).Buffer.Handle);
                    kernel.SetArg(2, ((DirectOpenClGpuBuffer)dst).Buffer.Handle);
                    kernel.SetArg(3, 1.0f);

                    int fastGlobalX = destOne / workPerThread;
                    int fastGlobalY = destTwo / workPerThread;
                    kernel.Execute2D(fastGlobalX, fastGlobalY, dim, dim);
                    return;
                }
            }

            string kernelName = pad ? "TransposePadMatrix" : "TransposeMatrix";
            var padKernel = _kernelCache[kernelName];
            padKernel.SetArg(0, srcOne);
            padKernel.SetArg(1, srcTwo);
            padKernel.SetArg(2, srcLd);
            padKernel.SetArg(3, srcOffset);
            padKernel.SetArg(4, ((DirectOpenClGpuBuffer)src).Buffer.Handle);
            padKernel.SetArg(5, destOne);
            padKernel.SetArg(6, destTwo);
            padKernel.SetArg(7, destLd);
            padKernel.SetArg(8, destOffset);
            padKernel.SetArg(9, ((DirectOpenClGpuBuffer)dst).Buffer.Handle);
            padKernel.SetArg(10, 1.0f);
            if (pad)
            {
                padKernel.SetArg(11, 0);
            }
            else
            {
                padKernel.SetArg(11, 0);
                padKernel.SetArg(12, 0);
                padKernel.SetArg(13, 0);
            }

            var (globalX, globalY) = GetClBlastPadTransposeGlobal(destOne, destTwo);
            int tile = _clblastPadTransposeParams.Tile;
            padKernel.Execute2D(globalX, globalY, tile, tile);
        }

        private bool TryExecuteClBlastDirectGemm(
            IGpuBuffer A,
            IGpuBuffer B,
            IGpuBuffer C,
            int M,
            int N,
            int K,
            float alpha,
            float beta)
        {
            if (!EnsureClBlastDirectKernels())
                return false;

            if (_clblastDirectParams.Wgd <= 0 || _clblastDirectParams.MdimCd <= 0 || _clblastDirectParams.NdimCd <= 0)
                return false;

            const int aOffset = 0;
            const int bOffset = 0;
            const int cOffset = 0;
            int aLd = K;
            int bLd = N;
            int cLd = N;

            // Row-major layout: A and C are treated as transposed, B is not.
            var kernel = _kernelCache["XgemmDirectTN"];
            kernel.SetArg(0, M);
            kernel.SetArg(1, N);
            kernel.SetArg(2, K);
            kernel.SetArg(3, alpha);
            kernel.SetArg(4, beta);
            kernel.SetArg(5, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            kernel.SetArg(6, aOffset);
            kernel.SetArg(7, aLd);
            kernel.SetArg(8, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            kernel.SetArg(9, bOffset);
            kernel.SetArg(10, bLd);
            kernel.SetArg(11, ((DirectOpenClGpuBuffer)C).Buffer.Handle);
            kernel.SetArg(12, cOffset);
            kernel.SetArg(13, cLd);
            kernel.SetArg(14, 1);
            kernel.SetArg(15, 0);
            kernel.SetArg(16, 0);

            int mCeiled = CeilToMultiple(M, _clblastDirectParams.Wgd);
            int nCeiled = CeilToMultiple(N, _clblastDirectParams.Wgd);
            int globalX = (mCeiled * _clblastDirectParams.MdimCd) / _clblastDirectParams.Wgd;
            int globalY = (nCeiled * _clblastDirectParams.NdimCd) / _clblastDirectParams.Wgd;
            kernel.Execute2D(globalX, globalY, _clblastDirectParams.MdimCd, _clblastDirectParams.NdimCd);
            return true;
        }

        private bool TryExecuteClBlastBaselineGemm(
            IGpuBuffer A,
            IGpuBuffer B,
            IGpuBuffer C,
            int M,
            int N,
            int K,
            float alpha,
            float beta,
            GemmConfig config)
        {
            // CLBlast uses MinIndirectSize threshold to decide between XgemmDirect and Xgemm kernels.
            // XgemmDirect is faster for small matrices (no packing overhead).
            // Xgemm (indirect) is faster for large matrices (better cache utilization despite packing).
            // For gfx1012 (RX 5500 XT), threshold is 448 - use indirect for M >= 448 or N >= 448.
            bool traceEnabled = Environment.GetEnvironmentVariable("AIDOTNET_GEMM_TRACE") == "1";
            bool forceDirect = Environment.GetEnvironmentVariable("AIDOTNET_FORCE_DIRECT") == "1";

            // Use indirect path for matrices at or above the MinIndirectSize threshold (CLBlast behavior)
            bool useIndirectPath = _clblastMinIndirectSize > 0 && (M >= _clblastMinIndirectSize || N >= _clblastMinIndirectSize);

            // Try direct path only for small matrices (below MinIndirectSize threshold)
            if (!useIndirectPath || forceDirect)
            {
                if (traceEnabled)
                    Console.WriteLine($"[GEMM-TRACE {M}x{N}x{K}] Trying DIRECT path (M/N < {_clblastMinIndirectSize} or forceDirect={forceDirect})");
                if (TryExecuteClBlastDirectGemm(A, B, C, M, N, K, alpha, beta))
                {
                    if (traceEnabled)
                        Console.WriteLine($"[GEMM-TRACE {M}x{N}x{K}] SUCCESS: DIRECT path executed");
                    return true;
                }
                if (traceEnabled)
                    Console.WriteLine($"[GEMM-TRACE {M}x{N}x{K}] DIRECT path failed, trying INDIRECT");
            }
            else if (traceEnabled)
            {
                Console.WriteLine($"[GEMM-TRACE {M}x{N}x{K}] Skipping DIRECT path (M/N >= {_clblastMinIndirectSize}), using INDIRECT");
            }

            if (_dynamicGemm == null)
            {
                return false;
            }

            if (!EnsureClBlastPackingKernels())
            {
                return false;
            }

            int kReg = config.KReg > 0 ? config.KReg : 1;
            int kUnit = config.TileK * kReg;
            if (config.TileM <= 0 || config.TileN <= 0 || kUnit <= 0)
                return false;

            bool kernel1 = IsClBlastBaselineKernel1(config);

            // ROW-MAJOR SWAP TRICK (Zero-Copy Optimization):
            // For row-major GEMM C = A × B where A(M×K), B(K×N), C(M×N):
            //   C^T = (A × B)^T = B^T × A^T
            // Row-major data reinterpreted as column-major is already transposed!
            // So: column-major computation with swapped args gives row-major result.
            //
            // For GEMMK=0 (kernel1=false):
            // - OLD: Physical transpose A and C (48% overhead!)
            // - NEW: Swap A↔B, M↔N, pass to kernel. Zero data movement!
            //
            // The kernel computes C' = B' × A' in column-major where:
            // - B' = B(K×N row-major) reinterpreted as N×K column-major
            // - A' = A(M×K row-major) reinterpreted as K×M column-major
            // - C' = C(M×N row-major) reinterpreted as N×M column-major
            // Result: C' (column-major N×M) = C (row-major M×N) ✓
            bool useRowMajorSwap = !kernel1;

            // Timing diagnostics (enable with AIDOTNET_GEMM_TIMING=1)
            var timingEnabled = Environment.GetEnvironmentVariable("AIDOTNET_GEMM_TIMING") == "1";
            var sw = timingEnabled ? System.Diagnostics.Stopwatch.StartNew() : null;
            long allocTime = 0, packATime = 0, packBTime = 0, packCTime = 0, gemmTime = 0, unpackCTime = 0;

            if (useRowMajorSwap)
            {
                // Row-major swap trick: Swap A↔B and M↔N for zero-copy operation.
                // After swap: new_M = old_N, new_N = old_M
                // The kernel sees: "A"=B (N×K col-major), "B"=A (K×M col-major), "C"=C (N×M col-major)
                int swappedM = N;
                int swappedN = M;
                int mCeiled = CeilDiv(swappedM, config.TileM) * config.TileM;
                int nCeiled = CeilDiv(swappedN, config.TileN) * config.TileN;
                int kCeiled = CeilDiv(K, kUnit) * kUnit;

                // For the swapped computation, check if padding is needed
                // A (originally B): K×N row-major → N×K column-major, needs to be mCeiled×kCeiled
                // B (originally A): M×K row-major → K×M column-major, needs to be nCeiled×kCeiled
                // C: M×N row-major → N×M column-major, needs to be mCeiled×nCeiled
                bool aNeedsPad = N != mCeiled || K != kCeiled;
                bool bNeedsPad = M != nCeiled || K != kCeiled;
                bool cNeedsPad = N != mCeiled || M != nCeiled;

                long aSize = (long)mCeiled * kCeiled;
                long bSize = (long)nCeiled * kCeiled;
                long cSize = (long)mCeiled * nCeiled;
                if (aSize > int.MaxValue || bSize > int.MaxValue || cSize > int.MaxValue)
                    return false;

                IGpuBuffer? aTemp = null;
                IGpuBuffer? bTemp = null;
                IGpuBuffer? cTemp = null;
                IGpuBuffer aBuf = B;  // Swapped! Original B becomes kernel's A
                IGpuBuffer bBuf = A;  // Swapped! Original A becomes kernel's B
                IGpuBuffer cBuf = C;

                try
                {
                    // Pad "A" (originally B) if needed - NO TRANSPOSE, just copy with padding
                    if (aNeedsPad)
                    {
                        if (timingEnabled) { Synchronize(); sw!.Restart(); }
                        aTemp = AllocateBuffer((int)aSize);
                        if (timingEnabled) { allocTime += sw!.ElapsedTicks; sw.Restart(); }
                        // B is K×N row-major. Reinterpreted as column-major, it's N×K.
                        // We need mCeiled×kCeiled = swappedM_ceiled × K_ceiled.
                        // Copy K rows of N elements, with output stride mCeiled.
                        ClBlastCopyMatrix(B, aTemp, N, K, N, 0, mCeiled, kCeiled, mCeiled, 0, true);
                        aBuf = aTemp;
                        if (timingEnabled) { Synchronize(); packATime = sw!.ElapsedTicks; }
                    }

                    // Pad "B" (originally A) if needed - NO TRANSPOSE, just copy with padding
                    if (bNeedsPad)
                    {
                        if (timingEnabled) { sw!.Restart(); }
                        bTemp = AllocateBuffer((int)bSize);
                        if (timingEnabled) { allocTime += sw!.ElapsedTicks; sw.Restart(); }
                        // A is M×K row-major. Reinterpreted as column-major, it's K×M.
                        // We need nCeiled×kCeiled = swappedN_ceiled × K_ceiled.
                        // Copy M rows of K elements, with output stride nCeiled.
                        ClBlastCopyMatrix(A, bTemp, K, M, K, 0, nCeiled, kCeiled, nCeiled, 0, true);
                        bBuf = bTemp;
                        if (timingEnabled) { Synchronize(); packBTime = sw!.ElapsedTicks; }
                    }

                    // Pad C if needed (for beta != 0) - NO TRANSPOSE
                    if (cNeedsPad)
                    {
                        if (timingEnabled) { sw!.Restart(); }
                        cTemp = AllocateBuffer((int)cSize);
                        if (timingEnabled) { allocTime += sw!.ElapsedTicks; sw.Restart(); }
                        if (beta != 0.0f)
                        {
                            // C is M×N row-major. Reinterpreted as column-major, it's N×M.
                            // We need mCeiled×nCeiled.
                            ClBlastCopyMatrix(C, cTemp, N, M, N, 0, mCeiled, nCeiled, mCeiled, 0, true);
                            if (timingEnabled) { Synchronize(); packCTime = sw!.ElapsedTicks; }
                        }
                        cBuf = cTemp;
                    }

                    // Execute GEMM with swapped dimensions
                    if (timingEnabled) { sw!.Restart(); }
                    if (!TryExecuteDynamicGemm(aBuf, bBuf, cBuf, mCeiled, nCeiled, kCeiled, alpha, beta, config))
                        return false;
                    if (timingEnabled) { Synchronize(); gemmTime = sw!.ElapsedTicks; }

                    // Copy result back if we used temp buffer - NO TRANSPOSE
                    if (cNeedsPad)
                    {
                        if (timingEnabled) { sw!.Restart(); }
                        // Result is mCeiled×nCeiled column-major = nCeiled×mCeiled row-major.
                        // But we want M×N row-major. Copy with original dimensions.
                        ClBlastCopyMatrix(cBuf, C, mCeiled, nCeiled, mCeiled, 0, N, M, N, 0, false);
                        if (timingEnabled) { Synchronize(); unpackCTime = sw!.ElapsedTicks; }
                    }

                    if (timingEnabled)
                    {
                        double ticksPerMs = System.Diagnostics.Stopwatch.Frequency / 1000.0;
                        var total = allocTime + packATime + packBTime + packCTime + gemmTime + unpackCTime;
                        double flops = 2.0 * M * N * K;
                        double gflops = flops / (total / ticksPerMs) / 1e6;
                        Console.WriteLine($"[TIMING-SWAP {M}x{N}x{K}] Alloc={allocTime / ticksPerMs:F2}ms PackA={packATime / ticksPerMs:F2}ms PackB={packBTime / ticksPerMs:F2}ms GEMM={gemmTime / ticksPerMs:F2}ms UnpackC={unpackCTime / ticksPerMs:F2}ms Total={total / ticksPerMs:F2}ms ({gflops:F0} GFLOPS)");
                    }

                    return true;
                }
                finally
                {
                    cTemp?.Dispose();
                    bTemp?.Dispose();
                    aTemp?.Dispose();
                }
            }
            else
            {
                // GEMMK=1 path (kernel1=true): Original logic without row-major swap
                int mCeiled = CeilDiv(M, config.TileN) * config.TileN;
                int nCeiled = CeilDiv(N, config.TileM) * config.TileM;
                int kCeiled = CeilDiv(K, kUnit) * kUnit;

                int aOne = K;
                int aTwo = M;
                int bOne = N;
                int bTwo = K;
                int cOne = N;
                int cTwo = M;

                int aOneI = kCeiled;
                int aTwoI = mCeiled;
                int bOneI = nCeiled;
                int bTwoI = kCeiled;
                int cOneI = nCeiled;
                int cTwoI = mCeiled;

                bool aNoTemp = aOne == aOneI && aTwo == aTwoI;
                bool bNoTemp = bOne == bOneI && bTwo == bTwoI;
                bool cNoTemp = cOne == cOneI && cTwo == cTwoI;

                long aSize = (long)aOneI * aTwoI;
                long bSize = (long)bOneI * bTwoI;
                long cSize = (long)cOneI * cTwoI;
                if (aSize > int.MaxValue || bSize > int.MaxValue || cSize > int.MaxValue)
                    return false;

                IGpuBuffer? aTemp = null;
                IGpuBuffer? bTemp = null;
                IGpuBuffer? cTemp = null;
                IGpuBuffer aBuf = A;
                IGpuBuffer bBuf = B;
                IGpuBuffer cBuf = C;

                try
                {
                    if (!aNoTemp)
                    {
                        if (timingEnabled) { Synchronize(); sw!.Restart(); }
                        aTemp = AllocateBuffer((int)aSize);
                        aBuf = aTemp;
                        if (timingEnabled) { allocTime += sw!.ElapsedTicks; sw.Restart(); }
                        ClBlastCopyMatrix(A, aBuf, aOne, aTwo, aOne, 0, aOneI, aTwoI, aOneI, 0, true);
                        if (timingEnabled) { Synchronize(); packATime = sw!.ElapsedTicks; }
                    }

                    if (!bNoTemp)
                    {
                        if (timingEnabled) { sw!.Restart(); }
                        bTemp = AllocateBuffer((int)bSize);
                        bBuf = bTemp;
                        if (timingEnabled) { allocTime += sw!.ElapsedTicks; sw.Restart(); }
                        ClBlastCopyMatrix(B, bBuf, bOne, bTwo, bOne, 0, bOneI, bTwoI, bOneI, 0, true);
                        if (timingEnabled) { Synchronize(); packBTime = sw!.ElapsedTicks; }
                    }

                    if (!cNoTemp)
                    {
                        if (timingEnabled) { sw!.Restart(); }
                        cTemp = AllocateBuffer((int)cSize);
                        cBuf = cTemp;
                        if (timingEnabled) { allocTime += sw!.ElapsedTicks; sw.Restart(); }
                        if (beta != 0.0f)
                        {
                            ClBlastCopyMatrix(C, cBuf, cOne, cTwo, cOne, 0, cOneI, cTwoI, cOneI, 0, true);
                            if (timingEnabled) { Synchronize(); packCTime = sw!.ElapsedTicks; }
                        }
                    }

                    if (timingEnabled) { sw!.Restart(); }
                    if (!TryExecuteDynamicGemm(aBuf, bBuf, cBuf, mCeiled, nCeiled, kCeiled, alpha, beta, config))
                        return false;
                    if (timingEnabled) { Synchronize(); gemmTime = sw!.ElapsedTicks; }

                    if (!cNoTemp)
                    {
                        if (timingEnabled) { sw!.Restart(); }
                        ClBlastCopyMatrix(cBuf, C, cOneI, cTwoI, cOneI, 0, cOne, cTwo, cOne, 0, false);
                        if (timingEnabled) { Synchronize(); unpackCTime = sw!.ElapsedTicks; }
                    }

                    if (timingEnabled)
                    {
                        double ticksPerMs = System.Diagnostics.Stopwatch.Frequency / 1000.0;
                        var total = allocTime + packATime + packBTime + packCTime + gemmTime + unpackCTime;
                        Console.WriteLine($"[TIMING {M}x{N}x{K}] Alloc={allocTime / ticksPerMs:F2}ms PackA={packATime / ticksPerMs:F2}ms PackB={packBTime / ticksPerMs:F2}ms GEMM={gemmTime / ticksPerMs:F2}ms UnpackC={unpackCTime / ticksPerMs:F2}ms Total={total / ticksPerMs:F2}ms");
                    }

                    return true;
                }
                finally
                {
                    cTemp?.Dispose();
                    bTemp?.Dispose();
                    aTemp?.Dispose();
                }
            }
        }

        private static int CeilDiv(int value, int divisor)
        {
            return (value + divisor - 1) / divisor;
        }

        private void PadCopyMatrix(IGpuBuffer src, IGpuBuffer dst, int srcRows, int srcCols, int dstRows, int dstCols)
        {
            var kernel = _kernelCache["pad_copy"];
            kernel.SetArg(0, ((DirectOpenClGpuBuffer)src).Buffer.Handle);
            kernel.SetArg(1, ((DirectOpenClGpuBuffer)dst).Buffer.Handle);
            kernel.SetArg(2, srcRows);
            kernel.SetArg(3, srcCols);
            kernel.SetArg(4, dstRows);
            kernel.SetArg(5, dstCols);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(dstRows, dstCols);
            kernel.Execute2D(dstRows, dstCols, localSizeX, localSizeY);
        }

        private void PadCopyTransposeMatrix(IGpuBuffer src, IGpuBuffer dst, int srcRows, int srcCols, int dstRows, int dstCols)
        {
            var kernel = _kernelCache["pad_copy_transpose"];
            kernel.SetArg(0, ((DirectOpenClGpuBuffer)src).Buffer.Handle);
            kernel.SetArg(1, ((DirectOpenClGpuBuffer)dst).Buffer.Handle);
            kernel.SetArg(2, srcRows);
            kernel.SetArg(3, srcCols);
            kernel.SetArg(4, dstRows);
            kernel.SetArg(5, dstCols);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(dstRows, dstCols);
            kernel.Execute2D(dstRows, dstCols, localSizeX, localSizeY);
        }

        private void PadCopyFromColumnMajorMatrix(IGpuBuffer src, IGpuBuffer dst, int srcRows, int srcCols, int dstRows, int dstCols)
        {
            var kernel = _kernelCache["pad_copy_from_column_major"];
            kernel.SetArg(0, ((DirectOpenClGpuBuffer)src).Buffer.Handle);
            kernel.SetArg(1, ((DirectOpenClGpuBuffer)dst).Buffer.Handle);
            kernel.SetArg(2, srcRows);
            kernel.SetArg(3, srcCols);
            kernel.SetArg(4, dstRows);
            kernel.SetArg(5, dstCols);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(dstRows, dstCols);
            kernel.Execute2D(dstRows, dstCols, localSizeX, localSizeY);
        }

        private void CopySubmatrix(IGpuBuffer src, IGpuBuffer dst, int rows, int cols, int srcStride, int dstStride)
        {
            var kernel = _kernelCache["copy_submatrix"];
            kernel.SetArg(0, ((DirectOpenClGpuBuffer)src).Buffer.Handle);
            kernel.SetArg(1, ((DirectOpenClGpuBuffer)dst).Buffer.Handle);
            kernel.SetArg(2, rows);
            kernel.SetArg(3, cols);
            kernel.SetArg(4, srcStride);
            kernel.SetArg(5, dstStride);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(rows, cols);
            kernel.Execute2D(rows, cols, localSizeX, localSizeY);
        }

        private bool TryExecutePackedDynamicGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha, float beta, GemmConfig config)
        {
            if (IsClBlastBaselineKernel(config))
                return TryExecuteClBlastBaselineGemm(A, B, C, M, N, K, alpha, beta, config);

            int kReg = config.KReg > 0 ? config.KReg : 1;
            int kUnit = config.TileK * kReg;
            if (config.TileM <= 0 || config.TileN <= 0 || kUnit <= 0)
                return false;

            int mPad = CeilDiv(M, config.TileM) * config.TileM;
            int nPad = CeilDiv(N, config.TileN) * config.TileN;
            int kPad = CeilDiv(K, kUnit) * kUnit;

            bool useColumnMajorA = config.UseColumnMajorA || IsClBlastBaselineKernel0(config);
            bool useColumnMajorC = IsClBlastBaselineKernel0(config);
            bool needsPadding = mPad != M || nPad != N || kPad != K;
            if (!needsPadding && !useColumnMajorA && !useColumnMajorC)
                return TryExecuteDynamicGemm(A, B, C, M, N, K, alpha, beta, config);

            long aSize = (long)mPad * kPad;
            long bSize = (long)kPad * nPad;
            long cSize = (long)mPad * nPad;
            if (aSize > int.MaxValue || bSize > int.MaxValue || cSize > int.MaxValue)
                return false;

            if (EnableTuningDiagnostics)
            {
                Console.WriteLine($"[GEMM] Packed GEMM: {M}x{N}x{K} -> {mPad}x{nPad}x{kPad}");
            }

            using var aPad = AllocateBuffer(mPad * kPad);
            using var bPad = AllocateBuffer(kPad * nPad);
            using var cPad = AllocateBuffer(mPad * nPad);

            if (useColumnMajorA)
                PadCopyTransposeMatrix(A, aPad, M, K, mPad, kPad);
            else
                PadCopyMatrix(A, aPad, M, K, mPad, kPad);
            PadCopyMatrix(B, bPad, K, N, kPad, nPad);
            if (useColumnMajorC)
            {
                if (beta != 0.0f)
                    PadCopyTransposeMatrix(C, cPad, M, N, mPad, nPad);
                else
                    PadCopyMatrix(C, cPad, 0, 0, mPad, nPad);
            }
            else
            {
                if (beta != 0.0f)
                    PadCopyMatrix(C, cPad, M, N, mPad, nPad);
                else
                    PadCopyMatrix(C, cPad, 0, 0, mPad, nPad);
            }

            if (!TryExecuteDynamicGemm(aPad, bPad, cPad, mPad, nPad, kPad, alpha, beta, config))
                return false;

            if (useColumnMajorC)
                PadCopyFromColumnMajorMatrix(cPad, C, mPad, nPad, M, N);
            else
                CopySubmatrix(cPad, C, M, N, nPad, N);
            return true;
        }

        private static bool IsClBlastBaselineKernel0(GemmConfig config)
        {
            return !string.IsNullOrWhiteSpace(config.KernelName) &&
                config.KernelName.StartsWith("clblast_baseline_k0", StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsClBlastBaselineKernel1(GemmConfig config)
        {
            return !string.IsNullOrWhiteSpace(config.KernelName) &&
                config.KernelName.StartsWith("clblast_baseline_k1", StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsClBlastBaselineKernel(GemmConfig config)
        {
            return IsClBlastBaselineKernel0(config) || IsClBlastBaselineKernel1(config);
        }

        private bool TryExecuteDynamicGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha, float beta, GemmConfig config)
        {
            if (_dynamicGemm == null)
                return false;

            var validationError = DynamicGemmKernel.ValidateConfig(config);
            if (validationError != null)
            {
                if (EnableTuningDiagnostics)
                {
                    Console.WriteLine($"[GEMM] Dynamic config invalid: {validationError}");
                }
                return false;
            }

            try
            {
                // Use size-aware kernel selection that falls back to baseline for small matrices
                var kernel = _dynamicGemm.GetKernelForSize(config, M, N, K);
                _dynamicGemm.Execute(kernel, config, A, B, C, M, N, K, alpha, beta);
                return true;
            }
            catch (Exception ex)
            {
                if (EnableTuningDiagnostics)
                {
                    Console.WriteLine($"[GEMM] Dynamic kernel failed ({config.KernelName}): {ex.Message}");
                }
                return false;
            }
        }

        public void Gemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;
            var bufferC = ((DirectOpenClGpuBuffer)C).Buffer;

            bool offlineEnabled = TryGetOfflineTuningMode(out _);

            // DIAGNOSTIC TRACING (always prints for debugging)
            bool traceEnabled = Environment.GetEnvironmentVariable("AIDOTNET_GEMM_TRACE") == "1";

            if (!offlineEnabled && TryGetClBlastBaselineConfig(out var baselineConfig))
            {
                if (traceEnabled)
                    Console.WriteLine($"[GEMM-TRACE {M}x{N}x{K}] Trying CLBlast baseline (TileM={baselineConfig.TileM}, TileN={baselineConfig.TileN}, TileK={baselineConfig.TileK})");

                if (TryExecuteClBlastBaselineGemm(A, B, C, M, N, K, alpha, beta, baselineConfig))
                {
                    if (traceEnabled)
                        Console.WriteLine($"[GEMM-TRACE {M}x{N}x{K}] SUCCESS: CLBlast baseline executed");
                    return;
                }
                if (traceEnabled)
                    Console.WriteLine($"[GEMM-TRACE {M}x{N}x{K}] FALLBACK: CLBlast baseline FAILED");
            }
            else
            {
                if (traceEnabled)
                    Console.WriteLine($"[GEMM-TRACE {M}x{N}x{K}] SKIP: CLBlast baseline not available (offline={offlineEnabled})");
            }

            if (_dynamicGemm != null && M >= 128 && N >= 128 && K >= 64 &&
                TryGetTunedConfig(M, N, K, out var tunedConfig))
            {
                if (traceEnabled)
                    Console.WriteLine($"[GEMM-TRACE {M}x{N}x{K}] Trying dynamic GEMM");
                if (TryExecutePackedDynamicGemm(A, B, C, M, N, K, alpha, beta, tunedConfig))
                {
                    if (traceEnabled)
                        Console.WriteLine($"[GEMM-TRACE {M}x{N}x{K}] SUCCESS: Dynamic GEMM executed");
                    return;
                }
                if (traceEnabled)
                    Console.WriteLine($"[GEMM-TRACE {M}x{N}x{K}] FALLBACK: Dynamic GEMM FAILED");
            }

            // FALLBACK: Using our own kernels (NOT CLBlast identical!)
            if (traceEnabled)
                Console.WriteLine($"[GEMM-TRACE {M}x{N}x{K}] FALLBACK: Using built-in kernel (NOT CLBlast!)");

            // Choose kernel based on matrix size
            // Use optimized kernel for matrices >= 128 in any dimension
            if (M >= 128 && N >= 128 && K >= 64)
            {
                // Large matrix - use CLBlast-style register-blocked kernel
                // Kernel uses 16x16 work group (256 threads), each computes 4x4 outputs = 64x64 tile
                _logger?.LogWarning("[GEMM {M}x{N}x{K}] FALLBACK kernel: gemm_double_buffered (expected ~30% slower than CLBlast)", M, N, K);
                var kernel = _kernelCache["gemm_double_buffered"];

                kernel.SetArg(0, bufferA.Handle);
                kernel.SetArg(1, bufferB.Handle);
                kernel.SetArg(2, bufferC.Handle);
                kernel.SetArg(3, M);
                kernel.SetArg(4, N);
                kernel.SetArg(5, K);
                kernel.SetArg(6, alpha);
                kernel.SetArg(7, beta);

                // CRITICAL: Correct global work size calculation for tiled kernel
                // Each work group processes a 64x64 output tile
                // Work group size is 16x16 (256 threads)
                // Each thread computes 4x4 outputs
                int numTilesM = (M + GemmKernel.TILE_M - 1) / GemmKernel.TILE_M;  // Number of 64-row tiles
                int numTilesN = (N + GemmKernel.TILE_N - 1) / GemmKernel.TILE_N;  // Number of 64-col tiles
                int globalSizeX = numTilesM * GemmKernel.WG_SIZE_M;  // 16 threads per tile in M
                int globalSizeY = numTilesN * GemmKernel.WG_SIZE_N;  // 16 threads per tile in N

                kernel.Execute2D(globalSizeX, globalSizeY, GemmKernel.WG_SIZE_M, GemmKernel.WG_SIZE_N);
            }
            else
            {
                // Small matrix - use simple kernel (one thread per output)
                var kernel = _kernelCache["gemm_small"];

                kernel.SetArg(0, bufferA.Handle);
                kernel.SetArg(1, bufferB.Handle);
                kernel.SetArg(2, bufferC.Handle);
                kernel.SetArg(3, M);
                kernel.SetArg(4, N);
                kernel.SetArg(5, K);
                kernel.SetArg(6, alpha);
                kernel.SetArg(7, beta);

                // Simple kernel: one thread per output element
                var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(M, N);
                kernel.Execute2D(M, N, localSizeX, localSizeY);
            }

            // Note: Synchronization removed from inner loop - caller should sync when needed
        }

        public IGpuBuffer MatMul(IGpuBuffer A, IGpuBuffer B, int M, int N, int K)
        {
            Console.WriteLine($"[OpenClBackend.MatMul] Called: {M}x{N}x{K}");
            var C = AllocateBuffer(M * N);
            Gemm(A, B, C, M, N, K, 1.0f, 0.0f);
            // Sync only when returning buffer that might be immediately read
            _context?.Finish();
            return C;
        }

        /// <summary>
        /// CLBlast-style GEMM with RDNA1-optimized parameters.
        /// </summary>
        public void GemmClblastRdna1(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            ExecuteGemmKernel("gemm_clblast_rdna1", A, B, C, M, N, K, alpha, beta, 64, 8);
        }

        /// <summary>
        /// Medium tile GEMM kernel.
        /// </summary>
        public void GemmMediumTile(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            ExecuteGemmKernel("gemm_medium_tile", A, B, C, M, N, K, alpha, beta, 32, 16);
        }

        /// <summary>
        /// Wide vector GEMM kernel.
        /// </summary>
        public void GemmWideVec(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            ExecuteGemmKernel("gemm_wide_vec", A, B, C, M, N, K, alpha, beta, 64, 16);
        }

        /// <summary>
        /// KREG4 GEMM kernel with 4-element register blocking.
        /// </summary>
        public void GemmKreg4(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            ExecuteGemmKernel("gemm_kreg4", A, B, C, M, N, K, alpha, beta, 64, 16);
        }

        /// <summary>
        /// Prefetching GEMM kernel.
        /// </summary>
        public void GemmPrefetch(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            ExecuteGemmKernel("gemm_prefetch", A, B, C, M, N, K, alpha, beta, 64, 16);
        }


        /// <summary>
        /// Simple tiled GEMM kernel.
        /// </summary>
        public void GemmSimple(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            ExecuteGemmKernel("gemm_tiled_simple", A, B, C, M, N, K, alpha, beta, 16, 16);
        }

        /// <summary>
        /// Small tile GEMM kernel.
        /// </summary>
        public void GemmSmallTile(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            ExecuteGemmKernel("gemm_small_tile", A, B, C, M, N, K, alpha, beta, 16, 16);
        }

        /// <summary>
        /// Coalesced GEMM kernel.
        /// </summary>
        public void GemmCoalesced(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            ExecuteGemmKernel("gemm_coalesced", A, B, C, M, N, K, alpha, beta, 32, 16);
        }

        /// <summary>
        /// Vectorized tile GEMM kernel.
        /// </summary>
        public void GemmVectorizedTile(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            ExecuteGemmKernel("gemm_vectorized_tile", A, B, C, M, N, K, alpha, beta, 32, 16);
        }

        /// <summary>
        /// Low register GEMM kernel.
        /// </summary>
        public void GemmLowRegister(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            ExecuteGemmKernel("gemm_low_register", A, B, C, M, N, K, alpha, beta, 32, 16);
        }

        private void ExecuteGemmKernel(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha, float beta, int tileSize, int wgSize)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;
            var bufferC = ((DirectOpenClGpuBuffer)C).Buffer;

            var kernel = _kernelCache[kernelName];

            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, bufferB.Handle);
            kernel.SetArg(2, bufferC.Handle);
            kernel.SetArg(3, M);
            kernel.SetArg(4, N);
            kernel.SetArg(5, K);
            kernel.SetArg(6, alpha);
            kernel.SetArg(7, beta);

            int numTilesM = (M + tileSize - 1) / tileSize;
            int numTilesN = (N + tileSize - 1) / tileSize;
            int globalSizeX = numTilesM * wgSize;
            int globalSizeY = numTilesN * wgSize;

            kernel.Execute2D(globalSizeX, globalSizeY, wgSize, wgSize);
        }

        /// <summary>
        /// GPU-resident GEMM that avoids synchronization.
        /// Use this in loops where result stays on GPU for further computation.
        /// Call Synchronize() explicitly when done.
        /// </summary>
        public void GemmAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            Gemm(A, B, C, M, N, K, alpha, beta);
            // No sync - caller will sync when needed
        }

        public void BatchedGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount, float alpha = 1.0f, float beta = 0.0f)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            if (batchCount <= 0)
                throw new ArgumentException("Batch count must be positive", nameof(batchCount));

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;
            var bufferC = ((DirectOpenClGpuBuffer)C).Buffer;

            // Allocate atomic counter buffer and initialize to 0
            var counterData = new int[] { 0 };
            using var counterBuffer = new DirectOpenClIntBuffer(_context, counterData);

            var kernel = _kernelCache["gemm_batched_persistent"];

            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, bufferB.Handle);
            kernel.SetArg(2, bufferC.Handle);
            kernel.SetArg(3, M);
            kernel.SetArg(4, N);
            kernel.SetArg(5, K);
            kernel.SetArg(6, batchCount);
            kernel.SetArg(7, counterBuffer.Handle);
            kernel.SetArg(8, alpha);
            kernel.SetArg(9, beta);

            // Calculate optimal work group size based on GPU capabilities
            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(M, N);

            // Launch enough workgroups to saturate the GPU
            // Each workgroup processes batches via work-stealing until all are done
            int numWorkgroups = Math.Max(batchCount, ComputeUnits * 2);
            int globalSizeX = numWorkgroups * localSizeX;
            int globalSizeY = localSizeY;

            kernel.Execute2D(globalSizeX, globalSizeY, localSizeX, localSizeY);
            _context.Finish();  // Sync for batched - ensures all batches complete
        }

        #endregion

        #region Fused Operations

        public IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        {
            return ExecuteFusedGemm("gemm_bias_relu", A, B, bias, M, N, K);
        }

        public IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        {
            return ExecuteFusedGemm("gemm_bias_gelu", A, B, bias, M, N, K);
        }

        public IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        {
            return ExecuteFusedGemm("gemm_bias_sigmoid", A, B, bias, M, N, K);
        }

        public IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        {
            return ExecuteFusedGemm("gemm_bias_tanh", A, B, bias, M, N, K);
        }

        public IGpuBuffer GemmBias(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        {
            return ExecuteFusedGemm("gemm_bias", A, B, bias, M, N, K);
        }

        private IGpuBuffer ExecuteFusedGemm(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;
            var bufferBias = ((DirectOpenClGpuBuffer)bias).Buffer;
            var C = AllocateBuffer(M * N);
            var bufferC = ((DirectOpenClGpuBuffer)C).Buffer;

            var kernel = _kernelCache[kernelName];

            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, bufferB.Handle);
            kernel.SetArg(2, bufferBias.Handle);
            kernel.SetArg(3, bufferC.Handle);
            kernel.SetArg(4, M);
            kernel.SetArg(5, N);
            kernel.SetArg(6, K);

            // Fused kernels use 16x16 tiles with one thread per output element
            // Local work size must be 16x16 to match kernel's TILE_SIZE
            const int FusedTileSize = 16;
            int localSizeX = Math.Min(FusedTileSize, M);
            int localSizeY = Math.Min(FusedTileSize, N);

            // Ensure local sizes fit within GPU limits
            while ((ulong)(localSizeX * localSizeY) > _maxWorkGroupSize && localSizeX > 1 && localSizeY > 1)
            {
                if (localSizeX >= localSizeY)
                    localSizeX /= 2;
                else
                    localSizeY /= 2;
            }

            // Global size is M x N (one thread per output), padded to local size
            int globalSizeX = ((M + localSizeX - 1) / localSizeX) * localSizeX;
            int globalSizeY = ((N + localSizeY - 1) / localSizeY) * localSizeY;

            kernel.Execute2D(globalSizeX, globalSizeY, localSizeX, localSizeY);
            _context.Finish();  // Sync when returning buffer for immediate use

            return C;
        }

        #endregion

        #region Broadcast Operations

        public void BiasAdd(IGpuBuffer A, IGpuBuffer bias, IGpuBuffer C, int M, int N)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferBias = ((DirectOpenClGpuBuffer)bias).Buffer;
            var bufferC = ((DirectOpenClGpuBuffer)C).Buffer;

            var kernel = _kernelCache["bias_add"];

            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, bufferBias.Handle);
            kernel.SetArg(2, bufferC.Handle);
            kernel.SetArg(3, M);
            kernel.SetArg(4, N);

            // Use 2D dispatch: one thread per output element
            const int localSize = 16;
            int localSizeX = Math.Min(localSize, M);
            int localSizeY = Math.Min(localSize, N);

            // Ensure local sizes fit within GPU limits
            while ((ulong)(localSizeX * localSizeY) > _maxWorkGroupSize && localSizeX > 1 && localSizeY > 1)
            {
                if (localSizeX >= localSizeY)
                    localSizeX /= 2;
                else
                    localSizeY /= 2;
            }

            // Global size padded to local size
            int globalSizeX = ((M + localSizeX - 1) / localSizeX) * localSizeX;
            int globalSizeY = ((N + localSizeY - 1) / localSizeY) * localSizeY;

            kernel.Execute2D(globalSizeX, globalSizeY, localSizeX, localSizeY);
        }

        public void Conv2DBiasAdd(IGpuBuffer output, IGpuBuffer bias, int batch, int channels, int spatialSize)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferOutput = ((DirectOpenClGpuBuffer)output).Buffer;
            var bufferBias = ((DirectOpenClGpuBuffer)bias).Buffer;

            var kernel = _kernelCache["conv2d_bias_add"];

            kernel.SetArg(0, bufferOutput.Handle);
            kernel.SetArg(1, bufferBias.Handle);
            kernel.SetArg(2, batch);
            kernel.SetArg(3, channels);
            kernel.SetArg(4, spatialSize);

            int totalSize = batch * channels * spatialSize;
            const int localSize = 256;
            int globalSize = ((totalSize + localSize - 1) / localSize) * localSize;

            kernel.Execute1D(globalSize, localSize);
        }

        #endregion

        #region Element-wise Operations

        public void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        {
            ExecuteElementwise("add_vectors", A, B, C, size);
        }

        public void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        {
            ExecuteElementwise("subtract_vectors", A, B, C, size);
        }

        public void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        {
            ExecuteElementwise("multiply_vectors", A, B, C, size);
        }

        public void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        {
            ExecuteElementwise("divide_vectors", A, B, C, size);
        }

        public void Min(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        {
            ExecuteElementwise("min_vectors", A, B, C, size);
        }

        public void Max(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        {
            ExecuteElementwise("max_vectors", A, B, C, size);
        }

        public void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;

            var kernel = _kernelCache["scale_vector"];
            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, bufferB.Handle);
            kernel.SetArg(2, scalar);
            kernel.SetArg(3, size);

            int localSize = CalculateOptimalWorkGroupSize1D(size);
            kernel.Execute1D(size, localSize);
            // No sync - element-wise ops can be chained asynchronously
        }

        public void Power(IGpuBuffer A, IGpuBuffer B, float exponent, int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;

            var kernel = _kernelCache["power_scalar"];
            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, bufferB.Handle);
            kernel.SetArg(2, exponent);
            kernel.SetArg(3, size);

            int localSize = CalculateOptimalWorkGroupSize1D(size);
            kernel.Execute1D(size, localSize);
        }

        public void Abs(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("abs_vector", A, B, size);
        }

        public void Exp(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("exp_vector", A, B, size);
        }

        public void Exp2(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("exp2_vector", A, B, size);
        }

        public void Exp10(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("exp10_vector", A, B, size);
        }

        public void ExpM1(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("expm1_vector", A, B, size);
        }

        public void Log(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("log_vector", A, B, size);
        }

        public void Log2(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("log2_vector", A, B, size);
        }

        public void Log1P(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("log1p_vector", A, B, size);
        }

        public void Sqrt(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("sqrt_vector", A, B, size);
        }

        public void Sign(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("sign_vector", A, B, size);
        }

        // Trigonometric operations
        public void Sin(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("sin_vector", A, B, size);
        }

        public void Cos(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("cos_vector", A, B, size);
        }

        public void Tan(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("tan_vector", A, B, size);
        }

        public void Asin(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("asin_vector", A, B, size);
        }

        public void Acos(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("acos_vector", A, B, size);
        }

        public void Atan(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("atan_vector", A, B, size);
        }

        // Hyperbolic operations
        public void Sinh(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("sinh_vector", A, B, size);
        }

        public void Cosh(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("cosh_vector", A, B, size);
        }

        public void Asinh(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("asinh_vector", A, B, size);
        }

        public void Acosh(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("acosh_vector", A, B, size);
        }

        public void Atanh(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("atanh_vector", A, B, size);
        }

        // Additional unary operations
        public void Reciprocal(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("reciprocal_vector", A, B, size);
        }

        public void Cbrt(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("cbrt_vector", A, B, size);
        }

        public void Log10(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("log10_vector", A, B, size);
        }

        public void Negate(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("negate_vector", A, B, size);
        }

        public void Floor(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("floor_vector", A, B, size);
        }

        public void Ceiling(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("ceil_vector", A, B, size);
        }

        public void Round(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("round_vector", A, B, size);
        }

        public void Truncate(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteUnary("trunc_vector", A, B, size);
        }

        private void ExecuteElementwise(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;
            var bufferC = ((DirectOpenClGpuBuffer)C).Buffer;

            var kernel = _kernelCache[kernelName];
            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, bufferB.Handle);
            kernel.SetArg(2, bufferC.Handle);
            kernel.SetArg(3, size);

            int localSize = CalculateOptimalWorkGroupSize1D(size);
            kernel.Execute1D(size, localSize);
            // No sync - element-wise ops can be chained asynchronously
        }

        private void ExecuteUnary(string kernelName, IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteActivation(kernelName, A, B, size);
        }

        public void Relu(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteActivation("relu", A, B, size);
        }

        public void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteActivation("sigmoid", A, B, size);
        }

        public void Tanh(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteActivation("tanh_activation", A, B, size);
        }

        public void Gelu(IGpuBuffer A, IGpuBuffer B, int size)
        {
            ExecuteActivation("gelu", A, B, size);
        }

        private void ExecuteActivation(string kernelName, IGpuBuffer A, IGpuBuffer B, int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;

            var kernel = _kernelCache[kernelName];
            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, bufferB.Handle);
            kernel.SetArg(2, size);

            int localSize = CalculateOptimalWorkGroupSize1D(size);
            kernel.Execute1D(size, localSize);
            // No sync - activations can be chained asynchronously
        }

        public void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;

            var kernel = _kernelCache["softmax"];
            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, bufferB.Handle);
            kernel.SetArg(2, batchSize);
            kernel.SetArg(3, features);

            // One work-item per batch
            kernel.Execute1D(batchSize, 1);
            // No sync - can be chained asynchronously
        }

        public void Squash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferInput = ((DirectOpenClGpuBuffer)input).Buffer;
            var bufferOutput = ((DirectOpenClGpuBuffer)output).Buffer;

            var kernel = _kernelCache["squash"];
            kernel.SetArg(0, bufferInput.Handle);
            kernel.SetArg(1, bufferOutput.Handle);
            kernel.SetArg(2, numCapsules);
            kernel.SetArg(3, capsuleDim);
            kernel.SetArg(4, epsilon);

            int localSize = CalculateOptimalWorkGroupSize1D(numCapsules);
            kernel.Execute1D(numCapsules, localSize);
        }

        public void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferGradOutput = ((DirectOpenClGpuBuffer)gradOutput).Buffer;
            var bufferInput = ((DirectOpenClGpuBuffer)input).Buffer;
            var bufferGradInput = ((DirectOpenClGpuBuffer)gradInput).Buffer;

            var kernel = _kernelCache["squash_backward"];
            kernel.SetArg(0, bufferGradOutput.Handle);
            kernel.SetArg(1, bufferInput.Handle);
            kernel.SetArg(2, bufferGradInput.Handle);
            kernel.SetArg(3, numCapsules);
            kernel.SetArg(4, capsuleDim);
            kernel.SetArg(5, epsilon);

            int localSize = CalculateOptimalWorkGroupSize1D(numCapsules);
            kernel.Execute1D(numCapsules, localSize);
        }

        public void CapsulePredictions(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
            int batchSize, int inputCapsules, int inputDim, int outputCapsules, int outputDim)
        {
            throw new NotImplementedException("CapsulePredictions kernel not yet implemented for OpenCL backend");
        }

        public void CapsuleTransform(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
            int batchSize, int inputCapsules, int inputDim, int numCapsules, int capsuleDim)
        {
            throw new NotImplementedException("CapsuleTransform kernel not yet implemented for OpenCL backend");
        }

        public void CapsuleWeightedSum(IGpuBuffer coupling, IGpuBuffer predictions, IGpuBuffer output,
            int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
        {
            throw new NotImplementedException("CapsuleWeightedSum kernel not yet implemented for OpenCL backend");
        }

        public void CapsuleAgreement(IGpuBuffer predictions, IGpuBuffer output, IGpuBuffer agreement,
            int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
        {
            throw new NotImplementedException("CapsuleAgreement kernel not yet implemented for OpenCL backend");
        }

        public void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferInput = ((DirectOpenClGpuBuffer)input).Buffer;
            var bufferOutput = ((DirectOpenClGpuBuffer)output).Buffer;

            var kernel = _kernelCache["tile_batch"];
            kernel.SetArg(0, bufferInput.Handle);
            kernel.SetArg(1, bufferOutput.Handle);
            kernel.SetArg(2, repeats);
            kernel.SetArg(3, innerSize);

            int totalSize = repeats * innerSize;
            int localSize = CalculateOptimalWorkGroupSize1D(totalSize);
            kernel.Execute1D(totalSize, localSize);
        }

        public void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferInput = ((DirectOpenClGpuBuffer)input).Buffer;
            var bufferOutput = ((DirectOpenClGpuBuffer)output).Buffer;

            var kernel = _kernelCache["tile_axis"];
            kernel.SetArg(0, bufferInput.Handle);
            kernel.SetArg(1, bufferOutput.Handle);
            kernel.SetArg(2, outerSize);
            kernel.SetArg(3, axisSize);
            kernel.SetArg(4, innerSize);
            kernel.SetArg(5, repeats);

            int totalSize = outerSize * axisSize * repeats * innerSize;
            int localSize = CalculateOptimalWorkGroupSize1D(totalSize);
            kernel.Execute1D(totalSize, localSize);
        }

        #endregion

        #region Reduction Operations

        public float Sum(IGpuBuffer A, int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            int localSize = CalculateOptimalWorkGroupSize1D(size);
            int groupCount = (size + localSize - 1) / localSize;

            using var partialBuffer = AllocateBuffer(groupCount);
            var partial = ((DirectOpenClGpuBuffer)partialBuffer).Buffer;

            var kernel = _kernelCache["reduce_sum"];
            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, partial.Handle);
            kernel.SetLocalArg(2, localSize * sizeof(float));
            kernel.SetArg(3, size);

            kernel.Execute1D(size, localSize);
            _context.Finish();

            var partials = DownloadBuffer(partialBuffer);
            float sum = 0.0f;
            for (int i = 0; i < partials.Length; i++)
                sum += partials[i];
            return sum;
        }

        public float Max(IGpuBuffer A, int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            int localSize = CalculateOptimalWorkGroupSize1D(size);
            int groupCount = (size + localSize - 1) / localSize;

            using var partialBuffer = AllocateBuffer(groupCount);
            var partial = ((DirectOpenClGpuBuffer)partialBuffer).Buffer;

            var kernel = _kernelCache["reduce_max"];
            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, partial.Handle);
            kernel.SetLocalArg(2, localSize * sizeof(float));
            kernel.SetArg(3, size);

            kernel.Execute1D(size, localSize);
            _context.Finish();

            var partials = DownloadBuffer(partialBuffer);
            float max = float.MinValue;
            for (int i = 0; i < partials.Length; i++)
                if (partials[i] > max) max = partials[i];
            return max;
        }

        public void SumAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;

            var kernel = _kernelCache["sum_axis"];
            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, bufferB.Handle);
            kernel.SetArg(2, outerSize);
            kernel.SetArg(3, reduceSize);

            int localSize = CalculateOptimalWorkGroupSize1D(outerSize);
            kernel.Execute1D(outerSize, localSize);
            // No sync - can be chained asynchronously
        }

        #endregion

        #region Sparse Operations (2:4 Structured Sparsity)

        public IGpuBuffer AllocateByteBuffer(int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            // Allocate as byte buffer (size in bytes, not floats)
            var buffer = new DirectOpenClByteBuffer(_context, size);
            return new DirectOpenClGpuByteBuffer(buffer);
        }

        public void Enforce2x4Sparsity(IGpuBuffer denseInput, IGpuBuffer sparseValues, IGpuBuffer sparseIndices, int M, int K)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            if (K % 4 != 0)
                throw new ArgumentException("K must be divisible by 4 for 2:4 sparsity", nameof(K));

            var bufferInput = ((DirectOpenClGpuBuffer)denseInput).Buffer;
            var bufferValues = ((DirectOpenClGpuBuffer)sparseValues).Buffer;
            var bufferIndices = ((DirectOpenClGpuByteBuffer)sparseIndices).Buffer;

            var kernel = _kernelCache["enforce_2_4_sparsity"];
            kernel.SetArg(0, bufferInput.Handle);
            kernel.SetArg(1, bufferValues.Handle);
            kernel.SetArg(2, bufferIndices.Handle);
            kernel.SetArg(3, M);
            kernel.SetArg(4, K);

            int numGroups = K / 4;
            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(M, numGroups);
            kernel.Execute2D(M, numGroups, localSizeX, localSizeY);
            // No sync - can be chained asynchronously
        }

        public void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            if (K % 4 != 0)
                throw new ArgumentException("K must be divisible by 4 for 2:4 sparsity", nameof(K));

            var bufferValues = ((DirectOpenClGpuBuffer)sparseValues).Buffer;
            var bufferIndices = ((DirectOpenClGpuByteBuffer)sparseIndices).Buffer;
            var bufferOutput = ((DirectOpenClGpuBuffer)denseOutput).Buffer;

            var kernel = _kernelCache["decompress_2_4_sparse"];
            kernel.SetArg(0, bufferValues.Handle);
            kernel.SetArg(1, bufferIndices.Handle);
            kernel.SetArg(2, bufferOutput.Handle);
            kernel.SetArg(3, M);
            kernel.SetArg(4, K);

            int numGroups = K / 4;
            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(M, numGroups);
            kernel.Execute2D(M, numGroups, localSizeX, localSizeY);
            // No sync - can be chained asynchronously
        }

        public void SparseGemm(
            IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices,
            IGpuBuffer B, IGpuBuffer C,
            int M, int N, int K,
            float alpha = 1.0f, float beta = 0.0f)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            if (K % 4 != 0)
                throw new ArgumentException("K must be divisible by 4 for 2:4 sparsity", nameof(K));

            var bufferAValues = ((DirectOpenClGpuBuffer)sparseAValues).Buffer;
            var bufferAIndices = ((DirectOpenClGpuByteBuffer)sparseAIndices).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;
            var bufferC = ((DirectOpenClGpuBuffer)C).Buffer;

            var kernel = _kernelCache["sparse_gemm_2_4"];
            kernel.SetArg(0, bufferAValues.Handle);
            kernel.SetArg(1, bufferAIndices.Handle);
            kernel.SetArg(2, bufferB.Handle);
            kernel.SetArg(3, bufferC.Handle);
            kernel.SetArg(4, M);
            kernel.SetArg(5, N);
            kernel.SetArg(6, K);
            kernel.SetArg(7, alpha);
            kernel.SetArg(8, beta);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(M, N);
            kernel.Execute2D(M, N, localSizeX, localSizeY);
            // No sync - can be chained asynchronously
        }

        public IGpuBuffer SparseGemmBiasRelu(
            IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices,
            IGpuBuffer B, IGpuBuffer bias,
            int M, int N, int K)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            if (K % 4 != 0)
                throw new ArgumentException("K must be divisible by 4 for 2:4 sparsity", nameof(K));

            var bufferAValues = ((DirectOpenClGpuBuffer)sparseAValues).Buffer;
            var bufferAIndices = ((DirectOpenClGpuByteBuffer)sparseAIndices).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;
            var bufferBias = ((DirectOpenClGpuBuffer)bias).Buffer;
            var C = AllocateBuffer(M * N);
            var bufferC = ((DirectOpenClGpuBuffer)C).Buffer;

            var kernel = _kernelCache["sparse_gemm_bias_relu"];
            kernel.SetArg(0, bufferAValues.Handle);
            kernel.SetArg(1, bufferAIndices.Handle);
            kernel.SetArg(2, bufferB.Handle);
            kernel.SetArg(3, bufferBias.Handle);
            kernel.SetArg(4, bufferC.Handle);
            kernel.SetArg(5, M);
            kernel.SetArg(6, N);
            kernel.SetArg(7, K);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(M, N);
            kernel.Execute2D(M, N, localSizeX, localSizeY);
            _context.Finish();  // Sync when returning buffer for immediate use

            return C;
        }

        #endregion

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
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var kernel = _kernelCache["csr_spmm"];
            var bufferValues = ((DirectOpenClGpuBuffer)csrValues).Buffer;
            var bufferColIndices = ((DirectOpenClGpuBuffer)csrColIndices).Buffer;
            var bufferRowPointers = ((DirectOpenClGpuBuffer)csrRowPointers).Buffer;
            var bufferDenseB = ((DirectOpenClGpuBuffer)denseB).Buffer;
            var bufferOutput = ((DirectOpenClGpuBuffer)output).Buffer;

            kernel.SetArg(0, bufferValues.Handle);
            kernel.SetArg(1, bufferColIndices.Handle);
            kernel.SetArg(2, bufferRowPointers.Handle);
            kernel.SetArg(3, bufferDenseB.Handle);
            kernel.SetArg(4, bufferOutput.Handle);
            kernel.SetArg(5, M);
            kernel.SetArg(6, K);
            kernel.SetArg(7, N);
            kernel.SetArg(8, nnz);

            // Global work size: [N, M] - columns then rows for coalesced memory access
            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(N, M);
            kernel.Execute2D(N, M, localSizeX, localSizeY);
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
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var kernel = _kernelCache["csr_spmm_bias"];
            var bufferValues = ((DirectOpenClGpuBuffer)csrValues).Buffer;
            var bufferColIndices = ((DirectOpenClGpuBuffer)csrColIndices).Buffer;
            var bufferRowPointers = ((DirectOpenClGpuBuffer)csrRowPointers).Buffer;
            var bufferDenseB = ((DirectOpenClGpuBuffer)denseB).Buffer;
            var bufferBias = ((DirectOpenClGpuBuffer)bias).Buffer;
            var bufferOutput = ((DirectOpenClGpuBuffer)output).Buffer;

            kernel.SetArg(0, bufferValues.Handle);
            kernel.SetArg(1, bufferColIndices.Handle);
            kernel.SetArg(2, bufferRowPointers.Handle);
            kernel.SetArg(3, bufferDenseB.Handle);
            kernel.SetArg(4, bufferBias.Handle);
            kernel.SetArg(5, bufferOutput.Handle);
            kernel.SetArg(6, M);
            kernel.SetArg(7, K);
            kernel.SetArg(8, N);
            kernel.SetArg(9, nnz);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(N, M);
            kernel.Execute2D(N, M, localSizeX, localSizeY);
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
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            // First zero the output buffer
            var zeroKernel = _kernelCache["zero_buffer"];
            var bufferOutput = ((DirectOpenClGpuBuffer)output).Buffer;
            int outputSize = numNodes * features;
            zeroKernel.SetArg(0, bufferOutput.Handle);
            zeroKernel.SetArg(1, outputSize);
            zeroKernel.Execute1D(outputSize, 256);
            _context.Finish();

            // Then scatter-add
            var kernel = _kernelCache["scatter_add_edges"];
            var bufferInput = ((DirectOpenClGpuBuffer)input).Buffer;
            var bufferSource = ((DirectOpenClGpuBuffer)sourceIndices).Buffer;
            var bufferTarget = ((DirectOpenClGpuBuffer)targetIndices).Buffer;

            kernel.SetArg(0, bufferInput.Handle);
            kernel.SetArg(1, bufferSource.Handle);
            kernel.SetArg(2, bufferTarget.Handle);

            if (edgeValues is not null)
            {
                var bufferEdgeValues = ((DirectOpenClGpuBuffer)edgeValues).Buffer;
                kernel.SetArg(3, bufferEdgeValues.Handle);
            }
            else
            {
                kernel.SetArg(3, IntPtr.Zero);
            }

            kernel.SetArg(4, bufferOutput.Handle);
            kernel.SetArg(5, numNodes);
            kernel.SetArg(6, numEdges);
            kernel.SetArg(7, features);
            kernel.SetArg(8, edgeValues is not null ? 1 : 0);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(features, numEdges);
            kernel.Execute2D(features, numEdges, localSizeX, localSizeY);
        }

        /// <inheritdoc/>
        public void CsrSegmentedMax(
            IGpuBuffer csrColIndices,
            IGpuBuffer csrRowPointers,
            IGpuBuffer input,
            IGpuBuffer output,
            int M, int K, int N)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var kernel = _kernelCache["csr_segmented_max"];
            var bufferColIndices = ((DirectOpenClGpuBuffer)csrColIndices).Buffer;
            var bufferRowPointers = ((DirectOpenClGpuBuffer)csrRowPointers).Buffer;
            var bufferInput = ((DirectOpenClGpuBuffer)input).Buffer;
            var bufferOutput = ((DirectOpenClGpuBuffer)output).Buffer;

            kernel.SetArg(0, bufferColIndices.Handle);
            kernel.SetArg(1, bufferRowPointers.Handle);
            kernel.SetArg(2, bufferInput.Handle);
            kernel.SetArg(3, bufferOutput.Handle);
            kernel.SetArg(4, M);
            kernel.SetArg(5, K);
            kernel.SetArg(6, N);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(N, M);
            kernel.Execute2D(N, M, localSizeX, localSizeY);
        }

        /// <inheritdoc/>
        public void CsrSegmentedMin(
            IGpuBuffer csrColIndices,
            IGpuBuffer csrRowPointers,
            IGpuBuffer input,
            IGpuBuffer output,
            int M, int K, int N)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var kernel = _kernelCache["csr_segmented_min"];
            var bufferColIndices = ((DirectOpenClGpuBuffer)csrColIndices).Buffer;
            var bufferRowPointers = ((DirectOpenClGpuBuffer)csrRowPointers).Buffer;
            var bufferInput = ((DirectOpenClGpuBuffer)input).Buffer;
            var bufferOutput = ((DirectOpenClGpuBuffer)output).Buffer;

            kernel.SetArg(0, bufferColIndices.Handle);
            kernel.SetArg(1, bufferRowPointers.Handle);
            kernel.SetArg(2, bufferInput.Handle);
            kernel.SetArg(3, bufferOutput.Handle);
            kernel.SetArg(4, M);
            kernel.SetArg(5, K);
            kernel.SetArg(6, N);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(N, M);
            kernel.Execute2D(N, M, localSizeX, localSizeY);
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
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var kernel = _kernelCache["csr_segmented_stddev"];
            var bufferColIndices = ((DirectOpenClGpuBuffer)csrColIndices).Buffer;
            var bufferRowPointers = ((DirectOpenClGpuBuffer)csrRowPointers).Buffer;
            var bufferInput = ((DirectOpenClGpuBuffer)input).Buffer;
            var bufferOutput = ((DirectOpenClGpuBuffer)output).Buffer;

            kernel.SetArg(0, bufferColIndices.Handle);
            kernel.SetArg(1, bufferRowPointers.Handle);
            kernel.SetArg(2, bufferInput.Handle);
            kernel.SetArg(3, bufferOutput.Handle);
            kernel.SetArg(4, M);
            kernel.SetArg(5, K);
            kernel.SetArg(6, N);
            kernel.SetArg(7, epsilon);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(N, M);
            kernel.Execute2D(N, M, localSizeX, localSizeY);
        }

        #endregion

        public void Synchronize()
        {
            _context?.Finish();
        }

        #region IAsyncGpuBackend Implementation

        /// <inheritdoc/>
        public IGpuStream CreateStream(GpuStreamType streamType)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            return new OpenClCommandQueue(this, _context.Context, _context.Device, streamType, enableProfiling: false);
        }

        /// <inheritdoc/>
        public IGpuStream CreateStream(GpuStreamType streamType, int priority)
        {
            // OpenCL doesn't support queue priorities, so we ignore the priority parameter
            return CreateStream(streamType);
        }

        /// <inheritdoc/>
        public IGpuEvent CreateEvent()
        {
            // Create an unrecorded event (will be recorded later)
            return new OpenClEvent(this, IntPtr.Zero, null, profilingEnabled: false);
        }

        /// <inheritdoc/>
        public IGpuEvent CreateEvent(bool enableTiming)
        {
            return new OpenClEvent(this, IntPtr.Zero, null, profilingEnabled: enableTiming);
        }

        /// <inheritdoc/>
        public void RecordEvent(IGpuEvent gpuEvent, IGpuStream stream)
        {
            if (gpuEvent is not OpenClEvent openClEvent)
                throw new ArgumentException("Event must be an OpenClEvent", nameof(gpuEvent));

            openClEvent.Record(stream);
        }

        /// <inheritdoc/>
        public void StreamWaitEvent(IGpuStream stream, IGpuEvent gpuEvent)
        {
            if (stream is not OpenClCommandQueue openClQueue)
                throw new ArgumentException("Stream must be an OpenClCommandQueue", nameof(stream));

            openClQueue.WaitEvent(gpuEvent);
        }

        /// <inheritdoc/>
        public GpuSyncPoint CreateSyncPoint(IGpuStream stream)
        {
            if (stream is not OpenClCommandQueue openClQueue)
                throw new ArgumentException("Stream must be an OpenClCommandQueue", nameof(stream));

            return new OpenClSyncPoint(this, openClQueue);
        }

        /// <inheritdoc/>
        public GpuSyncPoint CreateSyncPoint()
        {
            if (_defaultStream == null)
                throw new InvalidOperationException("Backend not initialized");

            return new OpenClSyncPoint(this, _defaultStream);
        }

        /// <inheritdoc/>
        public unsafe void UploadBufferAsync(float[] data, IGpuBuffer buffer, IGpuStream stream)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (buffer == null) throw new ArgumentNullException(nameof(buffer));
            if (stream == null) throw new ArgumentNullException(nameof(stream));

            var openClBuffer = (DirectOpenClGpuBuffer)buffer;
            fixed (float* dataPtr = data)
            {
                int err = OpenClNativeBindings.EnqueueWriteBuffer(
                    stream.Handle,
                    openClBuffer.Buffer.Handle,
                    0, // non-blocking
                    UIntPtr.Zero,
                    (UIntPtr)(data.Length * sizeof(float)),
                    (IntPtr)dataPtr,
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"clEnqueueWriteBuffer failed: {err}");
            }
        }

        /// <inheritdoc/>
        public unsafe void UploadBufferAsync(ReadOnlySpan<float> data, IGpuBuffer buffer, IGpuStream stream)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");
            if (buffer == null) throw new ArgumentNullException(nameof(buffer));
            if (stream == null) throw new ArgumentNullException(nameof(stream));

            var openClBuffer = (DirectOpenClGpuBuffer)buffer;
            fixed (float* dataPtr = data)
            {
                int err = OpenClNativeBindings.EnqueueWriteBuffer(
                    stream.Handle,
                    openClBuffer.Buffer.Handle,
                    0, // non-blocking
                    UIntPtr.Zero,
                    (UIntPtr)(data.Length * sizeof(float)),
                    (IntPtr)dataPtr,
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"clEnqueueWriteBuffer failed: {err}");
            }
        }

        /// <inheritdoc/>
        public unsafe void DownloadBufferAsync(IGpuBuffer buffer, float[] destination, IGpuStream stream)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");
            if (buffer == null) throw new ArgumentNullException(nameof(buffer));
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (stream == null) throw new ArgumentNullException(nameof(stream));

            var openClBuffer = (DirectOpenClGpuBuffer)buffer;
            fixed (float* destPtr = destination)
            {
                int err = OpenClNativeBindings.EnqueueReadBuffer(
                    stream.Handle,
                    openClBuffer.Buffer.Handle,
                    0, // non-blocking
                    UIntPtr.Zero,
                    (UIntPtr)(destination.Length * sizeof(float)),
                    (IntPtr)destPtr,
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"clEnqueueReadBuffer failed: {err}");
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
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (stream == null) throw new ArgumentNullException(nameof(stream));

            var srcBuffer = (DirectOpenClGpuBuffer)source;
            var dstBuffer = (DirectOpenClGpuBuffer)destination;

            int err = OpenClNativeBindings.EnqueueCopyBuffer(
                stream.Handle,
                srcBuffer.Buffer.Handle,
                dstBuffer.Buffer.Handle,
                UIntPtr.Zero,
                UIntPtr.Zero,
                (UIntPtr)(size * sizeof(float)),
                0,
                IntPtr.Zero,
                IntPtr.Zero);

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"clEnqueueCopyBuffer failed: {err}");
        }

        /// <inheritdoc/>
        public void GemmAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
            float alpha, float beta, IGpuStream stream)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;
            var bufferC = ((DirectOpenClGpuBuffer)C).Buffer;

            // Use the simple tiled kernel for async operations on specific streams
            var kernel = _kernelCache["gemm_tiled_simple"];

            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, bufferB.Handle);
            kernel.SetArg(2, bufferC.Handle);
            kernel.SetArg(3, M);
            kernel.SetArg(4, N);
            kernel.SetArg(5, K);
            kernel.SetArg(6, alpha);
            kernel.SetArg(7, beta);

            const int tileSize = 16;
            const int wgSize = 16;
            int numTilesM = (M + tileSize - 1) / tileSize;
            int numTilesN = (N + tileSize - 1) / tileSize;
            int globalSizeX = numTilesM * wgSize;
            int globalSizeY = numTilesN * wgSize;

            // Execute on the specified stream
            kernel.Execute2DOnQueue(stream.Handle, globalSizeX, globalSizeY, wgSize, wgSize);
        }

        /// <inheritdoc/>
        public void FusedGemmBiasActivationAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output,
            int M, int N, int K, FusedActivationType activation, IGpuStream stream)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            // Map activation to fused kernel name
            string kernelName = activation switch
            {
                FusedActivationType.ReLU => "gemm_bias_relu",
                FusedActivationType.Sigmoid => "gemm_bias_sigmoid",
                FusedActivationType.Tanh => "gemm_bias_tanh",
                FusedActivationType.None => "gemm_bias",
                _ => throw new NotSupportedException($"Activation type {activation} not supported for fused GEMM")
            };

            if (!_kernelCache.TryGetValue(kernelName, out var kernel))
                throw new InvalidOperationException($"Fused kernel not found: {kernelName}");

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;
            var bufferBias = ((DirectOpenClGpuBuffer)bias).Buffer;
            var bufferC = ((DirectOpenClGpuBuffer)output).Buffer;

            kernel.SetArg(0, bufferA.Handle);
            kernel.SetArg(1, bufferB.Handle);
            kernel.SetArg(2, bufferBias.Handle);
            kernel.SetArg(3, bufferC.Handle);
            kernel.SetArg(4, M);
            kernel.SetArg(5, N);
            kernel.SetArg(6, K);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(M, N);
            kernel.Execute2DOnQueue(stream.Handle, M, N, localSizeX, localSizeY);
        }

        /// <inheritdoc/>
        public void SynchronizeStream(IGpuStream stream)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));

            int err = OpenClNativeBindings.Finish(stream.Handle);
            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"clFinish failed: {err}");
        }

        /// <inheritdoc/>
        public bool QueryStreamComplete(IGpuStream stream)
        {
            if (stream is not OpenClCommandQueue openClQueue)
                throw new ArgumentException("Stream must be an OpenClCommandQueue", nameof(stream));

            return openClQueue.Query();
        }

        /// <inheritdoc/>
        public bool QueryEventComplete(IGpuEvent gpuEvent)
        {
            if (gpuEvent is not OpenClEvent openClEvent)
                throw new ArgumentException("Event must be an OpenClEvent", nameof(gpuEvent));

            return openClEvent.Query();
        }

        /// <inheritdoc/>
        public float GetEventElapsedTime(IGpuEvent start, IGpuEvent end)
        {
            if (start is not OpenClEvent openClStart)
                throw new ArgumentException("Start event must be an OpenClEvent", nameof(start));
            if (end is not OpenClEvent openClEnd)
                throw new ArgumentException("End event must be an OpenClEvent", nameof(end));

            return openClEnd.GetElapsedTime(openClStart);
        }

        #endregion

        #region Profiling and Diagnostics

        /// <summary>
        /// Gets whether profiling is enabled on this backend.
        /// Profiling requires a command queue created with CL_QUEUE_PROFILING_ENABLE.
        /// </summary>
        public bool IsProfilingEnabled => _context?.IsProfilingEnabled ?? false;

        /// <summary>
        /// Gets GPU device information for diagnostics.
        /// </summary>
        public GpuDeviceInfo GetDeviceInfo()
        {
            if (_context == null)
                return new GpuDeviceInfo();

            return new GpuDeviceInfo
            {
                DeviceName = DeviceName,
                DeviceVendor = DeviceVendor,
                ComputeUnits = ComputeUnits,
                GlobalMemoryBytes = GlobalMemoryBytes,
                LocalMemoryBytes = LocalMemoryBytes,
                MaxWorkGroupSize = (int)_maxWorkGroupSize,
                MaxWorkItemSizes = _maxWorkItemSizes,
                ClockFrequencyMHz = _context.ClockFrequencyMHz,
                SupportsFp16 = _supportsFp16,
                SupportsSubgroups = _supportsSubgroups,
                // Theoretical peak GFLOPS: 2 ops per FMA * CUs * SIMDs per CU * wavefront size * clock
                // For AMD RDNA: ~64 FMA ops per CU per cycle
                // For NVIDIA: ~128 FMA ops per SM per cycle
                TheoreticalPeakGflops = EstimateTheoreticalGflops()
            };
        }

        private double EstimateTheoreticalGflops()
        {
            if (_context == null) return 0;

            // Estimate based on vendor and compute units
            double opsPerCuPerCycle;
            if (DeviceVendor.Contains("NVIDIA"))
            {
                // NVIDIA: ~128 FP32 ops per SM per cycle (Ampere/Ada)
                opsPerCuPerCycle = 128;
            }
            else if (DeviceVendor.Contains("AMD") || DeviceVendor.Contains("Advanced Micro"))
            {
                // AMD RDNA: ~128 FP32 ops per CU per cycle (dual issue)
                opsPerCuPerCycle = 128;
            }
            else
            {
                // Intel/Other: ~32 FP32 ops per EU per cycle (conservative)
                opsPerCuPerCycle = 32;
            }

            // GFLOPS = CUs * ops/CU/cycle * MHz * 1e-3
            return ComputeUnits * opsPerCuPerCycle * _context.ClockFrequencyMHz / 1000.0;
        }

        /// <summary>
        /// Executes GEMM with comprehensive profiling diagnostics.
        /// Uses OpenCL profiling events for accurate GPU timing.
        /// </summary>
        /// <param name="A">Input matrix A (M x K)</param>
        /// <param name="B">Input matrix B (K x N)</param>
        /// <param name="C">Output matrix C (M x N)</param>
        /// <param name="M">Number of rows in A and C</param>
        /// <param name="N">Number of columns in B and C</param>
        /// <param name="K">Number of columns in A / rows in B</param>
        /// <param name="alpha">Scalar multiplier for A*B</param>
        /// <param name="beta">Scalar multiplier for C</param>
        /// <returns>Detailed timing and performance diagnostics</returns>
        public GemmDiagnostics GemmWithDiagnostics(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var diagnostics = new GemmDiagnostics
            {
                M = M,
                N = N,
                K = K,
                KernelName = "unknown",
                IsProfilingAvailable = _context.IsProfilingEnabled
            };

            var sw = Stopwatch.StartNew();
            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;
            var bufferC = ((DirectOpenClGpuBuffer)C).Buffer;

            // Calculate theoretical metrics
            long flops = 2L * M * N * K; // 2 ops (multiply + add) per output element per K
            long bytesRead = (long)(M * K + K * N) * sizeof(float);  // A + B
            long bytesWritten = (long)M * N * sizeof(float);  // C
            long totalBytes = bytesRead + bytesWritten;
            diagnostics.FlopsRequired = flops;
            diagnostics.BytesTransferred = totalBytes;

            IntPtr kernelEvent = IntPtr.Zero;

            try
            {
                if (M >= 128 && N >= 128 && K >= 64)
                {
                    // Large matrix - use optimized kernel
                    diagnostics.KernelName = "gemm_double_buffered";
                    var kernel = _kernelCache["gemm_double_buffered"];

                    kernel.SetArg(0, bufferA.Handle);
                    kernel.SetArg(1, bufferB.Handle);
                    kernel.SetArg(2, bufferC.Handle);
                    kernel.SetArg(3, M);
                    kernel.SetArg(4, N);
                    kernel.SetArg(5, K);
                    kernel.SetArg(6, alpha);
                    kernel.SetArg(7, beta);

                    int numTilesM = (M + GemmKernel.TILE_M - 1) / GemmKernel.TILE_M;
                    int numTilesN = (N + GemmKernel.TILE_N - 1) / GemmKernel.TILE_N;
                    int globalSizeX = numTilesM * GemmKernel.WG_SIZE_M;
                    int globalSizeY = numTilesN * GemmKernel.WG_SIZE_N;

                    diagnostics.GlobalSizeX = globalSizeX;
                    diagnostics.GlobalSizeY = globalSizeY;
                    diagnostics.LocalSizeX = GemmKernel.WG_SIZE_M;
                    diagnostics.LocalSizeY = GemmKernel.WG_SIZE_N;
                    diagnostics.WorkItemsLaunched = globalSizeX * globalSizeY;
                    diagnostics.WorkGroupsLaunched = numTilesM * numTilesN;

                    kernelEvent = kernel.Execute2DProfiled(globalSizeX, globalSizeY, GemmKernel.WG_SIZE_M, GemmKernel.WG_SIZE_N);
                }
                else
                {
                    // Small matrix - use simple kernel
                    diagnostics.KernelName = "gemm_small";
                    var kernel = _kernelCache["gemm_small"];

                    kernel.SetArg(0, bufferA.Handle);
                    kernel.SetArg(1, bufferB.Handle);
                    kernel.SetArg(2, bufferC.Handle);
                    kernel.SetArg(3, M);
                    kernel.SetArg(4, N);
                    kernel.SetArg(5, K);
                    kernel.SetArg(6, alpha);
                    kernel.SetArg(7, beta);

                    var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(M, N);
                    diagnostics.GlobalSizeX = M;
                    diagnostics.GlobalSizeY = N;
                    diagnostics.LocalSizeX = localSizeX;
                    diagnostics.LocalSizeY = localSizeY;
                    diagnostics.WorkItemsLaunched = M * N;
                    diagnostics.WorkGroupsLaunched = ((M + localSizeX - 1) / localSizeX) * ((N + localSizeY - 1) / localSizeY);

                    kernelEvent = kernel.Execute2DProfiled(M, N, localSizeX, localSizeY);
                }

                // Wait for kernel completion and get profiling info
                if (kernelEvent != IntPtr.Zero && _context.IsProfilingEnabled)
                {
                    // Wait for event to complete
                    var events = new IntPtr[] { kernelEvent };
                    int waitErr = OpenClNativeBindings.WaitForEvents(1, events);

                    if (waitErr == OpenClNativeBindings.CL_SUCCESS)
                    {
                        // Get GPU timestamps (in nanoseconds)
                        ulong queued = OpenClNativeBindings.GetEventProfilingInfoULong(kernelEvent, OpenClNativeBindings.CL_PROFILING_COMMAND_QUEUED);
                        ulong submit = OpenClNativeBindings.GetEventProfilingInfoULong(kernelEvent, OpenClNativeBindings.CL_PROFILING_COMMAND_SUBMIT);
                        ulong start = OpenClNativeBindings.GetEventProfilingInfoULong(kernelEvent, OpenClNativeBindings.CL_PROFILING_COMMAND_START);
                        ulong end = OpenClNativeBindings.GetEventProfilingInfoULong(kernelEvent, OpenClNativeBindings.CL_PROFILING_COMMAND_END);

                        diagnostics.QueueToSubmitNs = submit - queued;
                        diagnostics.SubmitToStartNs = start - submit;
                        diagnostics.KernelExecutionNs = end - start;
                        diagnostics.TotalGpuTimeNs = end - queued;

                        // Calculate achieved performance
                        double kernelTimeSeconds = diagnostics.KernelExecutionNs / 1e9;
                        if (kernelTimeSeconds > 0)
                        {
                            diagnostics.AchievedGflops = flops / kernelTimeSeconds / 1e9;
                            diagnostics.AchievedBandwidthGBps = totalBytes / kernelTimeSeconds / 1e9;
                        }
                    }
                    else
                    {
                        diagnostics.ProfilingError = $"WaitForEvents failed with error {waitErr}";
                    }
                }
                else
                {
                    // Fallback to wall clock timing
                    _context.Finish();
                    diagnostics.WallClockMs = sw.Elapsed.TotalMilliseconds;
                    if (diagnostics.WallClockMs > 0)
                    {
                        diagnostics.AchievedGflops = flops / (diagnostics.WallClockMs / 1000.0) / 1e9;
                        diagnostics.AchievedBandwidthGBps = totalBytes / (diagnostics.WallClockMs / 1000.0) / 1e9;
                    }
                }
            }
            finally
            {
                // Release the event
                if (kernelEvent != IntPtr.Zero)
                {
                    OpenClNativeBindings.ReleaseEvent(kernelEvent);
                }
            }

            sw.Stop();
            if (diagnostics.WallClockMs == 0)
            {
                diagnostics.WallClockMs = sw.Elapsed.TotalMilliseconds;
            }

            // Calculate efficiency metrics
            var deviceInfo = GetDeviceInfo();
            if (deviceInfo.TheoreticalPeakGflops > 0)
            {
                diagnostics.ComputeEfficiency = diagnostics.AchievedGflops / deviceInfo.TheoreticalPeakGflops * 100;
            }

            // Estimate memory bound vs compute bound
            // Arithmetic intensity = FLOPS / Bytes
            double arithmeticIntensity = (double)flops / totalBytes;
            diagnostics.ArithmeticIntensity = arithmeticIntensity;

            // Roofline analysis: if achieved GFLOPS is limited by bandwidth
            // peak_compute_bound_gflops = theoretical_peak
            // peak_memory_bound_gflops = bandwidth * arithmetic_intensity
            // If achieved is closer to memory bound, we're memory limited
            if (diagnostics.AchievedBandwidthGBps > 0)
            {
                double memoryBoundPeakGflops = diagnostics.AchievedBandwidthGBps * arithmeticIntensity;
                diagnostics.IsLikelyMemoryBound = diagnostics.AchievedGflops < memoryBoundPeakGflops * 0.8;
            }

            return diagnostics;
        }

        /// <summary>
        /// Prints comprehensive profiling diagnostics for a GEMM operation.
        /// </summary>
        public void PrintGemmDiagnostics(GemmDiagnostics diagnostics)
        {
            void WriteColored(string message, ConsoleColor color)
            {
                if (Console.IsOutputRedirected)
                {
                    Console.WriteLine(message);
                    return;
                }

                var previous = Console.ForegroundColor;
                Console.ForegroundColor = color;
                Console.WriteLine(message);
                Console.ForegroundColor = previous;
            }

            Console.WriteLine();
            Console.WriteLine("=== OpenCL GEMM Diagnostics ===");
            Console.WriteLine($"Matrix dimensions: M={diagnostics.M}, N={diagnostics.N}, K={diagnostics.K}");
            Console.WriteLine($"Kernel: {diagnostics.KernelName}");
            Console.WriteLine($"Work configuration: Global({diagnostics.GlobalSizeX}x{diagnostics.GlobalSizeY}), Local({diagnostics.LocalSizeX}x{diagnostics.LocalSizeY})");
            Console.WriteLine($"Work items launched: {diagnostics.WorkItemsLaunched:N0}");
            Console.WriteLine($"Work groups launched: {diagnostics.WorkGroupsLaunched:N0}");
            Console.WriteLine();

            if (diagnostics.IsProfilingAvailable && diagnostics.KernelExecutionNs > 0)
            {
                Console.WriteLine("--- GPU Timing (from OpenCL events) ---");
                Console.WriteLine($"Queue to Submit: {diagnostics.QueueToSubmitNs / 1e6:F3} ms");
                Console.WriteLine($"Submit to Start (launch overhead): {diagnostics.SubmitToStartNs / 1e6:F3} ms");
                Console.WriteLine($"Kernel Execution: {diagnostics.KernelExecutionNs / 1e6:F3} ms");
                Console.WriteLine($"Total GPU Time: {diagnostics.TotalGpuTimeNs / 1e6:F3} ms");
            }
            else if (!string.IsNullOrEmpty(diagnostics.ProfilingError))
            {
                Console.WriteLine($"Profiling error: {diagnostics.ProfilingError}");
            }

            Console.WriteLine($"Wall clock time: {diagnostics.WallClockMs:F3} ms");
            Console.WriteLine();

            Console.WriteLine("--- Performance Metrics ---");
            Console.WriteLine($"FLOPS required: {diagnostics.FlopsRequired:N0} ({diagnostics.FlopsRequired / 1e9:F2} GFLOP)");
            Console.WriteLine($"Bytes transferred: {diagnostics.BytesTransferred:N0} ({diagnostics.BytesTransferred / 1e6:F2} MB)");
            Console.WriteLine($"Arithmetic intensity: {diagnostics.ArithmeticIntensity:F2} FLOP/byte");
            Console.WriteLine($"Achieved GFLOPS: {diagnostics.AchievedGflops:F2}");
            Console.WriteLine($"Achieved bandwidth: {diagnostics.AchievedBandwidthGBps:F2} GB/s");
            Console.WriteLine($"Compute efficiency: {diagnostics.ComputeEfficiency:F1}% of theoretical peak");
            Console.WriteLine();

            Console.WriteLine("--- Bottleneck Analysis ---");
            if (diagnostics.SubmitToStartNs > diagnostics.KernelExecutionNs * 0.5 && diagnostics.KernelExecutionNs > 0)
            {
                WriteColored("WARNING: High launch overhead detected (>50% of kernel time)", ConsoleColor.Yellow);
                Console.WriteLine("  -> Consider batching multiple small operations");
            }
            if (diagnostics.IsLikelyMemoryBound)
            {
                WriteColored("LIKELY MEMORY BOUND: Achieved GFLOPS limited by memory bandwidth", ConsoleColor.Yellow);
                Console.WriteLine("  -> Consider using data tiling, caching, or reducing data movement");
            }
            else if (diagnostics.ComputeEfficiency < 50)
            {
                WriteColored("LIKELY COMPUTE BOUND with low efficiency:", ConsoleColor.Red);
                Console.WriteLine("  -> Check for bank conflicts, divergent warps, or suboptimal work group size");
            }
            else if (diagnostics.ComputeEfficiency < 80)
            {
                WriteColored("MODERATE EFFICIENCY: Some room for optimization", ConsoleColor.Yellow);
            }
            else
            {
                WriteColored("GOOD EFFICIENCY: Kernel is well-optimized", ConsoleColor.Green);
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Runs a diagnostic benchmark on the GEMM kernel with various matrix sizes.
        /// </summary>
        public void RunGemmBenchmark(int[] sizes, int warmupIterations = 3, int benchmarkIterations = 10)
        {
            if (_context == null)
            {
                Console.WriteLine("OpenCL context not available");
                return;
            }

            var deviceInfo = GetDeviceInfo();
            Console.WriteLine("=== OpenCL GEMM Benchmark ===");
            Console.WriteLine($"Device: {deviceInfo.DeviceName}");
            Console.WriteLine($"Vendor: {deviceInfo.DeviceVendor}");
            Console.WriteLine($"Compute Units: {deviceInfo.ComputeUnits}");
            Console.WriteLine($"Clock: {deviceInfo.ClockFrequencyMHz} MHz");
            Console.WriteLine($"Theoretical Peak: {deviceInfo.TheoreticalPeakGflops:F0} GFLOPS");
            Console.WriteLine($"Profiling enabled: {IsProfilingEnabled}");
            Console.WriteLine();

            int sizeIndex = 0;
            foreach (int size in sizes)
            {
                sizeIndex++;
                int M = size, N = size, K = size;
                Console.WriteLine($"[Progress] {sizeIndex}/{sizes.Length} size {size}x{size}x{size}");
                Console.WriteLine($"--- Matrix size: {size}x{size}x{size} ---");

                // Allocate buffers
                var dataA = new float[M * K];
                var dataB = new float[K * N];
                for (int i = 0; i < dataA.Length; i++) dataA[i] = 1.0f;
                for (int i = 0; i < dataB.Length; i++) dataB[i] = 1.0f;

                using var bufA = AllocateBuffer(dataA);
                using var bufB = AllocateBuffer(dataB);
                using var bufC = AllocateBuffer(M * N);

                // Warmup
                for (int i = 0; i < warmupIterations; i++)
                {
                    Gemm(bufA, bufB, bufC, M, N, K);
                }
                Synchronize();

                // Benchmark with diagnostics
                var allDiagnostics = new List<GemmDiagnostics>();
                for (int i = 0; i < benchmarkIterations; i++)
                {
                    var diag = GemmWithDiagnostics(bufA, bufB, bufC, M, N, K);
                    allDiagnostics.Add(diag);
                }

                // Summarize results
                double avgKernelTimeMs = 0;
                double avgGflops = 0;
                double avgBandwidth = 0;
                double avgLaunchOverhead = 0;

                foreach (var d in allDiagnostics)
                {
                    if (d.KernelExecutionNs > 0)
                    {
                        avgKernelTimeMs += d.KernelExecutionNs / 1e6;
                        avgLaunchOverhead += d.SubmitToStartNs / 1e6;
                    }
                    else
                    {
                        avgKernelTimeMs += d.WallClockMs;
                    }
                    avgGflops += d.AchievedGflops;
                    avgBandwidth += d.AchievedBandwidthGBps;
                }

                avgKernelTimeMs /= benchmarkIterations;
                avgGflops /= benchmarkIterations;
                avgBandwidth /= benchmarkIterations;
                avgLaunchOverhead /= benchmarkIterations;

                Console.WriteLine($"  Kernel: {allDiagnostics[0].KernelName}");
                Console.WriteLine($"  Avg kernel time: {avgKernelTimeMs:F3} ms");
                if (avgLaunchOverhead > 0)
                {
                    Console.WriteLine($"  Avg launch overhead: {avgLaunchOverhead:F3} ms");
                }
                Console.WriteLine($"  Avg GFLOPS: {avgGflops:F2}");
                Console.WriteLine($"  Avg bandwidth: {avgBandwidth:F2} GB/s");
                Console.WriteLine($"  Efficiency: {avgGflops / deviceInfo.TheoreticalPeakGflops * 100:F1}%");
                Console.WriteLine();
            }
        }


        /// <summary>
        /// Execute GEMM using dynamically compiled kernel with parameters baked in.
        /// </summary>
        public double GemmWithDynamicKernel(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, GemmConfig config)
        {
            if (_context == null || _dynamicGemm == null)
                throw new InvalidOperationException("OpenCL context or dynamic GEMM not available");

            var sw = Stopwatch.StartNew();
            if (!TryExecutePackedDynamicGemm(A, B, C, M, N, K, 1.0f, 0.0f, config))
                throw new InvalidOperationException("Packed GEMM execution failed");
            _dynamicGemm.Synchronize();
            sw.Stop();

            return sw.Elapsed.TotalMilliseconds;
        }

        /// <summary>
        /// Runs Bayesian optimization to find the optimal GEMM kernel configuration.
        /// </summary>
        /// <summary>
        /// Enable verbose diagnostics for GEMM tuning and kernel compilation.
        /// </summary>
        public static bool EnableTuningDiagnostics
        {
            get => GemmAutoTuner.EnableDiagnostics;
            set
            {
                GemmAutoTuner.EnableDiagnostics = value;
                DynamicGemmKernel.EnableDiagnostics = value;
            }
        }

        public TuningResult[] RunBayesianGemmOptimization(int M, int N, int K, int maxTrials = 20, int warmupRuns = 2, int benchmarkRuns = 3)
        {
            if (_context == null || _dynamicGemm == null)
                throw new InvalidOperationException("OpenCL context or dynamic GEMM not available");

            ConfigureOfflineTuning();

            var capabilities = GpuCapabilities.Detect(ComputeUnits, GlobalMemoryBytes, (int)LocalMemoryBytes,
                (int)_maxWorkGroupSize, DeviceVendor, DeviceName, _context.Extensions);

            Console.WriteLine("=== Bayesian GEMM Optimization ===");
            Console.WriteLine($"Matrix: {M}x{N}x{K}, Device: {DeviceName}, Max trials: {maxTrials}");

            // Print GPU capabilities if diagnostics enabled
            if (EnableTuningDiagnostics)
            {
                Console.WriteLine("[GPU Capabilities]");
                Console.Write(capabilities.GetDiagnosticString());
            }

            var dataA = new float[M * K];
            var dataB = new float[K * N];
            for (int i = 0; i < dataA.Length; i++) dataA[i] = 1.0f;
            for (int i = 0; i < dataB.Length; i++) dataB[i] = 1.0f;

            using var bufA = AllocateBuffer(dataA);
            using var bufB = AllocateBuffer(dataB);
            using var bufC = AllocateBuffer(M * N);

            string deviceSignature = GetDeviceSignature();
            using var database = new GemmTuningDatabase(deviceSignature: deviceSignature);
            if (GetEnvBool(OfflineTuningResetEnvVar) && !_tuningDbResetDone)
            {
                database.Clear();
                lock (_tunedConfigLock)
                {
                    _tunedConfigCache.Clear();
                }
                _tuningDbResetDone = true;
            }
            var tuner = new GemmAutoTuner();

            int benchmarkAttempts = 0;
            int benchmarkFailures = 0;
            double ops = 2.0 * M * N * K;

            double BenchmarkConfig(GemmConfig config, bool allowCached)
            {
                benchmarkAttempts++;

                // Validate config before attempting execution
                var validationError = DynamicGemmKernel.ValidateConfig(config);
                if (validationError != null)
                {
                    benchmarkFailures++;
                    if (EnableTuningDiagnostics)
                    {
                        Console.WriteLine($"  [Validation] {config.KernelName}: {validationError}");
                    }
                    database.MarkAsTested(M, N, K, config, 0);
                    return double.NaN;
                }

                if (allowCached && database.HasBeenTested(M, N, K, config))
                {
                    var cachedGflops = database.GetCachedGflops(M, N, K, config);
                    if (cachedGflops.HasValue && cachedGflops.Value > 0)
                    {
                        if (EnableTuningDiagnostics)
                        {
                            Console.WriteLine($"  [Cache] {config.KernelName}: {cachedGflops.Value:F2} GFLOPS");
                        }

                        return ops / (cachedGflops.Value * 1e6);
                    }

                    benchmarkFailures++;
                    return double.NaN;
                }

                try
                {
                    for (int i = 0; i < warmupRuns; i++)
                        GemmWithDynamicKernel(bufA, bufB, bufC, M, N, K, config);

                    double totalTimeMs = 0;
                    for (int i = 0; i < benchmarkRuns; i++)
                        totalTimeMs += GemmWithDynamicKernel(bufA, bufB, bufC, M, N, K, config);

                    double avgTimeMs = totalTimeMs / benchmarkRuns;
                    if (double.IsNaN(avgTimeMs) || double.IsInfinity(avgTimeMs) || avgTimeMs <= 0)
                    {
                        benchmarkFailures++;
                        database.MarkAsTested(M, N, K, config, 0);
                        return double.NaN;
                    }

                    double gflops = ops / (avgTimeMs * 1e6);
                    database.MarkAsTested(M, N, K, config, gflops);
                    return avgTimeMs;
                }
                catch (Exception ex)
                {
                    benchmarkFailures++;
                    Console.WriteLine($"  Config {config} failed: {ex.Message}");

                    // Print kernel stats on failure
                    if (EnableTuningDiagnostics && _dynamicGemm != null)
                    {
                        Console.WriteLine($"  [DynamicGemm Stats] {_dynamicGemm.GetDiagnosticStats()}");
                    }

                    database.MarkAsTested(M, N, K, config, 0);
                    return double.NaN;
                }
            }

            double BenchmarkConfigCached(GemmConfig config) => BenchmarkConfig(config, true);
            double BenchmarkConfigNoCache(GemmConfig config) => BenchmarkConfig(config, false);

            // Check for cached result first - use DATABASE GFLOPS as baseline
            var cachedEntry = database.GetBestConfigWithGflops(M, N, K);
            TuningResult? cachedResult = null;
            double databaseGflops = 0;  // The historical best from database - this is the threshold to beat

            if (cachedEntry.HasValue)
            {
                var cachedConfig = cachedEntry.Value.Config;
                databaseGflops = cachedEntry.Value.GFlops;  // Use stored GFLOPS as baseline
                Console.WriteLine($"Using cached configuration: {cachedConfig}");
                Console.WriteLine($"Database best: {databaseGflops:F2} GFLOPS (threshold to beat)");

                // Re-benchmark to validate config works and add to result set
                var cachedTimeMs = BenchmarkConfigNoCache(cachedConfig);
                if (!double.IsNaN(cachedTimeMs))
                {
                    double revalidatedGflops = (ops / (cachedTimeMs / 1000.0)) / 1e9;
                    cachedResult = new TuningResult
                    {
                        Config = cachedConfig,
                        TimeMs = cachedTimeMs,
                        GFlops = revalidatedGflops,
                        IsValid = true
                    };
                    Console.WriteLine($"Revalidated: {revalidatedGflops:F2} GFLOPS");
                }
            }

            // Run Bayesian optimization
            int initialRandomSamples = Math.Min(maxTrials, Math.Max(12, maxTrials / 10));
            var results = tuner.TuneWithBayesianOptimization(M, N, K, capabilities, BenchmarkConfigCached, maxTrials, initialRandomSamples, 0, 1);

            // Print final statistics
            if (EnableTuningDiagnostics)
            {
                Console.WriteLine($"\n[Benchmark Stats] Attempts: {benchmarkAttempts}, Failures: {benchmarkFailures}");
                if (_dynamicGemm != null)
                {
                    Console.WriteLine($"[DynamicGemm Stats] {_dynamicGemm.GetDiagnosticStats()}");
                }
            }

            // Merge cached result with optimization results
            var allResults = new List<TuningResult>(results);
            if (cachedResult.HasValue)
            {
                allResults.Add(cachedResult.Value);
            }

            // Sort by GFLOPS descending
            allResults.Sort((a, b) => b.GFlops.CompareTo(a.GFlops));

            if (allResults.Count > 0 && allResults[0].IsValid)
            {
                var best = allResults[0];
                Console.WriteLine($"Best: {best.Config} - {best.GFlops:F2} GFLOPS");

                // Only update database if we found something better than the DATABASE best
                // Note: We compare against databaseGflops (historical best), NOT re-benchmarked value
                if (best.GFlops > databaseGflops)
                {
                    Console.WriteLine($"NEW GLOBAL BEST! {best.GFlops:F2} > {databaseGflops:F2} GFLOPS (previous best)");
                    database.StoreResult(M, N, K, best.Config, best.GFlops);
                }
                else
                {
                    Console.WriteLine($"No improvement: {best.GFlops:F2} <= {databaseGflops:F2} GFLOPS (database best)");
                }
            }

            return allResults.ToArray();
        }

        /// <summary>
        /// Runs EXHAUSTIVE optimization (tests ALL configurations) like CLBlast does.
        /// This is slower but guarantees finding the global optimum.
        /// </summary>
        public TuningResult[] RunExhaustiveGemmOptimization(int M, int N, int K, int warmupRuns = 2, int benchmarkRuns = 3)
        {
            if (_context == null || _dynamicGemm == null)
                throw new InvalidOperationException("OpenCL context or dynamic GEMM not available");

            ConfigureOfflineTuning();

            var capabilities = GpuCapabilities.Detect(ComputeUnits, GlobalMemoryBytes, (int)LocalMemoryBytes,
                (int)_maxWorkGroupSize, DeviceVendor, DeviceName, _context.Extensions);

            Console.WriteLine("=== EXHAUSTIVE GEMM Optimization (CLBlast-style) ===");
            Console.WriteLine($"Matrix: {M}x{N}x{K}, Device: {DeviceName}");

            var dataA = new float[M * K];
            var dataB = new float[K * N];
            for (int i = 0; i < dataA.Length; i++) dataA[i] = 1.0f;
            for (int i = 0; i < dataB.Length; i++) dataB[i] = 1.0f;

            using var bufA = AllocateBuffer(dataA);
            using var bufB = AllocateBuffer(dataB);
            using var bufC = AllocateBuffer(M * N);

            string deviceSignature = GetDeviceSignature();
            using var database = new GemmTuningDatabase(deviceSignature: deviceSignature);
            if (GetEnvBool(OfflineTuningResetEnvVar) && !_tuningDbResetDone)
            {
                database.Clear();
                lock (_tunedConfigLock)
                {
                    _tunedConfigCache.Clear();
                }
                _tuningDbResetDone = true;
            }
            var tuner = new GemmAutoTuner();
            double ops = 2.0 * M * N * K;

            double BenchmarkConfig(GemmConfig config, bool allowCached)
            {
                // Validate config before attempting execution
                var validationError = DynamicGemmKernel.ValidateConfig(config);
                if (validationError != null)
                {
                    if (EnableTuningDiagnostics)
                    {
                        Console.WriteLine($"  [Validation] {config.KernelName}: {validationError}");
                    }
                    database.MarkAsTested(M, N, K, config, 0);
                    return double.NaN;
                }

                if (allowCached && database.HasBeenTested(M, N, K, config))
                {
                    var cachedGflops = database.GetCachedGflops(M, N, K, config);
                    if (cachedGflops.HasValue && cachedGflops.Value > 0)
                    {
                        if (EnableTuningDiagnostics)
                        {
                            Console.WriteLine($"  [Cache] {config.KernelName}: {cachedGflops.Value:F2} GFLOPS");
                        }

                        return ops / (cachedGflops.Value * 1e6);
                    }

                    return double.NaN;
                }

                try
                {
                    for (int i = 0; i < warmupRuns; i++)
                        GemmWithDynamicKernel(bufA, bufB, bufC, M, N, K, config);

                    double totalTimeMs = 0;
                    for (int i = 0; i < benchmarkRuns; i++)
                        totalTimeMs += GemmWithDynamicKernel(bufA, bufB, bufC, M, N, K, config);

                    double avgTimeMs = totalTimeMs / benchmarkRuns;
                    if (double.IsNaN(avgTimeMs) || double.IsInfinity(avgTimeMs) || avgTimeMs <= 0)
                    {
                        database.MarkAsTested(M, N, K, config, 0);
                        return double.NaN;
                    }

                    double gflops = ops / (avgTimeMs * 1e6);
                    database.MarkAsTested(M, N, K, config, gflops);
                    return avgTimeMs;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  Config {config} failed: {ex.Message}");
                    database.MarkAsTested(M, N, K, config, 0);
                    return double.NaN;
                }
            }

            double BenchmarkConfigCached(GemmConfig config) => BenchmarkConfig(config, true);

            // Use exhaustive search - tests ALL configurations
            var results = tuner.TuneForDimensions(M, N, K, capabilities, BenchmarkConfigCached, 0, 1, useFullSearchSpace: true);

            if (results.Length > 0 && results[0].IsValid)
            {
                var best = results[0];
                Console.WriteLine($"EXHAUSTIVE Best: {best.Config} - {best.GFlops:F2} GFLOPS");

                database.StoreResult(M, N, K, best.Config, best.GFlops);
            }

            return results;
        }

        /// <summary>
        /// Executes GEMM using the actual CLBlast library for head-to-head comparison.
        /// Returns the execution time in milliseconds, or -1 if CLBlast is not available.
        /// </summary>
        public double GemmWithClBlast(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            if (!ClBlastNative.IsAvailable)
                return -1.0;

            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;
            var bufferC = ((DirectOpenClGpuBuffer)C).Buffer;

            IntPtr queue = _context.CommandQueue;

            var sw = Stopwatch.StartNew();
            var status = ClBlastNative.Sgemm(
                ClBlastNative.Layout.RowMajor,
                ClBlastNative.Transpose.No,
                ClBlastNative.Transpose.No,
                (UIntPtr)M, (UIntPtr)N, (UIntPtr)K,
                alpha,
                bufferA.Handle, UIntPtr.Zero, (UIntPtr)K,
                bufferB.Handle, UIntPtr.Zero, (UIntPtr)N,
                beta,
                bufferC.Handle, UIntPtr.Zero, (UIntPtr)N,
                ref queue,
                IntPtr.Zero);

            OpenClNativeBindings.Finish(_context.CommandQueue);
            sw.Stop();

            if (status != ClBlastNative.StatusCode.Success)
            {
                Console.WriteLine($"CLBlast SGEMM failed with status: {status}");
                return -1.0;
            }

            return sw.Elapsed.TotalMilliseconds;
        }

        /// <summary>
        /// Checks if the CLBlast library is available for use.
        /// </summary>
        public static bool IsClBlastAvailable => ClBlastNative.IsAvailable;

        #endregion

        #region Enhanced Diagnostics and Roofline Analysis

        /// <summary>
        /// Environment variable to enable deep profiling: AIDOTNET_GEMM_PROFILE=1
        /// This enables detailed roofline model analysis and bottleneck detection.
        /// </summary>
        private const string GemmProfileEnvVar = "AIDOTNET_GEMM_PROFILE";

        /// <summary>
        /// Environment variable to set theoretical peak GFLOPS: AIDOTNET_GPU_PEAK_GFLOPS=5196
        /// If not set, estimates based on compute units and assumed clock speed.
        /// </summary>
        private const string GpuPeakGflopsEnvVar = "AIDOTNET_GPU_PEAK_GFLOPS";

        /// <summary>
        /// Environment variable to set memory bandwidth in GB/s: AIDOTNET_GPU_BANDWIDTH_GBS=224
        /// If not set, uses a conservative estimate of 200 GB/s.
        /// </summary>
        private const string GpuBandwidthEnvVar = "AIDOTNET_GPU_BANDWIDTH_GBS";

        /// <summary>
        /// Roofline model analysis result containing bottleneck information.
        /// </summary>
        public sealed class RooflineAnalysis
        {
            /// <summary>Matrix dimensions (M x N x K).</summary>
            public int M { get; init; }
            public int N { get; init; }
            public int K { get; init; }

            /// <summary>Achieved performance in GFLOPS.</summary>
            public double AchievedGflops { get; init; }

            /// <summary>Theoretical peak performance in GFLOPS.</summary>
            public double PeakGflops { get; init; }

            /// <summary>Memory bandwidth in GB/s.</summary>
            public double MemoryBandwidthGBs { get; init; }

            /// <summary>Arithmetic intensity (FLOPs per byte).</summary>
            public double ArithmeticIntensity { get; init; }

            /// <summary>Ridge point (AI where memory and compute roofs meet).</summary>
            public double RidgePoint { get; init; }

            /// <summary>Maximum attainable performance based on roofline model.</summary>
            public double RooflineLimit { get; init; }

            /// <summary>Efficiency as percentage of roofline limit.</summary>
            public double RooflineEfficiency { get; init; }

            /// <summary>Efficiency as percentage of theoretical peak.</summary>
            public double PeakEfficiency { get; init; }

            /// <summary>Whether the operation is memory-bound (AI < ridge point).</summary>
            public bool IsMemoryBound { get; init; }

            /// <summary>Estimated memory bytes transferred.</summary>
            public long MemoryBytesTransferred { get; init; }

            /// <summary>Effective memory bandwidth achieved (GB/s).</summary>
            public double AchievedBandwidthGBs { get; init; }

            /// <summary>Total FLOPs performed (2*M*N*K for GEMM).</summary>
            public long TotalFlops { get; init; }

            /// <summary>Execution time in milliseconds.</summary>
            public double TimeMs { get; init; }

            /// <summary>Primary bottleneck identified.</summary>
            public string Bottleneck { get; init; } = string.Empty;

            /// <summary>Recommendations for improvement.</summary>
            public List<string> Recommendations { get; init; } = new();

            public override string ToString()
            {
                var sb = new System.Text.StringBuilder();
                sb.AppendLine($"=== GEMM Roofline Analysis ({M}x{N}x{K}) ===");
                sb.AppendLine($"Achieved:    {AchievedGflops:F2} GFLOPS ({PeakEfficiency:F1}% of peak, {RooflineEfficiency:F1}% of roofline)");
                sb.AppendLine($"Peak:        {PeakGflops:F0} GFLOPS");
                sb.AppendLine($"Roofline:    {RooflineLimit:F2} GFLOPS (at AI={ArithmeticIntensity:F2})");
                sb.AppendLine($"Ridge Point: {RidgePoint:F2} FLOPs/byte");
                sb.AppendLine($"Bound:       {(IsMemoryBound ? "MEMORY" : "COMPUTE")}");
                sb.AppendLine($"Bandwidth:   {AchievedBandwidthGBs:F2} GB/s (of {MemoryBandwidthGBs:F0} GB/s peak)");
                sb.AppendLine($"Time:        {TimeMs:F3} ms");
                sb.AppendLine($"Bottleneck:  {Bottleneck}");
                if (Recommendations.Count > 0)
                {
                    sb.AppendLine("Recommendations:");
                    foreach (var rec in Recommendations)
                        sb.AppendLine($"  - {rec}");
                }
                return sb.ToString();
            }
        }

        /// <summary>
        /// Performs GEMM with comprehensive profiling and returns roofline analysis.
        /// Enable with AIDOTNET_GEMM_PROFILE=1 for automatic analysis output.
        /// </summary>
        public RooflineAnalysis ProfileGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f, int warmupRuns = 2, int benchmarkRuns = 5)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            // Get GPU specs from environment or estimate
            double peakGflops = GetEnvInt(GpuPeakGflopsEnvVar, 0);
            if (peakGflops <= 0)
            {
                // Use architecture-specific clock speed for accurate peak GFLOPS
                var arch = Profiling.GpuArchitectureSpec.DetectFromDeviceName(DeviceName);
                peakGflops = arch.CalculatePeakGflops(ComputeUnits);
            }

            double bandwidthGBs = GetEnvInt(GpuBandwidthEnvVar, 0);
            if (bandwidthGBs <= 0)
            {
                // Conservative estimate for GDDR6
                bandwidthGBs = 200.0;
            }

            // Warmup runs
            for (int i = 0; i < warmupRuns; i++)
            {
                Gemm(A, B, C, M, N, K, alpha, beta);
                Synchronize();
            }

            // Benchmark runs
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < benchmarkRuns; i++)
            {
                Gemm(A, B, C, M, N, K, alpha, beta);
                Synchronize();
            }
            sw.Stop();

            double avgTimeMs = sw.Elapsed.TotalMilliseconds / benchmarkRuns;
            long totalFlops = 2L * M * N * K;
            double achievedGflops = totalFlops / (avgTimeMs * 1e6);

            // Memory bytes: A(M*K) + B(K*N) + C(M*N read + write)
            // For beta=0: A + B + C_write = M*K + K*N + M*N floats
            // For beta!=0: A + B + C_read + C_write = M*K + K*N + 2*M*N floats
            long memBytes = ((long)M * K + (long)K * N + (beta == 0 ? (long)M * N : 2L * M * N)) * sizeof(float);
            double achievedBandwidthGBs = memBytes / (avgTimeMs * 1e6);

            // Arithmetic intensity (FLOPs per byte)
            double ai = (double)totalFlops / memBytes;

            // Ridge point: where memory and compute roofs meet
            double ridgePoint = peakGflops / bandwidthGBs;

            // Roofline limit
            double rooflineLimit = Math.Min(peakGflops, ai * bandwidthGBs);
            bool isMemoryBound = ai < ridgePoint;

            // Efficiency
            double rooflineEfficiency = 100.0 * achievedGflops / rooflineLimit;
            double peakEfficiency = 100.0 * achievedGflops / peakGflops;

            // Bottleneck analysis
            var recommendations = new List<string>();
            string bottleneck;

            if (rooflineEfficiency >= 90)
            {
                bottleneck = "Near-optimal - approaching roofline";
            }
            else if (isMemoryBound)
            {
                bottleneck = $"MEMORY BANDWIDTH (AI={ai:F2} < ridge={ridgePoint:F2})";
                if (M * N < 256 * 256)
                    recommendations.Add("Use batched GEMM to amortize memory transfer overhead");
                if (achievedBandwidthGBs < bandwidthGBs * 0.7)
                    recommendations.Add("Check memory access patterns - may have uncoalesced accesses");
                recommendations.Add("Consider fusing with subsequent operations to reduce memory traffic");
            }
            else
            {
                bottleneck = "COMPUTE (AI > ridge point, should be compute-bound)";
                if (rooflineEfficiency < 50)
                {
                    recommendations.Add("Check occupancy - may need smaller tile sizes for better wave scheduling");
                    recommendations.Add("Verify LDS bank conflicts with AMD rocprof-compute");
                    recommendations.Add("Check register spilling - may need fewer registers per thread");
                }
                if (rooflineEfficiency < 70)
                {
                    recommendations.Add("Try larger thread tiles (16x8 or 8x16) for more ILP");
                    recommendations.Add("Enable Wave32 mode if on RDNA architecture");
                    recommendations.Add("Check loop unrolling factor - may need more aggressive unrolling");
                }
                if (peakEfficiency < 50)
                    recommendations.Add("Check for instruction mix - may have too many non-FMA instructions");
            }

            // Additional checks
            if (M < 128 || N < 128)
                recommendations.Add($"Small matrix ({M}x{N}) - consider XgemmDirect kernel for reduced overhead");
            if (K < 64)
                recommendations.Add($"Small K dimension ({K}) - reduction overhead may dominate");
            if ((long)M * N * K < 1000000)
                recommendations.Add("Matrix too small to saturate GPU - batch multiple operations");

            var result = new RooflineAnalysis
            {
                M = M,
                N = N,
                K = K,
                AchievedGflops = achievedGflops,
                PeakGflops = peakGflops,
                MemoryBandwidthGBs = bandwidthGBs,
                ArithmeticIntensity = ai,
                RidgePoint = ridgePoint,
                RooflineLimit = rooflineLimit,
                RooflineEfficiency = rooflineEfficiency,
                PeakEfficiency = peakEfficiency,
                IsMemoryBound = isMemoryBound,
                MemoryBytesTransferred = memBytes,
                AchievedBandwidthGBs = achievedBandwidthGBs,
                TotalFlops = totalFlops,
                TimeMs = avgTimeMs,
                Bottleneck = bottleneck,
                Recommendations = recommendations
            };

            // Auto-print if profiling is enabled
            if (GetEnvBool(GemmProfileEnvVar))
            {
                Console.WriteLine(result);
            }

            return result;
        }

        /// <summary>
        /// Compares our GEMM implementation against CLBlast and returns detailed analysis.
        /// </summary>
        public string CompareWithClBlast(int M, int N, int K, int warmupRuns = 3, int benchmarkRuns = 10)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"=== GEMM Comparison: Our Implementation vs CLBlast ({M}x{N}x{K}) ===");

            // Allocate test matrices
            var random = new Random(42);
            var dataA = new float[M * K];
            var dataB = new float[K * N];
            for (int i = 0; i < dataA.Length; i++) dataA[i] = (float)(random.NextDouble() - 0.5);
            for (int i = 0; i < dataB.Length; i++) dataB[i] = (float)(random.NextDouble() - 0.5);

            var bufA = AllocateBuffer(dataA);
            var bufB = AllocateBuffer(dataB);
            var bufC_ours = AllocateBuffer(M * N);
            var bufC_clblast = AllocateBuffer(M * N);

            try
            {
                long flops = 2L * M * N * K;

                // Benchmark our implementation
                for (int i = 0; i < warmupRuns; i++)
                {
                    Gemm(bufA, bufB, bufC_ours, M, N, K, 1.0f, 0.0f);
                    Synchronize();
                }

                var sw = Stopwatch.StartNew();
                for (int i = 0; i < benchmarkRuns; i++)
                {
                    Gemm(bufA, bufB, bufC_ours, M, N, K, 1.0f, 0.0f);
                    Synchronize();
                }
                sw.Stop();
                double ourTimeMs = sw.Elapsed.TotalMilliseconds / benchmarkRuns;
                double ourGflops = flops / (ourTimeMs * 1e6);

                sb.AppendLine($"Our Implementation: {ourGflops:F2} GFLOPS ({ourTimeMs:F3} ms)");

                // Benchmark CLBlast if available
                if (IsClBlastAvailable)
                {
                    for (int i = 0; i < warmupRuns; i++)
                    {
                        GemmWithClBlast(bufA, bufB, bufC_clblast, M, N, K, 1.0f, 0.0f);
                    }

                    double totalClBlastTime = 0;
                    for (int i = 0; i < benchmarkRuns; i++)
                    {
                        totalClBlastTime += GemmWithClBlast(bufA, bufB, bufC_clblast, M, N, K, 1.0f, 0.0f);
                    }
                    double clblastTimeMs = totalClBlastTime / benchmarkRuns;
                    double clblastGflops = flops / (clblastTimeMs * 1e6);

                    sb.AppendLine($"CLBlast:            {clblastGflops:F2} GFLOPS ({clblastTimeMs:F3} ms)");

                    double speedup = ourGflops / clblastGflops;
                    sb.AppendLine($"Speedup:            {speedup:F2}x ({(speedup > 1 ? "FASTER" : "SLOWER")} than CLBlast)");

                    // Verify correctness
                    var resultOurs = ((DirectOpenClGpuBuffer)bufC_ours).Download();
                    var resultClBlast = ((DirectOpenClGpuBuffer)bufC_clblast).Download();
                    double maxDiff = 0;
                    for (int i = 0; i < resultOurs.Length; i++)
                    {
                        maxDiff = Math.Max(maxDiff, Math.Abs(resultOurs[i] - resultClBlast[i]));
                    }
                    sb.AppendLine($"Max Diff:           {maxDiff:E3} (should be < 1e-4)");
                }
                else
                {
                    sb.AppendLine("CLBlast: NOT AVAILABLE");
                }

                return sb.ToString();
            }
            finally
            {
                bufA.Dispose();
                bufB.Dispose();
                bufC_ours.Dispose();
                bufC_clblast.Dispose();
            }
        }

        /// <summary>
        /// Prints a summary of available diagnostic environment variables.
        /// </summary>
        public static void PrintDiagnosticHelp()
        {
            Console.WriteLine(@"
=== AiDotNet GPU Diagnostic Environment Variables ===

TIMING & TRACING:
  AIDOTNET_GEMM_TIMING=1      Show timing breakdown for each GEMM operation
  AIDOTNET_GEMM_TRACE=1       Trace kernel path selection (DIRECT vs INDIRECT)
  AIDOTNET_GEMM_PROFILE=1     Enable roofline model analysis output

TUNING:
  AIDOTNET_GPU_TUNE=1         Enable offline auto-tuning (Bayesian search)
  AIDOTNET_GPU_TUNE=exhaustive Full exhaustive search (slow but thorough)
  AIDOTNET_GPU_TUNE_TRIALS=N  Number of tuning trials (default: 500)
  AIDOTNET_GPU_TUNE_DIAG=1    Verbose tuning diagnostics
  AIDOTNET_GPU_TUNE_LOG=path  Log tuning output to file
  AIDOTNET_GPU_TUNE_CSV=path  Log trial results to CSV

GPU SPECS (for accurate roofline analysis):
  AIDOTNET_GPU_PEAK_GFLOPS=N  Theoretical peak FP32 GFLOPS (e.g., 5196 for RX 5500 XT)
  AIDOTNET_GPU_BANDWIDTH_GBS=N Memory bandwidth in GB/s (e.g., 224 for RX 5500 XT)

DEBUG:
  AIDOTNET_FORCE_DIRECT=1     Force XgemmDirect path (skip indirect path)

KERNEL VARIANTS (A/B testing):
  AIDOTNET_GEMM_VARIANT=0     Original CLBlast baseline
  AIDOTNET_GEMM_VARIANT=1     XOR LDS swizzling (eliminates bank conflicts)
  AIDOTNET_GEMM_VARIANT=2     RDNA1 optimized (swizzle + Wave32 hints)
");
        }

        /// <summary>
        /// A/B tests different kernel optimization variants and returns comparative results.
        /// Tests: Original CLBlast vs XOR Swizzle vs RDNA1 Optimized
        /// </summary>
        public string ABTestKernelVariants(int M, int N, int K, int warmupRuns = 3, int benchmarkRuns = 10)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"=== A/B Testing Kernel Variants ({M}x{N}x{K}) ===");
            sb.AppendLine($"Device: {DeviceName} ({DeviceVendor})");
            sb.AppendLine($"Warmup: {warmupRuns} runs, Benchmark: {benchmarkRuns} runs");
            sb.AppendLine();

            // Allocate test matrices
            var random = new Random(42);
            var dataA = new float[M * K];
            var dataB = new float[K * N];
            for (int i = 0; i < dataA.Length; i++) dataA[i] = (float)(random.NextDouble() - 0.5);
            for (int i = 0; i < dataB.Length; i++) dataB[i] = (float)(random.NextDouble() - 0.5);

            var bufA = AllocateBuffer(dataA);
            var bufB = AllocateBuffer(dataB);
            var bufC = AllocateBuffer(M * N);

            try
            {
                long flops = 2L * M * N * K;
                var results = new List<(string Name, double Gflops, double TimeMs)>();
                float[]? referenceResult = null;

                // Test each kernel variant
                for (int variant = 0; variant <= 2; variant++)
                {
                    string variantName = variant switch
                    {
                        0 => "CLBlast Baseline",
                        1 => "XOR Swizzle",
                        2 => "RDNA1 Optimized",
                        _ => "Unknown"
                    };

                    // Set kernel variant
                    int originalVariant = DynamicGemmKernel.KernelVariant;
                    bool originalDiag = DynamicGemmKernel.EnableDiagnostics;
                    DynamicGemmKernel.KernelVariant = variant;
                    DynamicGemmKernel.EnableDiagnostics = true; // Enable for debugging

                    // Clear kernel cache to force recompilation with new variant
                    _dynamicGemm?.ClearCache();

                    try
                    {
                        // Zero output buffer
                        ((DirectOpenClGpuBuffer)bufC).Buffer.CopyFromHost(new float[M * N]);

                        // Warmup
                        for (int i = 0; i < warmupRuns; i++)
                        {
                            Gemm(bufA, bufB, bufC, M, N, K, 1.0f, 0.0f);
                            Synchronize();
                        }

                        // Benchmark
                        var sw = Stopwatch.StartNew();
                        for (int i = 0; i < benchmarkRuns; i++)
                        {
                            Gemm(bufA, bufB, bufC, M, N, K, 1.0f, 0.0f);
                            Synchronize();
                        }
                        sw.Stop();

                        double timeMs = sw.Elapsed.TotalMilliseconds / benchmarkRuns;
                        double gflops = flops / (timeMs * 1e6);
                        results.Add((variantName, gflops, timeMs));

                        // Store reference result for verification
                        if (variant == 0)
                        {
                            referenceResult = ((DirectOpenClGpuBuffer)bufC).Download();
                        }
                        else if (referenceResult != null)
                        {
                            // Verify correctness against baseline
                            var currentResult = ((DirectOpenClGpuBuffer)bufC).Download();
                            double maxDiff = 0;
                            for (int i = 0; i < currentResult.Length; i++)
                            {
                                maxDiff = Math.Max(maxDiff, Math.Abs(currentResult[i] - referenceResult[i]));
                            }
                            sb.AppendLine($"{variantName}: {gflops:F2} GFLOPS ({timeMs:F3} ms) [MaxDiff: {maxDiff:E2}]");
                            continue;
                        }

                        sb.AppendLine($"{variantName}: {gflops:F2} GFLOPS ({timeMs:F3} ms)");
                    }
                    catch (Exception ex)
                    {
                        sb.AppendLine($"{variantName}: FAILED - {ex.Message}");
                    }
                    finally
                    {
                        DynamicGemmKernel.KernelVariant = originalVariant;
                        DynamicGemmKernel.EnableDiagnostics = originalDiag;
                    }
                }

                // Summary
                if (results.Count > 1)
                {
                    sb.AppendLine();
                    sb.AppendLine("=== Summary ===");
                    var baseline = results[0];
                    foreach (var result in results)
                    {
                        double speedup = result.Gflops / baseline.Gflops;
                        string status = speedup > 1.01 ? "FASTER" : speedup < 0.99 ? "SLOWER" : "SAME";
                        sb.AppendLine($"{result.Name}: {result.Gflops:F2} GFLOPS ({speedup:F2}x vs baseline) [{status}]");
                    }

                    var best = results.OrderByDescending(r => r.Gflops).First();
                    sb.AppendLine();
                    sb.AppendLine($"** WINNER: {best.Name} at {best.Gflops:F2} GFLOPS **");

                    // Recommendation
                    sb.AppendLine();
                    sb.AppendLine("To use the best variant, set environment variable:");
                    int bestVariant = results.IndexOf(best);
                    sb.AppendLine($"  AIDOTNET_GEMM_VARIANT={bestVariant}");
                }

                return sb.ToString();
            }
            finally
            {
                bufA.Dispose();
                bufB.Dispose();
                bufC.Dispose();
            }
        }

        /// <summary>
        /// Comprehensive multi-size A/B testing with profiling and bottleneck analysis.
        /// Tests all kernel variants across multiple matrix sizes to identify optimal configuration per size.
        /// </summary>
        /// <param name="sizes">Matrix sizes to test (M=N=K). Default: common sizes from 128 to 4096.</param>
        /// <param name="warmupRuns">Warmup runs per test.</param>
        /// <param name="benchmarkRuns">Benchmark runs per test.</param>
        /// <returns>Comprehensive analysis report.</returns>
        public string ComprehensiveAbTest(int[]? sizes = null, int warmupRuns = 2, int benchmarkRuns = 5)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            sizes ??= new[] { 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096 };

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("=".PadRight(100, '='));
            sb.AppendLine("COMPREHENSIVE GEMM A/B TESTING - ALL SIZES & VARIANTS");
            sb.AppendLine("=".PadRight(100, '='));
            sb.AppendLine($"Device: {DeviceName} ({DeviceVendor})");
            sb.AppendLine($"Compute Units: {ComputeUnits}");
            sb.AppendLine($"Sizes: {string.Join(", ", sizes)}");
            sb.AppendLine($"Warmup: {warmupRuns}, Benchmark: {benchmarkRuns}");
            sb.AppendLine();

            // Detect architecture and estimate peak using architecture-specific clock speed
            var arch = Profiling.GpuArchitectureSpec.DetectFromDeviceName(DeviceName);
            double estimatedPeakGflops = arch.CalculatePeakGflops(ComputeUnits);
            var envPeak = Environment.GetEnvironmentVariable("AIDOTNET_GPU_PEAK_GFLOPS");
            if (int.TryParse(envPeak, out int peakVal) && peakVal > 0)
                estimatedPeakGflops = peakVal;

            double estimatedBandwidth = 224.0; // Default for GDDR6
            var envBw = Environment.GetEnvironmentVariable("AIDOTNET_GPU_BANDWIDTH_GBS");
            if (int.TryParse(envBw, out int bwVal) && bwVal > 0)
                estimatedBandwidth = bwVal;

            var roofline = new Profiling.RooflineAnalyzer(estimatedPeakGflops, estimatedBandwidth);
            sb.AppendLine($"Architecture: {arch.Name}");
            sb.AppendLine($"Estimated Peak: {estimatedPeakGflops:F0} GFLOPS, Bandwidth: {estimatedBandwidth:F0} GB/s");
            sb.AppendLine($"Ridge Point: {roofline.RidgePoint:F1} FLOPS/byte");
            sb.AppendLine();

            // Track results per size
            var allResults = new Dictionary<int, List<(string Variant, double Gflops, double TimeMs, bool Correct)>>();
            var winnerBySize = new Dictionary<int, (string Variant, double Gflops)>();

            // Test each size
            foreach (var size in sizes)
            {
                sb.AppendLine($"--- Size: {size}x{size}x{size} ---");

                int M = size, N = size, K = size;
                long flops = 2L * M * N * K;
                double ai = Profiling.RooflineAnalyzer.CalculateGemmArithmeticIntensity(M, N, K);
                double rooflineLimit = roofline.GetRooflineLimitGflops(ai);
                bool isMemoryBound = ai < roofline.RidgePoint;

                sb.AppendLine($"  AI: {ai:F1} FLOPS/byte, Roofline: {rooflineLimit:F0} GFLOPS, Bound: {(isMemoryBound ? "MEMORY" : "COMPUTE")}");

                // Allocate test matrices
                var random = new Random(42);
                var dataA = new float[M * K];
                var dataB = new float[K * N];
                for (int i = 0; i < dataA.Length; i++) dataA[i] = (float)(random.NextDouble() - 0.5);
                for (int i = 0; i < dataB.Length; i++) dataB[i] = (float)(random.NextDouble() - 0.5);

                var bufA = AllocateBuffer(dataA);
                var bufB = AllocateBuffer(dataB);
                var bufC = AllocateBuffer(M * N);

                try
                {
                    var sizeResults = new List<(string Variant, double Gflops, double TimeMs, bool Correct)>();
                    float[]? referenceResult = null;

                    // Test each kernel variant
                    for (int variant = 0; variant <= 2; variant++)
                    {
                        string variantName = variant switch
                        {
                            0 => "CLBlast",
                            1 => "XOR Swizzle",
                            2 => "RDNA1 Opt",
                            _ => "Unknown"
                        };

                        int originalVariant = DynamicGemmKernel.KernelVariant;
                        DynamicGemmKernel.KernelVariant = variant;
                        _dynamicGemm?.ClearCache();

                        try
                        {
                            ((DirectOpenClGpuBuffer)bufC).Buffer.CopyFromHost(new float[M * N]);

                            // Warmup
                            for (int i = 0; i < warmupRuns; i++)
                            {
                                Gemm(bufA, bufB, bufC, M, N, K, 1.0f, 0.0f);
                                Synchronize();
                            }

                            // Benchmark
                            var sw = Stopwatch.StartNew();
                            for (int i = 0; i < benchmarkRuns; i++)
                            {
                                Gemm(bufA, bufB, bufC, M, N, K, 1.0f, 0.0f);
                                Synchronize();
                            }
                            sw.Stop();

                            double timeMs = sw.Elapsed.TotalMilliseconds / benchmarkRuns;
                            double gflops = flops / (timeMs * 1e6);
                            double efficiency = 100.0 * gflops / estimatedPeakGflops;

                            // Verify correctness
                            bool correct = true;
                            if (variant == 0)
                            {
                                referenceResult = ((DirectOpenClGpuBuffer)bufC).Download();
                            }
                            else if (referenceResult != null)
                            {
                                var currentResult = ((DirectOpenClGpuBuffer)bufC).Download();
                                double maxDiff = 0;
                                for (int i = 0; i < Math.Min(currentResult.Length, referenceResult.Length); i++)
                                {
                                    maxDiff = Math.Max(maxDiff, Math.Abs(currentResult[i] - referenceResult[i]));
                                }
                                correct = maxDiff < 0.01f; // Allow small numerical differences
                            }

                            sizeResults.Add((variantName, gflops, timeMs, correct));
                            sb.AppendLine($"    {variantName,-12}: {gflops,7:F0} GFLOPS ({efficiency,5:F1}%) {timeMs,8:F3} ms {(correct ? "OK" : "WRONG!")}");
                        }
                        catch (Exception ex)
                        {
                            sb.AppendLine($"    {variantName,-12}: FAILED - {ex.Message}");
                            sizeResults.Add((variantName, 0, 0, false));
                        }
                        finally
                        {
                            DynamicGemmKernel.KernelVariant = originalVariant;
                        }
                    }

                    allResults[size] = sizeResults;

                    // Determine winner for this size
                    var validResults = sizeResults.Where(r => r.Correct && r.Gflops > 0).ToList();
                    if (validResults.Any())
                    {
                        var best = validResults.OrderByDescending(r => r.Gflops).First();
                        winnerBySize[size] = (best.Variant, best.Gflops);
                        sb.AppendLine($"    ** Winner: {best.Variant} ({best.Gflops:F0} GFLOPS) **");
                    }
                }
                finally
                {
                    bufA.Dispose();
                    bufB.Dispose();
                    bufC.Dispose();
                }
                sb.AppendLine();
            }

            // Summary table
            sb.AppendLine("=".PadRight(100, '='));
            sb.AppendLine("SUMMARY: BEST VARIANT PER SIZE");
            sb.AppendLine("=".PadRight(100, '='));
            sb.AppendLine();
            sb.AppendLine(string.Format("{0,-8} {1,-12} {2,10} {3,10} {4,10} {5,-15}",
                "Size", "Winner", "GFLOPS", "Eff%", "AI", "Bound"));
            sb.AppendLine("-".PadRight(70, '-'));

            foreach (var size in sizes)
            {
                if (winnerBySize.TryGetValue(size, out var winner))
                {
                    double ai = Profiling.RooflineAnalyzer.CalculateGemmArithmeticIntensity(size, size, size);
                    bool isMemBound = ai < roofline.RidgePoint;
                    double eff = 100.0 * winner.Gflops / estimatedPeakGflops;

                    sb.AppendLine(string.Format("{0,-8} {1,-12} {2,10:F0} {3,9:F1}% {4,10:F1} {5,-15}",
                        size, winner.Variant, winner.Gflops, eff, ai, isMemBound ? "MEMORY" : "COMPUTE"));
                }
            }

            // Analyze patterns
            sb.AppendLine();
            sb.AppendLine("=".PadRight(100, '='));
            sb.AppendLine("ANALYSIS & RECOMMENDATIONS");
            sb.AppendLine("=".PadRight(100, '='));

            // Group by winner
            var byWinner = winnerBySize.GroupBy(kv => kv.Value.Variant).ToDictionary(g => g.Key, g => g.Select(kv => kv.Key).ToList());
            foreach (var group in byWinner)
            {
                sb.AppendLine($"  {group.Key}: wins at sizes {string.Join(", ", group.Value)}");
            }

            // Size-specific recommendations
            sb.AppendLine();
            var smallSizes = winnerBySize.Where(kv => kv.Key <= 512).ToList();
            var mediumSizes = winnerBySize.Where(kv => kv.Key > 512 && kv.Key <= 1536).ToList();
            var largeSizes = winnerBySize.Where(kv => kv.Key > 1536).ToList();

            if (smallSizes.Any())
            {
                var avgEff = smallSizes.Average(kv => 100.0 * kv.Value.Gflops / estimatedPeakGflops);
                sb.AppendLine($"  Small (<=512): Avg efficiency {avgEff:F1}%");
                if (avgEff < 30)
                    sb.AppendLine($"    -> RECOMMENDATION: Consider CPU fallback or batching for sizes < 256");
            }

            if (mediumSizes.Any())
            {
                var avgEff = mediumSizes.Average(kv => 100.0 * kv.Value.Gflops / estimatedPeakGflops);
                sb.AppendLine($"  Medium (513-1536): Avg efficiency {avgEff:F1}%");
                if (avgEff < 50)
                    sb.AppendLine($"    -> RECOMMENDATION: Focus on LDS tiling and register blocking");
            }

            if (largeSizes.Any())
            {
                var avgEff = largeSizes.Average(kv => 100.0 * kv.Value.Gflops / estimatedPeakGflops);
                sb.AppendLine($"  Large (>1536): Avg efficiency {avgEff:F1}%");
                if (avgEff < 60)
                    sb.AppendLine($"    -> RECOMMENDATION: Optimize memory prefetching and double buffering");
            }

            sb.AppendLine();
            sb.AppendLine("=".PadRight(100, '='));

            return sb.ToString();
        }

        #endregion

        #region Convolution Operations

        public void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW,
            int strideH, int strideW, int padH, int padW,
            int dilationH, int dilationW)
        {
            var k = _kernelCache["conv2d_direct"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)kernel).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, dilationH);
            k.SetArg(arg++, dilationW);

            // Work distribution: (outWidth, outHeight, batch * outChannels)
            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * outChannels, localX, localY, localZ);
        }

        public void Conv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW,
            int strideH, int strideW, int padH, int padW,
            int dilationH, int dilationW)
        {
            var k = _kernelCache["conv2d_backward_input"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)kernel).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, dilationH);
            k.SetArg(arg++, dilationW);

            // Work distribution: (inWidth, inHeight, batch * inChannels)
            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(inWidth, inHeight, batch * inChannels, localX, localY, localZ);
        }

        public void Conv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW,
            int strideH, int strideW, int padH, int padW,
            int dilationH, int dilationW)
        {
            var k = _kernelCache["conv2d_backward_kernel"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradKernel).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, dilationH);
            k.SetArg(arg++, dilationW);

            // Work distribution: (kernelW, kernelH, outChannels * inChannels)
            int localX = Math.Min(8, kernelW);
            int localY = Math.Min(8, kernelH);
            int localZ = 1;
            k.Execute3D(kernelW, kernelH, outChannels * inChannels, localX, localY, localZ);
        }

        public void Conv3D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
            int batch, int inChannels, int inDepth, int inHeight, int inWidth,
            int outChannels, int outDepth, int outHeight, int outWidth,
            int kernelD, int kernelH, int kernelW,
            int strideD, int strideH, int strideW,
            int padD, int padH, int padW,
            int dilationD, int dilationH, int dilationW)
        {
            var k = _kernelCache["conv3d_direct"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)kernel).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inDepth);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outDepth);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelD);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideD);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padD);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, dilationD);
            k.SetArg(arg++, dilationH);
            k.SetArg(arg++, dilationW);

            // Work distribution: (outWidth, outDepth * outHeight, batch * outChannels)
            int localX = 8, localY = 4, localZ = 1;
            k.Execute3D(outWidth, outDepth * outHeight, batch * outChannels, localX, localY, localZ);
        }

        public void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
            int batch, int channels, int inHeight, int inWidth,
            int outHeight, int outWidth,
            int kernelH, int kernelW,
            int strideH, int strideW, int padH, int padW)
        {
            var k = _kernelCache["depthwise_conv2d"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)kernel).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * channels, localX, localY, localZ);
        }

        public void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW,
            int strideH, int strideW, int padH, int padW,
            int outputPadH, int outputPadW)
        {
            var k = _kernelCache["conv_transpose2d"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)kernel).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * outChannels, localX, localY, localZ);
        }

        public void ConvTranspose2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW,
            int strideH, int strideW, int padH, int padW,
            int outputPadH, int outputPadW)
        {
            int totalInput = batch * inChannels * inHeight * inWidth;

            var k = _kernelCache["conv_transpose2d_backward_input"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)kernel).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, outputPadH);
            k.SetArg(arg++, outputPadW);
            k.SetArg(arg++, totalInput);

            int blockSize = 256;
            int numBlocks = (totalInput + blockSize - 1) / blockSize;
            k.Execute1D(numBlocks * blockSize, blockSize);
        }

        public void ConvTranspose2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW,
            int strideH, int strideW, int padH, int padW,
            int outputPadH, int outputPadW)
        {
            int totalKernel = inChannels * outChannels * kernelH * kernelW;

            // Zero the gradient buffer first (kernel uses atomic add to accumulate)
            Fill(gradKernel, 0.0f, totalKernel);

            var k = _kernelCache["conv_transpose2d_backward_weights"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradKernel).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, outputPadH);
            k.SetArg(arg++, outputPadW);
            k.SetArg(arg++, totalKernel);

            int blockSize = 256;
            int numBlocks = (totalKernel + blockSize - 1) / blockSize;
            k.Execute1D(numBlocks * blockSize, blockSize);
        }

        public void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW, int strideH, int strideW)
        {
            var k = _kernelCache["locally_connected_conv2d"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)weights).Buffer.Handle);
            k.SetArg(arg++, bias != null ? ((DirectOpenClGpuBuffer)bias).Buffer.Handle : IntPtr.Zero);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, bias != null ? 1 : 0);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * outChannels, localX, localY, localZ);
        }

        public void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW, int strideH, int strideW)
        {
            var k = _kernelCache["locally_connected_conv2d_backward_input"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)weights).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);

            int totalInputSize = batch * inChannels * inHeight * inWidth;
            int localSize = 256;
            int globalSize = ((totalInputSize + localSize - 1) / localSize) * localSize;
            k.Execute1D(globalSize, localSize);
        }

        public void LocallyConnectedConv2DBackwardWeights(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradWeights,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW, int strideH, int strideW)
        {
            var k = _kernelCache["locally_connected_conv2d_backward_weights"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradWeights).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);

            int totalWeights = outHeight * outWidth * outChannels * inChannels * kernelH * kernelW;
            int localSize = 256;
            int globalSize = ((totalWeights + localSize - 1) / localSize) * localSize;
            k.Execute1D(globalSize, localSize);
        }

        public void LocallyConnectedConv2DBackwardBias(IGpuBuffer gradOutput, IGpuBuffer gradBias,
            int batch, int outChannels, int outHeight, int outWidth)
        {
            var k = _kernelCache["locally_connected_conv2d_backward_bias"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradBias).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);

            int localSize = Math.Min(256, outChannels);
            int globalSize = ((outChannels + localSize - 1) / localSize) * localSize;
            k.Execute1D(globalSize, localSize);
        }

        public void DeformableConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer output,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
            int dilationH, int dilationW, int groups, int deformGroups)
        {
            var k = _kernelCache["deformable_conv2d"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)weights).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)offsets).Buffer.Handle);
            k.SetArg(arg++, mask != null ? ((DirectOpenClGpuBuffer)mask).Buffer.Handle : IntPtr.Zero);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, dilationH);
            k.SetArg(arg++, dilationW);
            k.SetArg(arg++, groups);
            k.SetArg(arg++, deformGroups);
            k.SetArg(arg++, mask != null ? 1 : 0);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * outChannels, localX, localY, localZ);
        }

        public void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
            int dilationH, int dilationW, int groups, int deformGroups)
        {
            var k = _kernelCache["deformable_conv2d_backward_input"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)weights).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)offsets).Buffer.Handle);
            k.SetArg(arg++, mask != null ? ((DirectOpenClGpuBuffer)mask).Buffer.Handle : IntPtr.Zero);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, dilationH);
            k.SetArg(arg++, dilationW);
            k.SetArg(arg++, groups);
            k.SetArg(arg++, deformGroups);
            k.SetArg(arg++, mask != null ? 1 : 0);

            int totalInputSize = batch * inChannels * inHeight * inWidth;
            int localSize = 256;
            int globalSize = ((totalInputSize + localSize - 1) / localSize) * localSize;
            k.Execute1D(globalSize, localSize);
        }

        public void DeformableConv2DBackwardWeights(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradWeights,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
            int dilationH, int dilationW, int groups, int deformGroups)
        {
            var k = _kernelCache["deformable_conv2d_backward_weights"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)offsets).Buffer.Handle);
            k.SetArg(arg++, mask != null ? ((DirectOpenClGpuBuffer)mask).Buffer.Handle : IntPtr.Zero);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradWeights).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, dilationH);
            k.SetArg(arg++, dilationW);
            k.SetArg(arg++, groups);
            k.SetArg(arg++, deformGroups);
            k.SetArg(arg++, mask != null ? 1 : 0);

            int totalWeights = (outChannels / groups) * inChannels * kernelH * kernelW;
            int localSize = 256;
            int globalSize = ((totalWeights + localSize - 1) / localSize) * localSize;
            k.Execute1D(globalSize, localSize);
        }

        public void DeformableConv2DBackwardOffset(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradOutput, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffset,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
            int dilationH, int dilationW, int groups, int deformGroups)
        {
            var k = _kernelCache["deformable_conv2d_backward_offset"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)weights).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)offsets).Buffer.Handle);
            k.SetArg(arg++, mask != null ? ((DirectOpenClGpuBuffer)mask).Buffer.Handle : IntPtr.Zero);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOffset).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, dilationH);
            k.SetArg(arg++, dilationW);
            k.SetArg(arg++, groups);
            k.SetArg(arg++, deformGroups);
            k.SetArg(arg++, mask != null ? 1 : 0);

            int totalOffsets = batch * deformGroups * 2 * kernelH * kernelW * outHeight * outWidth;
            int localSize = 256;
            int globalSize = ((totalOffsets + localSize - 1) / localSize) * localSize;
            k.Execute1D(globalSize, localSize);
        }

        public void DeformableConv2DBackwardMask(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradOutput, IGpuBuffer offsets, IGpuBuffer gradMask,
            int batch, int inChannels, int inHeight, int inWidth,
            int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
            int dilationH, int dilationW, int groups, int deformGroups)
        {
            var k = _kernelCache["deformable_conv2d_backward_mask"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)weights).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)offsets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradMask).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, inChannels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outChannels);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, dilationH);
            k.SetArg(arg++, dilationW);
            k.SetArg(arg++, groups);
            k.SetArg(arg++, deformGroups);

            int totalMask = batch * deformGroups * kernelH * kernelW * outHeight * outWidth;
            int localSize = 256;
            int globalSize = ((totalMask + localSize - 1) / localSize) * localSize;
            k.Execute1D(globalSize, localSize);
        }

        #endregion

        #region Pooling Operations

        public void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
            int batch, int channels, int inHeight, int inWidth,
            int outHeight, int outWidth,
            int kernelH, int kernelW,
            int strideH, int strideW, int padH, int padW)
        {
            var k = _kernelCache["maxpool2d"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, indices != null ? ((DirectOpenClGpuBuffer)indices).Buffer.Handle : IntPtr.Zero);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, indices != null ? 1 : 0);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * channels, localX, localY, localZ);
        }

        public void MaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
            int batch, int channels, int inHeight, int inWidth,
            int outHeight, int outWidth,
            int kernelH, int kernelW,
            int strideH, int strideW, int padH, int padW)
        {
            if (!_kernelCache.TryGetValue("maxpool2d_backward", out var k))
                throw new InvalidOperationException("OpenCL kernel not found: maxpool2d_backward");
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)indices).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * channels, localX, localY, localZ);
        }

        public void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
            int batch, int channels, int inHeight, int inWidth,
            int outHeight, int outWidth,
            int kernelH, int kernelW,
            int strideH, int strideW, int padH, int padW,
            bool countIncludePad)
        {
            var k = _kernelCache["avgpool2d"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, countIncludePad ? 1 : 0);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * channels, localX, localY, localZ);
        }

        public void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
            int batch, int channels, int inHeight, int inWidth,
            int outHeight, int outWidth,
            int kernelH, int kernelW,
            int strideH, int strideW, int padH, int padW,
            bool countIncludePad)
        {
            if (!_kernelCache.TryGetValue("avgpool2d_backward", out var k))
                throw new InvalidOperationException("OpenCL kernel not found: avgpool2d_backward");
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, padH);
            k.SetArg(arg++, padW);
            k.SetArg(arg++, countIncludePad ? 1 : 0);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(inWidth, inHeight, batch * channels, localX, localY, localZ);
        }

        public void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
        {
            var k = _kernelCache["global_avgpool2d"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, height);
            k.SetArg(arg++, width);

            k.Execute1D(batch * channels, Math.Min(64, batch * channels));
        }

        public void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
        {
            var k = _kernelCache["global_maxpool2d"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, height);
            k.SetArg(arg++, width);

            k.Execute1D(batch * channels, Math.Min(64, batch * channels));
        }

        public void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer indices, int batch, int channels, int height, int width)
        {
            var k = _kernelCache["global_maxpool2d_with_indices"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)indices).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, height);
            k.SetArg(arg++, width);

            k.Execute1D(batch * channels, Math.Min(64, batch * channels));
        }

        public void GlobalAvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int height, int width)
        {
            if (!_kernelCache.TryGetValue("global_avgpool2d_backward", out var k))
                throw new InvalidOperationException("OpenCL kernel not found: global_avgpool2d_backward");
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, height);
            k.SetArg(arg++, width);

            int totalElements = batch * channels * height * width;
            k.Execute1D(totalElements, Math.Min(64, totalElements));
        }

        public void GlobalMaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput, int batch, int channels, int height, int width)
        {
            if (!_kernelCache.TryGetValue("global_maxpool2d_backward", out var k))
                throw new InvalidOperationException("OpenCL kernel not found: global_maxpool2d_backward");

            // First zero out the gradient input
            Fill(gradInput, 0f, batch * channels * height * width);
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)indices).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, height);
            k.SetArg(arg++, width);

            int totalOutputs = batch * channels;
            k.Execute1D(totalOutputs, Math.Min(64, totalOutputs));
        }

        public void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
        {
            var k = _kernelCache["adaptive_avgpool2d"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * channels, localX, localY, localZ);
        }

        public void MaxPool3D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
            int batch, int channels,
            int inDepth, int inHeight, int inWidth,
            int outDepth, int outHeight, int outWidth,
            int kernelD, int kernelH, int kernelW,
            int strideD, int strideH, int strideW)
        {
            var k = _kernelCache["maxpool3d"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, indices is not null ? ((DirectOpenClGpuBuffer)indices).Buffer.Handle : IntPtr.Zero);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, inDepth);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outDepth);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, kernelD);
            k.SetArg(arg++, kernelH);
            k.SetArg(arg++, kernelW);
            k.SetArg(arg++, strideD);
            k.SetArg(arg++, strideH);
            k.SetArg(arg++, strideW);
            k.SetArg(arg++, indices is not null ? 1 : 0);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * channels * outDepth, localX, localY, localZ);
        }

        public void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
            int batch, int channels,
            int inDepth, int inHeight, int inWidth,
            int outDepth, int outHeight, int outWidth)
        {
            var k = _kernelCache["maxpool3d_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)indices).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, inDepth);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outDepth);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * channels * outDepth, localX, localY, localZ);
        }

        public void NearestNeighborUpsample3D(IGpuBuffer input, IGpuBuffer output,
            int batch, int channels,
            int inDepth, int inHeight, int inWidth,
            int scaleD, int scaleH, int scaleW)
        {
            var k = _kernelCache["nearest_upsample3d"];
            int outDepth = inDepth * scaleD;
            int outHeight = inHeight * scaleH;
            int outWidth = inWidth * scaleW;

            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, inDepth);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, scaleD);
            k.SetArg(arg++, scaleH);
            k.SetArg(arg++, scaleW);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * channels * outDepth, localX, localY, localZ);
        }

        public void NearestNeighborUpsample3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
            int batch, int channels,
            int inDepth, int inHeight, int inWidth,
            int scaleD, int scaleH, int scaleW)
        {
            var k = _kernelCache["nearest_upsample3d_backward"];
            int outDepth = inDepth * scaleD;
            int outHeight = inHeight * scaleH;
            int outWidth = inWidth * scaleW;

            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, inDepth);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, scaleD);
            k.SetArg(arg++, scaleH);
            k.SetArg(arg++, scaleW);

            int localX = 8, localY = 8, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * channels * outDepth, localX, localY, localZ);
        }

        #endregion

        #region Spatial Transformer Operations

        public void AffineGrid(IGpuBuffer theta, IGpuBuffer grid, int batch, int outputHeight, int outputWidth)
        {
            var k = _kernelCache["affine_grid"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)theta).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)grid).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, outputHeight);
            k.SetArg(arg++, outputWidth);

            int localX = 16, localY = 16, localZ = 1;
            k.Execute3D(outputWidth, outputHeight, batch, localX, localY, localZ);
        }

        public void GridSample(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
            int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
            int paddingMode = 0, bool alignCorners = false)
        {
            var k = _kernelCache["grid_sample"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)grid).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, paddingMode);
            k.SetArg(arg++, alignCorners ? 1 : 0);

            int localX = 16, localY = 16, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch * channels, localX, localY, localZ);
        }

        public void GridSampleBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
            IGpuBuffer gradInput, IGpuBuffer gradGrid,
            int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
            int paddingMode = 0, bool alignCorners = false)
        {
            var k = _kernelCache["grid_sample_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)grid).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradGrid).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, inHeight);
            k.SetArg(arg++, inWidth);
            k.SetArg(arg++, outHeight);
            k.SetArg(arg++, outWidth);
            k.SetArg(arg++, paddingMode);
            k.SetArg(arg++, alignCorners ? 1 : 0);

            int localX = 16, localY = 16, localZ = 1;
            k.Execute3D(outWidth, outHeight, batch, localX, localY, localZ);
        }

        #endregion

        #region Normalization Operations

        public void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
            IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
            int batch, int channels, int spatialSize, float epsilon, float momentum, bool training)
        {
            var k = _kernelCache["batchnorm_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gamma).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)beta).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)runningMean).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)runningVar).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveMean).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveInvVar).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, spatialSize);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, momentum);
            k.SetArg(arg++, training ? 1 : 0);

            k.Execute1D(channels, Math.Min(64, channels));
        }

        public void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
            IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
            int batch, int channels, int spatialSize, float epsilon)
        {
            var k = _kernelCache["batchnorm_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gamma).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveMean).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveInvVar).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradGamma).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradBeta).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, spatialSize);
            k.SetArg(arg++, epsilon);

            k.Execute1D(channels, Math.Min(64, channels));
        }

        public void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
            IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
        {
            var k = _kernelCache["layernorm_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gamma).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)beta).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveMean).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveInvVar).Buffer.Handle);
            k.SetArg(arg++, batchSize);
            k.SetArg(arg++, normalizedSize);
            k.SetArg(arg++, epsilon);

            k.Execute1D(batchSize, Math.Min(64, batchSize));
        }

        public void LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
            IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
            int batchSize, int normalizedSize, float epsilon)
        {
            // First compute gradInput
            var k = _kernelCache["layernorm_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gamma).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveMean).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveInvVar).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradGamma).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradBeta).Buffer.Handle);
            k.SetArg(arg++, batchSize);
            k.SetArg(arg++, normalizedSize);
            k.SetArg(arg++, epsilon);

            k.Execute1D(batchSize, Math.Min(64, batchSize));

            // Then accumulate gradient params
            var kp = _kernelCache["layernorm_grad_params"];
            arg = 0;
            kp.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            kp.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kp.SetArg(arg++, ((DirectOpenClGpuBuffer)saveMean).Buffer.Handle);
            kp.SetArg(arg++, ((DirectOpenClGpuBuffer)saveInvVar).Buffer.Handle);
            kp.SetArg(arg++, ((DirectOpenClGpuBuffer)gradGamma).Buffer.Handle);
            kp.SetArg(arg++, ((DirectOpenClGpuBuffer)gradBeta).Buffer.Handle);
            kp.SetArg(arg++, batchSize);
            kp.SetArg(arg++, normalizedSize);

            kp.Execute1D(normalizedSize, Math.Min(64, normalizedSize));
        }

        public void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
            IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
        {
            var k = _kernelCache["groupnorm_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gamma).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)beta).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveMean).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveInvVar).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, numGroups);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, spatialSize);
            k.SetArg(arg++, epsilon);

            k.Execute1D(batch * numGroups, Math.Min(64, batch * numGroups));
        }

        public void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
            IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon)
        {
            var k = _kernelCache["instancenorm_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gamma).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)beta).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveMean).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveInvVar).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, channels);
            k.SetArg(arg++, spatialSize);
            k.SetArg(arg++, epsilon);

            k.Execute1D(batch * channels, Math.Min(64, batch * channels));
        }

        public void InstanceNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
            IGpuBuffer saveMean, IGpuBuffer saveInvVar,
            IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
            int batch, int channels, int spatialSize, float epsilon)
        {
            // Fallback: implement using CPU operations
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

            // Upload results back to GPU buffers
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            // Upload gradInput to GPU
            var handleGradInput = GCHandle.Alloc(gradInputData, GCHandleType.Pinned);
            try
            {
                int err = OpenClNativeBindings.EnqueueWriteBuffer(
                    _context.CommandQueue,
                    ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle,
                    1, // blocking
                    UIntPtr.Zero,
                    (UIntPtr)(gradInputData.Length * sizeof(float)),
                    handleGradInput.AddrOfPinnedObject(),
                    0, IntPtr.Zero, IntPtr.Zero);
                if (err != 0)
                    throw new InvalidOperationException($"OpenCL EnqueueWriteBuffer failed for gradInput: {err}");
            }
            finally
            {
                handleGradInput.Free();
            }

            // Upload gradGamma to GPU
            var handleGradGamma = GCHandle.Alloc(gradGammaData, GCHandleType.Pinned);
            try
            {
                int err = OpenClNativeBindings.EnqueueWriteBuffer(
                    _context.CommandQueue,
                    ((DirectOpenClGpuBuffer)gradGamma).Buffer.Handle,
                    1, // blocking
                    UIntPtr.Zero,
                    (UIntPtr)(gradGammaData.Length * sizeof(float)),
                    handleGradGamma.AddrOfPinnedObject(),
                    0, IntPtr.Zero, IntPtr.Zero);
                if (err != 0)
                    throw new InvalidOperationException($"OpenCL EnqueueWriteBuffer failed for gradGamma: {err}");
            }
            finally
            {
                handleGradGamma.Free();
            }

            // Upload gradBeta to GPU
            var handleGradBeta = GCHandle.Alloc(gradBetaData, GCHandleType.Pinned);
            try
            {
                int err = OpenClNativeBindings.EnqueueWriteBuffer(
                    _context.CommandQueue,
                    ((DirectOpenClGpuBuffer)gradBeta).Buffer.Handle,
                    1, // blocking
                    UIntPtr.Zero,
                    (UIntPtr)(gradBetaData.Length * sizeof(float)),
                    handleGradBeta.AddrOfPinnedObject(),
                    0, IntPtr.Zero, IntPtr.Zero);
                if (err != 0)
                    throw new InvalidOperationException($"OpenCL EnqueueWriteBuffer failed for gradBeta: {err}");
            }
            finally
            {
                handleGradBeta.Free();
            }
        }

        public void RmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
            int batchSize, int normalizedSize, float epsilon)
        {
            var k = _kernelCache["rmsnorm_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gamma).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveRms).Buffer.Handle);
            k.SetArg(arg++, batchSize);
            k.SetArg(arg++, normalizedSize);
            k.SetArg(arg++, epsilon);

            k.Execute1D(batchSize, Math.Min(64, batchSize));
        }

        public void RmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
            IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize, float epsilon)
        {
            // Compute gradInput
            var k = _kernelCache["rmsnorm_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gamma).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)saveRms).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradGamma).Buffer.Handle);
            k.SetArg(arg++, batchSize);
            k.SetArg(arg++, normalizedSize);
            k.SetArg(arg++, epsilon);

            k.Execute1D(batchSize, Math.Min(64, batchSize));

            // Compute gradGamma
            var k2 = _kernelCache["rmsnorm_grad_gamma"];
            arg = 0;
            k2.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k2.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k2.SetArg(arg++, ((DirectOpenClGpuBuffer)saveRms).Buffer.Handle);
            k2.SetArg(arg++, ((DirectOpenClGpuBuffer)gradGamma).Buffer.Handle);
            k2.SetArg(arg++, batchSize);
            k2.SetArg(arg++, normalizedSize);

            k2.Execute1D(normalizedSize, Math.Min(256, normalizedSize));
        }

        #endregion

        #region Dropout Operations

        public void Dropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask, int size, float dropoutRate, ulong seed, bool training)
        {
            var k = _kernelCache["dropout_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)mask).Buffer.Handle);
            k.SetArg(arg++, size);
            k.SetArg(arg++, dropoutRate);
            k.SetArg(arg++, (int)(seed & 0xFFFFFFFF));
            k.SetArg(arg++, training ? 1 : 0);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void DropoutBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size, float dropoutRate)
        {
            var k = _kernelCache["dropout_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)mask).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);
            k.SetArg(arg++, dropoutRate);

            k.Execute1D(size, Math.Min(256, size));
        }

        #endregion

        #region Embedding Operations

        public void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim)
        {
            var k = _kernelCache["embedding_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)indices).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)embeddingTable).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, numIndices);
            k.SetArg(arg++, embeddingDim);

            int localX = Math.Min(16, embeddingDim);
            int localY = Math.Min(8, numIndices);
            k.Execute2D(embeddingDim, numIndices, localX, localY);
        }

        public void EmbeddingBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize)
        {
            var k = _kernelCache["embedding_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)indices).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradEmbedding).Buffer.Handle);
            k.SetArg(arg++, numIndices);
            k.SetArg(arg++, embeddingDim);
            k.SetArg(arg++, vocabSize);

            int localX = Math.Min(16, embeddingDim);
            int localY = Math.Min(8, numIndices);
            k.Execute2D(embeddingDim, numIndices, localX, localY);
        }

        public IGpuBuffer AllocateIntBuffer(int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var buffer = new DirectOpenClBuffer(_context, size);
            return new DirectOpenClGpuBuffer(buffer);
        }

        public IGpuBuffer AllocateIntBuffer(int[] data)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            // Convert int array to float array for storage (net471 compatible)
            var floatData = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                byte[] bytes = BitConverter.GetBytes(data[i]);
                floatData[i] = BitConverter.ToSingle(bytes, 0);
            }

            var buffer = new DirectOpenClBuffer(_context, floatData);
            return new DirectOpenClGpuBuffer(buffer);
        }

        #endregion

        #region Attention Operations

        public void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
            IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
            int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
        {
            var k = _kernelCache["scaled_dot_product_attention"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)query).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)key).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)value).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, attentionWeights != null ? ((DirectOpenClGpuBuffer)attentionWeights).Buffer.Handle : IntPtr.Zero);
            k.SetArg(arg++, mask != null ? ((DirectOpenClGpuBuffer)mask).Buffer.Handle : IntPtr.Zero);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, numHeads);
            k.SetArg(arg++, seqLen);
            k.SetArg(arg++, headDim);
            k.SetArg(arg++, scale);
            k.SetArg(arg++, isCausal ? 1 : 0);
            k.SetArg(arg++, attentionWeights != null ? 1 : 0);
            k.SetArg(arg++, mask != null ? 1 : 0);

            int localX = Math.Min(16, headDim);
            int localY = Math.Min(8, seqLen);
            k.Execute3D(headDim, seqLen, batch * numHeads, localX, localY, 1);
        }

        public void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
            IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
            int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
        {
            var k = _kernelCache["attention_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)query).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)key).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)value).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)attentionWeights).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradQuery).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradKey).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradValue).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, numHeads);
            k.SetArg(arg++, seqLen);
            k.SetArg(arg++, headDim);
            k.SetArg(arg++, scale);
            k.SetArg(arg++, isCausal ? 1 : 0);

            int localX = Math.Min(16, headDim);
            int localY = Math.Min(8, seqLen);
            k.Execute3D(headDim, seqLen, batch * numHeads, localX, localY, 1);
        }

        public void FlashAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
            IGpuBuffer output, IGpuBuffer? mask, int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
        {
            // FlashAttention is a memory-efficient version - for OpenCL, use tiled implementation
            var k = _kernelCache["flash_attention_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)query).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)key).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)value).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, mask != null ? ((DirectOpenClGpuBuffer)mask).Buffer.Handle : IntPtr.Zero);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, numHeads);
            k.SetArg(arg++, seqLen);
            k.SetArg(arg++, headDim);
            k.SetArg(arg++, scale);
            k.SetArg(arg++, isCausal ? 1 : 0);
            k.SetArg(arg++, mask != null ? 1 : 0);

            int localX = Math.Min(32, seqLen);
            k.Execute2D(seqLen, batch * numHeads, localX, 1);
        }

        public void FlashAttentionV2(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
            IGpuBuffer output, IGpuBuffer softmaxStats,
            int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
        {
            // FlashAttention V2 with online softmax and log-sum-exp statistics
            // Uses tiled computation for O(N) memory complexity
            var k = _kernelCache["flash_attention_v2"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)query).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)key).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)value).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)softmaxStats).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, numHeads);
            k.SetArg(arg++, seqQ);
            k.SetArg(arg++, seqK);
            k.SetArg(arg++, headDim);
            k.SetArg(arg++, scale);
            k.SetArg(arg++, isCausal ? 1 : 0);

            // Each work item handles one query position
            int localX = Math.Min(64, seqQ);
            k.Execute2D(seqQ, batch * numHeads, localX, 1);
        }

        public void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
            IGpuBuffer output, IGpuBuffer softmaxStats,
            IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
            int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
        {
            // FlashAttention backward with recomputation for memory efficiency
            var k = _kernelCache["flash_attention_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)query).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)key).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)value).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)softmaxStats).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradQuery).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradKey).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradValue).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, numHeads);
            k.SetArg(arg++, seqQ);
            k.SetArg(arg++, seqK);
            k.SetArg(arg++, headDim);
            k.SetArg(arg++, scale);
            k.SetArg(arg++, isCausal ? 1 : 0);

            // Process in blocks
            int localX = Math.Min(64, seqQ);
            k.Execute2D(seqQ, batch * numHeads, localX, 1);
        }

        public void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
            IGpuBuffer output, IGpuBuffer? attentionWeights,
            int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
        {
            // GQA: Multiple query heads share same KV heads
            // numQHeads must be divisible by numKVHeads
            var k = _kernelCache["grouped_query_attention"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)query).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)key).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)value).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, attentionWeights != null ? ((DirectOpenClGpuBuffer)attentionWeights).Buffer.Handle : IntPtr.Zero);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, numQHeads);
            k.SetArg(arg++, numKVHeads);
            k.SetArg(arg++, numQHeads / numKVHeads); // queries per KV head
            k.SetArg(arg++, seqQ);
            k.SetArg(arg++, seqK);
            k.SetArg(arg++, headDim);
            k.SetArg(arg++, scale);
            k.SetArg(arg++, isCausal ? 1 : 0);
            k.SetArg(arg++, attentionWeights != null ? 1 : 0);

            // Each work item handles one query head position
            int localX = Math.Min(16, headDim);
            int localY = Math.Min(8, seqQ);
            k.Execute3D(headDim, seqQ, batch * numQHeads, localX, localY, 1);
        }

        public void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
            IGpuBuffer attentionWeights,
            IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
            int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale)
        {
            // GQA backward - accumulate gradients for K,V across shared query heads
            var k = _kernelCache["grouped_query_attention_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)query).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)key).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)value).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)attentionWeights).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradQuery).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradKey).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradValue).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, numQHeads);
            k.SetArg(arg++, numKVHeads);
            k.SetArg(arg++, numQHeads / numKVHeads);
            k.SetArg(arg++, seqQ);
            k.SetArg(arg++, seqK);
            k.SetArg(arg++, headDim);
            k.SetArg(arg++, scale);

            int localX = Math.Min(16, headDim);
            int localY = Math.Min(8, seqQ);
            k.Execute3D(headDim, seqQ, batch * numQHeads, localX, localY, 1);
        }

        #endregion

        #region Transpose and Reshape Operations

        public void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols)
        {
            var k = _kernelCache["transpose2d"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, rows);
            k.SetArg(arg++, cols);

            int localX = Math.Min(16, cols);
            int localY = Math.Min(16, rows);
            k.Execute2D(cols, rows, localX, localY);
        }

        public void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols)
        {
            var k = _kernelCache["batched_transpose"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, batch);
            k.SetArg(arg++, rows);
            k.SetArg(arg++, cols);

            int localX = Math.Min(16, cols);
            int localY = Math.Min(16, rows);
            k.Execute3D(cols, rows, batch, localX, localY, 1);
        }

        public void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation)
        {
            // Permute is a general operation - handle common cases
            if (shape.Length == 2 && permutation[0] == 1 && permutation[1] == 0)
            {
                // 2D transpose
                Transpose(input, output, shape[0], shape[1]);
                return;
            }

            // For general permute, use CPU fallback (could implement general permute kernel)
            var srcBuffer = ((DirectOpenClGpuBuffer)input).Buffer;
            var dstBuffer = ((DirectOpenClGpuBuffer)output).Buffer;
            var data = srcBuffer.ToArray();

            int totalSize = 1;
            foreach (var dim in shape) totalSize *= dim;

            var result = new float[totalSize];

            // Calculate strides for original and permuted tensors
            int[] strides = new int[shape.Length];
            int[] permutedShape = new int[shape.Length];
            int[] permutedStrides = new int[shape.Length];

            strides[shape.Length - 1] = 1;
            for (int i = shape.Length - 2; i >= 0; i--)
                strides[i] = strides[i + 1] * shape[i + 1];

            for (int i = 0; i < shape.Length; i++)
                permutedShape[i] = shape[permutation[i]];

            permutedStrides[shape.Length - 1] = 1;
            for (int i = shape.Length - 2; i >= 0; i--)
                permutedStrides[i] = permutedStrides[i + 1] * permutedShape[i + 1];

            // Perform permutation
            for (int i = 0; i < totalSize; i++)
            {
                int[] coords = new int[shape.Length];
                int idx = i;
                for (int d = 0; d < shape.Length; d++)
                {
                    coords[d] = idx / strides[d];
                    idx %= strides[d];
                }

                int newIdx = 0;
                for (int d = 0; d < shape.Length; d++)
                    newIdx += coords[permutation[d]] * permutedStrides[d];

                result[newIdx] = data[i];
            }

            dstBuffer.CopyFromHost(result);
        }

        public void Copy(IGpuBuffer source, IGpuBuffer destination, int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            // Use CPU fallback for copy - download from source and upload to destination
            var srcBuffer = ((DirectOpenClGpuBuffer)source).Buffer;
            var dstBuffer = ((DirectOpenClGpuBuffer)destination).Buffer;

            var data = srcBuffer.ToArray();
            dstBuffer.CopyFromHost(data);
        }

        public void Fill(IGpuBuffer buffer, float value, int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            // Use CPU fallback for fill
            var data = new float[size];
            for (int i = 0; i < size; i++)
            {
                data[i] = value;
            }

            var clBuffer = ((DirectOpenClGpuBuffer)buffer).Buffer;
            clBuffer.CopyFromHost(data);
        }

        /// <inheritdoc/>
        public void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows, int srcCols, int destTotalCols, int destColOffset)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            var kernel = _kernelCache["copy_2d_strided"];
            var bufferSrc = ((DirectOpenClGpuBuffer)source).Buffer;
            var bufferDst = ((DirectOpenClGpuBuffer)destination).Buffer;

            kernel.SetArg(0, bufferSrc.Handle);
            kernel.SetArg(1, bufferDst.Handle);
            kernel.SetArg(2, numRows);
            kernel.SetArg(3, srcCols);
            kernel.SetArg(4, destTotalCols);
            kernel.SetArg(5, destColOffset);

            var (localSizeX, localSizeY) = CalculateOptimalWorkGroupSize(srcCols, numRows);
            kernel.Execute2D(srcCols, numRows, localSizeX, localSizeY);
        }

        /// <inheritdoc/>
        public void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            // Try to use kernel if available, otherwise fallback to CPU
            if (!_kernelCache.TryGetValue("nearest_neighbor_upsample", out var kernel))
            {
                NearestNeighborUpsampleFallback(input, output, batchChannels, height, width, scaleFactor);
                return;
            }

            var bufferIn = ((DirectOpenClGpuBuffer)input).Buffer;
            var bufferOut = ((DirectOpenClGpuBuffer)output).Buffer;

            int outHeight = height * scaleFactor;
            int outWidth = width * scaleFactor;
            int outputSize = batchChannels * outHeight * outWidth;

            kernel.SetArg(0, bufferIn.Handle);
            kernel.SetArg(1, bufferOut.Handle);
            kernel.SetArg(2, batchChannels);
            kernel.SetArg(3, height);
            kernel.SetArg(4, width);
            kernel.SetArg(5, scaleFactor);
            kernel.SetArg(6, outputSize);

            kernel.Execute1D(outputSize, Math.Min(256, outputSize));
        }

        /// <summary>
        /// CPU fallback for nearest-neighbor upsampling when kernel is not available.
        /// </summary>
        private void NearestNeighborUpsampleFallback(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
        {
            int inputSize = batchChannels * height * width;
            int outHeight = height * scaleFactor;
            int outWidth = width * scaleFactor;
            int outputSize = batchChannels * outHeight * outWidth;

            // Download input
            var inputData = new float[inputSize];
            var bufferIn = ((DirectOpenClGpuBuffer)input).Buffer;
            bufferIn.CopyToHost(inputData);

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

            // Upload output
            var bufferOut = ((DirectOpenClGpuBuffer)output).Buffer;
            bufferOut.CopyFromHost(outputData);
        }

        /// <inheritdoc/>
        public void NearestNeighborUpsampleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");

            if (!_kernelCache.TryGetValue("nearest_neighbor_upsample_backward", out var kernel))
            {
                NearestNeighborUpsampleBackwardFallback(gradOutput, gradInput, batchChannels, height, width, scaleFactor);
                return;
            }

            int inputSize = batchChannels * height * width;

            // Kernel iterates over input elements, accumulating from scaleFactor x scaleFactor output regions
            // No zeroing needed - kernel writes directly (not +=)
            var bufferIn = ((DirectOpenClGpuBuffer)gradOutput).Buffer;
            var bufferOut = ((DirectOpenClGpuBuffer)gradInput).Buffer;

            kernel.SetArg(0, bufferIn.Handle);
            kernel.SetArg(1, bufferOut.Handle);
            kernel.SetArg(2, batchChannels);
            kernel.SetArg(3, height);
            kernel.SetArg(4, width);
            kernel.SetArg(5, scaleFactor);
            kernel.SetArg(6, inputSize);

            kernel.Execute1D(inputSize, Math.Min(256, inputSize));
        }

        /// <summary>
        /// CPU fallback for nearest-neighbor upsampling backward when kernel is not available.
        /// </summary>
        private void NearestNeighborUpsampleBackwardFallback(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
        {
            int inputSize = batchChannels * height * width;
            int outHeight = height * scaleFactor;
            int outWidth = width * scaleFactor;
            int outputSize = batchChannels * outHeight * outWidth;

            // Download gradient output
            var gradOutputData = new float[outputSize];
            var bufferIn = ((DirectOpenClGpuBuffer)gradOutput).Buffer;
            bufferIn.CopyToHost(gradOutputData);

            // Perform CPU backward (accumulate gradients)
            var gradInputData = new float[inputSize];
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
                        gradInputData[inputIdx] += gradOutputData[outputIdx];
                    }
                }
            }

            // Upload gradient input
            var bufferOut = ((DirectOpenClGpuBuffer)gradInput).Buffer;
            bufferOut.CopyFromHost(gradInputData);
        }

        #endregion

        #region Activation Gradient Operations

        public void ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        {
            var k = _kernelCache["relu_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void SigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
        {
            var k = _kernelCache["sigmoid_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void TanhBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
        {
            var k = _kernelCache["tanh_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        {
            var k = _kernelCache["gelu_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
        {
            var k = _kernelCache["softmax_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, batchSize);
            k.SetArg(arg++, features);

            k.Execute1D(batchSize, Math.Min(64, batchSize));
        }

        public void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
        {
            var k = _kernelCache["leaky_relu_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, alpha);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void LeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, int size)
        {
            var k = _kernelCache["leaky_relu_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, alpha);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void Elu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
        {
            var k = _kernelCache["elu_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, alpha);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void EluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer output, IGpuBuffer gradInput, float alpha, int size)
        {
            var k = _kernelCache["elu_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, alpha);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void Swish(IGpuBuffer A, IGpuBuffer B, int size)
        {
            var k = _kernelCache["swish_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void SwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        {
            var k = _kernelCache["swish_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void Silu(IGpuBuffer A, IGpuBuffer B, int size)
        {
            var k = _kernelCache["silu_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void Mish(IGpuBuffer A, IGpuBuffer B, int size)
        {
            var k = _kernelCache["mish_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void Softplus(IGpuBuffer A, IGpuBuffer B, int size)
        {
            var k = _kernelCache["softplus_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void Hardswish(IGpuBuffer A, IGpuBuffer B, int size)
        {
            var k = _kernelCache["hardswish_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void Selu(IGpuBuffer A, IGpuBuffer B, float alpha, float scale, int size)
        {
            var k = _kernelCache["selu_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, alpha);
            k.SetArg(arg++, scale);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void Hardsigmoid(IGpuBuffer A, IGpuBuffer B, int size)
        {
            var k = _kernelCache["hardsigmoid_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void Hardtanh(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size)
        {
            var k = _kernelCache["hardtanh_forward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, minVal);
            k.SetArg(arg++, maxVal);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // SiLU backward uses SwishBackward since they're mathematically equivalent
        public void SiluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        {
            SwishBackward(gradOutput, input, gradInput, size);
        }

        public void MishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        {
            var k = _kernelCache["mish_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void SoftplusBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        {
            var k = _kernelCache["softplus_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void HardswishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        {
            var k = _kernelCache["hardswish_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void SeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, float scale, int size)
        {
            var k = _kernelCache["selu_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, alpha);
            k.SetArg(arg++, scale);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void HardsigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        {
            var k = _kernelCache["hardsigmoid_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void HardtanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float minVal, float maxVal, int size)
        {
            var k = _kernelCache["hardtanh_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, minVal);
            k.SetArg(arg++, maxVal);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        #endregion

        #region Loss Function Operations

        public float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses)
        {
            // Allocate output buffer for per-sample losses
            using var lossBuffer = AllocateBuffer(batchSize);

            var k = _kernelCache["cross_entropy_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)lossBuffer).Buffer.Handle);
            k.SetArg(arg++, batchSize);
            k.SetArg(arg++, numClasses);

            k.Execute1D(batchSize, Math.Min(64, batchSize));

            // Download and compute mean
            var losses = new float[batchSize];
            DownloadBuffer(lossBuffer, losses);
            float sum = 0;
            for (int i = 0; i < batchSize; i++) sum += losses[i];
            return sum / batchSize;
        }

        public void CrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int batchSize, int numClasses)
        {
            var k = _kernelCache["cross_entropy_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, batchSize);
            k.SetArg(arg++, numClasses);
        }

        public float BinaryCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        {
            using var lossBuffer = AllocateBuffer(size);

            var k = _kernelCache["bce_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)lossBuffer).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            var losses = new float[size];
            DownloadBuffer(lossBuffer, losses);
            float sum = 0;
            for (int i = 0; i < size; i++) sum += losses[i];
            return sum / size;
        }

        public void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        {
            var k = _kernelCache["bce_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        {
            using var lossBuffer = AllocateBuffer(size);

            var k = _kernelCache["mse_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)lossBuffer).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            var losses = new float[size];
            DownloadBuffer(lossBuffer, losses);
            float sum = 0;
            for (int i = 0; i < size; i++) sum += losses[i];
            return sum / size;
        }

        public void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        {
            var k = _kernelCache["mse_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta)
        {
            using var lossBuffer = AllocateBuffer(size);

            var k = _kernelCache["smooth_l1_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)lossBuffer).Buffer.Handle);
            k.SetArg(arg++, size);
            k.SetArg(arg++, beta);

            k.Execute1D(size, Math.Min(256, size));

            var losses = new float[size];
            DownloadBuffer(lossBuffer, losses);
            float sum = 0;
            for (int i = 0; i < size; i++) sum += losses[i];
            return sum / size;
        }

        public void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta)
        {
            var k = _kernelCache["smooth_l1_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);
            k.SetArg(arg++, beta);

            k.Execute1D(size, Math.Min(256, size));
        }

        public float TripletLoss(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative, int batchSize, int embeddingDim, float margin)
        {
            if (anchor is null) throw new ArgumentNullException(nameof(anchor));
            if (positive is null) throw new ArgumentNullException(nameof(positive));
            if (negative is null) throw new ArgumentNullException(nameof(negative));
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
            if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim), "Embedding dimension must be positive.");

            using var lossBuffer = AllocateBuffer(batchSize);

            var k = _kernelCache["triplet_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)anchor).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)positive).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)negative).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)lossBuffer).Buffer.Handle);
            k.SetArg(arg++, batchSize);
            k.SetArg(arg++, embeddingDim);
            k.SetArg(arg++, margin);

            k.Execute1D(batchSize, Math.Min(256, batchSize));

            var losses = new float[batchSize];
            DownloadBuffer(lossBuffer, losses);
            float sum = 0;
            for (int i = 0; i < batchSize; i++) sum += losses[i];
            return sum / batchSize;
        }

        public void TripletLossBackward(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative,
            IGpuBuffer gradAnchor, IGpuBuffer gradPositive, IGpuBuffer gradNegative,
            int batchSize, int embeddingDim, float margin)
        {
            if (anchor is null) throw new ArgumentNullException(nameof(anchor));
            if (positive is null) throw new ArgumentNullException(nameof(positive));
            if (negative is null) throw new ArgumentNullException(nameof(negative));
            if (gradAnchor is null) throw new ArgumentNullException(nameof(gradAnchor));
            if (gradPositive is null) throw new ArgumentNullException(nameof(gradPositive));
            if (gradNegative is null) throw new ArgumentNullException(nameof(gradNegative));
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
            if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim), "Embedding dimension must be positive.");

            var k = _kernelCache["triplet_loss_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)anchor).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)positive).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)negative).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradAnchor).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradPositive).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradNegative).Buffer.Handle);
            k.SetArg(arg++, batchSize);
            k.SetArg(arg++, embeddingDim);
            k.SetArg(arg++, margin);

            k.Execute1D(batchSize, Math.Min(256, batchSize));
        }

        // Huber Loss
        public float HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float delta)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["huber_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, delta);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void HuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float delta)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["huber_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, delta);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // Focal Loss
        public float FocalLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float alpha, float gamma)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["focal_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, alpha);
            k.SetArg(arg++, gamma);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void FocalBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float alpha, float gamma)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["focal_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, alpha);
            k.SetArg(arg++, gamma);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // MAE Loss
        public float MaeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["mae_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void MaeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["mae_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // Log-Cosh Loss
        public float LogCoshLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["log_cosh_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void LogCoshBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["log_cosh_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // Quantile Loss
        public float QuantileLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float quantile)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["quantile_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, quantile);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void QuantileBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float quantile)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["quantile_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, quantile);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // Hinge Loss
        public float HingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["hinge_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void HingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["hinge_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // Squared Hinge Loss
        public float SquaredHingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["squared_hinge_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void SquaredHingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["squared_hinge_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // Poisson Loss
        public float PoissonLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["poisson_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void PoissonBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["poisson_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // Exponential Loss
        public float ExponentialLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["exponential_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void ExponentialBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["exponential_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // Modified Huber Loss
        public float ModifiedHuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["modified_huber_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void ModifiedHuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["modified_huber_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // Categorical Cross-Entropy Loss
        public float CategoricalCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["categorical_cross_entropy_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void CategoricalCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["categorical_cross_entropy_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // Charbonnier Loss
        public float CharbonnierLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float epsilon)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["charbonnier_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void CharbonnierBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float epsilon)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["charbonnier_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // Elastic Net Loss
        public float ElasticNetLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float l1Weight, float l2Weight)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));

            using var outputBuffer = AllocateBuffer(size);

            var k = _kernelCache["elastic_net_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, l1Weight);
            k.SetArg(arg++, l2Weight);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            return Sum(outputBuffer, size) / size;
        }

        public void ElasticNetBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float l1Weight, float l2Weight)
        {
            if (predictions is null) throw new ArgumentNullException(nameof(predictions));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));

            var k = _kernelCache["elastic_net_gradient"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)predictions).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)targets).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            k.SetArg(arg++, l1Weight);
            k.SetArg(arg++, l2Weight);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        // Contrastive Loss
        public float ContrastiveLoss(IGpuBuffer anchor, IGpuBuffer other, IGpuBuffer labels, int batchSize, int embeddingDim, float margin)
        {
            if (anchor is null) throw new ArgumentNullException(nameof(anchor));
            if (other is null) throw new ArgumentNullException(nameof(other));
            if (labels is null) throw new ArgumentNullException(nameof(labels));

            using var outputBuffer = AllocateBuffer(batchSize);

            var k = _kernelCache["contrastive_loss"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)anchor).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)other).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)labels).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)outputBuffer).Buffer.Handle);
            k.SetArg(arg++, batchSize);
            k.SetArg(arg++, embeddingDim);
            k.SetArg(arg++, margin);

            k.Execute1D(batchSize, Math.Min(256, batchSize));

            return Sum(outputBuffer, batchSize) / batchSize;
        }

        public void ContrastiveBackward(IGpuBuffer anchor, IGpuBuffer other, IGpuBuffer labels,
            IGpuBuffer gradAnchor, IGpuBuffer gradOther,
            int batchSize, int embeddingDim, float margin)
        {
            if (anchor is null) throw new ArgumentNullException(nameof(anchor));
            if (other is null) throw new ArgumentNullException(nameof(other));
            if (labels is null) throw new ArgumentNullException(nameof(labels));
            if (gradAnchor is null) throw new ArgumentNullException(nameof(gradAnchor));
            if (gradOther is null) throw new ArgumentNullException(nameof(gradOther));

            int totalSize = batchSize * embeddingDim;
            var k = _kernelCache["contrastive_loss_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)anchor).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)other).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)labels).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradAnchor).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradOther).Buffer.Handle);
            k.SetArg(arg++, batchSize);
            k.SetArg(arg++, embeddingDim);
            k.SetArg(arg++, margin);

            k.Execute1D(totalSize, Math.Min(256, totalSize));
        }

        #endregion

        #region Gradient Clipping and Utility Operations

        public void Clamp(IGpuBuffer A, IGpuBuffer B, float min, float max, int size)
        {
            var k = _kernelCache["clamp"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, min);
            k.SetArg(arg++, max);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public float L2Norm(IGpuBuffer A, int size)
        {
            using var squaredBuffer = AllocateBuffer(size);

            var k = _kernelCache["l2_norm_squared"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)squaredBuffer).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));

            var squared = new float[size];
            DownloadBuffer(squaredBuffer, squared);
            float sum = 0;
            for (int i = 0; i < size; i++) sum += squared[i];
            return (float)Math.Sqrt(sum);
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
                var k = _kernelCache["scale"];
                uint arg = 0;
                k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
                k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
                k.SetArg(arg++, scale);
                k.SetArg(arg++, size);

                k.Execute1D(size, Math.Min(256, size));
            }
            else
            {
                Copy(A, B, size);
            }
        }

        public void Fma(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int size)
        {
            var k = _kernelCache["fma"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)C).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)D).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void ScatterAdd(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer destination, int sourceSize, int destSize)
        {
            var k = _kernelCache["scatter_add"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)source).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)indices).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)destination).Buffer.Handle);
            k.SetArg(arg++, sourceSize);
            k.SetArg(arg++, destSize);

            k.Execute1D(sourceSize, Math.Min(256, sourceSize));
        }

        public void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource,
            int numIndices, int featureSize)
        {
            var k = _kernelCache["scatter_add_backward"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradDestination).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)indices).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradSource).Buffer.Handle);
            k.SetArg(arg++, numIndices);
            k.SetArg(arg++, featureSize);

            k.Execute1D(numIndices, Math.Min(256, numIndices));
        }

        public void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize)
        {
            var k = _kernelCache["gather"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)source).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)indices).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, numIndices);
            k.SetArg(arg++, featureSize);

            k.Execute2D(featureSize, numIndices, Math.Min(16, featureSize), Math.Min(16, numIndices));
        }

        #endregion

        #region Comparison Operations

        public void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        {
            var k = _kernelCache["greater_than"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)C).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        {
            var k = _kernelCache["less_than"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)C).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        {
            var k = _kernelCache["equal"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)C).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        {
            var k = _kernelCache["where_select"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)condition).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)C).Buffer.Handle);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size)
        {
            var k = _kernelCache["not_equal_scalar"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)C).Buffer.Handle);
            k.SetArg(arg++, scalar);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        #endregion

        #region Statistics Operations

        public void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
        {
            var k = _kernelCache["mean_axis"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, outerSize);
            k.SetArg(arg++, reduceSize);

            k.Execute1D(outerSize, Math.Min(64, outerSize));
        }

        public void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize)
        {
            var k = _kernelCache["var_axis"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)mean).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)variance).Buffer.Handle);
            k.SetArg(arg++, outerSize);
            k.SetArg(arg++, reduceSize);

            k.Execute1D(outerSize, Math.Min(64, outerSize));
        }

        public void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
        {
            var k = _kernelCache["argmax"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)indices).Buffer.Handle);
            k.SetArg(arg++, outerSize);
            k.SetArg(arg++, reduceSize);

            k.Execute1D(outerSize, Math.Min(64, outerSize));
        }

        public void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
        {
            var k = _kernelCache["argmin"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)indices).Buffer.Handle);
            k.SetArg(arg++, outerSize);
            k.SetArg(arg++, reduceSize);

            k.Execute1D(outerSize, Math.Min(64, outerSize));
        }

        public void MaxAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var aBuf = ((DirectOpenClGpuBuffer)A).Buffer;
            var bBuf = ((DirectOpenClGpuBuffer)B).Buffer;

            // Reduce is simple element-wise for now (single work item per row)
            // For large reduceSize, this should be a parallel reduction.
            // Current kernel uses 1 thread per outer element.
            var kernel = _kernelCache["max_axis"];
            kernel.SetArg(0, aBuf.Handle);
            kernel.SetArg(1, bBuf.Handle);
            kernel.SetArg(2, outerSize);
            kernel.SetArg(3, reduceSize);

            // Execute 1D over outerSize
            kernel.Execute1D(outerSize, Math.Min(64, outerSize));
        }

        public void ArgMaxAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var aBuf = ((DirectOpenClGpuBuffer)A).Buffer;
            var idxBuf = ((DirectOpenClGpuBuffer)indices).Buffer;

            var kernel = _kernelCache["argmax_axis"];
            kernel.SetArg(0, aBuf.Handle);
            kernel.SetArg(1, idxBuf.Handle);
            kernel.SetArg(2, outerSize);
            kernel.SetArg(3, reduceSize);

            // Execute 1D over outerSize
            kernel.Execute1D(outerSize, Math.Min(64, outerSize));
        }

        #endregion

        public void TopK(IGpuBuffer A, IGpuBuffer values, IGpuBuffer indices, int outerSize, int reduceSize, int k, bool sorted = true)
        {
            var kernel = _kernelCache["topk"];
            uint arg = 0;
            kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)values).Buffer.Handle);
            kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)indices).Buffer.Handle);
            kernel.SetArg(arg++, outerSize);
            kernel.SetArg(arg++, reduceSize);
            kernel.SetArg(arg++, k);
            kernel.SetArg(arg++, sorted ? 1 : 0);
            // Local memory for top-k values and indices
            int localMemSize = k * (sizeof(float) + sizeof(int));
            kernel.SetLocalArg(arg++, localMemSize);
            kernel.SetLocalArg(arg++, localMemSize);

            // One work group per row
            kernel.Execute1D(outerSize * 64, 64);
        }

        public void BroadcastMultiplyLastAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
        {
            var k = _kernelCache["broadcast_multiply_last_axis"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)C).Buffer.Handle);
            k.SetArg(arg++, outerSize);
            k.SetArg(arg++, innerSize);

            int totalSize = outerSize * innerSize;
            k.Execute1D(totalSize, Math.Min(256, totalSize));
        }

        public void BroadcastMultiplyFirstAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
        {
            var k = _kernelCache["broadcast_multiply_first_axis"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)A).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)B).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)C).Buffer.Handle);
            k.SetArg(arg++, outerSize);
            k.SetArg(arg++, innerSize);

            int totalSize = outerSize * innerSize;
            k.Execute1D(totalSize, Math.Min(256, totalSize));
        }

        #region Optimizer Operations

        public void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
            float learningRate, float momentum, float weightDecay, int size)
        {
            var k = _kernelCache["sgd_momentum_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)velocity).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, momentum);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
            float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        {
            var k = _kernelCache["adam_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)m).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)v).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, beta1);
            k.SetArg(arg++, beta2);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, step);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        public void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
            float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        {
            var k = _kernelCache["adamw_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)m).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)v).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, beta1);
            k.SetArg(arg++, beta2);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, step);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        /// <inheritdoc/>
        public void RmspropUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer squaredAvg,
            float learningRate, float rho, float epsilon, float weightDecay, int size)
        {
            var k = _kernelCache["rmsprop_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)squaredAvg).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, rho);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        /// <inheritdoc/>
        public void AdagradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumulatedGrad,
            float learningRate, float epsilon, float weightDecay, int size)
        {
            var k = _kernelCache["adagrad_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)accumulatedGrad).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        /// <inheritdoc/>
        public void NagUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
            float learningRate, float momentum, float weightDecay, int size)
        {
            var k = _kernelCache["nag_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)velocity).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, momentum);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        /// <inheritdoc/>
        public void LarsUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
            float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
        {
            var k = _kernelCache["lars_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)velocity).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, momentum);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, trustCoeff);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        /// <inheritdoc/>
        public void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
            float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        {
            var k = _kernelCache["lamb_update"];
            float trustRatio = 1.0f; // Default: no layer-wise scaling (degenerates to AdamW)
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)m).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)v).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, beta1);
            k.SetArg(arg++, beta2);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, trustRatio);
            k.SetArg(arg++, step);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        /// <inheritdoc/>
        public void SgdUpdate(IGpuBuffer param, IGpuBuffer gradient,
            float learningRate, float weightDecay, int size)
        {
            var k = _kernelCache["sgd_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        /// <inheritdoc/>
        public void AdadeltaUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
            float rho, float epsilon, float weightDecay, int size)
        {
            var k = _kernelCache["adadelta_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)accumGrad).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)accumUpdate).Buffer.Handle);
            k.SetArg(arg++, rho);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        /// <inheritdoc/>
        public void AmsgradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
            float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        {
            var k = _kernelCache["amsgrad_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)m).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)v).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)vMax).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, beta1);
            k.SetArg(arg++, beta2);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, step);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        /// <inheritdoc/>
        public void AdamaxUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer u,
            float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        {
            var k = _kernelCache["adamax_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)m).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)u).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, beta1);
            k.SetArg(arg++, beta2);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, step);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        /// <inheritdoc/>
        public void LionUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m,
            float learningRate, float beta1, float beta2, float weightDecay, int size)
        {
            var k = _kernelCache["lion_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)m).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, beta1);
            k.SetArg(arg++, beta2);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        /// <inheritdoc/>
        public void NadamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
            float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        {
            var k = _kernelCache["nadam_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)m).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)v).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, beta1);
            k.SetArg(arg++, beta2);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, step);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        /// <inheritdoc/>
        public void FtrlUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer z, IGpuBuffer n,
            float learningRate, float l1Reg, float l2Reg, float beta, int size)
        {
            var k = _kernelCache["ftrl_update"];
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)param).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradient).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)z).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)n).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, l1Reg);
            k.SetArg(arg++, l2Reg);
            k.SetArg(arg++, beta);
            k.SetArg(arg++, size);

            k.Execute1D(size, Math.Min(256, size));
        }

        #endregion

        #region FFT and Signal Processing

        /// <inheritdoc/>
        public void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var inReal = ((DirectOpenClGpuBuffer)inputReal).Buffer;
            var inImag = ((DirectOpenClGpuBuffer)inputImag).Buffer;
            var outReal = ((DirectOpenClGpuBuffer)outputReal).Buffer;
            var outImag = ((DirectOpenClGpuBuffer)outputImag).Buffer;

            // Copy input to output for in-place FFT
            CopyBuffer(inputReal, outputReal, n);
            CopyBuffer(inputImag, outputImag, n);

            int log2n = (int)MathHelper.Log2(n);

            // Bit-reversal permutation
            var bitRevKernel = _kernelCache["bit_reverse_permutation"];
            bitRevKernel.SetArg(0, outReal.Handle);
            bitRevKernel.SetArg(1, outImag.Handle);
            bitRevKernel.SetArg(2, n);
            bitRevKernel.SetArg(3, log2n);
            bitRevKernel.Execute1D(n, Math.Min(256, n));

            // FFT butterfly stages
            var butterflyKernel = _kernelCache["fft_butterfly"];
            for (int stride = 2; stride <= n; stride *= 2)
            {
                int halfStride = stride / 2;
                butterflyKernel.SetArg(0, outReal.Handle);
                butterflyKernel.SetArg(1, outImag.Handle);
                butterflyKernel.SetArg(2, n);
                butterflyKernel.SetArg(3, stride);
                butterflyKernel.SetArg(4, inverse ? 1 : 0);
                butterflyKernel.Execute1D(n / 2, Math.Min(256, n / 2));
            }

            // Scale for inverse FFT
            if (inverse)
            {
                ScaleBuffer(outputReal, 1.0f / n, n);
                ScaleBuffer(outputImag, 1.0f / n, n);
            }
        }

        /// <inheritdoc/>
        public void RFFT(IGpuBuffer input, IGpuBuffer outputReal, IGpuBuffer outputImag, int n)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            // Allocate temporary buffers for full complex FFT
            var tempReal = AllocateBuffer(n);
            var tempImag = AllocateBuffer(n);

            try
            {
                // Copy real input and zero imaginary
                CopyBuffer(input, tempReal, n);
                ZeroBuffer(tempImag, n);

                // Perform full complex FFT
                FFT(tempReal, tempImag, tempReal, tempImag, n, false);

                // Extract positive frequencies (n/2+1 elements)
                var rfftKernel = _kernelCache["rfft_postprocess"];
                rfftKernel.SetArg(0, ((DirectOpenClGpuBuffer)tempReal).Buffer.Handle);
                rfftKernel.SetArg(1, ((DirectOpenClGpuBuffer)tempImag).Buffer.Handle);
                rfftKernel.SetArg(2, ((DirectOpenClGpuBuffer)outputReal).Buffer.Handle);
                rfftKernel.SetArg(3, ((DirectOpenClGpuBuffer)outputImag).Buffer.Handle);
                rfftKernel.SetArg(4, n);
                rfftKernel.Execute1D(n / 2 + 1, Math.Min(256, n / 2 + 1));
            }
            finally
            {
                tempReal.Dispose();
                tempImag.Dispose();
            }
        }

        /// <inheritdoc/>
        public void IRFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer output, int n)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            // Allocate temporary buffers for full complex FFT
            var tempReal = AllocateBuffer(n);
            var tempImag = AllocateBuffer(n);

            try
            {
                // Reconstruct negative frequencies using conjugate symmetry
                var irfftKernel = _kernelCache["irfft_preprocess"];
                irfftKernel.SetArg(0, ((DirectOpenClGpuBuffer)inputReal).Buffer.Handle);
                irfftKernel.SetArg(1, ((DirectOpenClGpuBuffer)inputImag).Buffer.Handle);
                irfftKernel.SetArg(2, ((DirectOpenClGpuBuffer)tempReal).Buffer.Handle);
                irfftKernel.SetArg(3, ((DirectOpenClGpuBuffer)tempImag).Buffer.Handle);
                irfftKernel.SetArg(4, n);
                irfftKernel.Execute1D(n, Math.Min(256, n));

                // Perform inverse FFT
                FFT(tempReal, tempImag, tempReal, tempImag, n, true);

                // Copy real part to output
                CopyBuffer(tempReal, output, n);
            }
            finally
            {
                tempReal.Dispose();
                tempImag.Dispose();
            }
        }

        /// <inheritdoc/>
        public void BatchedFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int batch, int n, bool inverse)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var inReal = ((DirectOpenClGpuBuffer)inputReal).Buffer;
            var inImag = ((DirectOpenClGpuBuffer)inputImag).Buffer;
            var outReal = ((DirectOpenClGpuBuffer)outputReal).Buffer;
            var outImag = ((DirectOpenClGpuBuffer)outputImag).Buffer;

            // Copy input to output for in-place FFT
            CopyBuffer(inputReal, outputReal, batch * n);
            CopyBuffer(inputImag, outputImag, batch * n);

            int log2n = (int)MathHelper.Log2(n);

            // Batched bit-reversal
            var bitRevKernel = _kernelCache["batched_bit_reverse"];
            bitRevKernel.SetArg(0, outReal.Handle);
            bitRevKernel.SetArg(1, outImag.Handle);
            bitRevKernel.SetArg(2, batch);
            bitRevKernel.SetArg(3, n);
            bitRevKernel.SetArg(4, log2n);
            bitRevKernel.Execute2D(n, batch, Math.Min(256, n), 1);

            // Batched FFT butterfly stages
            var butterflyKernel = _kernelCache["batched_fft_butterfly"];
            for (int stride = 2; stride <= n; stride *= 2)
            {
                butterflyKernel.SetArg(0, outReal.Handle);
                butterflyKernel.SetArg(1, outImag.Handle);
                butterflyKernel.SetArg(2, batch);
                butterflyKernel.SetArg(3, n);
                butterflyKernel.SetArg(4, stride);
                butterflyKernel.SetArg(5, inverse ? 1 : 0);
                butterflyKernel.Execute2D(n / 2, batch, Math.Min(256, n / 2), 1);
            }

            // Scale for inverse FFT
            if (inverse)
            {
                ScaleBuffer(outputReal, 1.0f / n, batch * n);
                ScaleBuffer(outputImag, 1.0f / n, batch * n);
            }
        }

        /// <inheritdoc/>
        public void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int height, int width, bool inverse)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var inReal = ((DirectOpenClGpuBuffer)inputReal).Buffer;
            var inImag = ((DirectOpenClGpuBuffer)inputImag).Buffer;
            var outReal = ((DirectOpenClGpuBuffer)outputReal).Buffer;
            var outImag = ((DirectOpenClGpuBuffer)outputImag).Buffer;

            // Copy input to output for in-place FFT
            CopyBuffer(inputReal, outputReal, height * width);
            CopyBuffer(inputImag, outputImag, height * width);

            int log2Width = (int)MathHelper.Log2(width);
            int log2Height = (int)MathHelper.Log2(height);

            // Row-wise bit reversal
            var bitRevRowsKernel = _kernelCache["bit_reverse_rows"];
            bitRevRowsKernel.SetArg(0, outReal.Handle);
            bitRevRowsKernel.SetArg(1, outImag.Handle);
            bitRevRowsKernel.SetArg(2, height);
            bitRevRowsKernel.SetArg(3, width);
            bitRevRowsKernel.SetArg(4, log2Width);
            bitRevRowsKernel.Execute2D(width, height, Math.Min(16, width), Math.Min(16, height));

            // Row-wise FFT
            var rowButterfly = _kernelCache["fft_rows_butterfly"];
            for (int stride = 2; stride <= width; stride *= 2)
            {
                rowButterfly.SetArg(0, outReal.Handle);
                rowButterfly.SetArg(1, outImag.Handle);
                rowButterfly.SetArg(2, height);
                rowButterfly.SetArg(3, width);
                rowButterfly.SetArg(4, stride);
                rowButterfly.SetArg(5, inverse ? 1 : 0);
                rowButterfly.Execute2D(width / 2, height, Math.Min(16, width / 2), Math.Min(16, height));
            }

            // Column-wise bit reversal
            var bitRevColsKernel = _kernelCache["bit_reverse_cols"];
            bitRevColsKernel.SetArg(0, outReal.Handle);
            bitRevColsKernel.SetArg(1, outImag.Handle);
            bitRevColsKernel.SetArg(2, height);
            bitRevColsKernel.SetArg(3, width);
            bitRevColsKernel.SetArg(4, log2Height);
            bitRevColsKernel.Execute2D(width, height, Math.Min(16, width), Math.Min(16, height));

            // Column-wise FFT
            var colButterfly = _kernelCache["fft_cols_butterfly"];
            for (int stride = 2; stride <= height; stride *= 2)
            {
                colButterfly.SetArg(0, outReal.Handle);
                colButterfly.SetArg(1, outImag.Handle);
                colButterfly.SetArg(2, height);
                colButterfly.SetArg(3, width);
                colButterfly.SetArg(4, stride);
                colButterfly.SetArg(5, inverse ? 1 : 0);
                colButterfly.Execute2D(height / 2, width, Math.Min(16, height / 2), Math.Min(16, width));
            }

            // Scale for inverse FFT
            if (inverse)
            {
                ScaleBuffer(outputReal, 1.0f / (height * width), height * width);
                ScaleBuffer(outputImag, 1.0f / (height * width), height * width);
            }
        }

        /// <inheritdoc/>
        public void ApplyWindow(IGpuBuffer input, IGpuBuffer window, IGpuBuffer output, int n)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var inBuf = ((DirectOpenClGpuBuffer)input).Buffer;
            var winBuf = ((DirectOpenClGpuBuffer)window).Buffer;
            var outBuf = ((DirectOpenClGpuBuffer)output).Buffer;

            var kernel = _kernelCache["apply_window"];
            kernel.SetArg(0, inBuf.Handle);
            kernel.SetArg(1, winBuf.Handle);
            kernel.SetArg(2, outBuf.Handle);
            kernel.SetArg(3, n);
            kernel.Execute1D(n, Math.Min(256, n));
        }

        /// <inheritdoc/>
        public void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var realBuf = ((DirectOpenClGpuBuffer)real).Buffer;
            var imagBuf = ((DirectOpenClGpuBuffer)imag).Buffer;
            var magBuf = ((DirectOpenClGpuBuffer)magnitude).Buffer;

            var kernel = _kernelCache["complex_magnitude"];
            kernel.SetArg(0, realBuf.Handle);
            kernel.SetArg(1, imagBuf.Handle);
            kernel.SetArg(2, magBuf.Handle);
            kernel.SetArg(3, n);
            kernel.Execute1D(n, Math.Min(256, n));
        }

        /// <inheritdoc/>
        public void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var realBuf = ((DirectOpenClGpuBuffer)real).Buffer;
            var imagBuf = ((DirectOpenClGpuBuffer)imag).Buffer;
            var phaseBuf = ((DirectOpenClGpuBuffer)phase).Buffer;

            var kernel = _kernelCache["complex_phase"];
            kernel.SetArg(0, realBuf.Handle);
            kernel.SetArg(1, imagBuf.Handle);
            kernel.SetArg(2, phaseBuf.Handle);
            kernel.SetArg(3, n);
            kernel.Execute1D(n, Math.Min(256, n));
        }

        /// <inheritdoc/>
        public void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var magBuf = ((DirectOpenClGpuBuffer)magnitude).Buffer;
            var phaseBuf = ((DirectOpenClGpuBuffer)phase).Buffer;
            var realBuf = ((DirectOpenClGpuBuffer)real).Buffer;
            var imagBuf = ((DirectOpenClGpuBuffer)imag).Buffer;

            var kernel = _kernelCache["polar_to_complex"];
            kernel.SetArg(0, magBuf.Handle);
            kernel.SetArg(1, phaseBuf.Handle);
            kernel.SetArg(2, realBuf.Handle);
            kernel.SetArg(3, imagBuf.Handle);
            kernel.SetArg(4, n);
            kernel.Execute1D(n, Math.Min(256, n));
        }

        /// <inheritdoc/>
        public void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec, int numFrames, int numFreqs, int nMels)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var powerBuf = ((DirectOpenClGpuBuffer)powerSpec).Buffer;
            var fbBuf = ((DirectOpenClGpuBuffer)filterbank).Buffer;
            var melBuf = ((DirectOpenClGpuBuffer)melSpec).Buffer;

            var kernel = _kernelCache["apply_mel_filterbank"];
            kernel.SetArg(0, powerBuf.Handle);
            kernel.SetArg(1, fbBuf.Handle);
            kernel.SetArg(2, melBuf.Handle);
            kernel.SetArg(3, numFrames);
            kernel.SetArg(4, numFreqs);
            kernel.SetArg(5, nMels);
            kernel.Execute2D(nMels, numFrames, Math.Min(32, nMels), 1);
        }

        /// <inheritdoc/>
        public void PowerToDb(IGpuBuffer power, IGpuBuffer db, int n, float refValue, float minDb)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var powerBuf = ((DirectOpenClGpuBuffer)power).Buffer;
            var dbBuf = ((DirectOpenClGpuBuffer)db).Buffer;

            var kernel = _kernelCache["power_to_db"];
            kernel.SetArg(0, powerBuf.Handle);
            kernel.SetArg(1, dbBuf.Handle);
            kernel.SetArg(2, n);
            kernel.SetArg(3, refValue);
            kernel.SetArg(4, minDb);
            kernel.Execute1D(n, Math.Min(256, n));
        }

        /// <inheritdoc/>
        public void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var dbBuf = ((DirectOpenClGpuBuffer)db).Buffer;
            var powerBuf = ((DirectOpenClGpuBuffer)power).Buffer;

            var kernel = _kernelCache["db_to_power"];
            kernel.SetArg(0, dbBuf.Handle);
            kernel.SetArg(1, powerBuf.Handle);
            kernel.SetArg(2, n);
            kernel.SetArg(3, refValue);
            kernel.Execute1D(n, Math.Min(256, n));
        }

        public void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");
            if (!_supportsFp16)
                throw new NotSupportedException("FP16 conversion is not supported on this device (device does not support FP16).");
            if (!_mixedPrecisionKernelsAvailable)
                throw new NotSupportedException("FP16 conversion is not available (mixed precision kernel compilation failed).");
            if (!_kernelCache.TryGetValue("convert_fp32_to_fp16", out var k))
                throw new InvalidOperationException("OpenCL kernel not found: convert_fp32_to_fp16. FP16 conversion requires a proper GPU kernel.");

            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, size);
            k.Execute1D(size, Math.Min(256, size));
        }

        public void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");
            if (!_supportsFp16)
                throw new NotSupportedException("FP32 conversion from FP16 is not supported on this device (device does not support FP16).");
            if (!_mixedPrecisionKernelsAvailable)
                throw new NotSupportedException("FP32 conversion from FP16 is not available (mixed precision kernel compilation failed).");
            if (!_kernelCache.TryGetValue("convert_fp16_to_fp32", out var k))
                throw new InvalidOperationException("OpenCL kernel not found: convert_fp16_to_fp32. FP32 conversion requires a proper GPU kernel.");

            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            k.SetArg(arg++, size);
            k.Execute1D(size, Math.Min(256, size));
        }

        private void CopyBuffer(IGpuBuffer src, IGpuBuffer dst, int size)
        {
            if (_context == null) return;
            var srcBuf = ((DirectOpenClGpuBuffer)src).Buffer;
            var dstBuf = ((DirectOpenClGpuBuffer)dst).Buffer;

            // Use EnqueueCopyBuffer for device-to-device copy
            int err = OpenClNativeBindings.EnqueueCopyBuffer(
                _context.CommandQueue,
                srcBuf.Handle,
                dstBuf.Handle,
                UIntPtr.Zero,
                UIntPtr.Zero,
                (UIntPtr)(size * sizeof(float)),
                0,
                IntPtr.Zero,
                IntPtr.Zero);

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to copy OpenCL buffer: {err}");
        }

        private void ZeroBuffer(IGpuBuffer buffer, int size)
        {
            var data = new float[size];
            var buf = ((DirectOpenClGpuBuffer)buffer).Buffer;
            buf.CopyFromHost(data);
        }

        private void ScaleBuffer(IGpuBuffer buffer, float scale, int size)
        {
            if (_context == null) return;
            var buf = ((DirectOpenClGpuBuffer)buffer).Buffer;
            var kernel = _kernelCache["scale_vector"];
            kernel.SetArg(0, buf.Handle);
            kernel.SetArg(1, buf.Handle);
            kernel.SetArg(2, scale);
            kernel.SetArg(3, size);
            kernel.Execute1D(size, Math.Min(256, size));
        }

        #endregion

        #region Random Number Generation

        public void GenerateRandomUniform(IGpuBuffer output, int size, float min, float max, ulong seed)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var bufferOut = ((DirectOpenClGpuBuffer)output).Buffer;
            var kernel = _kernelCache["GenerateRandomUniform"];

            kernel.SetArg(0, bufferOut.Handle);
            kernel.SetArg(1, size);
            kernel.SetArg(2, min);
            kernel.SetArg(3, max);
            kernel.SetArg(4, seed);

            kernel.Execute1D(size, Math.Min(256, size));
        }

        public void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var bufferOut = ((DirectOpenClGpuBuffer)output).Buffer;
            var kernel = _kernelCache["GenerateRandomNormal"];

            kernel.SetArg(0, bufferOut.Handle);
            kernel.SetArg(1, size);
            kernel.SetArg(2, mean);
            kernel.SetArg(3, stdDev);
            kernel.SetArg(4, seed);

            // Each thread generates 2 numbers
            int numThreads = (size + 1) / 2;
            kernel.Execute1D(numThreads, Math.Min(256, numThreads));
        }

        #endregion

        #region Specialized Layer Operations

        public void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output,
            int batchSize, int numCenters, int inputDim)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var inBuf = ((DirectOpenClGpuBuffer)input).Buffer;
            var cBuf = ((DirectOpenClGpuBuffer)centers).Buffer;
            var eBuf = ((DirectOpenClGpuBuffer)epsilons).Buffer;
            var outBuf = ((DirectOpenClGpuBuffer)output).Buffer;

            var kernel = _kernelCache["rbf_forward"];
            kernel.SetArg(0, inBuf.Handle);
            kernel.SetArg(1, cBuf.Handle);
            kernel.SetArg(2, eBuf.Handle);
            kernel.SetArg(3, outBuf.Handle);
            kernel.SetArg(4, batchSize);
            kernel.SetArg(5, numCenters);
            kernel.SetArg(6, inputDim);

            kernel.Execute2D(batchSize, numCenters, Math.Min(16, batchSize), Math.Min(16, numCenters));
        }

        public void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input,
            float decay, float threshold, int size)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var trBuf = ((DirectOpenClGpuBuffer)traces).Buffer;
            var spBuf = ((DirectOpenClGpuBuffer)spikes).Buffer;
            var inBuf = ((DirectOpenClGpuBuffer)input).Buffer;

            var kernel = _kernelCache["update_traces"];
            kernel.SetArg(0, trBuf.Handle);
            kernel.SetArg(1, spBuf.Handle);
            kernel.SetArg(2, inBuf.Handle);
            kernel.SetArg(3, decay);
            kernel.SetArg(4, threshold);
            kernel.SetArg(5, size);

            int localSize = Math.Min(256, size);
            kernel.Execute1D(size, localSize);
        }

        public void StdpUpdate(IGpuBuffer weights, IGpuBuffer preTrace, IGpuBuffer postTrace,
            IGpuBuffer preSpike, IGpuBuffer postSpike,
            float ltpRate, float ltdRate, float homeostasisRate,
            float minWeight, float maxWeight,
            int numPre, int numPost)
        {
            if (_context == null) throw new InvalidOperationException("OpenCL context not available");

            var wBuf = ((DirectOpenClGpuBuffer)weights).Buffer;
            var preT = ((DirectOpenClGpuBuffer)preTrace).Buffer;
            var postT = ((DirectOpenClGpuBuffer)postTrace).Buffer;
            var preS = ((DirectOpenClGpuBuffer)preSpike).Buffer;
            var postS = ((DirectOpenClGpuBuffer)postSpike).Buffer;

            var kernel = _kernelCache["stdp_update"];
            kernel.SetArg(0, wBuf.Handle);
            kernel.SetArg(1, preT.Handle);
            kernel.SetArg(2, postT.Handle);
            kernel.SetArg(3, preS.Handle);
            kernel.SetArg(4, postS.Handle);
            kernel.SetArg(5, ltpRate);
            kernel.SetArg(6, ltdRate);
            kernel.SetArg(7, homeostasisRate);
            kernel.SetArg(8, minWeight);
            kernel.SetArg(9, maxWeight);
            kernel.SetArg(10, numPre);
            kernel.SetArg(11, numPost);

            kernel.Execute2D(numPre, numPost, Math.Min(16, numPre), Math.Min(16, numPost));
        }

        #endregion

        #region Hyperbolic Geometry Operations

        public void PoincareProject(IGpuBuffer input, IGpuBuffer output, int batchSize, int dim, float curvature, float epsilon = 1e-5f)
        {
            if (!_kernelCache.TryGetValue("poincare_project", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: poincare_project");

            int localSize = CalculateOptimalWorkGroupSize1D(batchSize);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetArg(2u, batchSize);
            kernel.SetArg(3u, dim);
            kernel.SetArg(4u, curvature);
            kernel.SetArg(5u, epsilon);
            kernel.Execute1D(batchSize, localSize);
        }

        public void MobiusAdd(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
        {
            if (!_kernelCache.TryGetValue("mobius_add", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: mobius_add");

            int localSize = CalculateOptimalWorkGroupSize1D(batchSize);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)x).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)y).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetArg(3u, batchSize);
            kernel.SetArg(4u, dim);
            kernel.SetArg(5u, curvature);
            kernel.Execute1D(batchSize, localSize);
        }

        public void PoincareExpMap(IGpuBuffer basePoint, IGpuBuffer tangentVec, IGpuBuffer output, int batchSize, int dim, float curvature)
        {
            if (!_kernelCache.TryGetValue("poincare_exp_map", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: poincare_exp_map");

            int localSize = CalculateOptimalWorkGroupSize1D(batchSize);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)basePoint).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)tangentVec).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetArg(3u, batchSize);
            kernel.SetArg(4u, dim);
            kernel.SetArg(5u, curvature);
            kernel.Execute1D(batchSize, localSize);
        }

        public void PoincareDistance(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
        {
            if (!_kernelCache.TryGetValue("poincare_distance", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: poincare_distance");

            int localSize = CalculateOptimalWorkGroupSize1D(batchSize);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)x).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)y).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetArg(3u, batchSize);
            kernel.SetArg(4u, dim);
            kernel.SetArg(5u, curvature);
            kernel.Execute1D(batchSize, localSize);
        }

        public void HyperbolicLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
            int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
        {
            if (!_kernelCache.TryGetValue("hyperbolic_linear_forward", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: hyperbolic_linear_forward");

            int totalThreads = batchSize * outputFeatures;
            int localSize = CalculateOptimalWorkGroupSize1D(totalThreads);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)weights).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)biases).Buffer.Handle);
            kernel.SetArg(3u, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetArg(4u, batchSize);
            kernel.SetArg(5u, inputFeatures);
            kernel.SetArg(6u, outputFeatures);
            kernel.SetArg(7u, curvature);
            kernel.SetArg(8u, epsilon);
            kernel.Execute1D(totalThreads, localSize);
        }

        /// <inheritdoc/>
        public void HyperbolicLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
            int batchSize, int inputFeatures, int outputFeatures, float curvature)
        {
            if (!_kernelCache.TryGetValue("hyperbolic_linear_backward_input", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: hyperbolic_linear_backward_input");

            int totalThreads = batchSize * inputFeatures;
            int localSize = CalculateOptimalWorkGroupSize1D(totalThreads);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)weights).Buffer.Handle);
            kernel.SetArg(3u, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            kernel.SetArg(4u, batchSize);
            kernel.SetArg(5u, inputFeatures);
            kernel.SetArg(6u, outputFeatures);
            kernel.SetArg(7u, curvature);
            kernel.Execute1D(totalThreads, localSize);
        }

        /// <inheritdoc/>
        public void HyperbolicLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
            int batchSize, int inputFeatures, int outputFeatures, float curvature)
        {
            if (!_kernelCache.TryGetValue("hyperbolic_linear_backward_weights", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: hyperbolic_linear_backward_weights");

            int totalThreads = outputFeatures * inputFeatures;
            int localSize = CalculateOptimalWorkGroupSize1D(totalThreads);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)gradWeights).Buffer.Handle);
            kernel.SetArg(3u, batchSize);
            kernel.SetArg(4u, inputFeatures);
            kernel.SetArg(5u, outputFeatures);
            kernel.SetArg(6u, curvature);
            kernel.Execute1D(totalThreads, localSize);
        }

        /// <inheritdoc/>
        public void HyperbolicLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradBiases,
            int batchSize, int inputFeatures, int outputFeatures, float curvature)
        {
            if (!_kernelCache.TryGetValue("hyperbolic_linear_backward_biases", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: hyperbolic_linear_backward_biases");

            // Bias gradients: one per output feature (not outputFeatures * inputFeatures)
            int totalThreads = outputFeatures;
            int localSize = CalculateOptimalWorkGroupSize1D(totalThreads);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)gradBiases).Buffer.Handle);
            kernel.SetArg(3u, batchSize);
            kernel.SetArg(4u, inputFeatures);
            kernel.SetArg(5u, outputFeatures);
            kernel.SetArg(6u, curvature);
            kernel.Execute1D(totalThreads, localSize);
        }

        #endregion

        #region Octonion Algebra Operations

        public void OctonionMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
        {
            if (!_kernelCache.TryGetValue("octonion_multiply", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: octonion_multiply");

            int localSize = CalculateOptimalWorkGroupSize1D(count);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)a).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)b).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetArg(3u, count);
            kernel.Execute1D(count, localSize);
        }

        public void OctonionAdd(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
        {
            if (!_kernelCache.TryGetValue("octonion_add", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: octonion_add");

            int totalElements = count * 8;
            int localSize = CalculateOptimalWorkGroupSize1D(totalElements);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)a).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)b).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetArg(3u, count);
            kernel.Execute1D(totalElements, localSize);
        }

        public void OctonionLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
            int batchSize, int inputFeatures, int outputFeatures)
        {
            if (!_kernelCache.TryGetValue("octonion_linear_forward", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: octonion_linear_forward");

            int totalThreads = batchSize * outputFeatures;
            int localSize = CalculateOptimalWorkGroupSize1D(totalThreads);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)weights).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)biases).Buffer.Handle);
            kernel.SetArg(3u, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetArg(4u, batchSize);
            kernel.SetArg(5u, inputFeatures);
            kernel.SetArg(6u, outputFeatures);
            kernel.Execute1D(totalThreads, localSize);
        }

        public void OctonionLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
            int batchSize, int inputFeatures, int outputFeatures)
        {
            if (!_kernelCache.TryGetValue("octonion_linear_backward_input", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: octonion_linear_backward_input");

            int totalThreads = batchSize * inputFeatures;
            int localSize = CalculateOptimalWorkGroupSize1D(totalThreads);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)weights).Buffer.Handle);
            kernel.SetArg(3u, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            kernel.SetArg(4u, batchSize);
            kernel.SetArg(5u, inputFeatures);
            kernel.SetArg(6u, outputFeatures);
            kernel.Execute1D(totalThreads, localSize);
        }

        public void OctonionLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
            int batchSize, int inputFeatures, int outputFeatures)
        {
            if (!_kernelCache.TryGetValue("octonion_linear_backward_weights", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: octonion_linear_backward_weights");

            int totalThreads = outputFeatures * inputFeatures;
            int localSize = CalculateOptimalWorkGroupSize1D(totalThreads);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)gradWeights).Buffer.Handle);
            kernel.SetArg(3u, batchSize);
            kernel.SetArg(4u, inputFeatures);
            kernel.SetArg(5u, outputFeatures);
            kernel.Execute1D(totalThreads, localSize);
        }

        public void OctonionLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer gradBiases,
            int batchSize, int outputFeatures)
        {
            if (!_kernelCache.TryGetValue("octonion_linear_backward_biases", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: octonion_linear_backward_biases");

            int localSize = CalculateOptimalWorkGroupSize1D(outputFeatures);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)gradBiases).Buffer.Handle);
            kernel.SetArg(2u, batchSize);
            kernel.SetArg(3u, outputFeatures);
            kernel.Execute1D(outputFeatures, localSize);
        }

        #endregion

        #region Quantum Computing Operations

        public void QuantumMeasurement(IGpuBuffer realPart, IGpuBuffer imagPart, IGpuBuffer probabilities, int batchSize, int stateSize)
        {
            if (!_kernelCache.TryGetValue("quantum_measurement", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: quantum_measurement");

            int totalElements = batchSize * stateSize;
            int localSize = CalculateOptimalWorkGroupSize1D(totalElements);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)realPart).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)imagPart).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)probabilities).Buffer.Handle);
            kernel.SetArg(3u, batchSize);
            kernel.SetArg(4u, stateSize);
            kernel.Execute1D(totalElements, localSize);
        }

        public void NormalizeProbabilities(IGpuBuffer probabilities, int batchSize, int stateSize)
        {
            if (!_kernelCache.TryGetValue("normalize_probabilities", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: normalize_probabilities");

            int localSize = CalculateOptimalWorkGroupSize1D(batchSize);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)probabilities).Buffer.Handle);
            kernel.SetLocalArg(1u, localSize * sizeof(float));
            kernel.SetArg(2u, batchSize);
            kernel.SetArg(3u, stateSize);
            // Execute with batchSize work groups, each of localSize items
            kernel.Execute1D(batchSize * localSize, localSize);
        }

        public void ComplexMatVec(IGpuBuffer matReal, IGpuBuffer matImag, IGpuBuffer vecReal, IGpuBuffer vecImag,
            IGpuBuffer outReal, IGpuBuffer outImag, int batchSize, int dim)
        {
            if (!_kernelCache.TryGetValue("complex_matvec", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: complex_matvec");

            int totalElements = batchSize * dim;
            int localSize = CalculateOptimalWorkGroupSize1D(totalElements);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)matReal).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)matImag).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)vecReal).Buffer.Handle);
            kernel.SetArg(3u, ((DirectOpenClGpuBuffer)vecImag).Buffer.Handle);
            kernel.SetArg(4u, ((DirectOpenClGpuBuffer)outReal).Buffer.Handle);
            kernel.SetArg(5u, ((DirectOpenClGpuBuffer)outImag).Buffer.Handle);
            kernel.SetArg(6u, batchSize);
            kernel.SetArg(7u, dim);
            kernel.Execute1D(totalElements, localSize);
        }

        public void QuantumRotation(IGpuBuffer stateReal, IGpuBuffer stateImag, IGpuBuffer outReal, IGpuBuffer outImag,
            IGpuBuffer angles, int numQubits, int batchSize)
        {
            if (!_kernelCache.TryGetValue("quantum_rotation", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: quantum_rotation");

            int localSize = CalculateOptimalWorkGroupSize1D(batchSize);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)stateReal).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)stateImag).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)outReal).Buffer.Handle);
            kernel.SetArg(3u, ((DirectOpenClGpuBuffer)outImag).Buffer.Handle);
            kernel.SetArg(4u, ((DirectOpenClGpuBuffer)angles).Buffer.Handle);
            kernel.SetArg(5u, numQubits);
            kernel.SetArg(6u, batchSize);
            // Execute with batchSize work groups, each of localSize items
            kernel.Execute1D(batchSize * localSize, localSize);
        }

        public void MeasurementForward(IGpuBuffer input, IGpuBuffer output, int batchSize, int stateSize)
        {
            if (!_kernelCache.TryGetValue("measurement_forward", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: measurement_forward");

            int localSize = CalculateOptimalWorkGroupSize1D(batchSize);
            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetLocalArg(2u, localSize * sizeof(float));
            kernel.SetArg(3u, batchSize);
            kernel.SetArg(4u, stateSize);
            // Execute with batchSize work groups, each of localSize items
            kernel.Execute1D(batchSize * localSize, localSize);
        }

        #endregion

        #region RNN (LSTM/GRU) Sequence Operations

        public void LstmForwardSequence(
            IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
            IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
            IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer cFinal,
            IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
            int seqLen, int batch, int inputSize, int hiddenSize)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context is not initialized. Cannot execute LstmForwardSequence.");

            if (seqLen <= 0)
                throw new ArgumentException($"seqLen must be positive, got {seqLen}", nameof(seqLen));
            if (batch <= 0)
                throw new ArgumentException($"batch must be positive, got {batch}", nameof(batch));
            if (inputSize <= 0)
                throw new ArgumentException($"inputSize must be positive, got {inputSize}", nameof(inputSize));
            if (hiddenSize <= 0)
                throw new ArgumentException($"hiddenSize must be positive, got {hiddenSize}", nameof(hiddenSize));

            if (!_kernelCache.TryGetValue("lstm_forward_sequence", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: lstm_forward_sequence");

            int totalThreads = batch * hiddenSize;
            int localSize = CalculateOptimalWorkGroupSize1D(totalThreads);

            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)hInit).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)cInit).Buffer.Handle);
            kernel.SetArg(3u, ((DirectOpenClGpuBuffer)weightsIh).Buffer.Handle);
            kernel.SetArg(4u, ((DirectOpenClGpuBuffer)weightsHh).Buffer.Handle);
            kernel.SetArg(5u, ((DirectOpenClGpuBuffer)biasIh).Buffer.Handle);
            kernel.SetArg(6u, ((DirectOpenClGpuBuffer)biasHh).Buffer.Handle);
            kernel.SetArg(7u, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetArg(8u, ((DirectOpenClGpuBuffer)hFinal).Buffer.Handle);
            kernel.SetArg(9u, ((DirectOpenClGpuBuffer)cFinal).Buffer.Handle);
            kernel.SetArg(10u, ((DirectOpenClGpuBuffer)allH).Buffer.Handle);
            kernel.SetArg(11u, ((DirectOpenClGpuBuffer)allC).Buffer.Handle);
            kernel.SetArg(12u, ((DirectOpenClGpuBuffer)cacheGates).Buffer.Handle);
            kernel.SetArg(13u, seqLen);
            kernel.SetArg(14u, batch);
            kernel.SetArg(15u, inputSize);
            kernel.SetArg(16u, hiddenSize);

            int globalSize = ((totalThreads + localSize - 1) / localSize) * localSize;
            kernel.Execute1D(globalSize, localSize);
        }

        public void LstmBackwardSequence(
            IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
            IGpuBuffer hInit, IGpuBuffer cInit,
            IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
            IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit,
            IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
            int seqLen, int batch, int inputSize, int hiddenSize)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context is not initialized. Cannot execute LstmBackwardSequence.");

            if (seqLen <= 0)
                throw new ArgumentException($"seqLen must be positive, got {seqLen}", nameof(seqLen));
            if (batch <= 0)
                throw new ArgumentException($"batch must be positive, got {batch}", nameof(batch));
            if (inputSize <= 0)
                throw new ArgumentException($"inputSize must be positive, got {inputSize}", nameof(inputSize));
            if (hiddenSize <= 0)
                throw new ArgumentException($"hiddenSize must be positive, got {hiddenSize}", nameof(hiddenSize));

            if (!_kernelCache.TryGetValue("lstm_backward_sequence", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: lstm_backward_sequence");

            int totalThreads = batch * hiddenSize;
            int localSize = CalculateOptimalWorkGroupSize1D(totalThreads);

            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)allH).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)allC).Buffer.Handle);
            kernel.SetArg(3u, ((DirectOpenClGpuBuffer)cacheGates).Buffer.Handle);
            kernel.SetArg(4u, ((DirectOpenClGpuBuffer)hInit).Buffer.Handle);
            kernel.SetArg(5u, ((DirectOpenClGpuBuffer)cInit).Buffer.Handle);
            kernel.SetArg(6u, ((DirectOpenClGpuBuffer)weightsIh).Buffer.Handle);
            kernel.SetArg(7u, ((DirectOpenClGpuBuffer)weightsHh).Buffer.Handle);
            kernel.SetArg(8u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(9u, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            kernel.SetArg(10u, ((DirectOpenClGpuBuffer)gradHInit).Buffer.Handle);
            kernel.SetArg(11u, ((DirectOpenClGpuBuffer)gradCInit).Buffer.Handle);
            kernel.SetArg(12u, ((DirectOpenClGpuBuffer)gradWeightsIh).Buffer.Handle);
            kernel.SetArg(13u, ((DirectOpenClGpuBuffer)gradWeightsHh).Buffer.Handle);
            kernel.SetArg(14u, ((DirectOpenClGpuBuffer)gradBiasIh).Buffer.Handle);
            kernel.SetArg(15u, ((DirectOpenClGpuBuffer)gradBiasHh).Buffer.Handle);
            kernel.SetArg(16u, seqLen);
            kernel.SetArg(17u, batch);
            kernel.SetArg(18u, inputSize);
            kernel.SetArg(19u, hiddenSize);

            int globalSize = ((totalThreads + localSize - 1) / localSize) * localSize;
            kernel.Execute1D(globalSize, localSize);
        }

        public void GruForwardSequence(
            IGpuBuffer input, IGpuBuffer hInit,
            IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
            IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer allH, IGpuBuffer cacheGates,
            int seqLen, int batch, int inputSize, int hiddenSize)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context is not initialized. Cannot execute GruForwardSequence.");

            if (seqLen <= 0)
                throw new ArgumentException($"seqLen must be positive, got {seqLen}", nameof(seqLen));
            if (batch <= 0)
                throw new ArgumentException($"batch must be positive, got {batch}", nameof(batch));
            if (inputSize <= 0)
                throw new ArgumentException($"inputSize must be positive, got {inputSize}", nameof(inputSize));
            if (hiddenSize <= 0)
                throw new ArgumentException($"hiddenSize must be positive, got {hiddenSize}", nameof(hiddenSize));

            if (!_kernelCache.TryGetValue("gru_forward_sequence", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: gru_forward_sequence");

            int totalThreads = batch * hiddenSize;
            int localSize = CalculateOptimalWorkGroupSize1D(totalThreads);

            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)hInit).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)weightsIh).Buffer.Handle);
            kernel.SetArg(3u, ((DirectOpenClGpuBuffer)weightsHh).Buffer.Handle);
            kernel.SetArg(4u, ((DirectOpenClGpuBuffer)biasIh).Buffer.Handle);
            kernel.SetArg(5u, ((DirectOpenClGpuBuffer)biasHh).Buffer.Handle);
            kernel.SetArg(6u, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetArg(7u, ((DirectOpenClGpuBuffer)hFinal).Buffer.Handle);
            kernel.SetArg(8u, ((DirectOpenClGpuBuffer)allH).Buffer.Handle);
            kernel.SetArg(9u, ((DirectOpenClGpuBuffer)cacheGates).Buffer.Handle);
            kernel.SetArg(10u, seqLen);
            kernel.SetArg(11u, batch);
            kernel.SetArg(12u, inputSize);
            kernel.SetArg(13u, hiddenSize);

            int globalSize = ((totalThreads + localSize - 1) / localSize) * localSize;
            kernel.Execute1D(globalSize, localSize);
        }

        public void GruBackwardSequence(
            IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer cacheGates,
            IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
            IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer dHBuffer,
            IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
            int seqLen, int batch, int inputSize, int hiddenSize)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context is not initialized. Cannot execute GruBackwardSequence.");

            if (seqLen <= 0)
                throw new ArgumentException($"seqLen must be positive, got {seqLen}", nameof(seqLen));
            if (batch <= 0)
                throw new ArgumentException($"batch must be positive, got {batch}", nameof(batch));
            if (inputSize <= 0)
                throw new ArgumentException($"inputSize must be positive, got {inputSize}", nameof(inputSize));
            if (hiddenSize <= 0)
                throw new ArgumentException($"hiddenSize must be positive, got {hiddenSize}", nameof(hiddenSize));

            if (!_kernelCache.TryGetValue("gru_backward_sequence", out var kernel))
                throw new InvalidOperationException("OpenCL kernel not found: gru_backward_sequence");

            int totalThreads = batch * hiddenSize;
            int localSize = CalculateOptimalWorkGroupSize1D(totalThreads);

            kernel.SetArg(0u, ((DirectOpenClGpuBuffer)gradOutput).Buffer.Handle);
            kernel.SetArg(1u, ((DirectOpenClGpuBuffer)allH).Buffer.Handle);
            kernel.SetArg(2u, ((DirectOpenClGpuBuffer)cacheGates).Buffer.Handle);
            kernel.SetArg(3u, ((DirectOpenClGpuBuffer)weightsIh).Buffer.Handle);
            kernel.SetArg(4u, ((DirectOpenClGpuBuffer)weightsHh).Buffer.Handle);
            kernel.SetArg(5u, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
            kernel.SetArg(6u, ((DirectOpenClGpuBuffer)gradInput).Buffer.Handle);
            kernel.SetArg(7u, ((DirectOpenClGpuBuffer)gradHInit).Buffer.Handle);
            kernel.SetArg(8u, ((DirectOpenClGpuBuffer)dHBuffer).Buffer.Handle);
            kernel.SetArg(9u, ((DirectOpenClGpuBuffer)gradWeightsIh).Buffer.Handle);
            kernel.SetArg(10u, ((DirectOpenClGpuBuffer)gradWeightsHh).Buffer.Handle);
            kernel.SetArg(11u, ((DirectOpenClGpuBuffer)gradBiasIh).Buffer.Handle);
            kernel.SetArg(12u, ((DirectOpenClGpuBuffer)gradBiasHh).Buffer.Handle);
            kernel.SetArg(13u, seqLen);
            kernel.SetArg(14u, batch);
            kernel.SetArg(15u, inputSize);
            kernel.SetArg(16u, hiddenSize);

            int globalSize = ((totalThreads + localSize - 1) / localSize) * localSize;
            kernel.Execute1D(globalSize, localSize);
        }

        public void GruCellBackward(
            IGpuBuffer gradH, IGpuBuffer gateR, IGpuBuffer gateZ, IGpuBuffer gateN, IGpuBuffer prevH,
            IGpuBuffer weightsHh,
            IGpuBuffer gradPrevH, IGpuBuffer gradGateR, IGpuBuffer gradGateZ, IGpuBuffer gradGateN,
            int batch, int hiddenSize)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context is not initialized. Cannot execute GruCellBackward.");

            if (batch <= 0)
                throw new ArgumentException($"batch must be positive, got {batch}", nameof(batch));
            if (hiddenSize <= 0)
                throw new ArgumentException($"hiddenSize must be positive, got {hiddenSize}", nameof(hiddenSize));

            // Step 1: Call gru_cell_backward to compute gate gradients and partial gradPrevH (direct path only)
            if (!_kernelCache.TryGetValue("gru_cell_backward", out var cellBackwardKernel))
                throw new InvalidOperationException("OpenCL kernel not found: gru_cell_backward");

            int totalThreads = batch * hiddenSize;
            int localSize = CalculateOptimalWorkGroupSize1D(totalThreads);
            int globalSize = ((totalThreads + localSize - 1) / localSize) * localSize;

            cellBackwardKernel.SetArg(0u, ((DirectOpenClGpuBuffer)gradH).Buffer.Handle);
            cellBackwardKernel.SetArg(1u, ((DirectOpenClGpuBuffer)gateR).Buffer.Handle);
            cellBackwardKernel.SetArg(2u, ((DirectOpenClGpuBuffer)gateZ).Buffer.Handle);
            cellBackwardKernel.SetArg(3u, ((DirectOpenClGpuBuffer)gateN).Buffer.Handle);
            cellBackwardKernel.SetArg(4u, ((DirectOpenClGpuBuffer)prevH).Buffer.Handle);
            cellBackwardKernel.SetArg(5u, ((DirectOpenClGpuBuffer)weightsHh).Buffer.Handle);
            cellBackwardKernel.SetArg(6u, ((DirectOpenClGpuBuffer)gradPrevH).Buffer.Handle);
            cellBackwardKernel.SetArg(7u, ((DirectOpenClGpuBuffer)gradGateR).Buffer.Handle);
            cellBackwardKernel.SetArg(8u, ((DirectOpenClGpuBuffer)gradGateZ).Buffer.Handle);
            cellBackwardKernel.SetArg(9u, ((DirectOpenClGpuBuffer)gradGateN).Buffer.Handle);
            cellBackwardKernel.SetArg(10u, batch);
            cellBackwardKernel.SetArg(11u, hiddenSize);

            cellBackwardKernel.Execute1D(globalSize, localSize);

            // Step 2: Call gru_backward_prevh to compute full gradPrevH using all gate gradients
            // This overwrites the partial result from step 1 with the complete gradient
            if (!_kernelCache.TryGetValue("gru_backward_prevh", out var prevhKernel))
                throw new InvalidOperationException("OpenCL kernel not found: gru_backward_prevh");

            prevhKernel.SetArg(0u, ((DirectOpenClGpuBuffer)gradGateR).Buffer.Handle);
            prevhKernel.SetArg(1u, ((DirectOpenClGpuBuffer)gradGateZ).Buffer.Handle);
            prevhKernel.SetArg(2u, ((DirectOpenClGpuBuffer)gradGateN).Buffer.Handle);
            prevhKernel.SetArg(3u, ((DirectOpenClGpuBuffer)gradH).Buffer.Handle);
            prevhKernel.SetArg(4u, ((DirectOpenClGpuBuffer)gateR).Buffer.Handle);
            prevhKernel.SetArg(5u, ((DirectOpenClGpuBuffer)gateZ).Buffer.Handle);
            prevhKernel.SetArg(6u, ((DirectOpenClGpuBuffer)weightsHh).Buffer.Handle);
            prevhKernel.SetArg(7u, ((DirectOpenClGpuBuffer)gradPrevH).Buffer.Handle);
            prevhKernel.SetArg(8u, batch);
            prevhKernel.SetArg(9u, hiddenSize);

            prevhKernel.Execute1D(globalSize, localSize);
        }

        #endregion

        public void Dispose()
        {
            if (_disposed) return;

            _dynamicGemm?.Dispose();

            foreach (var kernel in _kernelCache.Values)
            {
                kernel.Dispose();
            }
            _kernelCache.Clear();

            foreach (var program in _programs)
            {
                program.Dispose();
            }
            _programs.Clear();

            _context?.Dispose();
            _disposed = true;
        }
    }

    /// <summary>
    /// OpenCL GPU buffer wrapper implementing IGpuBuffer.
    /// Uses pure P/Invoke with no managed GPU runtime dependency.
    /// </summary>
    internal sealed class DirectOpenClGpuBuffer : IGpuBuffer
    {
        internal readonly DirectOpenClBuffer Buffer;

        public int Size => Buffer.Length;
        public long SizeInBytes => Buffer.Length * sizeof(float);
        public IntPtr Handle => Buffer.Handle;

        public DirectOpenClGpuBuffer(DirectOpenClBuffer buffer)
        {
            Buffer = buffer;
        }

        public float[] Download()
        {
            return Buffer.ToArray();
        }

        public void Download(float[] destination)
        {
            Buffer.CopyToHost(destination);
        }

        public void Dispose()
        {
            Buffer.Dispose();
        }
    }

    /// <summary>
    /// Contains GPU device information for diagnostics.
    /// </summary>
    public sealed class GpuDeviceInfo
    {
        /// <summary>
        /// Device name (e.g., "AMD Radeon RX 7900 XTX").
        /// </summary>
        public string DeviceName { get; set; } = string.Empty;

        /// <summary>
        /// Device vendor (e.g., "AMD", "NVIDIA", "Intel").
        /// </summary>
        public string DeviceVendor { get; set; } = string.Empty;

        /// <summary>
        /// Number of compute units (CUs for AMD, SMs for NVIDIA).
        /// </summary>
        public int ComputeUnits { get; set; }

        /// <summary>
        /// Global memory size in bytes.
        /// </summary>
        public long GlobalMemoryBytes { get; set; }

        /// <summary>
        /// Local (shared) memory size in bytes per work group.
        /// </summary>
        public long LocalMemoryBytes { get; set; }

        /// <summary>
        /// Maximum work items per work group.
        /// </summary>
        public int MaxWorkGroupSize { get; set; }

        /// <summary>
        /// Maximum work items per dimension.
        /// </summary>
        public ulong[] MaxWorkItemSizes { get; set; } = Array.Empty<ulong>();

        /// <summary>
        /// GPU clock frequency in MHz.
        /// </summary>
        public uint ClockFrequencyMHz { get; set; }

        /// <summary>
        /// Whether FP16 (half precision) is supported.
        /// </summary>
        public bool SupportsFp16 { get; set; }

        /// <summary>
        /// Whether subgroups/wavefronts are supported.
        /// </summary>
        public bool SupportsSubgroups { get; set; }

        /// <summary>
        /// Estimated theoretical peak GFLOPS (FP32).
        /// </summary>
        public double TheoreticalPeakGflops { get; set; }
    }

    /// <summary>
    /// Contains comprehensive GEMM profiling diagnostics.
    /// </summary>
    public sealed class GemmDiagnostics
    {
        // Matrix dimensions
        public int M { get; set; }
        public int N { get; set; }
        public int K { get; set; }

        // Kernel configuration
        public string KernelName { get; set; } = string.Empty;
        public int GlobalSizeX { get; set; }
        public int GlobalSizeY { get; set; }
        public int LocalSizeX { get; set; }
        public int LocalSizeY { get; set; }
        public int WorkItemsLaunched { get; set; }
        public int WorkGroupsLaunched { get; set; }

        // GPU timing (nanoseconds) - from OpenCL profiling events
        public bool IsProfilingAvailable { get; set; }
        public ulong QueueToSubmitNs { get; set; }
        public ulong SubmitToStartNs { get; set; }
        public ulong KernelExecutionNs { get; set; }
        public ulong TotalGpuTimeNs { get; set; }
        public string ProfilingError { get; set; } = string.Empty;

        // Wall clock timing (milliseconds) - fallback
        public double WallClockMs { get; set; }

        // Performance metrics
        public long FlopsRequired { get; set; }
        public long BytesTransferred { get; set; }
        public double ArithmeticIntensity { get; set; }
        public double AchievedGflops { get; set; }
        public double AchievedBandwidthGBps { get; set; }
        public double ComputeEfficiency { get; set; }

        // Bottleneck analysis
        public bool IsLikelyMemoryBound { get; set; }
    }

    /// <summary>
    /// OpenCL byte buffer wrapper implementing IGpuBuffer.
    /// Used for sparse matrix indices (1 byte per group of 4 elements).
    /// </summary>
    internal sealed class DirectOpenClGpuByteBuffer : IGpuBuffer
    {
        internal readonly DirectOpenClByteBuffer Buffer;

        public int Size => Buffer.Length;
        public long SizeInBytes => Buffer.Length;
        public IntPtr Handle => Buffer.Handle;

        public DirectOpenClGpuByteBuffer(DirectOpenClByteBuffer buffer)
        {
            Buffer = buffer;
        }

        public byte[] Download()
        {
            return Buffer.ToArray();
        }

        public void Download(byte[] destination)
        {
            Buffer.CopyToHost(destination);
        }

        public void Dispose()
        {
            Buffer.Dispose();
        }
    }
}

