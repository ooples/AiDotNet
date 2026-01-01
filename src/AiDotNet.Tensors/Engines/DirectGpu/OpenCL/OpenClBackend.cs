// Copyright (c) AiDotNet. All rights reserved.
// OpenCL backend using pure P/Invoke - NO ILGPU dependency.
// Works on ALL .NET versions including .NET Framework 4.6.2.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// OpenCL backend for direct GPU access on AMD, Intel, and NVIDIA GPUs.
    /// Uses pure P/Invoke with no ILGPU dependency.
    /// </summary>
    /// <remarks>
    /// <para><b>Key Features:</b></para>
    /// <list type="bullet">
    /// <item>Works on ALL .NET versions (4.6.2, 4.7.1, net8.0, etc.)</item>
    /// <item>No ILGPU dependency - pure P/Invoke</item>
    /// <item>Double-buffered GEMM for compute/memory overlap</item>
    /// <item>Fused operations (GEMM+Bias+Activation)</item>
    /// <item>Bank-conflict-free shared memory</item>
    /// </list>
    /// </remarks>
    public sealed class OpenClBackend : IDirectGpuBackend
    {
        private DirectOpenClContext? _context;
        private readonly Dictionary<string, DirectOpenClKernel> _kernelCache;   
        private readonly List<DirectOpenClProgram> _programs;
        private DynamicGemmKernel? _dynamicGemm;
        private bool _disposed;
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

        public bool IsAvailable { get; }
        public string BackendName => "OpenCL";
        public string DeviceName { get; }
        public string DeviceVendor { get; }
        public int ComputeUnits { get; }
        public long GlobalMemoryBytes { get; }
        public long LocalMemoryBytes { get; }

        // Dynamic GPU capabilities - initialized from device queries
        private readonly ulong _maxWorkGroupSize;
        private readonly ulong[] _maxWorkItemSizes;
        private readonly bool _supportsFp16;
        private readonly bool _supportsSubgroups;

        /// <summary>
        /// Gets whether OpenCL is available on this system.
        /// </summary>
        public static bool IsOpenClAvailable => DirectOpenClContext.IsAvailable;

        public OpenClBackend(ILogger? logger = null)
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
                Console.WriteLine("[OpenClBackend] Creating DirectOpenClContext...");
                _context = new DirectOpenClContext();
                Console.WriteLine($"[OpenClBackend] Context created: Device={_context.DeviceName}, Vendor={_context.DeviceVendor}");

                IsAvailable = true;
                DeviceName = _context.DeviceName;
                DeviceVendor = _context.DeviceVendor;
                ComputeUnits = (int)_context.MaxComputeUnits;
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
                var kernel = _dynamicGemm.GetKernel(config);
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

            if (_dynamicGemm != null && M >= 128 && N >= 128 && K >= 64 &&
                TryGetTunedConfig(M, N, K, out var tunedConfig))
            {
                if (TryExecutePackedDynamicGemm(A, B, C, M, N, K, alpha, beta, tunedConfig))
                    return;
            }

            // Choose kernel based on matrix size
            // Use optimized kernel for matrices >= 128 in any dimension
            if (M >= 128 && N >= 128 && K >= 64)
            {
                // Large matrix - use CLBlast-style register-blocked kernel
                // Kernel uses 16x16 work group (256 threads), each computes 4x4 outputs = 64x64 tile
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

        public void Synchronize()
        {
            _context?.Finish();
        }

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
    /// Uses pure P/Invoke with no ILGPU dependency.
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
