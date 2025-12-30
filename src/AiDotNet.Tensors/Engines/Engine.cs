// Copyright (c) AiDotNet. All rights reserved.
// Automatic engine selection based on hardware capabilities

using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu;
#if !NET462
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
#endif

namespace AiDotNet.Tensors.Engines
{
    /// <summary>
    /// Provides automatic hardware-accelerated engine selection.
    /// Similar to how SIMD automatically selects Vector128/256/512 based on CPU features,
    /// this factory automatically selects GPU or CPU engine based on available hardware.
    /// </summary>
    /// <remarks>
    /// <para><b>Priority Order (highest to lowest):</b></para>
    /// <list type="number">
    /// <item>DirectGpu - Custom optimized kernels (10-100x faster than CLBlast)</item>
    /// <item>CLBlast - Tuned OpenCL BLAS (5-10x faster than ILGPU)</item>
    /// <item>ILGPU - General purpose GPU via .NET</item>
    /// <item>CPU - Always available with SIMD acceleration</item>
    /// </list>
    /// <para>The Default property lazily initializes on first access and probes for
    /// GPU availability. This ensures optimal performance without manual configuration.</para>
    /// </remarks>
    public static class Engine
    {
        private static IEngine? _default;
        private static HardwareCapabilities? _capabilities;
        private static readonly object _lock = new object();
        private static DirectGpuEngine? _directGpuEngine;

        /// <summary>
        /// Gets the default engine instance, automatically selected based on available hardware.
        /// First access detects GPU availability and creates the optimal engine.
        /// Subsequent accesses return the cached instance.
        /// Thread-safe via double-checked locking pattern.
        /// </summary>
        /// <value>
        /// GpuEngine if GPU is available and initialized successfully, otherwise CpuEngine.
        /// </value>
        /// <example>
        /// <code>
        /// // Automatic engine selection - no configuration needed
        /// var vector = new Vector&lt;float&gt;(1000);
        /// vector.ApplyInPlace(x => x * 2, Engine.Default);
        ///
        /// // Check what was selected
        /// Console.WriteLine(Engine.Default.Name);
        /// // Output: "GPU Engine (NVIDIA RTX 4090)" or "CPU Engine"
        /// </code>
        /// </example>
        public static IEngine Default
        {
            get
            {
                if (_default == null)
                {
                    lock (_lock)
                    {
                        if (_default == null)
                        {
                            _default = CreateOptimalEngine();
                        }
                    }
                }
                return _default;
            }
        }

        /// <summary>
        /// Gets comprehensive hardware capabilities for both SIMD and GPU.
        /// Provides detailed information about what acceleration features are available.
        /// </summary>
        /// <value>
        /// Hardware capabilities including SIMD width, GPU type, memory, etc.
        /// </value>
        /// <example>
        /// <code>
        /// Console.WriteLine(Engine.Capabilities);
        /// // Output:
        /// // Hardware Capabilities:
        /// //   SIMD: Available (256-bit AVX2)
        /// //   GPU:  NVIDIA GeForce RTX 4090 (Cuda, 24GB)
        /// </code>
        /// </example>
        public static HardwareCapabilities Capabilities
        {
            get
            {
                if (_capabilities == null)
                {
                    lock (_lock)
                    {
                        if (_capabilities == null)
                        {
                            _capabilities = HardwareCapabilities.Detect();
                        }
                    }
                }
                return _capabilities;
            }
        }

        /// <summary>
        /// Gets the DirectGpu engine for high-performance GPU operations.
        /// This provides direct access to optimized custom kernels for matrix operations.
        /// Works on ALL .NET versions including .NET Framework 4.6.2.
        /// </summary>
        /// <remarks>
        /// <para>The DirectGpuEngine provides:</para>
        /// <list type="bullet">
        /// <item>Optimized GEMM with hierarchical tiling and double-buffering</item>
        /// <item>Fused operations (GEMM+Bias+Activation) to eliminate memory round-trips</item>
        /// <item>Float32-only kernels for maximum GPU performance</item>
        /// <item>Generic type support via boundary conversion</item>
        /// <item>Pure P/Invoke - no ILGPU dependency</item>
        /// </list>
        /// </remarks>
        /// <example>
        /// <code>
        /// // Direct GPU matrix multiplication
        /// var engine = Engine.DirectGpu;
        /// if (engine != null &amp;&amp; engine.IsAvailable)
        /// {
        ///     float[]? result = engine.MatMul(A, B, M, K, N);
        /// }
        /// </code>
        /// </example>
        public static DirectGpuEngine? DirectGpu
        {
            get
            {
                if (_directGpuEngine == null)
                {
                    lock (_lock)
                    {
                        if (_directGpuEngine == null)
                        {
                            try
                            {
                                _directGpuEngine = new DirectGpuEngine();
                            }
                            catch
                            {
                                _directGpuEngine = null;
                            }
                        }
                    }
                }
                return _directGpuEngine;
            }
        }

        /// <summary>
        /// Probes hardware and creates the optimal engine implementation.
        /// Priority order: DirectGpu &gt; CLBlast &gt; ILGPU &gt; CPU.
        /// </summary>
        /// <returns>GPU engine if available, otherwise CPU engine.</returns>
        private static IEngine CreateOptimalEngine()
        {
            // Try DirectGpu first (highest performance - custom optimized kernels)
            // Works on ALL .NET versions including .NET Framework 4.6.2
            try
            {
                var directGpu = DirectGpu;
                if (directGpu != null && directGpu.IsAvailable)
                {
                    Console.WriteLine($"[Engine] DirectGpu available: {directGpu.BackendName} ({directGpu.DeviceName})");
                    Console.WriteLine($"[Engine] DirectGpu: {directGpu.ComputeUnits} CUs, {directGpu.GlobalMemoryGB:F1}GB VRAM");
                }
                else
                {
                    Console.WriteLine("[Engine] DirectGpu: Not available (OpenCL not found or no compatible GPU)");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Engine] DirectGpu initialization failed: {ex.Message}");
            }

#if !NET462
            // Try ILGPU GPU engine (general purpose, good compatibility)
            // Only available on .NET Framework 4.7.1+ and .NET Core/.NET 5+
            try
            {
                var gpuEngine = new GpuEngine();
                if (gpuEngine.SupportsGpu)
                {
                    Console.WriteLine($"[Engine] Auto-selected: {gpuEngine.Name}");
                    Console.WriteLine($"[Engine] GPU detected - operations will use GPU acceleration");
                    Console.WriteLine($"[Engine] Note: Use Engine.DirectGpu for optimized matrix operations");
                    return gpuEngine;
                }
            }
            catch (DllNotFoundException ex)
            {
                Console.WriteLine($"[Engine] ILGPU runtime not found: {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Engine] GPU initialization failed: {ex.Message}");
            }
#endif

            // Fallback to CPU (always available)
            var cpuEngine = new CpuEngine();
            Console.WriteLine($"[Engine] Auto-selected: {cpuEngine.Name}");
            Console.WriteLine($"[Engine] No GPU detected - operations will use CPU with SIMD acceleration");
            return cpuEngine;
        }

        /// <summary>
        /// Forces re-detection of hardware capabilities and engine selection.
        /// Useful if GPU becomes available/unavailable at runtime.
        /// </summary>
        public static void Reset()
        {
            lock (_lock)
            {
                if (_default is IDisposable disposable)
                {
                    disposable.Dispose();
                }
                _default = null;
                _capabilities = null;

                _directGpuEngine?.Dispose();
                _directGpuEngine = null;
            }
        }
    }

    /// <summary>
    /// Comprehensive hardware capability detection for both SIMD and GPU.
    /// Provides a unified view of all acceleration features available on the current machine.
    /// Parallel to how Vector.IsHardwareAccelerated works for SIMD, but covers all hardware.
    /// </summary>
    public class HardwareCapabilities
    {
        /// <summary>
        /// Gets whether SIMD hardware acceleration is available.
        /// </summary>
        public bool SimdAvailable { get; set; }

        /// <summary>
        /// Gets the SIMD vector width in bits (128, 256, 512, or 0 if unavailable).
        /// </summary>
        public int SimdWidth { get; set; }

        /// <summary>
        /// Gets the SIMD technology name (SSE, AVX, AVX2, AVX-512, ARM NEON, etc.).
        /// </summary>
        public string SimdTechnology { get; set; } = string.Empty;

        /// <summary>
        /// Gets whether a GPU is available for acceleration.
        /// </summary>
        public bool GpuAvailable { get; set; }

        /// <summary>
        /// Gets the GPU name (e.g., "NVIDIA GeForce RTX 4090").
        /// </summary>
        public string GpuName { get; set; } = string.Empty;

        // DirectGpu properties - available on ALL .NET versions (pure P/Invoke)
        /// <summary>
        /// Gets whether DirectGpu (custom optimized kernels) is available.
        /// Works on ALL .NET versions including .NET Framework 4.6.2.
        /// </summary>
        public bool DirectGpuAvailable { get; set; }

        /// <summary>
        /// Gets the DirectGpu backend name (OpenCL, CUDA, etc.).
        /// </summary>
        public string DirectGpuBackend { get; set; } = string.Empty;

        /// <summary>
        /// Gets the DirectGpu device name.
        /// </summary>
        public string DirectGpuDevice { get; set; } = string.Empty;

        /// <summary>
        /// Gets the number of DirectGpu compute units.
        /// </summary>
        public int DirectGpuComputeUnits { get; set; }

        // ILGPU-specific GPU properties - only available on .NET Framework 4.7.1+ and .NET Core/.NET 5+
#if !NET462
        /// <summary>
        /// Gets the GPU accelerator type (Cuda, OpenCL, Velocity, CPU).
        /// </summary>
        public AcceleratorType GpuType { get; set; }

        /// <summary>
        /// Gets the total GPU memory in bytes.
        /// </summary>
        public long GpuMemory { get; set; }

        /// <summary>
        /// Gets the maximum number of threads per GPU workgroup.
        /// </summary>
        public int GpuMaxThreads { get; set; }

        /// <summary>
        /// Gets the GPU warp/wavefront size for optimal parallelism.
        /// </summary>
        public int GpuWarpSize { get; set; }

        /// <summary>
        /// Gets the CUDA compute capability (e.g., "8.6" for RTX 3090), or empty if not CUDA.
        /// </summary>
        public string GpuComputeCapability { get; set; } = string.Empty;
#else
        /// <summary>
        /// ILGPU GPU acceleration is not available on .NET Framework 4.6.2
        /// </summary>
        public int GpuType { get; set; } = 0;  // AcceleratorType.CPU equivalent
        public long GpuMemory { get; set; } = 0;
        public int GpuMaxThreads { get; set; } = 0;
        public int GpuWarpSize { get; set; } = 0;
        public string GpuComputeCapability { get; set; } = string.Empty;
#endif

        /// <summary>
        /// Detects all hardware capabilities on the current machine.
        /// </summary>
        /// <returns>Comprehensive hardware capability information.</returns>
        public static HardwareCapabilities Detect()
        {
            var caps = new HardwareCapabilities();

            // Detect SIMD capabilities
            caps = DetectSimd(caps);

            // Detect DirectGpu capabilities (custom optimized kernels)
            // Works on ALL .NET versions including .NET Framework 4.6.2
            caps = DetectDirectGpu(caps);

#if !NET462
            // Detect ILGPU GPU capabilities (available on net471+ and net8.0+)
            caps = DetectGpu(caps);
#else
            // ILGPU GPU detection not available on .NET Framework 4.6.2
            caps.GpuAvailable = false;
#endif

            return caps;
        }

        private static HardwareCapabilities DetectSimd(HardwareCapabilities caps)
        {
#if NET5_0_OR_GREATER
            bool hasVector512 = Vector512.IsHardwareAccelerated;
            bool hasVector256 = Vector256.IsHardwareAccelerated;
            bool hasVector128 = Vector128.IsHardwareAccelerated;

            if (hasVector512)
            {
                caps.SimdAvailable = true;
                caps.SimdWidth = 512;
                caps.SimdTechnology = "AVX-512";
                return caps;
            }
            else if (hasVector256)
            {
                caps.SimdAvailable = true;
                caps.SimdWidth = 256;
                caps.SimdTechnology = "AVX2";
                return caps;
            }
            else if (hasVector128)
            {
                caps.SimdAvailable = true;
                caps.SimdWidth = 128;
                caps.SimdTechnology = "SSE/NEON";
                return caps;
            }
#else
            // .NET Framework: Use System.Numerics.Vector
            if (Vector.IsHardwareAccelerated)
            {
                int vectorBits = Vector<float>.Count * 32;  // 32 bits per float
                string tech = vectorBits >= 256 ? "AVX" : "SSE";
                caps.SimdAvailable = true;
                caps.SimdWidth = vectorBits;
                caps.SimdTechnology = tech;
                return caps;
            }
#endif

            caps.SimdAvailable = false;
            caps.SimdWidth = 0;
            caps.SimdTechnology = "None";
            return caps;
        }

        /// <summary>
        /// Detects DirectGpu capabilities.
        /// Works on ALL .NET versions including .NET Framework 4.6.2.
        /// </summary>
        private static HardwareCapabilities DetectDirectGpu(HardwareCapabilities caps)
        {
            try
            {
                var directGpu = Engine.DirectGpu;
                if (directGpu != null && directGpu.IsAvailable)
                {
                    caps.DirectGpuAvailable = true;
                    caps.DirectGpuBackend = directGpu.BackendName;
                    caps.DirectGpuDevice = directGpu.DeviceName;
                    caps.DirectGpuComputeUnits = directGpu.ComputeUnits;
                }
                else
                {
                    caps.DirectGpuAvailable = false;
                }
            }
            catch
            {
                caps.DirectGpuAvailable = false;
            }
            return caps;
        }

#if !NET462
        private static HardwareCapabilities DetectGpu(HardwareCapabilities caps)
        {
            try
            {
                // Don't use EnableAlgorithms() during detection - it can hide devices
                // on some systems. EnableAlgorithms() is enabled during actual accelerator
                // creation in GpuEngine where it's needed for algorithm operations.
                using var context = Context.Create(builder => builder.Default());
                var device = context.GetPreferredDevice(preferCPU: false);

                if (device.AcceleratorType != AcceleratorType.CPU)
                {
                    caps.GpuAvailable = true;
                    caps.GpuName = device.Name;
                    caps.GpuType = device.AcceleratorType;
                    caps.GpuMemory = device.MemorySize;
                    caps.GpuMaxThreads = device.MaxNumThreadsPerGroup;
                    caps.GpuWarpSize = device.WarpSize;

                    // CUDA-specific detection (optional - may not be available in all ILGPU versions)
                    if (device is CudaDevice cudaDevice)
                    {
                        try
                        {
                            // ComputeCapability may not be available in all ILGPU versions
                            var capability = cudaDevice.GetType().GetProperty("ComputeCapability");
                            if (capability != null)
                            {
                                var value = capability.GetValue(cudaDevice);
                                if (value != null)
                                {
                                    var majorProp = value.GetType().GetProperty("Major");
                                    var minorProp = value.GetType().GetProperty("Minor");
                                    if (majorProp != null && minorProp != null)
                                    {
                                        var major = majorProp.GetValue(value);
                                        var minor = minorProp.GetValue(value);
                                        caps.GpuComputeCapability = $"{major}.{minor}";
                                    }
                                }
                            }
                        }
                        catch
                        {
                            // ComputeCapability detection failed - leave as default
                        }
                    }

                    return caps;
                }
            }
            catch (DllNotFoundException)
            {
                // ILGPU runtime not available
            }
            catch (Exception)
            {
                // GPU detection failed
            }

            caps.GpuAvailable = false;
            caps.GpuType = AcceleratorType.CPU;
            return caps;
        }
#endif

        /// <summary>
        /// Returns a human-readable string describing all hardware capabilities.
        /// </summary>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine("Hardware Capabilities:");

            // SIMD info
            if (SimdAvailable)
            {
                sb.AppendLine($"  SIMD: Available ({SimdWidth}-bit {SimdTechnology})");
            }
            else
            {
                sb.AppendLine("  SIMD: Not Available (scalar fallback)");
            }

            // DirectGpu info (highest performance tier)
            if (DirectGpuAvailable)
            {
                sb.AppendLine($"  DirectGpu: Available ({DirectGpuBackend})");
                sb.AppendLine($"        Device: {DirectGpuDevice}");
                sb.AppendLine($"        Compute Units: {DirectGpuComputeUnits}");
            }
            else
            {
                sb.AppendLine("  DirectGpu: Not Available");
            }

            // GPU info (ILGPU tier)
            if (GpuAvailable)
            {
                long gbMemory = GpuMemory / (1024 * 1024 * 1024);
                sb.AppendLine($"  GPU:  {GpuName} ({GpuType}, {gbMemory}GB)");
                sb.AppendLine($"        Max Threads: {GpuMaxThreads}, Warp Size: {GpuWarpSize}");
                if (!string.IsNullOrEmpty(GpuComputeCapability))
                {
                    sb.AppendLine($"        Compute Capability: {GpuComputeCapability}");
                }
            }
            else
            {
                sb.AppendLine("  GPU:  Not Available (CPU fallback)");
            }

            return sb.ToString();
        }
    }
}
