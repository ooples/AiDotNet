// Copyright (c) AiDotNet. All rights reserved.
// Automatic engine selection based on hardware capabilities

using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Text;
#if NET471_OR_GREATER
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
    /// The Default property lazily initializes on first access and probes for:
    /// 1. GPU availability (via ILGPU) - preferred
    /// 2. CPU with SIMD fallback - if no GPU
    /// This ensures optimal performance without manual configuration.
    /// </remarks>
    public static class Engine
    {
        private static IEngine? _default;
        private static HardwareCapabilities? _capabilities;
        private static readonly object _lock = new object();

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
        /// Probes hardware and creates the optimal engine implementation.
        /// Priority order: GPU &gt; CPU.
        /// </summary>
        /// <returns>GPU engine if available, otherwise CPU engine.</returns>
        private static IEngine CreateOptimalEngine()
        {
#if NET471_OR_GREATER
            // Try GPU first (highest performance potential)
            try
            {
                var gpuEngine = new GpuEngine();
                if (gpuEngine.SupportsGpu)
                {
                    Console.WriteLine($"[Engine] Auto-selected: {gpuEngine.Name}");
                    Console.WriteLine($"[Engine] GPU detected - operations will use GPU acceleration");
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

#if NET471_OR_GREATER
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
        /// GPU acceleration is not available on .NET Framework 4.6.2
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

#if NET471_OR_GREATER
            // Detect GPU capabilities (available on net471+)
            caps = DetectGpu(caps);
#else
            // GPU detection not available on .NET Framework 4.6.2
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

#if NET471_OR_GREATER
        private static HardwareCapabilities DetectGpu(HardwareCapabilities caps)
        {
            try
            {
                using var context = Context.Create(builder => builder.Default().EnableAlgorithms());
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

            // GPU info
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
