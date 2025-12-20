using System;
using System.Runtime.InteropServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
#endif

namespace AiDotNet.Tensors.Engines
{
    /// <summary>
    /// Provides platform and hardware capability detection for optimizing
    /// tensor operations based on available SIMD instructions and cache sizes.
    /// </summary>
    public static class PlatformDetector
    {
        private static readonly Lazy<PlatformCapabilities> _capabilities =
            new Lazy<PlatformCapabilities>(DetectCapabilities);

        /// <summary>
        /// Gets the detected platform capabilities
        /// </summary>
        public static PlatformCapabilities Capabilities => _capabilities.Value;

        private static PlatformCapabilities DetectCapabilities()
        {
            var caps = new PlatformCapabilities
            {
                Architecture = RuntimeInformation.ProcessArchitecture,
                OSDescription = RuntimeInformation.OSDescription,
                FrameworkDescription = RuntimeInformation.FrameworkDescription,
                ProcessorCount = Environment.ProcessorCount,
                Is64BitProcess = Environment.Is64BitProcess,
                Is64BitOperatingSystem = Environment.Is64BitOperatingSystem
            };

#if NET5_0_OR_GREATER
            // Detect x86/x64 SIMD support
            if (caps.Architecture == Architecture.X64 || caps.Architecture == Architecture.X86)
            {
                caps.HasSSE = Sse.IsSupported;
                caps.HasSSE2 = Sse2.IsSupported;
                caps.HasSSE3 = Sse3.IsSupported;
                caps.HasSSSE3 = Ssse3.IsSupported;
                caps.HasSSE41 = Sse41.IsSupported;
                caps.HasSSE42 = Sse42.IsSupported;
                caps.HasAVX = Avx.IsSupported;
                caps.HasAVX2 = Avx2.IsSupported;
                caps.HasFMA = Fma.IsSupported;
                caps.HasAVX512F = Avx512F.IsSupported;
                caps.HasAVX512BW = Avx512BW.IsSupported;
                caps.HasAVX512DQ = Avx512DQ.IsSupported;
                // AVX-512VL is implied when other AVX-512 extensions are supported
                caps.HasAVX512VL = Avx512F.VL.IsSupported;
            }

            // Detect ARM SIMD support
            if (caps.Architecture == Architecture.Arm64 || caps.Architecture == Architecture.Arm)
            {
                caps.HasNeon = AdvSimd.IsSupported;
                caps.HasArmBase = ArmBase.IsSupported;
                caps.HasArmAes = System.Runtime.Intrinsics.Arm.Aes.IsSupported;
                caps.HasArmCrc32 = Crc32.IsSupported;
                caps.HasArmDp = caps.Architecture == Architecture.Arm64 && Dp.Arm64.IsSupported;
            }
#endif

            // Detect cache sizes (approximate based on typical values)
            caps.L1CacheSize = EstimateL1CacheSize(caps.Architecture);
            caps.L2CacheSize = EstimateL2CacheSize(caps.Architecture);
            caps.L3CacheSize = EstimateL3CacheSize(caps.Architecture);

            // Check for GPU support (requires additional libraries)
            caps.HasCudaSupport = DetectCudaSupport();
            caps.HasOpenCLSupport = DetectOpenCLSupport();

            return caps;
        }

        private static int EstimateL1CacheSize(Architecture arch)
        {
            // Typical L1 cache size is 32KB per core
            return 32 * 1024;
        }

        private static int EstimateL2CacheSize(Architecture arch)
        {
            // Typical L2 cache size is 256KB per core
            return 256 * 1024;
        }

        private static int EstimateL3CacheSize(Architecture arch)
        {
            // Typical L3 cache size is 2-8MB shared
            return 8 * 1024 * 1024;
        }

        /// <summary>
        /// Checks whether CUDA driver support appears to be available on this machine.
        ///
        /// Notes:
        /// - This attempts a lightweight runtime check for the CUDA driver library (not the toolkit).
        /// - It is intentionally conservative: if we cannot verify CUDA driver presence, we return false.
        /// - This does not guarantee that higher-level CUDA compute is usable (device selection, permissions, etc.).
        /// </summary>
        private static bool DetectCudaSupport()
        {
            if (!Environment.Is64BitProcess)
                return false;

#if NET5_0_OR_GREATER
            // Prefer checking for the CUDA driver library:
            // - Windows: nvcuda.dll
            // - Linux: libcuda.so.1 (or libcuda.so)
            try
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    return TryLoadNativeLibrary("nvcuda.dll");
                }

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    return TryLoadNativeLibrary("libcuda.so.1") || TryLoadNativeLibrary("libcuda.so");
                }

                return false;
            }
            catch
            {
                return false;
            }
#else
            // .NET Framework builds are conservative here; implement a native check if/when CUDA support is added for net471.
            return false;
#endif
        }

#if NET5_0_OR_GREATER
        private static bool TryLoadNativeLibrary(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return false;

            if (NativeLibrary.TryLoad(name, out var handle))
            {
                NativeLibrary.Free(handle);
                return true;
            }

            return false;
        }
#endif

        private static bool DetectOpenCLSupport()
        {
            // This would require OpenCL library calls
            // For now, we'll return false (requires additional implementation)
            return false;
        }

        /// <summary>
        /// Gets a human-readable description of the platform capabilities
        /// </summary>
        public static string GetCapabilitiesDescription()
        {
            var caps = Capabilities;
            var desc = new System.Text.StringBuilder();

            desc.AppendLine($"Platform: {caps.OSDescription}");
            desc.AppendLine($"Architecture: {caps.Architecture}");
            desc.AppendLine($"Framework: {caps.FrameworkDescription}");
            desc.AppendLine($"Processor Count: {caps.ProcessorCount}");
            desc.AppendLine($"64-bit Process: {caps.Is64BitProcess}");
            desc.AppendLine();

            if (caps.Architecture == Architecture.X64 || caps.Architecture == Architecture.X86)
            {
                desc.AppendLine("x86/x64 SIMD Support:");
                desc.AppendLine($"  SSE: {caps.HasSSE}");
                desc.AppendLine($"  SSE2: {caps.HasSSE2}");
                desc.AppendLine($"  SSE3: {caps.HasSSE3}");
                desc.AppendLine($"  SSSE3: {caps.HasSSSE3}");
                desc.AppendLine($"  SSE4.1: {caps.HasSSE41}");
                desc.AppendLine($"  SSE4.2: {caps.HasSSE42}");
                desc.AppendLine($"  AVX: {caps.HasAVX}");
                desc.AppendLine($"  AVX2: {caps.HasAVX2}");
                desc.AppendLine($"  FMA: {caps.HasFMA}");
                desc.AppendLine($"  AVX-512F: {caps.HasAVX512F}");
                desc.AppendLine($"  AVX-512BW: {caps.HasAVX512BW}");
                desc.AppendLine($"  AVX-512DQ: {caps.HasAVX512DQ}");
                desc.AppendLine($"  AVX-512VL: {caps.HasAVX512VL}");
            }

            if (caps.Architecture == Architecture.Arm64 || caps.Architecture == Architecture.Arm)
            {
                desc.AppendLine("ARM SIMD Support:");
                desc.AppendLine($"  NEON: {caps.HasNeon}");
                desc.AppendLine($"  ARM Base: {caps.HasArmBase}");
                desc.AppendLine($"  AES: {caps.HasArmAes}");
                desc.AppendLine($"  CRC32: {caps.HasArmCrc32}");
                desc.AppendLine($"  Dot Product: {caps.HasArmDp}");
            }

            desc.AppendLine();
            desc.AppendLine("GPU Support:");
            desc.AppendLine($"  CUDA: {caps.HasCudaSupport}");
            desc.AppendLine($"  OpenCL: {caps.HasOpenCLSupport}");

            return desc.ToString();
        }
    }

    /// <summary>
    /// Represents detected platform capabilities including SIMD support,
    /// cache sizes, and GPU availability.
    /// </summary>
    public class PlatformCapabilities
    {
        // Basic platform info
        public Architecture Architecture { get; set; }
        public string OSDescription { get; set; } = string.Empty;
        public string FrameworkDescription { get; set; } = string.Empty;
        public int ProcessorCount { get; set; }
        public bool Is64BitProcess { get; set; }
        public bool Is64BitOperatingSystem { get; set; }

        // x86/x64 SIMD capabilities
        public bool HasSSE { get; set; }
        public bool HasSSE2 { get; set; }
        public bool HasSSE3 { get; set; }
        public bool HasSSSE3 { get; set; }
        public bool HasSSE41 { get; set; }
        public bool HasSSE42 { get; set; }
        public bool HasAVX { get; set; }
        public bool HasAVX2 { get; set; }
        public bool HasFMA { get; set; }
        public bool HasAVX512F { get; set; }
        public bool HasAVX512BW { get; set; }
        public bool HasAVX512DQ { get; set; }
        public bool HasAVX512VL { get; set; }

        // ARM SIMD capabilities
        public bool HasNeon { get; set; }
        public bool HasArmBase { get; set; }
        public bool HasArmAes { get; set; }
        public bool HasArmCrc32 { get; set; }
        public bool HasArmDp { get; set; }

        // Cache information
        public int L1CacheSize { get; set; }
        public int L2CacheSize { get; set; }
        public int L3CacheSize { get; set; }

        // GPU capabilities
        public bool HasCudaSupport { get; set; }
        public bool HasOpenCLSupport { get; set; }

        /// <summary>
        /// Returns the best available SIMD instruction set
        /// </summary>
        public string GetBestSimdSet()
        {
            if (Architecture == Architecture.X64 || Architecture == Architecture.X86)
            {
                if (HasAVX512F) return "AVX-512";
                if (HasAVX2) return "AVX2";
                if (HasAVX) return "AVX";
                if (HasSSE42) return "SSE4.2";
                if (HasSSE41) return "SSE4.1";
                if (HasSSSE3) return "SSSE3";
                if (HasSSE3) return "SSE3";
                if (HasSSE2) return "SSE2";
                if (HasSSE) return "SSE";
            }
            else if (Architecture == Architecture.Arm64 || Architecture == Architecture.Arm)
            {
                if (HasArmDp) return "NEON with Dot Product";
                if (HasNeon) return "NEON";
            }

            return "None";
        }
    }
}
