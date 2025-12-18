#if NET6_0_OR_GREATER
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// SIMD capabilities detected on the current hardware.
/// </summary>
/// <remarks>
/// <para>
/// This class provides information about the SIMD (Single Instruction Multiple Data) capabilities
/// available on the current CPU. This information is used by the JIT compiler to select the most
/// efficient code paths for tensor operations.
/// </para>
/// <para><b>For Beginners:</b> Modern CPUs have special instructions that can process multiple
/// numbers at once. This class detects which of these special instructions are available:
///
/// - SSE: Can process 4 floats at once (128-bit)
/// - AVX: Can process 8 floats at once (256-bit)
/// - AVX-512: Can process 16 floats at once (512-bit)
/// - NEON: ARM's equivalent (for mobile/Apple Silicon)
///
/// The more advanced instructions available, the faster tensor operations can be.
/// </para>
/// </remarks>
public class SIMDCapabilities
{
    /// <summary>Whether SSE (128-bit) is available.</summary>
    public bool HasSSE { get; set; }

    /// <summary>Whether AVX (256-bit) is available.</summary>
    public bool HasAVX { get; set; }

    /// <summary>Whether AVX2 is available.</summary>
    public bool HasAVX2 { get; set; }

    /// <summary>Whether AVX-512 is available.</summary>
    public bool HasAVX512 { get; set; }

    /// <summary>Whether FMA (Fused Multiply-Add) is available.</summary>
    public bool HasFMA { get; set; }

    /// <summary>Whether ARM NEON is available.</summary>
    public bool HasNEON { get; set; }

    /// <summary>Maximum vector width in bytes.</summary>
    public int MaxVectorWidth { get; set; }

    /// <summary>
    /// Gets whether any hardware SIMD acceleration is available.
    /// </summary>
    public bool IsHardwareAccelerated => HasSSE || HasNEON;

    /// <summary>
    /// Detects SIMD capabilities of the current hardware.
    /// </summary>
    /// <returns>A SIMDCapabilities instance describing the current hardware.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method checks what SIMD features your CPU supports
    /// using the .NET runtime intrinsics API to directly query hardware capabilities.
    /// </para>
    /// </remarks>
    public static SIMDCapabilities Detect()
    {
        var caps = new SIMDCapabilities();

#if NET6_0_OR_GREATER
        // Detect x86/x64 SIMD capabilities using intrinsics
        caps.HasSSE = Sse.IsSupported;
        caps.HasAVX = Avx.IsSupported;
        caps.HasAVX2 = Avx2.IsSupported;
        caps.HasAVX512 = Avx512F.IsSupported;
        caps.HasFMA = Fma.IsSupported;

        // Detect ARM NEON capabilities
        caps.HasNEON = AdvSimd.IsSupported;

        // Determine maximum vector width based on capabilities
        if (caps.HasAVX512)
            caps.MaxVectorWidth = 64; // 512 bits = 64 bytes
        else if (caps.HasAVX)
            caps.MaxVectorWidth = 32; // 256 bits = 32 bytes
        else if (caps.HasSSE || caps.HasNEON)
            caps.MaxVectorWidth = 16; // 128 bits = 16 bytes
        else
            caps.MaxVectorWidth = 0;
#else
        // .NET Framework doesn't have intrinsics - disable SIMD acceleration
        caps.HasSSE = false;
        caps.HasAVX = false;
        caps.HasAVX2 = false;
        caps.HasAVX512 = false;
        caps.HasFMA = false;
        caps.HasNEON = false;
        caps.MaxVectorWidth = 0;
#endif

        return caps;
    }

    /// <summary>
    /// Gets the number of elements that fit in a SIMD register for the specified type size.
    /// </summary>
    /// <param name="typeSizeInBytes">The size of the element type in bytes.</param>
    /// <returns>The number of elements that fit in a SIMD register.</returns>
    public int GetVectorCount(int typeSizeInBytes)
    {
        if (typeSizeInBytes <= 0 || MaxVectorWidth <= 0)
            return 1;

        return MaxVectorWidth / typeSizeInBytes;
    }

    /// <summary>
    /// Gets a human-readable description of the capabilities.
    /// </summary>
    public override string ToString()
    {
        var features = new List<string>();
        if (HasSSE) features.Add("SSE");
        if (HasAVX) features.Add("AVX");
        if (HasAVX2) features.Add("AVX2");
        if (HasAVX512) features.Add("AVX-512");
        if (HasFMA) features.Add("FMA");
        if (HasNEON) features.Add("NEON");

        return features.Count > 0
            ? $"SIMD: {string.Join(", ", features)} (max width: {MaxVectorWidth} bytes)"
            : "SIMD: Not available";
    }
}
