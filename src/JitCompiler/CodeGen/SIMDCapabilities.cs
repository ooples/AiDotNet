using System.Numerics;

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
    /// Detects SIMD capabilities of the current hardware.
    /// </summary>
    /// <returns>A SIMDCapabilities instance describing the current hardware.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method checks what SIMD features your CPU supports.
    /// It uses the .NET runtime's Vector class to determine the maximum vector size,
    /// then infers which instruction sets are available based on that size.
    /// </para>
    /// </remarks>
    public static SIMDCapabilities Detect()
    {
        var caps = new SIMDCapabilities();

        if (Vector.IsHardwareAccelerated)
        {
            var vectorSize = Vector<float>.Count;

            // Infer capabilities from vector size
            caps.HasSSE = vectorSize >= 4;  // 128-bit = 4 floats
            caps.HasAVX = vectorSize >= 8;  // 256-bit = 8 floats
            caps.HasAVX512 = vectorSize >= 16; // 512-bit = 16 floats
            caps.HasAVX2 = caps.HasAVX;

            // .NET's System.Numerics.Vector doesn't directly expose FMA
            // but modern CPUs with AVX2 typically have FMA
            caps.HasFMA = caps.HasAVX2;

            caps.MaxVectorWidth = vectorSize * sizeof(float);
        }

        return caps;
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
