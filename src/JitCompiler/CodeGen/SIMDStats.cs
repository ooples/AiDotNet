using System.Numerics;

namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Statistics about SIMD optimization opportunities in an IR graph.
/// </summary>
/// <remarks>
/// <para>
/// This class provides insights into how much of a computation graph can benefit
/// from SIMD optimization. It helps developers understand the potential performance
/// improvements available from vectorization.
/// </para>
/// <para><b>For Beginners:</b> When the JIT compiler analyzes your computation graph,
/// it identifies operations that can be made faster using SIMD instructions.
/// This class summarizes what it found:
///
/// - How many total operations are in the graph
/// - How many can be accelerated with SIMD
/// - What the expected speedup might be
///
/// For example, if 80% of your operations are vectorizable and your CPU supports
/// 8-wide SIMD (AVX), you might see a 4-6x overall speedup.
/// </para>
/// </remarks>
public class SIMDStats
{
    /// <summary>Total number of operations in the graph.</summary>
    public int TotalOperations { get; set; }

    /// <summary>Number of operations that can be vectorized.</summary>
    public int VectorizableOperations { get; set; }

    /// <summary>Size of SIMD vectors on this hardware (number of elements).</summary>
    public int VectorSize { get; set; }

    /// <summary>Whether hardware acceleration is available.</summary>
    public bool HardwareAccelerated { get; set; }

    /// <summary>
    /// Gets the estimated speedup from vectorization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a rough estimate based on the ratio of vectorizable operations
    /// and the vector width. The actual speedup depends on many factors including
    /// memory bandwidth, operation complexity, and CPU cache efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This number gives you a rough idea of how much
    /// faster your code might run with SIMD optimization. A value of 3.0 means
    /// approximately 3x faster. Real-world results may vary.
    /// </para>
    /// </remarks>
    public double EstimatedSpeedup
    {
        get
        {
            if (!HardwareAccelerated || TotalOperations == 0)
                return 1.0;

            var vectorizableRatio = (double)VectorizableOperations / TotalOperations;
            var perOpSpeedup = VectorSize * 0.75; // Account for overhead
            return 1.0 + (vectorizableRatio * (perOpSpeedup - 1.0));
        }
    }

    /// <summary>
    /// Gets the ratio of vectorizable operations to total operations.
    /// </summary>
    public double VectorizableRatio => TotalOperations > 0
        ? (double)VectorizableOperations / TotalOperations
        : 0.0;

    /// <summary>
    /// Creates a new SIMDStats instance with default values.
    /// </summary>
    public SIMDStats()
    {
        HardwareAccelerated = Vector.IsHardwareAccelerated;
        VectorSize = Vector.IsHardwareAccelerated ? Vector<float>.Count : 1;
    }

    /// <summary>
    /// Gets a string representation of the SIMD statistics.
    /// </summary>
    public override string ToString()
    {
        return $"SIMD Stats: {VectorizableOperations}/{TotalOperations} operations vectorizable ({VectorizableRatio:P1}), " +
               $"Vector size: {VectorSize}, " +
               $"Estimated speedup: {EstimatedSpeedup:F2}x";
    }
}
