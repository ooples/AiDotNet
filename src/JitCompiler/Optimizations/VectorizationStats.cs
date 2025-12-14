namespace AiDotNet.JitCompiler.Optimizations;

/// <summary>
/// Statistics about vectorization opportunities.
/// </summary>
public class VectorizationStats
{
    /// <summary>Total number of operations in the graph.</summary>
    public int TotalOperations { get; set; }

    /// <summary>Number of operations that can be vectorized.</summary>
    public int VectorizableOperations { get; set; }

    /// <summary>Total elements that can be processed with vectors.</summary>
    public long TotalVectorizableElements { get; set; }

    /// <summary>Hardware vector width.</summary>
    public int HardwareVectorWidth { get; set; }

    /// <summary>Whether hardware acceleration is available.</summary>
    public bool IsHardwareAccelerated { get; set; }

    /// <summary>Estimated speedup from vectorization.</summary>
    public double EstimatedSpeedup
    {
        get
        {
            if (!IsHardwareAccelerated || TotalOperations == 0)
                return 1.0;

            var vectorizableRatio = (double)VectorizableOperations / TotalOperations;
            // Amdahl's law: Speedup = 1 / ((1 - P) + P/S) where P = parallel fraction, S = speedup factor
            var speedupFactor = HardwareVectorWidth * 0.7; // Account for overhead
            return 1.0 / ((1.0 - vectorizableRatio) + (vectorizableRatio / speedupFactor));
        }
    }

    /// <summary>Returns a string representation of the statistics.</summary>
    public override string ToString()
    {
        return $"Vectorization Stats: {VectorizableOperations}/{TotalOperations} ops vectorizable, " +
               $"Vector width: {HardwareVectorWidth}, " +
               $"Estimated speedup: {EstimatedSpeedup:F2}x";
    }
}
