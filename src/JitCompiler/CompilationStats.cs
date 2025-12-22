namespace AiDotNet.JitCompiler;

using System.Globalization;

/// <summary>
/// Statistics about a compilation operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Information about what happened during compilation.
///
/// Tells you:
/// - How many operations were optimized away
/// - What optimizations were applied
/// - How long compilation took
/// - Whether the result came from cache
/// </para>
/// </remarks>
public class CompilationStats
{
    /// <summary>
    /// Gets or sets the number of operations in the original graph.
    /// </summary>
    public int OriginalOperationCount { get; set; }

    /// <summary>
    /// Gets or sets the number of operations after optimization.
    /// </summary>
    public int OptimizedOperationCount { get; set; }

    /// <summary>
    /// Gets or sets the list of optimizations that were applied.
    /// </summary>
    public List<string> OptimizationsApplied { get; set; } = new();

    /// <summary>
    /// Gets or sets the time taken to compile the graph.
    /// </summary>
    public TimeSpan CompilationTime { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the compiled function came from cache.
    /// </summary>
    public bool CacheHit { get; set; }

    /// <summary>
    /// Gets the reduction in operation count from optimization.
    /// </summary>
    public int OperationsEliminated => OriginalOperationCount - OptimizedOperationCount;

    /// <summary>
    /// Gets the percentage reduction in operation count.
    /// </summary>
    public double OptimizationPercentage =>
        OriginalOperationCount > 0
            ? (double)OperationsEliminated / OriginalOperationCount * 100
            : 0;

    /// <summary>
    /// Gets a string representation of the compilation statistics.
    /// </summary>
    public override string ToString()
    {
        var optimizationPercentage = OptimizationPercentage.ToString("F1", CultureInfo.InvariantCulture);
        var compilationTimeMs = CompilationTime.TotalMilliseconds.ToString("F2", CultureInfo.InvariantCulture);

        return $"Compilation Stats:\n" +
               $"  Original operations: {OriginalOperationCount}\n" +
               $"  Optimized operations: {OptimizedOperationCount}\n" +
               $"  Operations eliminated: {OperationsEliminated} ({optimizationPercentage}%)\n" +
               $"  Optimizations applied: {string.Join(", ", OptimizationsApplied)}\n" +
               $"  Compilation time: {compilationTimeMs}ms\n" +
               $"  Cache hit: {CacheHit}";
    }
}
