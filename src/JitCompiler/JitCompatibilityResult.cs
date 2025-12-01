namespace AiDotNet.JitCompiler;

/// <summary>
/// Result of analyzing a graph for JIT compatibility.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Before compiling, you can check if your graph is compatible.
/// This result tells you:
/// - Whether full JIT compilation is possible
/// - What operations are supported/unsupported
/// - Whether hybrid mode can be used
/// </para>
/// </remarks>
public class JitCompatibilityResult
{
    /// <summary>
    /// Gets or sets whether all operations in the graph are supported.
    /// </summary>
    public bool IsFullySupported { get; set; }

    /// <summary>
    /// Gets or sets the list of supported operation types found in the graph.
    /// </summary>
    public List<string> SupportedOperations { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of unsupported operations found in the graph.
    /// </summary>
    public List<UnsupportedOperationInfo> UnsupportedOperations { get; set; } = new();

    /// <summary>
    /// Gets or sets whether hybrid mode can be used (some ops JIT, some interpreted).
    /// </summary>
    public bool CanUseHybridMode { get; set; }

    /// <summary>
    /// Gets the percentage of operations that can be JIT compiled.
    /// </summary>
    public double SupportedPercentage =>
        SupportedOperations.Count + UnsupportedOperations.Count > 0
            ? (double)SupportedOperations.Count / (SupportedOperations.Count + UnsupportedOperations.Count) * 100
            : 100;

    /// <summary>
    /// Returns a summary of the compatibility analysis.
    /// </summary>
    public override string ToString()
    {
        if (IsFullySupported)
        {
            return $"Fully JIT compatible ({SupportedOperations.Count} operations)";
        }

        return $"Partial JIT support: {SupportedPercentage:F1}% ({SupportedOperations.Count} supported, " +
               $"{UnsupportedOperations.Count} unsupported). Hybrid mode: {(CanUseHybridMode ? "available" : "not available")}";
    }
}
