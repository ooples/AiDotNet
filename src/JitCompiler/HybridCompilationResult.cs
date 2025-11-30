using AiDotNet.Tensors;

namespace AiDotNet.JitCompiler;

/// <summary>
/// Result of compiling with unsupported operation handling.
/// </summary>
/// <typeparam name="T">The numeric type for tensor elements.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> When you use CompileWithUnsupportedHandling, you get this result.
/// It tells you:
/// - The compiled function (always usable)
/// - Whether it's fully JIT compiled or uses fallback
/// - Compatibility details
/// - Any warnings about unsupported operations
/// </para>
/// </remarks>
public class HybridCompilationResult<T>
{
    /// <summary>
    /// Gets or sets the compiled function.
    /// This function is always usable regardless of execution mode.
    /// </summary>
    public Func<Tensor<T>[], Tensor<T>[]> CompiledFunc { get; set; } = null!;

    /// <summary>
    /// Gets or sets whether the function was fully JIT compiled.
    /// If false, some or all operations use interpreted execution.
    /// </summary>
    public bool IsFullyJitCompiled { get; set; }

    /// <summary>
    /// Gets or sets the execution mode: "JIT", "Interpreted", "Hybrid", or "JIT (skipped ops)".
    /// </summary>
    public string ExecutionMode { get; set; } = "Unknown";

    /// <summary>
    /// Gets or sets the compatibility analysis results.
    /// </summary>
    public JitCompatibilityResult Compatibility { get; set; } = new();

    /// <summary>
    /// Gets or sets any warnings generated during compilation.
    /// </summary>
    public List<string> Warnings { get; set; } = new();

    /// <summary>
    /// Returns a summary of the compilation result.
    /// </summary>
    public override string ToString()
    {
        var warnings = Warnings.Count > 0 ? $" ({Warnings.Count} warnings)" : "";
        return $"Execution: {ExecutionMode}, JIT: {(IsFullyJitCompiled ? "100%" : $"{Compatibility.SupportedPercentage:F1}%")}{warnings}";
    }
}
