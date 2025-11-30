namespace AiDotNet.JitCompiler;

/// <summary>
/// Information about an unsupported operation encountered during compilation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When the JIT compiler finds an operation it can't handle,
/// it creates one of these to record:
/// - What operation was unsupported
/// - Where it was in the graph
/// - Why it couldn't be compiled
///
/// Use this to diagnose compilation issues or to know which operations need fallback.
/// </para>
/// </remarks>
public class UnsupportedOperationInfo
{
    /// <summary>
    /// Gets or sets the name of the unsupported operation type.
    /// </summary>
    public string OperationType { get; set; } = "";

    /// <summary>
    /// Gets or sets the name of the computation node (if available).
    /// </summary>
    public string? NodeName { get; set; }

    /// <summary>
    /// Gets or sets the tensor ID that would have been assigned to this operation.
    /// </summary>
    public int TensorId { get; set; }

    /// <summary>
    /// Gets or sets the reason why this operation is not supported.
    /// </summary>
    public string Reason { get; set; } = "Operation type not implemented in JIT compiler";

    /// <summary>
    /// Gets or sets whether this operation can be executed via fallback.
    /// </summary>
    public bool CanFallback { get; set; } = true;

    /// <summary>
    /// Returns a string representation of the unsupported operation.
    /// </summary>
    public override string ToString()
    {
        var name = NodeName != null ? $" ({NodeName})" : "";
        return $"Unsupported: {OperationType}{name} at tensor {TensorId} - {Reason}";
    }
}
