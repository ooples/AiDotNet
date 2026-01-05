namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Defines the execution mode for GPU operations.
/// </summary>
public enum GpuExecutionMode
{
    /// <summary>
    /// Eager execution mode where each operation runs immediately and synchronously.
    /// This is the default fallback mode with maximum compatibility.
    /// </summary>
    Eager = 0,

    /// <summary>
    /// Deferred execution mode where operations are recorded to an execution graph.
    /// The graph is compiled and executed when explicitly flushed.
    /// Enables operation fusion and scheduling optimization.
    /// </summary>
    Deferred = 1,

    /// <summary>
    /// Scoped deferred execution where operations within a scope are batched.
    /// Provides a middle ground between eager and full deferred execution.
    /// Uses multi-stream parallelism without full graph compilation.
    /// </summary>
    ScopedDeferred = 2,

    /// <summary>
    /// Automatic mode that selects the best execution strategy based on:
    /// - GPU capabilities (async support, multi-stream)
    /// - Operation size (small ops stay eager)
    /// - Current workload patterns
    /// </summary>
    Auto = 3
}
