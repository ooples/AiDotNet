namespace AiDotNet.Enums;

/// <summary>
/// Controls how <c>AiModelBuilder</c> executes model forward, backward, and optimizer steps.
/// </summary>
public enum ExecutionMode
{
    /// <summary>
    /// Automatically compiles cross-layer fused graphs, multi-bucket CUDA Graph replay trees (when a DirectGpu
    /// backend is active), GPU-resident optimizer scalar updates, and <c>LayerBase&lt;T&gt;</c> versioned derived-weight
    /// caches, falling back gracefully to eager execution for dynamic host-only operations. Enabled by default on
    /// <c>AiModelBuilder</c>.
    /// </summary>
    AutoCompiled = 0,

    /// <summary>
    /// Executes every operation eagerly step-by-step without graph compilation or CUDA Graph capture. Recommended
    /// when stepping through custom layer logic in a debugger.
    /// </summary>
    Eager = 1,

    /// <summary>
    /// Compiles and captures execution graphs like <see cref="AutoCompiled"/>, but throws immediately if graph
    /// compilation fails instead of falling back to eager execution. Useful in regression tests and benchmarks.
    /// </summary>
    StrictCompiled = 2,
}
