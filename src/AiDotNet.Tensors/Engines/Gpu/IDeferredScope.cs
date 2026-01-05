using AiDotNet.Tensors.Engines.Gpu.Graph;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Interface for a deferred execution scope that records operations
/// and executes them as an optimized graph.
/// </summary>
/// <remarks>
/// <para><b>Phase 3: Deferred Execution API</b></para>
/// <para>
/// A deferred scope records all GPU operations instead of executing them immediately.
/// When the scope is completed (via Execute or Dispose), the recorded operations
/// are compiled into an optimized execution graph and executed.
/// </para>
/// <para><b>Example Usage:</b></para>
/// <code>
/// using (var scope = backend.BeginDeferredScope())
/// {
///     // Operations are RECORDED, not executed
///     var c1 = backend.GemmBiasRelu(a1, b1, bias1, M, N, K);
///     var c2 = backend.GemmBiasRelu(a2, b2, bias2, M, N, K);
///     var result = backend.Add(c1, c2, output, size);
///
///     // Graph is compiled, optimized, and executed here
///     await scope.ExecuteAsync();
/// }
/// </code>
/// </remarks>
public interface IDeferredScope : IDisposable
{
    /// <summary>
    /// Gets the execution graph builder for recording operations.
    /// </summary>
    ExecutionGraphBuilder GraphBuilder { get; }

    /// <summary>
    /// Gets the execution options for this scope.
    /// </summary>
    GpuExecutionOptions Options { get; }

    /// <summary>
    /// Gets whether this scope has been executed.
    /// </summary>
    bool IsExecuted { get; }

    /// <summary>
    /// Gets the number of operations recorded in this scope.
    /// </summary>
    int OperationCount { get; }

    /// <summary>
    /// Compiles and executes the recorded operations synchronously.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The execution graph is compiled with optimization passes (fusion, stream assignment, etc.)
    /// and then executed. After execution, the scope cannot be reused.
    /// </para>
    /// </remarks>
    void Execute();

    /// <summary>
    /// Compiles and executes the recorded operations asynchronously.
    /// </summary>
    /// <param name="cancellationToken">Token to cancel execution.</param>
    /// <returns>A task representing the asynchronous execution.</returns>
    /// <remarks>
    /// <para>
    /// The execution graph is compiled with optimization passes and executed asynchronously.
    /// GPU operations run in parallel where possible.
    /// </para>
    /// </remarks>
    Task ExecuteAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the compiled execution graph without executing it.
    /// </summary>
    /// <returns>The compiled and optimized execution graph.</returns>
    /// <remarks>
    /// <para>
    /// Useful for inspecting the optimization results or caching the compiled graph.
    /// Call <see cref="Execute"/> or <see cref="ExecuteAsync"/> to run the graph.
    /// </para>
    /// </remarks>
    ExecutionGraph Compile();

    /// <summary>
    /// Gets execution statistics after the scope has been executed.
    /// </summary>
    /// <returns>Statistics about the execution, or null if not yet executed.</returns>
    DeferredScopeStatistics? GetStatistics();
}

/// <summary>
/// Statistics collected from a deferred scope execution.
/// </summary>
public sealed class DeferredScopeStatistics
{
    /// <summary>
    /// Gets or sets the number of operations recorded.
    /// </summary>
    public int OperationsRecorded { get; set; }

    /// <summary>
    /// Gets or sets the number of nodes after compilation.
    /// </summary>
    public int NodesAfterCompilation { get; set; }

    /// <summary>
    /// Gets or sets the number of fused operations.
    /// </summary>
    public int FusedOperations { get; set; }

    /// <summary>
    /// Gets or sets the number of eliminated operations (dead code).
    /// </summary>
    public int EliminatedOperations { get; set; }

    /// <summary>
    /// Gets or sets the time spent compiling the graph.
    /// </summary>
    public TimeSpan CompilationTime { get; set; }

    /// <summary>
    /// Gets or sets the time spent executing the graph.
    /// </summary>
    public TimeSpan ExecutionTime { get; set; }

    /// <summary>
    /// Gets or sets the total time from recording to completion.
    /// </summary>
    public TimeSpan TotalTime { get; set; }

    /// <summary>
    /// Gets the estimated speedup from optimization.
    /// </summary>
    public double EstimatedSpeedup =>
        NodesAfterCompilation > 0 && OperationsRecorded > 0
            ? (double)OperationsRecorded / NodesAfterCompilation
            : 1.0;

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"DeferredScope: {OperationsRecorded} ops -> {NodesAfterCompilation} nodes " +
               $"({FusedOperations} fused, {EliminatedOperations} eliminated) " +
               $"Compile: {CompilationTime.TotalMilliseconds:F1}ms, " +
               $"Execute: {ExecutionTime.TotalMilliseconds:F1}ms, " +
               $"Speedup: {EstimatedSpeedup:F2}x";
    }
}
