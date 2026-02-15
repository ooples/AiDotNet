using AiDotNet.Enums;

namespace AiDotNet.CausalDiscovery;

/// <summary>
/// Contains the results of a causal discovery analysis, including the learned graph and convergence metrics.
/// </summary>
/// <remarks>
/// <para>
/// This result object is populated when <c>ConfigureCausalDiscovery()</c> is used on the
/// <c>AiModelBuilder</c>. It contains the discovered causal graph along with algorithm-specific
/// metrics about the optimization process.
/// </para>
/// <para>
/// <b>For Beginners:</b> After running causal discovery, this object tells you:
/// <list type="bullet">
/// <item>The causal graph itself (which variables cause which others)</item>
/// <item>Which algorithm was used</item>
/// <item>How well the algorithm converged (did it find a good solution?)</item>
/// <item>How sparse the graph is (fewer edges = more interpretable)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CausalDiscoveryResult<T>
{
    /// <summary>
    /// Gets the discovered causal graph (DAG).
    /// </summary>
    public CausalGraph<T> Graph { get; }

    /// <summary>
    /// Gets the algorithm that was used for discovery.
    /// </summary>
    public CausalDiscoveryAlgorithmType AlgorithmUsed { get; }

    /// <summary>
    /// Gets the category of the algorithm used.
    /// </summary>
    public CausalDiscoveryCategory Category { get; }

    /// <summary>
    /// Gets the number of iterations the algorithm performed.
    /// </summary>
    /// <remarks>
    /// <para>For augmented Lagrangian methods (NOTEARS), this is the number of outer loop iterations.</para>
    /// </remarks>
    public int Iterations { get; }

    /// <summary>
    /// Gets the final loss/objective value at convergence.
    /// </summary>
    public double FinalLoss { get; }

    /// <summary>
    /// Gets the final acyclicity constraint value h(W).
    /// </summary>
    /// <remarks>
    /// <para>For a perfect DAG, this should be very close to zero (below h_tol).</para>
    /// </remarks>
    public double AcyclicityConstraint { get; }

    /// <summary>
    /// Gets the number of edges in the discovered graph.
    /// </summary>
    public int EdgeCount => Graph.EdgeCount;

    /// <summary>
    /// Gets the graph density (fraction of possible edges present).
    /// </summary>
    public double GraphDensity => Graph.Density;

    /// <summary>
    /// Gets whether the algorithm converged successfully.
    /// </summary>
    public bool Converged { get; }

    /// <summary>
    /// Gets the wall-clock time the algorithm took to run.
    /// </summary>
    public TimeSpan ElapsedTime { get; }

    /// <summary>
    /// Gets optional additional metrics specific to the algorithm used.
    /// </summary>
    public Dictionary<string, double>? AdditionalMetrics { get; }

    /// <summary>
    /// Initializes a new CausalDiscoveryResult with full convergence metrics.
    /// </summary>
    public CausalDiscoveryResult(
        CausalGraph<T> graph,
        CausalDiscoveryAlgorithmType algorithmUsed,
        CausalDiscoveryCategory category,
        int iterations,
        double finalLoss,
        double acyclicityConstraint,
        bool converged,
        TimeSpan elapsedTime,
        Dictionary<string, double>? additionalMetrics = null)
    {
        Guard.NotNull(graph);
        Graph = graph;
        AlgorithmUsed = algorithmUsed;
        Category = category;
        Iterations = iterations;
        FinalLoss = finalLoss;
        AcyclicityConstraint = acyclicityConstraint;
        Converged = converged;
        ElapsedTime = elapsedTime;
        AdditionalMetrics = additionalMetrics;
    }

    /// <summary>
    /// Initializes a new CausalDiscoveryResult from a graph and elapsed time only.
    /// Convergence metrics are not available when the algorithm does not expose them.
    /// </summary>
    internal CausalDiscoveryResult(
        CausalGraph<T> graph,
        CausalDiscoveryAlgorithmType algorithmUsed,
        CausalDiscoveryCategory category,
        TimeSpan elapsedTime)
    {
        Guard.NotNull(graph);
        Graph = graph;
        AlgorithmUsed = algorithmUsed;
        Category = category;
        Converged = graph.IsDAG();
        ElapsedTime = elapsedTime;
    }
}
