using AiDotNet.CausalDiscovery;
using Newtonsoft.Json;

namespace AiDotNet.Models.Results;

/// <summary>
/// Causal discovery extensions for AiModelResult.
/// </summary>
public partial class AiModelResult<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the causal discovery result, if causal discovery was configured.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains the discovered causal graph (DAG), algorithm metadata, convergence info,
    /// and graph metrics. Only populated when <c>ConfigureCausalDiscovery()</c> was called
    /// on the builder.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After running causal discovery, this property contains the
    /// results. You can access:
    /// <list type="bullet">
    /// <item><c>CausalDiscoveryResult.Graph</c> — the causal graph itself</item>
    /// <item><c>CausalDiscoveryResult.Converged</c> — whether the algorithm found a good solution</item>
    /// <item><c>CausalDiscoveryResult.EdgeCount</c> — number of causal edges discovered</item>
    /// </list>
    /// </para>
    /// </remarks>
    [JsonIgnore]
    public CausalDiscoveryResult<T>? CausalDiscoveryResult { get; internal set; }

    /// <summary>
    /// Sets the causal discovery result. Used internally by the builder.
    /// </summary>
    internal void SetCausalDiscoveryResult(CausalDiscoveryResult<T>? result)
    {
        CausalDiscoveryResult = result;
    }
}
