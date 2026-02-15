using AiDotNet.CausalDiscovery;
using AiDotNet.Models.Options;

namespace AiDotNet;

/// <summary>
/// Causal discovery extensions for AiModelBuilder.
/// </summary>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    private CausalDiscoveryOptions? _causalDiscoveryOptions;

    /// <summary>
    /// Configures causal structure discovery to learn a DAG from the training data.
    /// </summary>
    /// <param name="configure">Action to configure causal discovery options.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// When configured, the builder will run a causal discovery algorithm during training
    /// to learn the causal structure (Directed Acyclic Graph) between variables. The result
    /// is available on <c>AiModelResult.CausalDiscoveryResult</c>.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This adds causal discovery as an analysis step when training
    /// your model. After training, you can inspect which variables cause which others.
    ///
    /// Example — discover causal structure with NOTEARS:
    /// <code>
    /// var result = new AiModelBuilder&lt;double, double[], double&gt;()
    ///     .ConfigureCausalDiscovery(options => {
    ///         options.Algorithm = CausalDiscoveryAlgorithmType.NOTEARSLinear;
    ///         options.SparsityPenalty = 0.1;
    ///         options.EdgeThreshold = 0.3;
    ///         options.FeatureNames = new[] { "age", "income", "education", "spending" };
    ///     })
    ///     .Build(inputData, outputData);
    ///
    /// // Access the discovered causal graph
    /// var dag = result.CausalDiscoveryResult.Graph;
    /// var parentsOfSpending = dag.GetParents("spending");
    /// var edges = dag.GetNamedEdges(minAbsWeight: 0.1);
    /// </code>
    ///
    /// Example — auto-detect algorithm:
    /// <code>
    /// var result = new AiModelBuilder&lt;double, double[], double&gt;()
    ///     .ConfigureCausalDiscovery()  // uses sensible defaults
    ///     .Build(inputData, outputData);
    /// </code>
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureCausalDiscovery(
        Action<CausalDiscoveryOptions>? configure = null)
    {
        _causalDiscoveryOptions = new CausalDiscoveryOptions();
        configure?.Invoke(_causalDiscoveryOptions);
        return this;
    }

    /// <summary>
    /// Configures causal structure discovery with a pre-built options object.
    /// </summary>
    /// <param name="options">The causal discovery options. If null, default options are used.</param>
    /// <returns>This builder instance for method chaining.</returns>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureCausalDiscovery(
        CausalDiscoveryOptions? options)
    {
        _causalDiscoveryOptions = options ?? new CausalDiscoveryOptions();
        return this;
    }
}
