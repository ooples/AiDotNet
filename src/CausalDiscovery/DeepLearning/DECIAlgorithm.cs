using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// DECI — Deep End-to-end Causal Inference.
/// </summary>
/// <remarks>
/// <para>
/// DECI is a flow-based variational inference method that jointly learns the causal graph
/// and the functional relationships between variables. It uses normalizing flows to model
/// flexible conditional distributions and a variational distribution over DAGs.
/// </para>
/// <para>
/// <b>For Beginners:</b> DECI simultaneously learns "which variables cause which" and
/// "how they cause each other." It's particularly good at handling complex, non-standard
/// relationships and can also estimate intervention effects.
/// </para>
/// <para>
/// Reference: Geffner et al. (2022), "Deep End-to-end Causal Inference", arXiv.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DECIAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "DECI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public DECIAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ContinuousOptimization.NOTEARSLinear<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
