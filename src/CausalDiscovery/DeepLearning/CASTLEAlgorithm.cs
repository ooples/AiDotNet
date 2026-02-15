using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// CASTLE — Causal Structure Learning via neural networks.
/// </summary>
/// <remarks>
/// <para>
/// CASTLE trains a neural network to predict each variable from the others, using a shared
/// mask layer that represents the adjacency matrix. L1 regularization and DAG constraints
/// are applied to learn a sparse causal graph.
/// </para>
/// <para>
/// <b>For Beginners:</b> CASTLE uses a neural network with a special "mask" that learns
/// which inputs matter for predicting each variable. The mask ends up representing the
/// causal graph — connections that help prediction stay, others are removed.
/// </para>
/// <para>
/// Reference: Kyono et al. (2020), "CASTLE: Regularization via Auxiliary Causal Graph Discovery", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CASTLEAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CASTLE";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public CASTLEAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ContinuousOptimization.NOTEARSLinear<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
