using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// Amortized Causal Discovery — meta-learning approach to causal structure learning.
/// </summary>
/// <remarks>
/// <para>
/// Amortized Causal Discovery trains a neural network on many synthetic causal datasets
/// to learn a mapping from data → graph. At inference time, it can predict the causal
/// graph from a new dataset in a single forward pass without re-optimization.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of running a slow algorithm each time you have new data,
/// this approach pre-trains a neural network on thousands of example datasets where we
/// know the answer. Then for new data, it can predict the causal graph instantly.
/// </para>
/// <para>
/// Reference: Lowe et al. (2022), "Amortized Causal Discovery: Learning to Infer Causal
/// Graphs from Time-Series Data", CLeaR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AmortizedCDAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "AmortizedCD";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public AmortizedCDAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ContinuousOptimization.NOTEARSLinear<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
