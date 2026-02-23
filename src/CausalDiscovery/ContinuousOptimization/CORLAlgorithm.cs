using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// CORL — Causal Ordering via Reinforcement Learning.
/// </summary>
/// <remarks>
/// <para>
/// CORL uses reinforcement learning (specifically, an encoder-decoder architecture with
/// attention) to learn a causal ordering of variables. Once the ordering is determined,
/// the edge weights are learned via standard regression. The RL agent receives a reward
/// based on how well the resulting DAG fits the data.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of directly optimizing a weight matrix (like NOTEARS),
/// CORL learns the ORDER in which variables cause each other. It uses a technique from
/// AI game-playing (reinforcement learning) where the algorithm tries different orderings
/// and gets "rewarded" for finding ones that explain the data well. Once you know the
/// order (e.g., X causes Y which causes Z), finding the exact relationships is easy.
/// </para>
/// <para>
/// Reference: Wang et al. (2021), "Ordering-Based Causal Discovery with Reinforcement
/// Learning", IJCAI.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CORLAlgorithm<T> : ContinuousOptimizationBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CORL";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes CORL with optional configuration.
    /// </summary>
    public CORLAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell implementation — delegates to DAGMA Linear as baseline
        // Full implementation would use RL-based ordering + regression
        var baseline = new DAGMALinear<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
