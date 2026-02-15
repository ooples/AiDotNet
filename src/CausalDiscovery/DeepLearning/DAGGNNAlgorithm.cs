using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// DAG-GNN — DAG Structure Learning with Graph Neural Networks.
/// </summary>
/// <remarks>
/// <para>
/// DAG-GNN uses a variational autoencoder with graph neural network encoder and decoder
/// to learn a DAG. The encoder generates an adjacency matrix, and the decoder reconstructs
/// data using this graph. The NOTEARS acyclicity constraint is applied to the learned adjacency.
/// </para>
/// <para>
/// <b>For Beginners:</b> DAG-GNN trains a special neural network (GNN) to simultaneously
/// figure out the graph structure AND generate data that matches the observed data.
/// The best graph is the one that lets the network most accurately recreate the data.
/// </para>
/// <para>
/// Reference: Yu et al. (2019), "DAG-GNN: DAG Structure Learning with Graph Neural Networks", ICML.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DAGGNNAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "DAG-GNN";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public DAGGNNAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ContinuousOptimization.NOTEARSLinear<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
