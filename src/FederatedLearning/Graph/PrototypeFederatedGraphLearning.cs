using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Graph;

/// <summary>
/// Prototype-based federated graph learning: clients share class prototypes instead of full model parameters.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Sharing full GNN model parameters raises two concerns: (1) model
/// parameters can leak information about the training graph, and (2) different clients may have
/// very different graph structures that benefit from different model architectures. Prototype-based
/// FGL addresses both:</para>
///
/// <list type="bullet">
/// <item><description><b>Prototypes:</b> Instead of sharing the full model, each client computes "prototypes" —
/// representative embedding vectors for each class in their data.</description></item>
/// <item><description><b>Server aggregation:</b> The server averages prototypes across clients for each class,
/// creating global prototypes.</description></item>
/// <item><description><b>Local training:</b> Clients use global prototypes as regularization targets, pulling their
/// local embeddings toward the global consensus while preserving local structure.</description></item>
/// </list>
///
/// <para><b>Benefits:</b> Prototypes are much smaller than full models (K prototypes of dimension D vs.
/// millions of parameters), more robust to topology heterogeneity, and reveal less about graph structure.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PrototypeFederatedGraphLearning<T> : FederatedLearningComponentBase<T>
{
    private readonly FederatedGraphOptions _options;
    private readonly int _prototypeDim;
    private readonly int _numClasses;
    private readonly Dictionary<int, Dictionary<int, Tensor<T>>> _clientPrototypes = new();
    private Dictionary<int, Tensor<T>> _globalPrototypes = new();

    /// <summary>
    /// Initializes a new instance of <see cref="PrototypeFederatedGraphLearning{T}"/>.
    /// </summary>
    /// <param name="options">Graph FL configuration.</param>
    /// <param name="prototypeDim">Dimensionality of prototype vectors.</param>
    /// <param name="numClasses">Number of classes in the task.</param>
    public PrototypeFederatedGraphLearning(
        FederatedGraphOptions options,
        int prototypeDim,
        int numClasses)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _prototypeDim = prototypeDim;
        _numClasses = numClasses;

        InitializeGlobalPrototypes();
    }

    /// <summary>
    /// Gets the current global prototypes.
    /// </summary>
    public IReadOnlyDictionary<int, Tensor<T>> GlobalPrototypes => _globalPrototypes;

    /// <summary>
    /// Registers a client's class prototypes for the current round.
    /// </summary>
    /// <param name="clientId">Client identifier.</param>
    /// <param name="prototypes">Class-to-prototype mapping from the client.</param>
    public void RegisterClientPrototypes(int clientId, Dictionary<int, Tensor<T>> prototypes)
    {
        if (prototypes is null) throw new ArgumentNullException(nameof(prototypes));
        _clientPrototypes[clientId] = prototypes;
    }

    /// <summary>
    /// Computes a client's class prototypes from node embeddings and labels.
    /// </summary>
    /// <param name="nodeEmbeddings">Node embedding matrix (flattened [numNodes * embDim]).</param>
    /// <param name="labels">Node labels (integer class IDs).</param>
    /// <param name="embeddingDim">Dimensionality of node embeddings.</param>
    /// <returns>Class-to-prototype mapping.</returns>
    public Dictionary<int, Tensor<T>> ComputePrototypes(
        Tensor<T> nodeEmbeddings, Tensor<T> labels, int embeddingDim)
    {
        if (nodeEmbeddings is null) throw new ArgumentNullException(nameof(nodeEmbeddings));
        if (labels is null) throw new ArgumentNullException(nameof(labels));

        int numNodes = labels.Shape[0];
        var prototypes = new Dictionary<int, Tensor<T>>();
        var classCounts = new Dictionary<int, int>();
        var classSums = new Dictionary<int, double[]>();

        // Accumulate embeddings per class
        for (int n = 0; n < numNodes; n++)
        {
            int classId = (int)NumOps.ToDouble(labels[n]);

            if (!classSums.ContainsKey(classId))
            {
                classSums[classId] = new double[embeddingDim];
                classCounts[classId] = 0;
            }

            classCounts[classId]++;

            for (int d = 0; d < embeddingDim; d++)
            {
                int idx = n * embeddingDim + d;
                if (idx < nodeEmbeddings.Shape[0])
                {
                    classSums[classId][d] += NumOps.ToDouble(nodeEmbeddings[idx]);
                }
            }
        }

        // Compute mean prototype per class
        foreach (var kvp in classSums)
        {
            int classId = kvp.Key;
            double[] sum = kvp.Value;
            int count = classCounts[classId];

            var proto = new Tensor<T>(new[] { embeddingDim });
            for (int d = 0; d < embeddingDim; d++)
            {
                proto[d] = NumOps.FromDouble(count > 0 ? sum[d] / count : 0);
            }

            prototypes[classId] = proto;
        }

        return prototypes;
    }

    /// <summary>
    /// Aggregates prototypes from all registered clients into global prototypes.
    /// </summary>
    /// <returns>Updated global prototypes.</returns>
    public Dictionary<int, Tensor<T>> AggregatePrototypes()
    {
        if (_clientPrototypes.Count == 0)
        {
            return _globalPrototypes;
        }

        var newGlobal = new Dictionary<int, Tensor<T>>();

        for (int c = 0; c < _numClasses; c++)
        {
            var sum = new double[_prototypeDim];
            int clientsWithClass = 0;

            foreach (var clientProtos in _clientPrototypes.Values)
            {
                if (clientProtos.ContainsKey(c))
                {
                    var proto = clientProtos[c];
                    clientsWithClass++;

                    for (int d = 0; d < _prototypeDim && d < proto.Shape[0]; d++)
                    {
                        sum[d] += NumOps.ToDouble(proto[d]);
                    }
                }
            }

            var globalProto = new Tensor<T>(new[] { _prototypeDim });
            for (int d = 0; d < _prototypeDim; d++)
            {
                globalProto[d] = NumOps.FromDouble(clientsWithClass > 0 ? sum[d] / clientsWithClass : 0);
            }

            newGlobal[c] = globalProto;
        }

        _globalPrototypes = newGlobal;
        _clientPrototypes.Clear(); // Reset for next round

        return _globalPrototypes;
    }

    /// <summary>
    /// Computes prototype regularization loss: pull client prototypes toward global prototypes.
    /// </summary>
    /// <param name="clientPrototypes">Client's local prototypes.</param>
    /// <param name="lambda">Regularization strength. Default 0.1.</param>
    /// <returns>Regularization loss value.</returns>
    public double ComputePrototypeLoss(Dictionary<int, Tensor<T>> clientPrototypes, double lambda = 0.1)
    {
        double loss = 0;

        foreach (var kvp in clientPrototypes)
        {
            int classId = kvp.Key;
            var localProto = kvp.Value;

            if (_globalPrototypes.ContainsKey(classId))
            {
                var globalProto = _globalPrototypes[classId];
                int dim = Math.Min(localProto.Shape[0], globalProto.Shape[0]);

                // L2 distance between local and global prototype
                double distSquared = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = NumOps.ToDouble(localProto[d]) - NumOps.ToDouble(globalProto[d]);
                    distSquared += diff * diff;
                }

                loss += lambda * distSquared;
            }
        }

        return loss;
    }

    private void InitializeGlobalPrototypes()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();

        for (int c = 0; c < _numClasses; c++)
        {
            var proto = new Tensor<T>(new[] { _prototypeDim });
            double scale = Math.Sqrt(2.0 / _prototypeDim);

            for (int d = 0; d < _prototypeDim; d++)
            {
                proto[d] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
            }

            _globalPrototypes[c] = proto;
        }
    }
}
