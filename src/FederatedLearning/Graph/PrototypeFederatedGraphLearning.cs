using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class PrototypeFederatedGraphLearning<T> : FederatedLearningComponentBase<T>
{
    private readonly FederatedGraphOptions _options;
    private readonly int _prototypeDim;
    private readonly int _numClasses;
    private readonly Dictionary<int, Dictionary<int, Tensor<T>>> _clientPrototypes = new();
    private readonly Dictionary<int, Dictionary<int, int>> _clientClassCounts = new();
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
    /// <remarks>
    /// FedProto (Tan et al. 2022) weights each client's prototype of class c by the number of local samples of
    /// that class, so the counts are part of the registration.
    /// </remarks>
    /// <param name="clientId">Client identifier.</param>
    /// <param name="prototypes">Class-to-prototype mapping from the client.</param>
    /// <param name="classSampleCounts">Number of local samples behind each prototype (one positive count per class).</param>
    public void RegisterClientPrototypes(int clientId, Dictionary<int, Tensor<T>> prototypes, Dictionary<int, int> classSampleCounts)
    {
        if (prototypes is null) throw new ArgumentNullException(nameof(prototypes));
        if (classSampleCounts is null) throw new ArgumentNullException(nameof(classSampleCounts));

        foreach (int classId in prototypes.Keys)
        {
            if (!classSampleCounts.TryGetValue(classId, out int count) || count <= 0)
            {
                throw new ArgumentException(
                    $"Client {clientId} must report a positive sample count for class {classId}.", nameof(classSampleCounts));
            }
        }

        _clientPrototypes[clientId] = prototypes;
        _clientClassCounts[clientId] = new Dictionary<int, int>(classSampleCounts);
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
        return ComputePrototypes(nodeEmbeddings, labels, embeddingDim, out _);
    }

    /// <summary>
    /// Computes a client's class prototypes and the number of nodes behind each one (the weights used by
    /// <see cref="RegisterClientPrototypes"/>).
    /// </summary>
    /// <param name="nodeEmbeddings">Node embedding matrix (flattened [numNodes * embDim]).</param>
    /// <param name="labels">Node labels (integer class IDs).</param>
    /// <param name="embeddingDim">Dimensionality of node embeddings.</param>
    /// <param name="classSampleCounts">Number of nodes of each class.</param>
    /// <returns>Class-to-prototype mapping.</returns>
    public Dictionary<int, Tensor<T>> ComputePrototypes(
        Tensor<T> nodeEmbeddings, Tensor<T> labels, int embeddingDim, out Dictionary<int, int> classSampleCounts)
    {
        if (nodeEmbeddings is null) throw new ArgumentNullException(nameof(nodeEmbeddings));
        if (labels is null) throw new ArgumentNullException(nameof(labels));

        int numNodes = labels.Shape[0];
        var prototypes = new Dictionary<int, Tensor<T>>();
        classSampleCounts = new Dictionary<int, int>();
        if (numNodes == 0 || embeddingDim <= 0)
        {
            return prototypes;
        }

        // Class of every node; classes are numbered in order of first appearance.
        var labelValues = labels.Reshape(new[] { labels.Length }).Cast<double>().ToArray();
        var classSlot = new Dictionary<int, int>();
        var classOrder = new List<int>();
        var classCounts = new List<int>();
        var nodeSlots = new int[numNodes];
        for (int n = 0; n < numNodes; n++)
        {
            int classId = (int)labelValues[n];
            if (!classSlot.TryGetValue(classId, out int slot))
            {
                slot = classOrder.Count;
                classSlot[classId] = slot;
                classOrder.Add(classId);
                classCounts.Add(0);
            }

            classCounts[slot]++;
            nodeSlots[n] = slot;
        }

        // Accumulate embeddings per class (entries past the end of the embedding tensor count as zero).
        var embeddings = FitToLength(nodeEmbeddings, numNodes * embeddingDim).Reshape(new[] { numNodes, embeddingDim });
        var classSums = Engine.TensorIndexAdd(
            new Tensor<T>(new[] { classOrder.Count, embeddingDim }), 0, new Tensor<int>(nodeSlots, new[] { numNodes }), embeddings);

        // Compute mean prototype per class
        var counts = new double[classOrder.Count];
        for (int s = 0; s < counts.Length; s++)
        {
            counts[s] = classCounts[s];
        }

        var means = Engine.TensorDivide(classSums, Engine.TensorBroadcastTo(
            new Tensor<double>(counts, new[] { counts.Length, 1 }).Cast<T>(), new[] { counts.Length, embeddingDim }));
        for (int s = 0; s < classOrder.Count; s++)
        {
            prototypes[classOrder[s]] = Engine.TensorSlice(means, new[] { s, 0 }, new[] { 1, embeddingDim }).Reshape(new[] { embeddingDim });
            classSampleCounts[classOrder[s]] = classCounts[s];
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

        // FedProto: p_c = Σ_i |D_{i,c}| p_{i,c} / Σ_i |D_{i,c}|. Counts are normalized by their class total before
        // scaling, so the running sum stays bounded for narrow T.
        for (int c = 0; c < _numClasses; c++)
        {
            long classTotal = 0;
            foreach (var (clientId, clientProtos) in _clientPrototypes)
            {
                if (clientProtos.ContainsKey(c))
                {
                    classTotal += _clientClassCounts[clientId][c];
                }
            }

            var sum = new Tensor<T>(new[] { _prototypeDim });
            if (classTotal > 0)
            {
                foreach (var (clientId, clientProtos) in _clientPrototypes)
                {
                    if (clientProtos.TryGetValue(c, out var proto))
                    {
                        double share = (double)_clientClassCounts[clientId][c] / classTotal;
                        sum = Engine.TensorAdd(sum, Engine.TensorMultiplyScalar(FitToLength(proto, _prototypeDim), NumOps.FromDouble(share)));
                    }
                }
            }

            newGlobal[c] = sum;
        }

        _globalPrototypes = newGlobal;
        _clientPrototypes.Clear(); // Reset for next round
        _clientClassCounts.Clear();

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

            if (_globalPrototypes.TryGetValue(classId, out var globalProto))
            {
                int dim = Math.Min(localProto.Shape[0], globalProto.Shape[0]);
                if (dim == 0)
                {
                    continue;
                }

                // L2 distance between local and global prototype, as max|diff|^2 * Σ (diff / max|diff|)^2 so the
                // sum of squares cannot overflow narrow T.
                var diff = Engine.TensorSubtract(FitToLength(localProto, dim), FitToLength(globalProto, dim));
                T maxAbs = Engine.TensorMaxValue(Engine.TensorAbs(diff));
                if (!NumOps.GreaterThan(maxAbs, NumOps.Zero))
                {
                    continue;
                }

                double scaledSumSq = NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(Engine.TensorDivideScalar(diff, maxAbs))));
                double maxValue = NumOps.ToDouble(maxAbs);
                loss += lambda * maxValue * maxValue * scaledSumSq;
            }
        }

        return loss;
    }

    /// <summary>
    /// Copies the first <paramref name="length"/> entries of a flattened tensor, zero-padding short ones.
    /// </summary>
    private Tensor<T> FitToLength(Tensor<T> source, int length)
    {
        var flat = source.Reshape(new[] { source.Length });
        int covered = Math.Min(length, source.Length);
        if (covered == length)
        {
            return covered == source.Length ? flat : Engine.TensorSlice(flat, new[] { 0 }, new[] { length });
        }

        return Engine.TensorConcatenate(new[] { flat, new Tensor<T>(new[] { length - covered }) }, 0);
    }

    private void InitializeGlobalPrototypes()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        double scale = Math.Sqrt(2.0 / _prototypeDim);

        for (int c = 0; c < _numClasses; c++)
        {
            var values = new double[_prototypeDim];
            for (int d = 0; d < _prototypeDim; d++)
            {
                values[d] = (rng.NextDouble() * 2 - 1) * scale;
            }

            _globalPrototypes[c] = new Tensor<double>(values, new[] { _prototypeDim }).Cast<T>();
        }
    }
}
