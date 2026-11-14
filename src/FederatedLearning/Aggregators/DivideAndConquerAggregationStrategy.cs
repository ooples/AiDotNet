using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements DnC (Divide and Conquer) aggregation strategy for Byzantine-robust FL.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Some poisoning attacks are hard to detect when you look at the
/// full high-dimensional update vectors — the malicious signal hides in the noise.
/// DnC looks at random subsets of the update coordinates, then uses spectral
/// analysis (top singular vector projection) to identify attackers that might evade
/// simpler coordinate-wise defenses like median or trimmed mean.</para>
///
/// <para>Algorithm:</para>
/// <list type="number">
/// <item>Flatten all client updates into vectors</item>
/// <item>Sample a random sorted subset of <c>SubspaceDimension</c> coordinates and center the sub-sampled updates</item>
/// <item>Compute the top right singular vector of the centered matrix</item>
/// <item>Score each client by its squared projection onto this singular vector</item>
/// <item>Keep the n - floor(c * NumByzantine) lowest-scoring clients; repeat for <c>Iterations</c> and intersect</item>
/// <item>Average the remaining (trusted) updates</item>
/// </list>
///
/// <para>Reference: Shejwalkar, V. &amp; Houmansadr, A. (2021). "Manipulating the Byzantine:
/// Optimizing Model Poisoning Attacks and Defenses for Federated Learning."
/// NDSS 2021, Algorithm 2.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class DivideAndConquerAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly int _numByzantine;
    private readonly int _subspaceDimension;
    private readonly int _seed;
    private readonly int _iterations;
    private readonly double _filterFraction;
    private int _roundCounter;

    /// <summary>
    /// Initializes a new instance of the <see cref="DivideAndConquerAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="numByzantine">Expected number of Byzantine clients m. Default: 1.</param>
    /// <param name="subspaceDimension">Number b of randomly sub-sampled coordinates per iteration. Default: 10000 (paper).</param>
    /// <param name="seed">Random seed for reproducibility. Default: 42.</param>
    /// <param name="iterations">Number of filtering iterations (niters). Default: 1 (paper).</param>
    /// <param name="filterFraction">Filtering fraction c; each iteration removes floor(c * m) clients. Default: 1.0 (paper).</param>
    public DivideAndConquerAggregationStrategy(
        int numByzantine = 1,
        int subspaceDimension = 10000,
        int seed = 42,
        int iterations = 1,
        double filterFraction = 1.0)
    {
        if (numByzantine < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numByzantine), "Byzantine count must be non-negative.");
        }

        if (subspaceDimension < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(subspaceDimension), "Subspace dimension must be at least 1.");
        }

        if (iterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(iterations), "Iterations must be at least 1.");
        }

        if (double.IsNaN(filterFraction) || filterFraction <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(filterFraction), "Filter fraction must be positive.");
        }

        _numByzantine = numByzantine;
        _subspaceDimension = subspaceDimension;
        _seed = seed;
        _iterations = iterations;
        _filterFraction = filterFraction;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        if (clientModels == null || clientModels.Count == 0)
        {
            throw new ArgumentException("Client models cannot be null or empty.", nameof(clientModels));
        }

        Guard.NotNull(clientWeights);

        if (clientModels.Count == 1)
        {
            var single = clientModels.First().Value;
            return single.ToDictionary(kv => kv.Key, kv => (T[])kv.Value.Clone());
        }

        var referenceModel = clientModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();

        // Validate all clients have matching layer structure before flattening.
        foreach (var (clientId, model) in clientModels)
        {
            if (model == null)
            {
                throw new ArgumentException($"Client {clientId} has null model.", nameof(clientModels));
            }

            foreach (var layerName in layerNames)
            {
                if (!model.TryGetValue(layerName, out var layer) || layer == null)
                {
                    throw new ArgumentException(
                        $"Client {clientId} missing layer '{layerName}'.", nameof(clientModels));
                }

                if (layer.Length != referenceModel[layerName].Length)
                {
                    throw new ArgumentException(
                        $"Client {clientId} layer '{layerName}' length mismatch: {layer.Length} != {referenceModel[layerName].Length}.",
                        nameof(clientModels));
                }
            }
        }

        // Flatten all client models into one [n, d] matrix of updates.
        int totalParams = layerNames.Sum(ln => referenceModel[ln].Length);
        var clientIds = clientModels.Keys.ToList();
        int n = clientIds.Count;
        var updates = Engine.TensorStack(
            clientIds.Select(id => FlattenLayers(clientModels[id], layerNames, totalParams).AsTensor()).ToArray(), 0);

        // Use round-varying seed for fresh randomness each aggregation.
        var rng = RandomHelper.CreateSeededRandom(_seed + _roundCounter++);

        // Algorithm 2: each iteration keeps the n - floor(c * m) clients with the lowest outlier scores on a
        // random sorted subset of b coordinates; the trusted set is the intersection over iterations.
        int removeCount = Math.Min((int)Math.Floor(_filterFraction * _numByzantine), n - 1);
        int subDim = Math.Min(_subspaceDimension, totalParams);
        var goodSet = new HashSet<int>(Enumerable.Range(0, n));

        for (int iter = 0; iter < _iterations && removeCount > 0 && subDim > 0; iter++)
        {
            var coordinates = SampleSortedCoordinates(rng, totalParams, subDim);
            var sub = Engine.TensorIndexSelect(updates, Vector<int>.Wrap(coordinates).AsTensor(), 1);

            var mu = Engine.ReduceMean(sub, new[] { 0 }, true);
            var centered = Engine.TensorSubtract(sub, Engine.TensorBroadcastTo(mu, new[] { n, subDim }));

            // Top right singular vector v of the centered sub-sampled updates.
            var (_, _, vh) = AiDotNet.Helpers.LinalgDispatch<T>.Svd(centered, false);
            var topRight = Engine.TensorSlice(vh, new[] { 0, 0 }, new[] { 1, subDim }).Reshape(subDim, 1);

            // s_i = ((g_i - mu) . v)^2
            var scores = Engine.TensorSquare(Engine.TensorMatMul(centered, topRight)).ToVector().ToArray();

            // LINQ OrderBy is stable: equal scores keep the lower client index first.
            var kept = Enumerable.Range(0, n).OrderBy(i => scores[i]).Take(n - removeCount);
            goodSet.IntersectWith(kept);
        }

        var activeIndices = goodSet.OrderBy(i => i).ToList();
        if (activeIndices.Count == 0)
        {
            throw new InvalidOperationException(
                "DnC filtering removed every client: the per-iteration trusted sets do not intersect.");
        }

        // Aggregate remaining (trusted) clients.
        var result = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            result[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
        }

        // Validate and accumulate weights, rejecting non-finite values.
        double trustedTotalWeight = 0;
        foreach (int c in activeIndices)
        {
            double w = clientWeights.TryGetValue(clientIds[c], out var cw) ? cw : 1.0;
            if (double.IsNaN(w) || double.IsInfinity(w) || w < 0)
            {
                throw new ArgumentException(
                    $"Client {clientIds[c]} has invalid weight: {w}.", nameof(clientWeights));
            }

            trustedTotalWeight += w;
        }

        foreach (int c in activeIndices)
        {
            double w = clientWeights.TryGetValue(clientIds[c], out var cw) ? cw : 1.0;
            double normalizedWeight = trustedTotalWeight > 0 ? w / trustedTotalWeight : 1.0 / activeIndices.Count;
            var nw = NumOps.FromDouble(normalizedWeight);
            var clientModel = clientModels[clientIds[c]];

            foreach (var layerName in layerNames)
            {
                AddScaledInto(result[layerName], clientModel[layerName], nw);
            }
        }

        return result;
    }

    /// <summary>
    /// Draws <paramref name="count"/> distinct coordinates out of <paramref name="dimension"/> uniformly without
    /// replacement (partial Fisher-Yates) and returns them sorted ascending.
    /// </summary>
    private static int[] SampleSortedCoordinates(Random rng, int dimension, int count)
    {
        var pool = Enumerable.Range(0, dimension).ToArray();
        for (int i = 0; i < count; i++)
        {
            int j = i + rng.Next(dimension - i);
            (pool[i], pool[j]) = (pool[j], pool[i]);
        }

        var chosen = new int[count];
        Array.Copy(pool, chosen, count);
        Array.Sort(chosen);
        return chosen;
    }

    /// <summary>Gets the expected number of Byzantine clients.</summary>
    public int NumByzantine => _numByzantine;

    /// <summary>Gets the number b of sub-sampled coordinates per iteration.</summary>
    public int SubspaceDimension => _subspaceDimension;

    /// <summary>Gets the number of filtering iterations (niters).</summary>
    public int Iterations => _iterations;

    /// <summary>Gets the filtering fraction c.</summary>
    public double FilterFraction => _filterFraction;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"DnC(f={_numByzantine},d={_subspaceDimension})";
}
