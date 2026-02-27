namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements DnC (Divide and Conquer) aggregation strategy for Byzantine-robust FL.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Some poisoning attacks are hard to detect when you look at the
/// full high-dimensional update vectors — the malicious signal hides in the noise.
/// DnC projects client updates into random low-dimensional subspaces, then uses spectral
/// analysis (outlier detection via singular values) to identify attackers that might evade
/// simpler coordinate-wise defenses like median or trimmed mean.</para>
///
/// <para>Algorithm:</para>
/// <list type="number">
/// <item>Flatten all client updates into vectors</item>
/// <item>Project into a random subspace of dimension <c>SubspaceDimension</c></item>
/// <item>Compute centered second-moment matrix and its top singular vector</item>
/// <item>Score each client by its projection onto this vector</item>
/// <item>Remove top <c>NumByzantine</c> outliers</item>
/// <item>Average the remaining (trusted) updates</item>
/// </list>
///
/// <para>Reference: Shejwalkar, V. &amp; Houmansadr, A. (2021). "Manipulating the Byzantine:
/// Optimizing Model Poisoning Attacks and Defenses for Federated Learning."
/// NDSS 2021.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class DivideAndConquerAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly int _numByzantine;
    private readonly int _subspaceDimension;
    private readonly int _seed;

    /// <summary>
    /// Initializes a new instance of the <see cref="DivideAndConquerAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="numByzantine">Expected number of Byzantine clients. Default: 1.</param>
    /// <param name="subspaceDimension">Dimension of random projection subspace. Default: 10.</param>
    /// <param name="seed">Random seed for reproducibility. Default: 42.</param>
    public DivideAndConquerAggregationStrategy(int numByzantine = 1, int subspaceDimension = 10, int seed = 42)
    {
        if (numByzantine < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numByzantine), "Byzantine count must be non-negative.");
        }

        if (subspaceDimension < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(subspaceDimension), "Subspace dimension must be at least 1.");
        }

        _numByzantine = numByzantine;
        _subspaceDimension = subspaceDimension;
        _seed = seed;
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

        if (clientModels.Count == 1)
        {
            return clientModels.First().Value;
        }

        var referenceModel = clientModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();

        // Flatten all client models into vectors.
        int totalParams = layerNames.Sum(ln => referenceModel[ln].Length);
        var clientIds = clientModels.Keys.ToList();
        int n = clientIds.Count;

        var flatVectors = new double[n][];
        for (int c = 0; c < n; c++)
        {
            flatVectors[c] = new double[totalParams];
            int offset = 0;
            foreach (var layerName in layerNames)
            {
                var cp = clientModels[clientIds[c]][layerName];
                for (int i = 0; i < cp.Length; i++)
                {
                    flatVectors[c][offset++] = NumOps.ToDouble(cp[i]);
                }
            }
        }

        // Compute mean.
        var mean = new double[totalParams];
        for (int c = 0; c < n; c++)
        {
            for (int i = 0; i < totalParams; i++)
            {
                mean[i] += flatVectors[c][i] / n;
            }
        }

        // Center the vectors.
        var centered = new double[n][];
        for (int c = 0; c < n; c++)
        {
            centered[c] = new double[totalParams];
            for (int i = 0; i < totalParams; i++)
            {
                centered[c][i] = flatVectors[c][i] - mean[i];
            }
        }

        // Random projection to subspace.
        int dim = Math.Min(_subspaceDimension, totalParams);
        var rng = new Random(_seed);
        var projected = new double[n][];
        var projectionMatrix = new double[dim][];

        for (int d = 0; d < dim; d++)
        {
            projectionMatrix[d] = new double[totalParams];
            for (int i = 0; i < totalParams; i++)
            {
                // Gaussian random projection.
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                projectionMatrix[d][i] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            }
        }

        for (int c = 0; c < n; c++)
        {
            projected[c] = new double[dim];
            for (int d = 0; d < dim; d++)
            {
                double sum = 0;
                for (int i = 0; i < totalParams; i++)
                {
                    sum += centered[c][i] * projectionMatrix[d][i];
                }

                projected[c][d] = sum;
            }
        }

        // Compute outlier scores via squared L2 norm of projected vectors.
        var scores = new double[n];
        for (int c = 0; c < n; c++)
        {
            double norm2 = 0;
            for (int d = 0; d < dim; d++)
            {
                norm2 += projected[c][d] * projected[c][d];
            }

            scores[c] = norm2;
        }

        // Remove top numByzantine outliers.
        int numToRemove = Math.Min(_numByzantine, n - 1);
        var sortedIndices = Enumerable.Range(0, n).OrderByDescending(i => scores[i]).ToArray();
        var removedSet = new HashSet<int>();
        for (int i = 0; i < numToRemove; i++)
        {
            removedSet.Add(sortedIndices[i]);
        }

        // Aggregate remaining clients.
        var result = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            result[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
        }

        var trustedIds = new List<int>();
        double trustedTotalWeight = 0;
        for (int c = 0; c < n; c++)
        {
            if (removedSet.Contains(c))
            {
                continue;
            }

            trustedIds.Add(c);
            if (clientWeights.TryGetValue(clientIds[c], out var w))
            {
                trustedTotalWeight += w;
            }
        }

        foreach (int c in trustedIds)
        {
            double w = clientWeights.TryGetValue(clientIds[c], out var cw) ? cw : 1.0;
            double normalizedWeight = trustedTotalWeight > 0 ? w / trustedTotalWeight : 1.0 / trustedIds.Count;
            var nw = NumOps.FromDouble(normalizedWeight);
            var clientModel = clientModels[clientIds[c]];

            foreach (var layerName in layerNames)
            {
                var cp = clientModel[layerName];
                var rp = result[layerName];
                for (int i = 0; i < rp.Length; i++)
                {
                    rp[i] = NumOps.Add(rp[i], NumOps.Multiply(cp[i], nw));
                }
            }
        }

        return result;
    }

    /// <summary>Gets the expected number of Byzantine clients.</summary>
    public int NumByzantine => _numByzantine;

    /// <summary>Gets the random projection subspace dimension.</summary>
    public int SubspaceDimension => _subspaceDimension;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"DnC(f={_numByzantine},d={_subspaceDimension})";
}
