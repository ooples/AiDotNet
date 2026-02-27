namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements DnC (Divide and Conquer) aggregation strategy for Byzantine-robust FL.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Some poisoning attacks are hard to detect when you look at the
/// full high-dimensional update vectors — the malicious signal hides in the noise.
/// DnC projects client updates into random low-dimensional subspaces, then uses spectral
/// analysis (top singular vector projection) to identify attackers that might evade
/// simpler coordinate-wise defenses like median or trimmed mean.</para>
///
/// <para>Algorithm:</para>
/// <list type="number">
/// <item>Flatten all client updates into vectors</item>
/// <item>Project into a random orthogonal subspace of dimension <c>SubspaceDimension</c></item>
/// <item>Compute centered second-moment matrix and its top right singular vector via power iteration</item>
/// <item>Score each client by squared projection onto this singular vector</item>
/// <item>Iteratively remove the top outlier, recompute, repeat for <c>NumByzantine</c> rounds</item>
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
    private int _roundCounter;

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
            var single = clientModels.First().Value;
            return single.ToDictionary(kv => kv.Key, kv => (T[])kv.Value.Clone());
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

        // Use round-varying seed for fresh randomness each aggregation.
        var rng = new Random(_seed + _roundCounter++);

        // Generate random orthogonal projection matrix via QR decomposition of Gaussian matrix.
        int dim = Math.Min(_subspaceDimension, totalParams);
        var projectionMatrix = GenerateOrthogonalProjection(rng, totalParams, dim);

        // Iterative spectral filtering: remove one outlier per iteration.
        var activeIndices = new List<int>(Enumerable.Range(0, n));
        int numToRemove = Math.Min(_numByzantine, n - 1);

        for (int iter = 0; iter < numToRemove; iter++)
        {
            if (activeIndices.Count <= 1)
            {
                break;
            }

            // Compute mean of active clients.
            var mean = new double[totalParams];
            foreach (int c in activeIndices)
            {
                for (int i = 0; i < totalParams; i++)
                {
                    mean[i] += flatVectors[c][i] / activeIndices.Count;
                }
            }

            // Center active vectors.
            var centered = new double[activeIndices.Count][];
            for (int idx = 0; idx < activeIndices.Count; idx++)
            {
                int c = activeIndices[idx];
                centered[idx] = new double[totalParams];
                for (int i = 0; i < totalParams; i++)
                {
                    centered[idx][i] = flatVectors[c][i] - mean[i];
                }
            }

            // Project into random subspace.
            var projected = new double[activeIndices.Count][];
            for (int idx = 0; idx < activeIndices.Count; idx++)
            {
                projected[idx] = new double[dim];
                for (int d = 0; d < dim; d++)
                {
                    double sum = 0;
                    for (int i = 0; i < totalParams; i++)
                    {
                        sum += centered[idx][i] * projectionMatrix[d][i];
                    }

                    projected[idx][d] = sum;
                }
            }

            // Find top right singular vector of projected matrix via power iteration.
            var topSingularVector = ComputeTopRightSingularVector(projected, activeIndices.Count, dim, rng);

            // Score each client by squared projection onto the top singular vector.
            double maxScore = double.NegativeInfinity;
            int maxIdx = 0;

            for (int idx = 0; idx < activeIndices.Count; idx++)
            {
                double projection = 0;
                for (int d = 0; d < dim; d++)
                {
                    projection += projected[idx][d] * topSingularVector[d];
                }

                double score = projection * projection;
                if (score > maxScore)
                {
                    maxScore = score;
                    maxIdx = idx;
                }
            }

            // Remove the top outlier.
            activeIndices.RemoveAt(maxIdx);
        }

        // Aggregate remaining (trusted) clients.
        var result = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            result[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
        }

        double trustedTotalWeight = 0;
        foreach (int c in activeIndices)
        {
            double w = clientWeights.TryGetValue(clientIds[c], out var cw) ? cw : 1.0;
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

    /// <summary>
    /// Generates an orthogonal projection matrix via modified Gram-Schmidt on Gaussian vectors.
    /// </summary>
    private static double[][] GenerateOrthogonalProjection(Random rng, int totalParams, int dim)
    {
        var basis = new double[dim][];

        for (int d = 0; d < dim; d++)
        {
            // Generate random Gaussian vector.
            basis[d] = new double[totalParams];
            for (int i = 0; i < totalParams; i++)
            {
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                basis[d][i] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            }

            // Gram-Schmidt orthogonalization against previous basis vectors.
            for (int prev = 0; prev < d; prev++)
            {
                double dot = 0;
                for (int i = 0; i < totalParams; i++)
                {
                    dot += basis[d][i] * basis[prev][i];
                }

                for (int i = 0; i < totalParams; i++)
                {
                    basis[d][i] -= dot * basis[prev][i];
                }
            }

            // Normalize.
            double norm = 0;
            for (int i = 0; i < totalParams; i++)
            {
                norm += basis[d][i] * basis[d][i];
            }

            norm = Math.Sqrt(norm);
            if (norm > 1e-12)
            {
                for (int i = 0; i < totalParams; i++)
                {
                    basis[d][i] /= norm;
                }
            }
        }

        return basis;
    }

    /// <summary>
    /// Computes the top right singular vector of the data matrix via power iteration.
    /// data is [numSamples x dim], we want the top right singular vector of A^T A.
    /// </summary>
    private static double[] ComputeTopRightSingularVector(double[][] data, int numSamples, int dim, Random rng)
    {
        const int maxIterations = 50;
        const double tolerance = 1e-8;

        // Initialize random vector.
        var v = new double[dim];
        for (int d = 0; d < dim; d++)
        {
            v[d] = rng.NextDouble() - 0.5;
        }

        NormalizeVector(v);

        double prevEigenvalue = 0;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Compute A^T A v: first compute u = A v, then result = A^T u.
            // u = data * v  (u is [numSamples])
            var u = new double[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                double sum = 0;
                for (int d = 0; d < dim; d++)
                {
                    sum += data[i][d] * v[d];
                }

                u[i] = sum;
            }

            // v_new = A^T u  (v_new is [dim])
            var vNew = new double[dim];
            for (int d = 0; d < dim; d++)
            {
                double sum = 0;
                for (int i = 0; i < numSamples; i++)
                {
                    sum += data[i][d] * u[i];
                }

                vNew[d] = sum;
            }

            double eigenvalue = NormalizeVector(vNew);

            if (Math.Abs(eigenvalue - prevEigenvalue) < tolerance * Math.Max(prevEigenvalue, 1e-10))
            {
                return vNew;
            }

            prevEigenvalue = eigenvalue;
            v = vNew;
        }

        return v;
    }

    private static double NormalizeVector(double[] vec)
    {
        double norm = 0;
        for (int i = 0; i < vec.Length; i++)
        {
            norm += vec[i] * vec[i];
        }

        norm = Math.Sqrt(norm);
        if (norm > 1e-15)
        {
            for (int i = 0; i < vec.Length; i++)
            {
                vec[i] /= norm;
            }
        }

        return norm;
    }

    /// <summary>Gets the expected number of Byzantine clients.</summary>
    public int NumByzantine => _numByzantine;

    /// <summary>Gets the random projection subspace dimension.</summary>
    public int SubspaceDimension => _subspaceDimension;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"DnC(f={_numByzantine},d={_subspaceDimension})";
}
