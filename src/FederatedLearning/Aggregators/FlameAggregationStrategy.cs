namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FLAME (Filtering via cosine similarity + Adaptive clipping + Noise) for
/// Byzantine-robust federated learning with backdoor resistance.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Backdoor attacks in federated learning try to implant hidden
/// triggers in the global model. FLAME defends against this with a three-step approach:
/// (1) use HDBSCAN-inspired clustering on cosine distances to identify the honest majority
/// cluster, (2) clip surviving updates to a common norm to prevent magnitude-based attacks,
/// and (3) add calibrated noise to the aggregated result to erase any residual backdoor signal.</para>
///
/// <para>Pipeline:</para>
/// <list type="number">
/// <item>Flatten all client updates into vectors and compute pairwise cosine distances</item>
/// <item>Compute core distances (k-th nearest neighbor) and mutual reachability distances</item>
/// <item>Build a minimum spanning tree (MST) and cut at the largest gap to extract clusters</item>
/// <item>Select the largest cluster as the honest majority</item>
/// <item>Clip remaining updates to the median norm (adaptive clipping)</item>
/// <item>Average the clipped updates</item>
/// <item>Add Gaussian noise with std = <c>NoiseMultiplier</c> * <c>clipNorm</c></item>
/// </list>
///
/// <para>Reference: Nguyen, T. D., et al. (2022). "FLAME: Taming Backdoors in Federated
/// Learning." USENIX Security 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FlameAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _noiseMultiplier;
    private readonly int _minClusterSize;
    private readonly int _seed;
    private int _roundCounter;

    /// <summary>
    /// Initializes a new instance of the <see cref="FlameAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="noiseMultiplier">Gaussian noise multiplier relative to clip norm. Default: 0.001.</param>
    /// <param name="minClusterSize">Minimum number of points to form a dense cluster in HDBSCAN.
    /// Default: 2 (the paper's min_samples parameter).</param>
    /// <param name="seed">Random seed for noise generation. Default: 42.</param>
    public FlameAggregationStrategy(double noiseMultiplier = 0.001, int minClusterSize = 2, int seed = 42)
    {
        if (noiseMultiplier < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(noiseMultiplier), "Noise multiplier must be non-negative.");
        }

        if (minClusterSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(minClusterSize), "Minimum cluster size must be at least 1.");
        }

        _noiseMultiplier = noiseMultiplier;
        _minClusterSize = minClusterSize;
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
        int totalParams = layerNames.Sum(ln => referenceModel[ln].Length);
        var clientIds = clientModels.Keys.ToList();
        int n = clientIds.Count;

        // Flatten to double vectors and compute norms.
        var flatVectors = new double[n][];
        var norms = new double[n];

        for (int c = 0; c < n; c++)
        {
            flatVectors[c] = new double[totalParams];
            int offset = 0;
            foreach (var layerName in layerNames)
            {
                var cp = clientModels[clientIds[c]][layerName];
                for (int i = 0; i < cp.Length; i++)
                {
                    double v = NumOps.ToDouble(cp[i]);
                    flatVectors[c][offset] = v;
                    norms[c] += v * v;
                    offset++;
                }
            }

            norms[c] = Math.Sqrt(norms[c]);
        }

        // Step 1: HDBSCAN-inspired clustering on cosine distances.
        var trusted = IdentifyHonestCluster(flatVectors, norms, n, totalParams);

        // If clustering fails (e.g., too few clients), fall back to all clients.
        if (trusted.Count == 0)
        {
            trusted = Enumerable.Range(0, n).ToList();
        }

        // Step 2: Adaptive clipping — clip to median norm of trusted clients.
        var trustedNorms = trusted.Select(c => norms[c]).OrderBy(x => x).ToArray();
        double medianNorm = trustedNorms.Length % 2 == 1
            ? trustedNorms[trustedNorms.Length / 2]
            : (trustedNorms[trustedNorms.Length / 2 - 1] + trustedNorms[trustedNorms.Length / 2]) / 2.0;

        double clipNorm = Math.Max(medianNorm, 1e-10);

        // Step 3: Average the clipped updates.
        var result = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            result[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
        }

        double weight = 1.0 / trusted.Count;
        foreach (int c in trusted)
        {
            double scale = norms[c] > clipNorm ? clipNorm / norms[c] : 1.0;
            double combinedScale = scale * weight;
            var csT = NumOps.FromDouble(combinedScale);
            var clientModel = clientModels[clientIds[c]];

            foreach (var layerName in layerNames)
            {
                var cp = clientModel[layerName];
                var rp = result[layerName];
                for (int i = 0; i < rp.Length; i++)
                {
                    rp[i] = NumOps.Add(rp[i], NumOps.Multiply(cp[i], csT));
                }
            }
        }

        // Step 4: Add calibrated Gaussian noise.
        if (_noiseMultiplier > 0)
        {
            double noiseStd = _noiseMultiplier * clipNorm;
            var rng = new Random(_seed + _roundCounter++);

            foreach (var layerName in layerNames)
            {
                var rp = result[layerName];
                for (int i = 0; i < rp.Length; i++)
                {
                    double u1 = 1.0 - rng.NextDouble();
                    double u2 = 1.0 - rng.NextDouble();
                    double noise = noiseStd * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    rp[i] = NumOps.Add(rp[i], NumOps.FromDouble(noise));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Identifies the honest majority cluster using HDBSCAN-inspired density-based clustering
    /// on cosine distances between client update vectors.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of using a fixed threshold to decide which clients
    /// are honest, this method automatically discovers groups (clusters) of similar updates.
    /// The idea is that honest clients will naturally form a dense cluster, while attackers
    /// will be outliers or form separate small clusters. The largest dense cluster is selected
    /// as the honest group.</para>
    ///
    /// <para>Steps:</para>
    /// <list type="number">
    /// <item>Compute pairwise cosine distances between all clients</item>
    /// <item>For each client, find its core distance (distance to k-th nearest neighbor)</item>
    /// <item>Compute mutual reachability distances: max(core_i, core_j, dist(i,j))</item>
    /// <item>Build MST on mutual reachability graph using Prim's algorithm</item>
    /// <item>Sort MST edges and cut at the largest gap to separate clusters</item>
    /// <item>Return the largest cluster</item>
    /// </list>
    /// </remarks>
    private List<int> IdentifyHonestCluster(double[][] vectors, double[] norms, int n, int dim)
    {
        if (n <= 2)
        {
            return Enumerable.Range(0, n).ToList();
        }

        // Compute pairwise cosine distances: d(i,j) = 1 - cos_sim(i,j).
        var cosineDistances = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (norms[i] <= 1e-15 || norms[j] <= 1e-15)
                {
                    cosineDistances[i, j] = 1.0;
                    cosineDistances[j, i] = 1.0;
                    continue;
                }

                double dot = 0;
                for (int p = 0; p < dim; p++)
                {
                    dot += vectors[i][p] * vectors[j][p];
                }

                double cosSim = dot / (norms[i] * norms[j]);
                double dist = 1.0 - cosSim; // Range [0, 2]
                cosineDistances[i, j] = dist;
                cosineDistances[j, i] = dist;
            }
        }

        // Compute core distances: distance to the k-th nearest neighbor.
        int k = Math.Min(_minClusterSize, n - 1);
        var coreDistances = new double[n];

        for (int i = 0; i < n; i++)
        {
            var dists = new double[n - 1];
            int idx = 0;
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    dists[idx++] = cosineDistances[i, j];
                }
            }

            Array.Sort(dists);
            coreDistances[i] = dists[Math.Min(k - 1, dists.Length - 1)];
        }

        // Compute mutual reachability distances: mrd(i,j) = max(core_i, core_j, d(i,j)).
        var mrd = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double d = Math.Max(coreDistances[i], Math.Max(coreDistances[j], cosineDistances[i, j]));
                mrd[i, j] = d;
                mrd[j, i] = d;
            }
        }

        // Build MST using Prim's algorithm on mutual reachability distances.
        var mstEdges = BuildMST(mrd, n);

        // Sort MST edges by weight (ascending).
        mstEdges.Sort((a, b) => a.weight.CompareTo(b.weight));

        // Find the largest gap in MST edge weights to determine where to cut.
        // The largest gap separates the honest cluster from outliers/attackers.
        double maxGap = 0;
        int cutIndex = mstEdges.Count; // Default: no cut (all in one cluster).

        for (int i = 1; i < mstEdges.Count; i++)
        {
            double gap = mstEdges[i].weight - mstEdges[i - 1].weight;
            if (gap > maxGap)
            {
                maxGap = gap;
                cutIndex = i;
            }
        }

        // Also check if the last edge itself is a large outlier.
        if (mstEdges.Count > 0)
        {
            double lastEdgeWeight = mstEdges[mstEdges.Count - 1].weight;
            double meanWeight = mstEdges.Average(e => e.weight);
            if (lastEdgeWeight > 2.0 * meanWeight && cutIndex == mstEdges.Count)
            {
                cutIndex = mstEdges.Count - 1;
            }
        }

        // Build adjacency from edges up to the cut point (exclude edges after the gap).
        var adjacency = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            adjacency[i] = [];
        }

        for (int i = 0; i < cutIndex; i++)
        {
            adjacency[mstEdges[i].u].Add(mstEdges[i].v);
            adjacency[mstEdges[i].v].Add(mstEdges[i].u);
        }

        // Find connected components (clusters).
        var visited = new bool[n];
        var clusters = new List<List<int>>();

        for (int i = 0; i < n; i++)
        {
            if (visited[i])
            {
                continue;
            }

            var cluster = new List<int>();
            var queue = new Queue<int>();
            queue.Enqueue(i);
            visited[i] = true;

            while (queue.Count > 0)
            {
                int node = queue.Dequeue();
                cluster.Add(node);
                foreach (int neighbor in adjacency[node])
                {
                    if (!visited[neighbor])
                    {
                        visited[neighbor] = true;
                        queue.Enqueue(neighbor);
                    }
                }
            }

            clusters.Add(cluster);
        }

        // Return the largest cluster (honest majority).
        var largestCluster = clusters.OrderByDescending(c => c.Count).First();

        // Only accept if the largest cluster has at least minClusterSize members.
        return largestCluster.Count >= _minClusterSize ? largestCluster : [];
    }

    /// <summary>
    /// Builds a minimum spanning tree using Prim's algorithm on the given distance matrix.
    /// </summary>
    private static List<(int u, int v, double weight)> BuildMST(double[,] distances, int n)
    {
        var edges = new List<(int u, int v, double weight)>(n - 1);
        var inMST = new bool[n];
        var minEdge = new double[n];
        var minFrom = new int[n];

        for (int i = 0; i < n; i++)
        {
            minEdge[i] = double.MaxValue;
            minFrom[i] = -1;
        }

        // Start from node 0.
        inMST[0] = true;
        for (int j = 1; j < n; j++)
        {
            minEdge[j] = distances[0, j];
            minFrom[j] = 0;
        }

        for (int iter = 0; iter < n - 1; iter++)
        {
            // Find the minimum edge from the MST frontier.
            int bestNode = -1;
            double bestDist = double.MaxValue;

            for (int j = 0; j < n; j++)
            {
                if (!inMST[j] && minEdge[j] < bestDist)
                {
                    bestDist = minEdge[j];
                    bestNode = j;
                }
            }

            if (bestNode < 0)
            {
                break;
            }

            inMST[bestNode] = true;
            edges.Add((minFrom[bestNode], bestNode, bestDist));

            // Update frontier distances.
            for (int j = 0; j < n; j++)
            {
                if (!inMST[j] && distances[bestNode, j] < minEdge[j])
                {
                    minEdge[j] = distances[bestNode, j];
                    minFrom[j] = bestNode;
                }
            }
        }

        return edges;
    }

    /// <summary>Gets the noise multiplier for DP noise injection.</summary>
    public double NoiseMultiplier => _noiseMultiplier;

    /// <summary>Gets the minimum cluster size for HDBSCAN.</summary>
    public int MinClusterSize => _minClusterSize;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FLAME(k={_minClusterSize},\u03c3={_noiseMultiplier})";
}
