using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Ensemble;

/// <summary>
/// Consensus (Ensemble) Clustering implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Consensus clustering combines multiple base clusterings to produce a more
/// robust final result. It builds a co-association matrix showing how often
/// pairs of points are clustered together, then clusters this matrix.
/// </para>
/// <para>
/// Algorithm:
/// 1. Generate multiple base clusterings (e.g., K-Means with different k or seeds)
/// 2. Build co-association matrix: C[i,j] = frequency of i and j in same cluster
/// 3. Apply final clustering to the co-association matrix
/// </para>
/// <para><b>For Beginners:</b> Consensus clustering combines multiple opinions.
///
/// Think of it like a jury:
/// - Multiple clusterings give their "verdict"
/// - We count how often they agree
/// - The final answer is based on majority agreement
///
/// The co-association matrix captures:
/// - "Points A and B were in the same cluster in 8 out of 10 runs"
/// - This creates a similarity measure between all pairs
/// - We then cluster based on this similarity
///
/// Benefits:
/// - More stable than single clustering
/// - Less sensitive to initialization
/// - Can combine different algorithms
/// </para>
/// </remarks>
public class ConsensusClustering<T> : ClusteringBase<T>
{
    private readonly ConsensusClusteringOptions<T> _options;
    private double[,]? _coAssociationMatrix;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new ConsensusClustering instance.
    /// </summary>
    /// <param name="options">The consensus clustering options.</param>
    public ConsensusClustering(ConsensusClusteringOptions<T>? options = null)
        : base(new ClusteringOptions<T>())
    {
        _options = options ?? new ConsensusClusteringOptions<T>();
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the co-association matrix.
    /// </summary>
    public double[,]? CoAssociationMatrix => _coAssociationMatrix;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new ConsensusClustering<T>(new ConsensusClusteringOptions<T>
        {
            NumBaseClusterings = _options.NumBaseClusterings,
            Method = _options.Method,
            FinalAlgorithm = _options.FinalAlgorithm,
            NumClusters = _options.NumClusters,
            RandomSeed = _options.RandomSeed
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (ConsensusClustering<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        var rand = _options.RandomSeed.HasValue
            ? new Random(_options.RandomSeed.Value)
            : new Random();

        // Generate base clusterings
        var baseClusterings = new List<int[]>();

        for (int b = 0; b < _options.NumBaseClusterings; b++)
        {
            var clustering = GenerateBaseClustering(x, n, rand);
            baseClusterings.Add(clustering);
        }

        // Build co-association matrix
        _coAssociationMatrix = BuildCoAssociationMatrix(baseClusterings, n);

        // Apply final clustering
        int targetClusters = _options.NumClusters ?? EstimateNumClusters(_coAssociationMatrix, n);
        var finalLabels = ApplyFinalClustering(_coAssociationMatrix, n, targetClusters);

        // Set results
        NumClusters = finalLabels.Distinct().Where(l => l >= 0).Count();
        Labels = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            Labels[i] = _numOps.FromDouble(finalLabels[i]);
        }

        // Compute cluster centers
        ComputeClusterCenters(x, finalLabels, n, d);

        IsTrained = true;
    }

    private int[] GenerateBaseClustering(Matrix<T> x, int n, Random rand)
    {
        // Use K-Means with random k and random seed
        int k = rand.Next(2, Math.Max(3, (int)Math.Sqrt(n)));

        var kmeans = new KMeans<T>(new KMeansOptions<T>
        {
            NumClusters = k,
            MaxIterations = 100,
            NumInitializations = 1,
            RandomState = rand.Next()
        });

        kmeans.Train(x);
        var labels = kmeans.Labels!;

        var result = new int[n];
        for (int i = 0; i < n; i++)
        {
            result[i] = (int)_numOps.ToDouble(labels[i]);
        }

        return result;
    }

    private double[,] BuildCoAssociationMatrix(List<int[]> clusterings, int n)
    {
        var matrix = new double[n, n];

        foreach (var clustering in clusterings)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = i; j < n; j++)
                {
                    if (clustering[i] == clustering[j])
                    {
                        matrix[i, j] += 1;
                        if (i != j)
                        {
                            matrix[j, i] += 1;
                        }
                    }
                }
            }
        }

        // Normalize by number of clusterings
        double numClusterings = clusterings.Count;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                matrix[i, j] /= numClusterings;
            }
        }

        return matrix;
    }

    private int EstimateNumClusters(double[,] coAssoc, int n)
    {
        // Use eigenvalue analysis or simple heuristic
        // Here we use a simple heuristic based on the matrix structure

        // Count distinct "strong" groups (high co-association)
        var visited = new bool[n];
        int numClusters = 0;
        double threshold = 0.5;

        for (int i = 0; i < n; i++)
        {
            if (visited[i]) continue;

            visited[i] = true;
            numClusters++;

            // Mark all strongly connected points
            for (int j = i + 1; j < n; j++)
            {
                if (!visited[j] && coAssoc[i, j] > threshold)
                {
                    visited[j] = true;
                }
            }
        }

        return Math.Max(2, numClusters);
    }

    private int[] ApplyFinalClustering(double[,] coAssoc, int n, int numClusters)
    {
        // Convert co-association to distance (1 - similarity)
        var distMatrix = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                distMatrix[i, j] = 1 - coAssoc[i, j];
            }
        }

        // Use hierarchical clustering (average linkage)
        return AgglomerativeCluster(distMatrix, n, numClusters);
    }

    private int[] AgglomerativeCluster(double[,] distMatrix, int n, int numClusters)
    {
        // Initialize: each point is its own cluster
        var clusterAssignments = new int[n];
        for (int i = 0; i < n; i++)
        {
            clusterAssignments[i] = i;
        }

        var activeClusters = new HashSet<int>(Enumerable.Range(0, n));
        var clusterDist = new double[n, n];
        Array.Copy(distMatrix, clusterDist, n * n);

        // Merge until we have the desired number of clusters
        while (activeClusters.Count > numClusters)
        {
            // Find closest pair
            double minDist = double.MaxValue;
            int merge1 = -1, merge2 = -1;

            var activeList = activeClusters.ToList();
            for (int i = 0; i < activeList.Count; i++)
            {
                for (int j = i + 1; j < activeList.Count; j++)
                {
                    int c1 = activeList[i];
                    int c2 = activeList[j];

                    if (clusterDist[c1, c2] < minDist)
                    {
                        minDist = clusterDist[c1, c2];
                        merge1 = c1;
                        merge2 = c2;
                    }
                }
            }

            if (merge1 < 0) break;

            // Merge cluster2 into cluster1
            for (int i = 0; i < n; i++)
            {
                if (clusterAssignments[i] == merge2)
                {
                    clusterAssignments[i] = merge1;
                }
            }

            // Update distances (average linkage)
            foreach (int c in activeClusters)
            {
                if (c != merge1 && c != merge2)
                {
                    clusterDist[merge1, c] = (clusterDist[merge1, c] + clusterDist[merge2, c]) / 2;
                    clusterDist[c, merge1] = clusterDist[merge1, c];
                }
            }

            activeClusters.Remove(merge2);
        }

        // Renumber clusters from 0
        var clusterMap = new Dictionary<int, int>();
        int clusterNum = 0;
        foreach (int c in activeClusters.OrderBy(x => x))
        {
            clusterMap[c] = clusterNum++;
        }

        var result = new int[n];
        for (int i = 0; i < n; i++)
        {
            int originalCluster = clusterAssignments[i];
            // Find the active cluster this belongs to
            while (!activeClusters.Contains(originalCluster))
            {
                // Find which cluster it merged into
                for (int j = 0; j < n; j++)
                {
                    if (clusterAssignments[j] == originalCluster && j != i)
                    {
                        originalCluster = clusterAssignments[j];
                        break;
                    }
                }
                if (!activeClusters.Contains(originalCluster))
                {
                    // Fallback: assign to first active cluster
                    originalCluster = activeClusters.First();
                    break;
                }
            }

            result[i] = clusterMap.ContainsKey(originalCluster) ? clusterMap[originalCluster] : 0;
        }

        return result;
    }

    private void ComputeClusterCenters(Matrix<T> x, int[] labels, int n, int d)
    {
        if (NumClusters == 0)
        {
            ClusterCenters = new Matrix<T>(0, d);
            return;
        }

        ClusterCenters = new Matrix<T>(NumClusters, d);
        var counts = new int[NumClusters];
        var sums = new double[NumClusters, d];

        for (int i = 0; i < n; i++)
        {
            int label = labels[i];
            if (label >= 0 && label < NumClusters)
            {
                counts[label]++;
                for (int j = 0; j < d; j++)
                {
                    sums[label, j] += _numOps.ToDouble(x[i, j]);
                }
            }
        }

        for (int k = 0; k < NumClusters; k++)
        {
            if (counts[k] > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    ClusterCenters[k, j] = _numOps.FromDouble(sums[k, j] / counts[k]);
                }
            }
        }
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        var labels = new Vector<T>(x.Rows);

        for (int i = 0; i < x.Rows; i++)
        {
            var point = GetRow(x, i);
            double minDist = double.MaxValue;
            int nearestCluster = 0;

            if (ClusterCenters is not null)
            {
                for (int k = 0; k < NumClusters; k++)
                {
                    var center = GetRow(ClusterCenters, k);
                    double dist = 0;
                    for (int j = 0; j < point.Length; j++)
                    {
                        double diff = _numOps.ToDouble(point[j]) - _numOps.ToDouble(center[j]);
                        dist += diff * diff;
                    }

                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearestCluster = k;
                    }
                }
            }

            labels[i] = _numOps.FromDouble(nearestCluster);
        }

        return labels;
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels!;
    }
}
