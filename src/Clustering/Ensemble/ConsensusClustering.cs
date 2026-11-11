using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new ConsensusClustering&lt;double&gt;(new ConsensusClusteringOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // the cluster each training row was assigned to (Predict places rows the model has not seen)
/// var labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Cluster Ensembles - A Knowledge Reuse Framework for Combining Multiple Partitions", "https://doi.org/10.1162/153244303321897735", Year = 2003, Authors = "Alexander Strehl, Joydeep Ghosh")]
public partial class ConsensusClustering<T> : ClusteringBase<T>
{
    private readonly ConsensusClusteringOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _coAssociationMatrix;
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
    public Matrix<T>? CoAssociationMatrix => _coAssociationMatrix;

    /// <inheritdoc />

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

        var rand = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

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
        Labels = ClusteringTensorOps.FromIntArray<T>(finalLabels);

        // Compute cluster centers
        ComputeClusterCenters(x, finalLabels, n, d);

        MergeDegenerateClusters(x);
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
            Seed = rand.Next()
        });

        kmeans.Train(x);
        return ClusteringTensorOps.ToIntArray(kmeans.Labels ?? new Vector<T>(n));
    }

    private Matrix<T> BuildCoAssociationMatrix(List<int[]> clusterings, int n)
    {
        // Fraction of clusterings that put each pair together: sum_b M_b M_b^T / B (M_b one-hot memberships)
        var together = Tensor<T>.CreateZeros(new[] { n, n });
        foreach (var clustering in clusterings)
        {
            if (n == 0) break;

            var members = ClusteringTensorOps.OneHotMask(ClusteringTensorOps.FromIntArray<T>(clustering), clustering.Max() + 1);
            together = Engine.TensorAdd(together, Engine.TensorMatMul(members, Engine.TensorTranspose(members)));
        }

        return Engine.TensorDivideScalar(together, _numOps.FromDouble(clusterings.Count)).ToMatrix();
    }

    private int EstimateNumClusters(Matrix<T> coAssoc, int n)
    {
        // Simple heuristic based on the matrix structure:
        // count distinct "strong" groups (high co-association)
        var visited = new bool[n];
        int numClusters = 0;
        var strong = ClusteringTensorOps.ToBoolArray(
            Engine.TensorGreaterThan(Tensor<T>.FromMatrix(coAssoc), _numOps.FromDouble(0.5)).ToVector());

        for (int i = 0; i < n; i++)
        {
            if (visited[i]) continue;

            visited[i] = true;
            numClusters++;

            // Mark all strongly connected points
            for (int j = i + 1; j < n; j++)
            {
                if (!visited[j] && strong[i * n + j])
                {
                    visited[j] = true;
                }
            }
        }

        return Math.Max(2, numClusters);
    }

    private int[] ApplyFinalClustering(Matrix<T> coAssoc, int n, int numClusters)
    {
        // Convert co-association to distance (1 - similarity)
        var distMatrix = Engine.ScalarMinusTensor(_numOps.One, Tensor<T>.FromMatrix(coAssoc));

        // Use hierarchical clustering (average linkage)
        return AgglomerativeCluster(distMatrix, n, numClusters);
    }

    private int[] AgglomerativeCluster(Tensor<T> distMatrix, int n, int numClusters)
    {
        // Initialize: each point is its own cluster
        var clusterAssignments = new int[n];
        for (int i = 0; i < n; i++)
        {
            clusterAssignments[i] = i;
        }

        var activeClusters = new HashSet<int>(Enumerable.Range(0, n));
        var clusterDist = distMatrix;
        var active = n == 0 ? new Vector<T>(0) : Engine.Fill(n, _numOps.One);

        // Track merge history: when cluster B merges into A, record mergedInto[B] = A
        var mergedInto = new Dictionary<int, int>();

        // Merge until we have the desired number of clusters
        while (activeClusters.Count > numClusters)
        {
            // Find closest pair
            if (!TryFindClosestPair(clusterDist, active, out int merge1, out int merge2)) break;

            // Track the merge before updating assignments
            mergedInto[merge2] = merge1;

            // Merge cluster2 into cluster1
            for (int i = 0; i < n; i++)
            {
                if (clusterAssignments[i] == merge2)
                {
                    clusterAssignments[i] = merge1;
                }
            }

            // Update distances (average linkage)
            clusterDist = AverageLinkageMerge(clusterDist, merge1, merge2);
            active = Deactivate(active, merge2);

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
            int cluster = clusterAssignments[i];

            // Follow the merge chain to find the final active cluster
            while (!activeClusters.Contains(cluster) && mergedInto.ContainsKey(cluster))
            {
                cluster = mergedInto[cluster];
            }

            // Fallback if still not in active clusters (should not happen normally)
            if (!activeClusters.Contains(cluster))
            {
                cluster = activeClusters.First();
            }

            result[i] = clusterMap.ContainsKey(cluster) ? clusterMap[cluster] : 0;
        }

        return result;
    }

    /// <summary>
    /// The closest pair of distinct active clusters (first in row-major order on ties, so merge1 &lt; merge2);
    /// false when no pair has a distance below MaxValue.
    /// </summary>
    private bool TryFindClosestPair(Tensor<T> clusterDist, Vector<T> active, out int merge1, out int merge2)
    {
        int n = active.Length;
        merge1 = -1;
        merge2 = -1;
        if (n < 2)
        {
            return false;
        }

        var shape = new[] { n, n };
        var activeRows = Engine.TensorBroadcastTo(Tensor<T>.FromVector(active, new[] { n, 1 }), shape);
        var activeColumns = Engine.TensorBroadcastTo(Tensor<T>.FromVector(active, new[] { 1, n }), shape);
        var offDiagonal = Engine.ScalarMinusTensor(_numOps.One, Engine.TensorEye<T>(n));
        var valid = Engine.TensorMultiply(Engine.TensorMultiply(activeRows, activeColumns), offDiagonal);
        var candidates = Engine.TensorWhere(valid, clusterDist, Engine.TensorFullLike(clusterDist, _numOps.MaxValue));

        var flat = Engine.Reshape(candidates, new[] { n * n }).ToVector();
        int best = ClusteringTensorOps.ArgMin(flat);
        if (!_numOps.LessThan(flat[best], _numOps.MaxValue))
        {
            return false;
        }

        merge1 = best / n;
        merge2 = best % n;
        return true;
    }

    /// <summary>Marks a cluster as merged away.</summary>
    private Vector<T> Deactivate(Vector<T> active, int index)
        => Engine.Scatter(active.AsTensor(), Vector<int>.Wrap(new[] { index }).AsTensor(), Tensor<T>.CreateZeros(new[] { 1 }), 0).ToVector();

    /// <summary>Average linkage: d(merge1, c) = (d(merge1, c) + d(merge2, c)) / 2 for every other cluster c.</summary>
    private Tensor<T> AverageLinkageMerge(Tensor<T> clusterDist, int merge1, int merge2)
    {
        int n = clusterDist.Shape[0];
        var shape = new[] { n, n };
        var distances = clusterDist.ToMatrix();
        var merged = Engine.TensorMultiplyScalar(
            Engine.TensorAdd(distances.GetRow(merge1).AsTensor(), distances.GetRow(merge2).AsTensor()), _numOps.FromDouble(0.5));

        // Keep the merged clusters' own entries (the diagonal and the merge1/merge2 pair) unchanged
        var ids = Engine.TensorArange<T>(0, n, 1);
        var keep = Engine.TensorAdd(
            Engine.TensorEquals(ids, _numOps.FromDouble(merge1)),
            Engine.TensorEquals(ids, _numOps.FromDouble(merge2)));
        var update = Engine.TensorWhere(keep, distances.GetRow(merge1).AsTensor(), merged);

        var rowMask = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorEquals(ids, _numOps.FromDouble(merge1)), new[] { n, 1 }), shape);
        var columnMask = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorEquals(ids, _numOps.FromDouble(merge1)), new[] { 1, n }), shape);
        var withRow = Engine.TensorWhere(rowMask, Engine.TensorBroadcastTo(Engine.Reshape(update, new[] { 1, n }), shape), clusterDist);
        return Engine.TensorWhere(columnMask, Engine.TensorBroadcastTo(Engine.Reshape(update, new[] { n, 1 }), shape), withRow);
    }

    private void ComputeClusterCenters(Matrix<T> x, int[] labels, int n, int d)
    {
        // Mean of each cluster's members (empty clusters stay at zero)
        ClusterCenters = NumClusters == 0
            ? new Matrix<T>(0, d)
            : ClusteringTensorOps.ClusterMeans(x, ClusteringTensorOps.FromIntArray<T>(labels), NumClusters, out _);
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        if (ClusterCenters is null || NumClusters == 0)
        {
            return new Vector<T>(x.Rows);
        }

        // Nearest center by squared Euclidean distance (first on ties)
        var centers = ClusterCenters.Rows == NumClusters
            ? ClusterCenters
            : ClusterCenters.GetSubMatrix(0, 0, NumClusters, ClusterCenters.Columns);
        return ClusteringTensorOps.FromIntArray<T>(
            ClusteringTensorOps.ArgMinRows(ClusteringTensorOps.SquaredDistances(x, centers)));
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }
}
