using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.SpatialIndex;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Density;

/// <summary>
/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DBSCAN is a density-based clustering algorithm that groups together points that are
/// closely packed together, marking points in low-density regions as outliers.
/// </para>
/// <para>
/// Algorithm steps:
/// 1. For each point, find all neighbors within epsilon distance
/// 2. If a point has at least MinPoints neighbors, it's a core point
/// 3. Core points form cluster seeds
/// 4. Expand clusters by adding density-reachable points
/// 5. Mark remaining points as noise (-1)
/// </para>
/// <para><b>For Beginners:</b> DBSCAN finds clusters by looking for dense regions.
///
/// Imagine you're at a party:
/// - Dense groups of people talking are clusters
/// - People standing alone are noise/outliers
/// - DBSCAN finds these groups automatically
///
/// Key advantages:
/// - Discovers clusters of any shape (not just circles)
/// - Doesn't require knowing the number of clusters
/// - Identifies outliers as noise
/// - Handles clusters of different sizes and densities
///
/// Limitations:
/// - Sensitive to epsilon and MinPoints parameters
/// - May struggle with clusters of varying density
/// - O(n²) complexity without spatial indexing
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new DBSCAN&lt;double&gt;(new DBSCANOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // the cluster each training row was assigned to (Predict places rows the model has not seen)
/// var labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise", "https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf", Year = 1996, Authors = "Martin Ester, Hans-Peter Kriegel, Jorg Sander, Xiaowei Xu")]
public partial class DBSCAN<T> : ClusteringBase<T>
{
    private readonly DBSCANOptions<T> _options;

    private T _fittedEpsilon;


    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private bool[]? _corePointMask;

    // Feature normalization state for scale-invariant distance computation
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _featureMeans;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _featureStds;

    // Cluster centers in normalized space for Predict comparison
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _normalizedClusterCenters;

    /// <summary>
    /// Noise label (points not assigned to any cluster).
    /// </summary>
    public const int NoiseLabel = -1;

    /// <summary>
    /// Undefined label (not yet processed).
    /// </summary>
    private const int UndefinedLabel = -2;

    /// <summary>
    /// Initializes a new DBSCAN instance with the specified options.
    /// </summary>
    /// <param name="options">The DBSCAN configuration options.</param>
    public DBSCAN(DBSCANOptions<T>? options = null)
        : base(options ??= new DBSCANOptions<T>())
    {
        _fittedEpsilon = NumOps.Zero;
        _options = options;

        if (_options.DistanceMetric is null)
        {
            _options.DistanceMetric = new EuclideanDistance<T>();
        }
    }

    /// <summary>
    /// Gets the epsilon neighborhood radius.
    /// </summary>
    public double Epsilon => _options.Epsilon;

    /// <summary>
    /// Gets the minimum points for core point classification.
    /// </summary>
    public int MinPoints => _options.MinPoints;

    /// <summary>
    /// Gets the mask indicating which points are core points.
    /// </summary>
    public bool[]? CorePointMask => _corePointMask;

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (DBSCAN<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputData(x);

        // Normalize features to zero-mean unit-variance for scale-invariant distance.
        // Without normalization, epsilon is scale-dependent: data scaled 100x needs
        // epsilon scaled 100x. Normalization makes epsilon a universal threshold.
        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        x = ClusteringTensorOps.StandardizeColumns(x, out var featureMeans, out var featureStds);
        _featureMeans = featureMeans;
        _featureStds = featureStds;

        var labels = new int[n];
        _corePointMask = new bool[n];

        // Initialize all points as undefined
        for (int i = 0; i < n; i++)
        {
            labels[i] = (int)UndefinedLabel;
        }

        // Build spatial index for efficient neighbor queries
        var neighborFinder = CreateNeighborFinder(x);

        // Auto-estimate epsilon from the normalized data.
        // Since we normalize features, the user's Epsilon was tuned for original scale
        // and is no longer meaningful. Always re-estimate on the normalized data.
        // Uses k-distance method: compute the MinPoints-nearest-neighbor distance for each point,
        // then use the mean * scaling factor as epsilon.
        var metric = _options.DistanceMetric ?? new AiDotNet.Clustering.DistanceMetrics.EuclideanDistance<T>();
        T epsilonT = TryEstimateEpsilon(x, metric, out T estimated) ? estimated : NumOps.FromDouble(_options.Epsilon);
        _fittedEpsilon = epsilonT;

        // Find neighbors for each point and identify core points
        var neighbors = new List<int>[n];

        for (int i = 0; i < n; i++)
        {
            var query = GetRow(x, i);
            neighbors[i] = FindNeighbors(neighborFinder, x, query, epsilonT);
            _corePointMask[i] = neighbors[i].Count >= _options.MinPoints;
        }

        // Cluster expansion
        int currentCluster = 0;

        for (int i = 0; i < n; i++)
        {
            if (labels[i] != UndefinedLabel)
            {
                continue; // Already processed
            }

            if (!_corePointMask[i])
            {
                labels[i] = NoiseLabel; // Mark as noise for now
                continue;
            }

            // Start a new cluster
            ExpandCluster(x, i, neighbors, labels, currentCluster, epsilonT, neighborFinder);
            currentCluster++;
        }

        // Convert to Vector<T>
        Labels = ClusteringTensorOps.FromIntArray<T>(labels);

        NumClusters = currentCluster;

        // Compute cluster centers for clusters with at least one point
        if (currentCluster > 0)
        {
            ComputeClusterCenters(x, labels, currentCluster);

            // Save normalized centers for Predict (where input is also normalized)
            if (_featureMeans is not null && _featureStds is not null && ClusterCenters is not null)
            {
                _normalizedClusterCenters = ClusterCenters.Clone();

                // De-normalize cluster centers back to original space so users get
                // meaningful values (not normalized values that only make sense internally).
                ClusterCenters = ClusteringTensorOps.Destandardize(ClusterCenters, _featureMeans, _featureStds);
            }
        }

        IsTrained = true;
    }

    /// <summary>
    /// 2.5 times the mean distance from (up to) the first 200 points to their MinPoints-th nearest other point;
    /// false when there are fewer than MinPoints other points.
    /// </summary>
    private bool TryEstimateEpsilon(Matrix<T> x, IDistanceMetric<T> metric, out T epsilon)
    {
        epsilon = NumOps.Zero;
        int n = x.Rows;
        int k = _options.MinPoints;
        if (n - 1 < k)
        {
            return false;
        }

        // Distances from the sampled points to every point, with the self-distance pushed past every other one
        int sampled = Math.Min(n, 200);
        var sample = x.Rows == sampled ? x : x.GetSubMatrix(0, 0, sampled, x.Columns);
        var distances = Tensor<T>.FromMatrix(metric.ComputePairwise(sample, x));
        var self = ClusteringTensorOps.OneHotMask(Engine.TensorArange<T>(0, sampled, 1).ToVector(), n);
        distances = Engine.TensorWhere(self, Engine.TensorFullLike(distances, NumOps.MaxValue), distances);

        // k-th smallest distance of every sampled point (msort orders along the first axis)
        var sorted = Engine.TensorMsort(Engine.TensorTranspose(distances)).ToMatrix();
        T meanKDistance = Engine.TensorMean(sorted.GetRow(k - 1).AsTensor());

        // Factor of 2.5 provides margin to include all same-cluster neighbors.
        epsilon = NumOps.Multiply(meanKDistance, NumOps.FromDouble(2.5));
        return true;
    }

    private void ExpandCluster(
        Matrix<T> x,
        int seedIndex,
        List<int>[] neighbors,
        int[] labels,
        int clusterId,
        T epsilon,
        object neighborFinder)
    {
        // Use a queue for BFS expansion
        var queue = new Queue<int>();
        queue.Enqueue(seedIndex);
        labels[seedIndex] = clusterId;

        while (queue.Count > 0)
        {
            int current = queue.Dequeue();

            // If current is a core point, expand to its neighbors
            if ((_corePointMask ?? throw new InvalidOperationException("_corePointMask has not been initialized."))[current])
            {
                foreach (int neighborIdx in neighbors[current])
                {
                    if (labels[neighborIdx] == UndefinedLabel || labels[neighborIdx] == NoiseLabel)
                    {
                        if (labels[neighborIdx] == UndefinedLabel)
                        {
                            queue.Enqueue(neighborIdx);
                        }
                        labels[neighborIdx] = clusterId;
                    }
                }
            }
        }
    }

    private void ComputeClusterCenters(Matrix<T> x, int[] labels, int numClusters)
    {
        // Mean of each cluster's members (noise is ignored; empty clusters stay at zero)
        ClusterCenters = ClusteringTensorOps.ClusterMeans(x, ClusteringTensorOps.FromIntArray<T>(labels), numClusters, out _);
    }

    private object CreateNeighborFinder(Matrix<T> x)
    {
        var algorithm = _options.Algorithm;

        // Auto-select algorithm based on data characteristics
        if (algorithm == NeighborAlgorithm.Auto)
        {
            if (x.Columns > 20)
            {
                algorithm = NeighborAlgorithm.BallTree;
            }
            else if (x.Rows < 50)
            {
                algorithm = NeighborAlgorithm.BruteForce;
            }
            else
            {
                algorithm = NeighborAlgorithm.KDTree;
            }
        }

        switch (algorithm)
        {
            case NeighborAlgorithm.KDTree:
                var kdTree = new KDTree<T>(_options.DistanceMetric, _options.LeafSize);
                kdTree.Build(x);
                return kdTree;

            case NeighborAlgorithm.BallTree:
                var ballTree = new BallTree<T>(_options.DistanceMetric, _options.LeafSize);
                ballTree.Build(x);
                return ballTree;

            case NeighborAlgorithm.BruteForce:
            default:
                return x; // Just return the data for brute force
        }
    }

    private List<int> FindNeighbors(object neighborFinder, Matrix<T> data, Vector<T> query, T epsilon)
    {
        if (neighborFinder is KDTree<T> kdTree)
        {
            var results = kdTree.QueryRadius(query, epsilon);
            return results.Select(r => r.Index).ToList();
        }
        else if (neighborFinder is BallTree<T> ballTree)
        {
            var results = ballTree.QueryRadius(query, epsilon);
            return results.Select(r => r.Index).ToList();
        }
        else
        {
            // Brute force
            return FindNeighborsBruteForce(data, query, epsilon);
        }
    }

    private List<int> FindNeighborsBruteForce(Matrix<T> data, Vector<T> query, T epsilon)
    {
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var within = ClusteringTensorOps.LessOrEqualFlags(
            ClusteringTensorOps.DistancesToPoint(distanceMetric, data, query), epsilon);

        var neighbors = new List<int>();
        for (int i = 0; i < within.Length; i++)
        {
            if (within[i])
            {
                neighbors.Add(i);
            }
        }

        return neighbors;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        // Normalize input using the same parameters from training
        if (_featureMeans is not null && _featureStds is not null)
        {
            if (x.Columns != _featureMeans.Length)
                throw new ArgumentException(
                    $"Input columns ({x.Columns}) must match training data columns ({_featureMeans.Length}).");

            x = ClusteringTensorOps.Standardize(x, _featureMeans, _featureStds);
        }

        // Use normalized centers when input was normalized, to ensure
        // distances are computed in the same space
        var centersForPrediction = _normalizedClusterCenters ?? ClusterCenters;
        int k = centersForPrediction is null ? 0 : Math.Min(NumClusters, centersForPrediction.Rows);
        if (centersForPrediction is null || k <= 0 || x.Rows == 0)
        {
            // No clusters: everything is noise
            var noise = new Vector<T>(x.Rows);
            return x.Rows == 0
                ? noise
                : Engine.TensorFullLike(noise.AsTensor(), NumOps.FromDouble(NoiseLabel)).ToVector();
        }

        // For new data, assign to nearest cluster or noise
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var centers = centersForPrediction.Rows == k
            ? centersForPrediction
            : centersForPrediction.GetSubMatrix(0, 0, k, centersForPrediction.Columns);
        var distances = distanceMetric.ComputePairwise(x, centers);
        var nearest = ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.ArgMinRows(distances)).AsTensor();

        // Prediction runs in normalized feature space, so compare against
        // the fitted normalized epsilon rather than the caller's raw-scale value.
        var tooFar = Engine.TensorGreaterThan(
            ClusteringTensorOps.MinRows(distances).AsTensor(), NumOps.Multiply(_fittedEpsilon, NumOps.FromDouble(2)));
        return Engine.TensorWhere(tooFar, Engine.TensorFullLike(nearest, NumOps.FromDouble(NoiseLabel)), nearest).ToVector();
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        if (Labels is null)
        {
            throw new InvalidOperationException("Train did not produce labels. This indicates a bug in the training implementation.");
        }
        return Labels;
    }

    /// <inheritdoc />
    public override Matrix<T> Transform(Matrix<T> x)
    {
        ValidateIsTrained();

        if (ClusterCenters is null || NumClusters == 0)
        {
            // No clusters found, return single column of zeros
            return new Matrix<T>(x.Rows, 1);
        }

        return base.Transform(x);
    }

    /// <summary>
    /// Gets the indices of core samples.
    /// </summary>
    /// <returns>Array of indices that are core points.</returns>
    public int[] GetCoreSampleIndices()
    {
        if (_corePointMask is null)
        {
            return Array.Empty<int>();
        }

        var indices = new List<int>();
        for (int i = 0; i < _corePointMask.Length; i++)
        {
            if (_corePointMask[i])
            {
                indices.Add(i);
            }
        }

        return indices.ToArray();
    }

    /// <summary>
    /// Gets the number of noise points (outliers).
    /// </summary>
    /// <returns>Count of points labeled as noise.</returns>
    public int GetNoiseCount()
    {
        if (Labels is null)
        {
            return 0;
        }

        int count = 0;
        var labelsIds = ClusteringTensorOps.ToIntArray(Labels);
        for (int i = 0; i < Labels.Length; i++)
        {
            if (labelsIds[i] == NoiseLabel)
            {
                count++;
            }
        }

        return count;
    }

    private void ValidateInputData(Matrix<T> x)
    {
        if (x.Rows < 1)
        {
            throw new ArgumentException("Data must have at least one sample.");
        }

        if (_options.Epsilon <= 0)
        {
            throw new ArgumentException("Epsilon must be positive.");
        }

        if (_options.MinPoints < 1)
        {
            throw new ArgumentException("MinPoints must be at least 1.");
        }
    }

    private void ValidatePredictInput(Matrix<T> x)
    {
        if (ClusterCenters is not null && x.Columns != ClusterCenters.Columns)
        {
            throw new ArgumentException(
                $"Input columns ({x.Columns}) must match training data columns ({ClusterCenters.Columns}).");
        }
    }
}
