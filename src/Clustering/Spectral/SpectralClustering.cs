using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Spectral;

/// <summary>
/// Spectral Clustering implementation using graph Laplacian eigendecomposition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Spectral clustering works by:
/// 1. Building an affinity (similarity) matrix
/// 2. Computing the graph Laplacian
/// 3. Finding eigenvectors corresponding to smallest eigenvalues
/// 4. Clustering points in the reduced eigenspace
/// </para>
/// <para><b>For Beginners:</b> Spectral clustering is like finding communities in a network.
///
/// The key insight: Similar points should be in the same cluster.
/// Instead of looking at distances directly, we:
/// 1. Build a "friendship network" where similar points are connected
/// 2. Find the natural groups in this network
/// 3. Use these groups as clusters
///
/// This works better than K-Means when:
/// - Clusters aren't round/spherical
/// - Clusters have complex shapes (moons, spirals)
/// - You care about connectivity, not just distance
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new SpectralClustering&lt;double&gt;(new SpectralOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// Vector&lt;double&gt;? labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("On Spectral Clustering: Analysis and an algorithm", "https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html", Year = 2002, Authors = "Andrew Ng, Michael Jordan, Yair Weiss")]
public partial class SpectralClustering<T> : ClusteringBase<T>
{
    private readonly SpectralOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Matrix<T>? _embedding;
    private Matrix<T>? _affinityMatrix;

    /// <summary>
    /// Initializes a new SpectralClustering instance.
    /// </summary>
    /// <param name="options">The spectral clustering options.</param>
    public SpectralClustering(SpectralOptions<T>? options = null)
        : base(options ??= new SpectralOptions<T>())
    {
        _options = options;
        NumClusters = _options.NumClusters;
    }

    /// <summary>
    /// Gets the spectral embedding.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets the affinity matrix.
    /// </summary>
    public Matrix<T>? AffinityMatrix => _affinityMatrix;

    /// <inheritdoc />

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (SpectralClustering<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        if (n < _options.NumClusters)
        {
            throw new ArgumentException($"Need at least {_options.NumClusters} samples for {_options.NumClusters} clusters.");
        }

        // Build affinity matrix
        _affinityMatrix = BuildAffinityMatrix(x);

        // Compute graph Laplacian
        var laplacian = ComputeLaplacian(_affinityMatrix, n);

        // Compute eigenvectors
        _embedding = ComputeEigenvectors(laplacian, n, _options.NumClusters);

        // Normalize rows of embedding for normalized cut
        if (_options.Normalization == LaplacianNormalization.Normalized)
        {
            _embedding = NormalizeRows(_embedding);
        }

        // Cluster the embedding
        Labels = ClusterEmbedding(_embedding, n, _options.NumClusters);

        // Compute cluster centers in original space
        ComputeClusterCenters(x);

        MergeDegenerateClusters(x);
        IsTrained = true;
    }

    private Matrix<T> BuildAffinityMatrix(Matrix<T> x)
    {
        int n = x.Rows;

        switch (_options.Affinity)
        {
            case AffinityType.RBF:
                return BuildRBFAffinity(x, n);

            case AffinityType.NearestNeighbors:
                return BuildNearestNeighborsAffinity(x, n);

            case AffinityType.Precomputed:
                // Assume x is the affinity matrix
                return x.GetSubMatrix(0, 0, n, n);

            default:
                return BuildRBFAffinity(x, n);
        }
    }

    private Matrix<T> BuildRBFAffinity(Matrix<T> x, int n)
    {
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Compute gamma if not specified. Use 1/(n_features * Var(X)) following
        // scikit-learn's "scale" convention. The naive 1/n_features doesn't account for
        // data scale, causing the RBF kernel to collapse when features are large.
        T gamma;
        if (_options.Gamma.HasValue)
        {
            gamma = NumOps.FromDouble(_options.Gamma.Value);
        }
        else
        {
            // Sum of the per-feature population variances
            var centered = ClusteringTensorOps.SubtractRow(x, ClusteringTensorOps.ColumnMeans(x));
            T totalVar = Engine.TensorSum(Engine.ReduceMean(Engine.TensorSquare(centered), new[] { 0 }, false));
            gamma = NumOps.GreaterThan(totalVar, NumOps.FromDouble(1e-10))
                ? NumOps.Divide(NumOps.One, totalVar)
                : NumOps.Divide(NumOps.One, NumOps.FromDouble(x.Columns));
        }

        // exp(-gamma * dist^2) off the diagonal, 1 on it
        var distances = Tensor<T>.FromMatrix(metric.ComputePairwise(x));
        var similarity = Engine.TensorExp(Engine.TensorNegate(Engine.TensorMultiplyScalar(Engine.TensorSquare(distances), gamma)));
        return Engine.TensorWhere(Engine.TensorEye<T>(n), Engine.TensorOnesLike(similarity), similarity).ToMatrix();
    }

    private Matrix<T> BuildNearestNeighborsAffinity(Matrix<T> x, int n)
    {
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        int k = Math.Min(_options.NumNeighbors, n - 1);
        if (k <= 0)
        {
            return new Matrix<T>(n, n);
        }

        // Pairwise distances; a point is never its own neighbour
        var shape = new[] { n, n };
        var distances = Tensor<T>.FromMatrix(metric.ComputePairwise(x));
        var offDiagonal = Engine.ScalarMinusTensor(NumOps.One, Engine.TensorEye<T>(n));
        var candidates = Engine.TensorWhere(offDiagonal, distances, Engine.TensorFullLike(distances, NumOps.MaxValue));

        // The k nearest per row, ties broken by lower index (stable ordering):
        // every distance below the k-th smallest, plus the first equal ones needed to reach k
        var (sorted, _) = Engine.TensorSort(candidates, 1, false);
        var kth = Engine.TensorBroadcastTo(Engine.TensorSlice(sorted, new[] { 0, k - 1 }, new[] { n, 1 }), shape);
        var below = Engine.TensorMultiply(Engine.TensorGreaterThan(kth, candidates), offDiagonal);
        var equal = Engine.TensorMultiply(Engine.TensorEquals(candidates, kth), offDiagonal);
        var needed = Engine.TensorBroadcastTo(
            Engine.ScalarMinusTensor(NumOps.FromDouble(k), Engine.ReduceSum(below, new[] { 1 }, true)), shape);
        var withinNeeded = Engine.TensorGreaterThan(
            Engine.TensorSubtract(needed, Engine.TensorCumSum(equal, 1)), NumOps.FromDouble(-0.5));
        var selected = Engine.TensorAdd(below, Engine.TensorMultiply(equal, withinNeeded));

        // Make symmetric
        return Engine.TensorMax(selected, Engine.TensorTranspose(selected)).ToMatrix();
    }

    private Matrix<T> ComputeLaplacian(Matrix<T> affinity, int n)
    {
        T epsilon = NumOps.FromDouble(1e-10);
        var shape = new[] { n, n };
        var weights = Tensor<T>.FromMatrix(affinity);
        var identity = Engine.TensorEye<T>(n);

        // Degree of each node (sum of each row), as an [n, 1] column
        var degree = Engine.ReduceSum(weights, new[] { 1 }, true);
        var flooredDegree = Engine.TensorMax(degree, epsilon);

        switch (_options.Normalization)
        {
            case LaplacianNormalization.Unnormalized:
                // L = D - W
                return Engine.TensorSubtract(
                    Engine.TensorMultiply(identity, Engine.TensorBroadcastTo(degree, shape)), weights).ToMatrix();

            case LaplacianNormalization.Normalized:
            {
                // L_sym = I - D^(-1/2) * W * D^(-1/2)
                var sqrtDegree = Engine.TensorSqrt(flooredDegree);
                var scale = Engine.TensorMatMul(sqrtDegree, Engine.TensorTranspose(sqrtDegree));
                return Engine.TensorSubtract(identity, Engine.TensorDivide(weights, scale)).ToMatrix();
            }

            case LaplacianNormalization.RandomWalk:
            {
                // L_rw = I - D^(-1) * W
                var inverseDegree = Engine.TensorReciprocal(flooredDegree);
                return Engine.TensorSubtract(
                    identity, Engine.TensorMultiply(weights, Engine.TensorBroadcastTo(inverseDegree, shape))).ToMatrix();
            }

            default:
                return new Matrix<T>(n, n);
        }
    }

    private Matrix<T> ComputeEigenvectors(Matrix<T> laplacian, int n, int k)
    {
        // Simple power iteration method for finding smallest eigenvalues/eigenvectors
        T epsilon = NumOps.FromDouble(1e-10);
        T tolerance = NumOps.FromDouble(1e-12);

        // Initialize with random vectors
        var rand = Random ?? RandomHelper.CreateSecureRandom();
        // [k, n] start vectors with entries rand.NextDouble() - 0.5, drawn in row-major order,
        // each row normalized to unit length
        var starts = Engine.TensorSubtractScalar(
            Tensor<T>.FromMatrix(Matrix<T>.CreateRandom(rand, k, n)), NumOps.FromDouble(0.5));
        var startNorms = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(starts), new[] { 1 }, true));
        var vectors = Engine.TensorDivide(starts, Engine.TensorBroadcastTo(startNorms, new[] { k, n })).ToMatrix();

        // Shifted power iteration: use (sigma I - L) to find smallest eigenvalues
        // as largest eigenvalues of the shifted matrix.
        // sigma = trace(L)/n + 1 is a safe upper bound for Laplacian eigenvalues.
        var laplacianTensor = Tensor<T>.FromMatrix(laplacian);
        T sigma = NumOps.Add(
            NumOps.Divide(Engine.TensorSum(Engine.TensorDiagonal(laplacianTensor)), NumOps.FromDouble(n)), NumOps.One);

        // Find eigenvectors corresponding to smallest eigenvalues
        var eigenvectors = new List<Vector<T>>();
        for (int vec = 0; vec < k; vec++)
        {
            var v = vectors.GetRow(vec);

            for (int iter = 0; iter < 300; iter++)
            {
                // Compute (sigma I - L) * v instead of L * v
                var lv = Engine.Reshape(Engine.TensorMatMul(laplacianTensor, Tensor<T>.FromVector(v, new[] { n, 1 })), new[] { n }).ToVector();
                var newV = Engine.Subtract(Engine.Multiply(v, sigma), lv);

                // Orthogonalize against previous eigenvectors (two passes for numerical stability)
                for (int pass = 0; pass < 2; pass++)
                {
                    foreach (var previous in eigenvectors)
                    {
                        newV = Engine.Subtract(newV, Engine.Multiply(previous, Engine.DotProduct(newV, previous)));
                    }
                }

                // Normalize, then check convergence: ||v_new - v_old|| < tolerance
                var normalized = Engine.Divide(newV, FlooredNorm(newV, epsilon));
                var change = Engine.Subtract(normalized, v);
                v = normalized;

                if (IsBelow(Engine.DotProduct(change, change), tolerance)) break;
            }

            // Store eigenvector
            eigenvectors.Add(v);
        }

        return k == 0
            ? new Matrix<T>(n, 0)
            : Engine.TensorStack(eigenvectors.Select(e => e.AsTensor()).ToArray(), 1).ToMatrix();
    }

    /// <summary>Euclidean norm of <paramref name="v"/>, raised to <paramref name="floor"/> when smaller.</summary>
    private T FlooredNorm(Vector<T> v, T floor)
    {
        T norm = NumOps.Sqrt(Engine.DotProduct(v, v));
        return NumOps.LessThan(norm, floor) ? floor : norm;
    }

    /// <summary>True when <paramref name="value"/> is strictly below <paramref name="threshold"/>.</summary>
    private bool IsBelow(T value, T threshold) => NumOps.LessThan(value, threshold);

    private Matrix<T> NormalizeRows(Matrix<T> embedding)
    {
        // Divide each row by sqrt(max(||row||^2, epsilon))
        T epsilon = NumOps.FromDouble(1e-10);
        var rows = Tensor<T>.FromMatrix(embedding);
        var norms = Engine.TensorSqrt(Engine.TensorMax(Engine.ReduceSum(Engine.TensorSquare(rows), new[] { 1 }, true), epsilon));
        return Engine.TensorDivide(rows, Engine.TensorBroadcastTo(norms, new[] { embedding.Rows, embedding.Columns })).ToMatrix();
    }

    private Vector<T> ClusterEmbedding(Matrix<T> embedding, int n, int k)
    {
        if (_options.AssignLabels == SpectralAssignment.KMeans)
        {
            // Use K-Means on the embedding
            var kmeans = new KMeans<T>(new KMeansOptions<T>
            {
                NumClusters = k,
                MaxIterations = Options.MaxIterations,
                Seed = Options.Seed
            });
            kmeans.Train(embedding);
            return kmeans.Labels ?? new Vector<T>(0);
        }
        else
        {
            // Discretization method
            return DiscretizeEmbedding(embedding, n, k);
        }
    }

    private Vector<T> DiscretizeEmbedding(Matrix<T> embedding, int n, int k)
    {
        // Simple discretization: assign each point to the cluster
        // corresponding to its largest eigenvector component (first on ties)
        if (k == 0)
        {
            return new Vector<T>(n);
        }

        return ClusteringTensorOps.FromIntArray<T>(
            Engine.TensorArgMax(Engine.TensorAbs(Tensor<T>.FromMatrix(embedding)), 1).ToArray());
    }

    private void ComputeClusterCenters(Matrix<T> x)
    {
        if (Labels is null || NumClusters <= 0)
        {
            return;
        }

        // Mean of each cluster's members (empty clusters stay at zero)
        ClusterCenters = ClusteringTensorOps.ClusterMeans(x, Labels, NumClusters, out _);
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        if (ClusterCenters is null || NumClusters == 0)
        {
            return new Vector<T>(x.Rows);
        }

        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var centers = ClusterCenters.Rows == NumClusters
            ? ClusterCenters
            : ClusterCenters.GetSubMatrix(0, 0, NumClusters, ClusterCenters.Columns);
        return ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.NearestCenters(metric, x, centers));
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }

    /// <summary>
    /// Gets the spectral embedding of the data.
    /// </summary>
    /// <returns>Matrix of embedded points [n_samples x n_clusters].</returns>
    public Matrix<T> GetSpectralEmbedding()
    {
        ValidateIsTrained();

        return _embedding is null ? new Matrix<T>(0, 0) : _embedding.Clone();
    }
}
