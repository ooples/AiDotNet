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
public class SpectralClustering<T> : ClusteringBase<T>
{
    private readonly SpectralOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private double[,]? _embedding;
    private double[,]? _affinityMatrix;

    /// <summary>
    /// Initializes a new SpectralClustering instance.
    /// </summary>
    /// <param name="options">The spectral clustering options.</param>
    public SpectralClustering(SpectralOptions<T>? options = null)
        : base(options ?? new SpectralOptions<T>())
    {
        _options = options ?? new SpectralOptions<T>();
        NumClusters = _options.NumClusters;
    }

    /// <summary>
    /// Gets the spectral embedding.
    /// </summary>
    public double[,]? Embedding => _embedding;

    /// <summary>
    /// Gets the affinity matrix.
    /// </summary>
    public double[,]? AffinityMatrix => _affinityMatrix;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new SpectralClustering<T>(new SpectralOptions<T>
        {
            NumClusters = _options.NumClusters,
            Affinity = _options.Affinity,
            Gamma = _options.Gamma,
            NumNeighbors = _options.NumNeighbors,
            EigenSolver = _options.EigenSolver,
            Normalization = _options.Normalization,
            AssignLabels = _options.AssignLabels,
            DistanceMetric = _options.DistanceMetric
        });
    }

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
            NormalizeRows(_embedding, n, _options.NumClusters);
        }

        // Cluster the embedding
        Labels = ClusterEmbedding(_embedding, n, _options.NumClusters);

        // Compute cluster centers in original space
        ComputeClusterCenters(x);

        IsTrained = true;
    }

    private double[,] BuildAffinityMatrix(Matrix<T> x)
    {
        int n = x.Rows;
        var affinity = new double[n, n];

        switch (_options.Affinity)
        {
            case AffinityType.RBF:
                return BuildRBFAffinity(x, n);

            case AffinityType.NearestNeighbors:
                return BuildNearestNeighborsAffinity(x, n);

            case AffinityType.Precomputed:
                // Assume x is the affinity matrix
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        affinity[i, j] = NumOps.ToDouble(x[i, j]);
                    }
                }
                return affinity;

            default:
                return BuildRBFAffinity(x, n);
        }
    }

    private double[,] BuildRBFAffinity(Matrix<T> x, int n)
    {
        var affinity = new double[n, n];
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Compute gamma if not specified
        double gamma = _options.Gamma ?? 1.0 / x.Columns;

        for (int i = 0; i < n; i++)
        {
            affinity[i, i] = 1.0; // Self-similarity
            var pointI = GetRow(x, i);

            for (int j = i + 1; j < n; j++)
            {
                var pointJ = GetRow(x, j);
                double dist = NumOps.ToDouble(metric.Compute(pointI, pointJ));
                double distSq = dist * dist;

                double similarity = Math.Exp(-gamma * distSq);
                affinity[i, j] = similarity;
                affinity[j, i] = similarity;
            }
        }

        return affinity;
    }

    private double[,] BuildNearestNeighborsAffinity(Matrix<T> x, int n)
    {
        var affinity = new double[n, n];
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        int k = _options.NumNeighbors;

        // Compute all pairwise distances
        var distances = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            var pointI = GetRow(x, i);
            for (int j = i + 1; j < n; j++)
            {
                var pointJ = GetRow(x, j);
                double dist = NumOps.ToDouble(metric.Compute(pointI, pointJ));
                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        // For each point, find k nearest neighbors
        for (int i = 0; i < n; i++)
        {
            var neighborDists = new List<(int Index, double Dist)>();
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    neighborDists.Add((j, distances[i, j]));
                }
            }

            var nearestNeighbors = neighborDists.OrderBy(nd => nd.Dist).Take(k).ToList();

            foreach (var (idx, _) in nearestNeighbors)
            {
                affinity[i, idx] = 1.0;
                affinity[idx, i] = 1.0; // Make symmetric
            }
        }

        return affinity;
    }

    private double[,] ComputeLaplacian(double[,] affinity, int n)
    {
        // Compute degree matrix (sum of each row)
        var degree = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                degree[i] += affinity[i, j];
            }
        }

        var laplacian = new double[n, n];

        switch (_options.Normalization)
        {
            case LaplacianNormalization.Unnormalized:
                // L = D - W
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (i == j)
                        {
                            laplacian[i, j] = degree[i] - affinity[i, j];
                        }
                        else
                        {
                            laplacian[i, j] = -affinity[i, j];
                        }
                    }
                }
                break;

            case LaplacianNormalization.Normalized:
                // L_sym = D^(-1/2) * L * D^(-1/2) = I - D^(-1/2) * W * D^(-1/2)
                for (int i = 0; i < n; i++)
                {
                    double sqrtDi = Math.Sqrt(Math.Max(degree[i], 1e-10));
                    for (int j = 0; j < n; j++)
                    {
                        double sqrtDj = Math.Sqrt(Math.Max(degree[j], 1e-10));
                        if (i == j)
                        {
                            laplacian[i, j] = 1.0 - affinity[i, j] / (sqrtDi * sqrtDj);
                        }
                        else
                        {
                            laplacian[i, j] = -affinity[i, j] / (sqrtDi * sqrtDj);
                        }
                    }
                }
                break;

            case LaplacianNormalization.RandomWalk:
                // L_rw = D^(-1) * L = I - D^(-1) * W
                for (int i = 0; i < n; i++)
                {
                    double invDi = 1.0 / Math.Max(degree[i], 1e-10);
                    for (int j = 0; j < n; j++)
                    {
                        if (i == j)
                        {
                            laplacian[i, j] = 1.0 - affinity[i, j] * invDi;
                        }
                        else
                        {
                            laplacian[i, j] = -affinity[i, j] * invDi;
                        }
                    }
                }
                break;
        }

        return laplacian;
    }

    private double[,] ComputeEigenvectors(double[,] laplacian, int n, int k)
    {
        // Simple power iteration method for finding smallest eigenvalues/eigenvectors
        // Note: For production, should use proper ARPACK implementation
        var eigenvectors = new double[n, k];
        var eigenvalues = new double[k];

        // Initialize with random vectors
        var rand = Random ?? RandomHelper.CreateSecureRandom();
        var vectors = new double[k, n];

        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
            {
                vectors[i, j] = rand.NextDouble() - 0.5;
            }
        }

        // Find eigenvectors corresponding to smallest eigenvalues
        // Using inverse power iteration with deflation
        for (int vec = 0; vec < k; vec++)
        {
            var v = new double[n];
            for (int j = 0; j < n; j++)
            {
                v[j] = vectors[vec, j];
            }

            // Normalize
            double norm = 0;
            for (int j = 0; j < n; j++)
            {
                norm += v[j] * v[j];
            }
            norm = Math.Sqrt(norm);
            for (int j = 0; j < n; j++)
            {
                v[j] /= norm;
            }

            // Power iteration (using shifted matrix to find smallest eigenvalues)
            for (int iter = 0; iter < 100; iter++)
            {
                var newV = MultiplyMatrixVector(laplacian, v, n);

                // Orthogonalize against previous eigenvectors
                for (int prev = 0; prev < vec; prev++)
                {
                    double dot = 0;
                    for (int j = 0; j < n; j++)
                    {
                        dot += newV[j] * eigenvectors[j, prev];
                    }
                    for (int j = 0; j < n; j++)
                    {
                        newV[j] -= dot * eigenvectors[j, prev];
                    }
                }

                // Normalize
                norm = 0;
                for (int j = 0; j < n; j++)
                {
                    norm += newV[j] * newV[j];
                }
                norm = Math.Sqrt(Math.Max(norm, 1e-10));

                for (int j = 0; j < n; j++)
                {
                    v[j] = newV[j] / norm;
                }
            }

            // Store eigenvector
            for (int j = 0; j < n; j++)
            {
                eigenvectors[j, vec] = v[j];
            }

            // Compute eigenvalue
            var Av = MultiplyMatrixVector(laplacian, v, n);
            double eigenval = 0;
            for (int j = 0; j < n; j++)
            {
                eigenval += v[j] * Av[j];
            }
            eigenvalues[vec] = eigenval;
        }

        return eigenvectors;
    }

    private double[] MultiplyMatrixVector(double[,] matrix, double[] vector, int n)
    {
        var result = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i] += matrix[i, j] * vector[j];
            }
        }
        return result;
    }

    private void NormalizeRows(double[,] embedding, int n, int k)
    {
        for (int i = 0; i < n; i++)
        {
            double norm = 0;
            for (int j = 0; j < k; j++)
            {
                norm += embedding[i, j] * embedding[i, j];
            }
            norm = Math.Sqrt(Math.Max(norm, 1e-10));

            for (int j = 0; j < k; j++)
            {
                embedding[i, j] /= norm;
            }
        }
    }

    private Vector<T> ClusterEmbedding(double[,] embedding, int n, int k)
    {
        // Convert embedding to Matrix<T>
        var embeddingMatrix = new Matrix<T>(n, k);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                embeddingMatrix[i, j] = NumOps.FromDouble(embedding[i, j]);
            }
        }

        if (_options.AssignLabels == SpectralAssignment.KMeans)
        {
            // Use K-Means on the embedding
            var kmeans = new KMeans<T>(new KMeansOptions<T>
            {
                NumClusters = k,
                MaxIterations = Options.MaxIterations,
                Seed = Options.Seed
            });
            kmeans.Train(embeddingMatrix);
            return kmeans.Labels!;
        }
        else
        {
            // Discretization method
            return DiscretizeEmbedding(embedding, n, k);
        }
    }

    private Vector<T> DiscretizeEmbedding(double[,] embedding, int n, int k)
    {
        // Simple discretization: assign each point to the cluster
        // corresponding to its largest eigenvector component
        var labels = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            int bestCluster = 0;
            double maxVal = Math.Abs(embedding[i, 0]);

            for (int j = 1; j < k; j++)
            {
                double absVal = Math.Abs(embedding[i, j]);
                if (absVal > maxVal)
                {
                    maxVal = absVal;
                    bestCluster = j;
                }
            }

            labels[i] = NumOps.FromDouble(bestCluster);
        }

        return labels;
    }

    private void ComputeClusterCenters(Matrix<T> x)
    {
        if (Labels is null || NumClusters <= 0)
        {
            return;
        }

        ClusterCenters = new Matrix<T>(NumClusters, x.Columns);
        var counts = new int[NumClusters];

        for (int i = 0; i < x.Rows; i++)
        {
            int cluster = (int)NumOps.ToDouble(Labels[i]);
            if (cluster >= 0 && cluster < NumClusters)
            {
                counts[cluster]++;
                for (int j = 0; j < x.Columns; j++)
                {
                    ClusterCenters[cluster, j] = NumOps.Add(ClusterCenters[cluster, j], x[i, j]);
                }
            }
        }

        for (int k = 0; k < NumClusters; k++)
        {
            if (counts[k] > 0)
            {
                T countT = NumOps.FromDouble(counts[k]);
                for (int j = 0; j < x.Columns; j++)
                {
                    ClusterCenters[k, j] = NumOps.Divide(ClusterCenters[k, j], countT);
                }
            }
        }
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        var labels = new Vector<T>(x.Rows);
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

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
                    double dist = NumOps.ToDouble(metric.Compute(point, center));

                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearestCluster = k;
                    }
                }
            }

            labels[i] = NumOps.FromDouble(nearestCluster);
        }

        return labels;
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels!;
    }

    /// <summary>
    /// Gets the spectral embedding of the data.
    /// </summary>
    /// <returns>Matrix of embedded points [n_samples x n_clusters].</returns>
    public Matrix<T> GetSpectralEmbedding()
    {
        ValidateIsTrained();

        int n = _embedding!.GetLength(0);
        int k = _embedding.GetLength(1);
        var result = new Matrix<T>(n, k);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                result[i, j] = NumOps.FromDouble(_embedding[i, j]);
            }
        }

        return result;
    }
}
