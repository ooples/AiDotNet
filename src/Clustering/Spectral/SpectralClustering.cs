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
/// var options = new SpectralOptions&lt;double&gt;();
/// var spectralClustering = new SpectralClustering&lt;double&gt;(options);
/// spectralClustering.Train(dataMatrix);
/// Vector&lt;double&gt;? labels = spectralClustering.Labels;
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("On Spectral Clustering: Analysis and an algorithm", "https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html", Year = 2002, Authors = "Andrew Ng, Michael Jordan, Yair Weiss")]
public class SpectralClustering<T> : ClusteringBase<T>
{
    private readonly SpectralOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private T[,]? _embedding;
    private T[,]? _affinityMatrix;

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
    public T[,]? Embedding => _embedding;

    /// <summary>
    /// Gets the affinity matrix.
    /// </summary>
    public T[,]? AffinityMatrix => _affinityMatrix;

    /// <inheritdoc />

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

        MergeDegenerateClusters(x);
        IsTrained = true;
    }

    private T[,] BuildAffinityMatrix(Matrix<T> x)
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
                var affinity = new T[n, n];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        affinity[i, j] = x[i, j];
                    }
                }
                return affinity;

            default:
                return BuildRBFAffinity(x, n);
        }
    }

    private T[,] BuildRBFAffinity(Matrix<T> x, int n)
    {
        var affinity = new T[n, n];
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Compute gamma if not specified. Use 1/(n_features * Var(X)) following
        // scikit-learn's "scale" convention. The naive 1/n_features doesn't account for
        // data scale, causing the RBF kernel to collapse when features are large.
        double gammaValue;
        if (_options.Gamma.HasValue)
        {
            gammaValue = _options.Gamma.Value;
        }
        else
        {
            double totalVar = 0;
            for (int j = 0; j < x.Columns; j++)
            {
                double mean = 0;
                for (int i2 = 0; i2 < n; i2++)
                    mean += NumOps.ToDouble(x[i2, j]);
                mean /= n;

                double featureVar = 0;
                for (int i2 = 0; i2 < n; i2++)
                {
                    double diff = NumOps.ToDouble(x[i2, j]) - mean;
                    featureVar += diff * diff;
                }
                totalVar += featureVar / n;
            }
            gammaValue = totalVar > 1e-10 ? 1.0 / totalVar : 1.0 / x.Columns;
        }
        T gamma = NumOps.FromDouble(gammaValue);
        int d = x.Columns;

        // Cache rows as arrays for allocation-free distance
        var rowArrays = new T[n][];
        for (int i = 0; i < n; i++)
        {
            rowArrays[i] = new T[d];
            for (int c = 0; c < d; c++)
                rowArrays[i][c] = x[i, c];
        }

        for (int i = 0; i < n; i++)
        {
            affinity[i, i] = NumOps.One;

            for (int j = i + 1; j < n; j++)
            {
                T dist = metric.ComputeInline(rowArrays[i], rowArrays[j], d);
                T distSq = NumOps.Multiply(dist, dist);

                T similarity = NumOps.Exp(NumOps.Negate(NumOps.Multiply(gamma, distSq)));
                affinity[i, j] = similarity;
                affinity[j, i] = similarity;
            }
        }

        return affinity;
    }

    private T[,] BuildNearestNeighborsAffinity(Matrix<T> x, int n)
    {
        var affinity = new T[n, n];
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        int k = _options.NumNeighbors;

        // Initialize to zero
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                affinity[i, j] = NumOps.Zero;

        // Compute all pairwise distances with allocation-free inline metric
        int d2 = x.Columns;
        var nnRowArrays = new T[n][];
        for (int i = 0; i < n; i++)
        {
            nnRowArrays[i] = new T[d2];
            for (int c = 0; c < d2; c++)
                nnRowArrays[i][c] = x[i, c];
        }

        var distances = new T[n, n];
        for (int i = 0; i < n; i++)
        {
            distances[i, i] = NumOps.Zero;
            for (int j = i + 1; j < n; j++)
            {
                T dist = metric.ComputeInline(nnRowArrays[i], nnRowArrays[j], d2);
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
                    neighborDists.Add((j, NumOps.ToDouble(distances[i, j])));
                }
            }

            var nearestNeighbors = neighborDists.OrderBy(nd => nd.Dist).Take(k).ToList();

            foreach (var (idx, _) in nearestNeighbors)
            {
                affinity[i, idx] = NumOps.One;
                affinity[idx, i] = NumOps.One; // Make symmetric
            }
        }

        return affinity;
    }

    private T[,] ComputeLaplacian(T[,] affinity, int n)
    {
        T epsilon = NumOps.FromDouble(1e-10);

        // Compute degree matrix (sum of each row)
        var degree = new T[n];
        for (int i = 0; i < n; i++)
        {
            degree[i] = NumOps.Zero;
            for (int j = 0; j < n; j++)
            {
                degree[i] = NumOps.Add(degree[i], affinity[i, j]);
            }
        }

        var laplacian = new T[n, n];

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
                            laplacian[i, j] = NumOps.Subtract(degree[i], affinity[i, j]);
                        }
                        else
                        {
                            laplacian[i, j] = NumOps.Negate(affinity[i, j]);
                        }
                    }
                }
                break;

            case LaplacianNormalization.Normalized:
                // L_sym = I - D^(-1/2) * W * D^(-1/2)
                for (int i = 0; i < n; i++)
                {
                    T di = NumOps.LessThan(degree[i], epsilon) ? epsilon : degree[i];
                    T sqrtDi = NumOps.Sqrt(di);
                    for (int j = 0; j < n; j++)
                    {
                        T dj = NumOps.LessThan(degree[j], epsilon) ? epsilon : degree[j];
                        T sqrtDj = NumOps.Sqrt(dj);
                        T normalized = NumOps.Divide(affinity[i, j], NumOps.Multiply(sqrtDi, sqrtDj));
                        if (i == j)
                        {
                            laplacian[i, j] = NumOps.Subtract(NumOps.One, normalized);
                        }
                        else
                        {
                            laplacian[i, j] = NumOps.Negate(normalized);
                        }
                    }
                }
                break;

            case LaplacianNormalization.RandomWalk:
                // L_rw = I - D^(-1) * W
                for (int i = 0; i < n; i++)
                {
                    T di = NumOps.LessThan(degree[i], epsilon) ? epsilon : degree[i];
                    T invDi = NumOps.Divide(NumOps.One, di);
                    for (int j = 0; j < n; j++)
                    {
                        if (i == j)
                        {
                            laplacian[i, j] = NumOps.Subtract(NumOps.One, NumOps.Multiply(affinity[i, j], invDi));
                        }
                        else
                        {
                            laplacian[i, j] = NumOps.Negate(NumOps.Multiply(affinity[i, j], invDi));
                        }
                    }
                }
                break;
        }

        return laplacian;
    }

    private T[,] ComputeEigenvectors(T[,] laplacian, int n, int k)
    {
        // Simple power iteration method for finding smallest eigenvalues/eigenvectors
        var eigenvectors = new T[n, k];
        T epsilon = NumOps.FromDouble(1e-10);

        // Initialize with random vectors
        var rand = Random ?? RandomHelper.CreateSecureRandom();
        var vectors = new T[k, n];

        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
            {
                vectors[i, j] = NumOps.FromDouble(rand.NextDouble() - 0.5);
            }
        }

        // Find eigenvectors corresponding to smallest eigenvalues
        for (int vec = 0; vec < k; vec++)
        {
            var v = new T[n];
            for (int j = 0; j < n; j++)
            {
                v[j] = vectors[vec, j];
            }

            // Normalize
            T norm = NumOps.Zero;
            for (int j = 0; j < n; j++)
            {
                norm = NumOps.Add(norm, NumOps.Multiply(v[j], v[j]));
            }
            norm = NumOps.Sqrt(norm);
            for (int j = 0; j < n; j++)
            {
                v[j] = NumOps.Divide(v[j], norm);
            }

            // Shifted power iteration: use (σI - L) to find smallest eigenvalues
            // as largest eigenvalues of the shifted matrix.
            // σ = trace(L)/n + 1 is a safe upper bound for Laplacian eigenvalues.
            double sigma = 0;
            for (int j = 0; j < n; j++)
                sigma += NumOps.ToDouble(laplacian[j, j]);
            sigma = sigma / n + 1.0;

            for (int iter = 0; iter < 100; iter++)
            {
                // Compute (σI - L) * v instead of L * v
                var lv = MultiplyMatrixVector(laplacian, v, n);
                var newV = new T[n];
                for (int j = 0; j < n; j++)
                    newV[j] = NumOps.Subtract(NumOps.Multiply(NumOps.FromDouble(sigma), v[j]), lv[j]);

                // Orthogonalize against previous eigenvectors
                for (int prev = 0; prev < vec; prev++)
                {
                    T dot = NumOps.Zero;
                    for (int j = 0; j < n; j++)
                    {
                        dot = NumOps.Add(dot, NumOps.Multiply(newV[j], eigenvectors[j, prev]));
                    }
                    for (int j = 0; j < n; j++)
                    {
                        newV[j] = NumOps.Subtract(newV[j], NumOps.Multiply(dot, eigenvectors[j, prev]));
                    }
                }

                // Normalize
                norm = NumOps.Zero;
                for (int j = 0; j < n; j++)
                {
                    norm = NumOps.Add(norm, NumOps.Multiply(newV[j], newV[j]));
                }
                if (NumOps.LessThan(norm, epsilon))
                {
                    norm = epsilon;
                }
                norm = NumOps.Sqrt(norm);

                for (int j = 0; j < n; j++)
                {
                    v[j] = NumOps.Divide(newV[j], norm);
                }
            }

            // Store eigenvector
            for (int j = 0; j < n; j++)
            {
                eigenvectors[j, vec] = v[j];
            }
        }

        return eigenvectors;
    }

    private T[] MultiplyMatrixVector(T[,] matrix, T[] vector, int n)
    {
        var result = new T[n];
        for (int i = 0; i < n; i++)
        {
            result[i] = NumOps.Zero;
            for (int j = 0; j < n; j++)
            {
                result[i] = NumOps.Add(result[i], NumOps.Multiply(matrix[i, j], vector[j]));
            }
        }
        return result;
    }

    private void NormalizeRows(T[,] embedding, int n, int k)
    {
        T epsilon = NumOps.FromDouble(1e-10);
        for (int i = 0; i < n; i++)
        {
            T norm = NumOps.Zero;
            for (int j = 0; j < k; j++)
            {
                norm = NumOps.Add(norm, NumOps.Multiply(embedding[i, j], embedding[i, j]));
            }
            if (NumOps.LessThan(norm, epsilon))
            {
                norm = epsilon;
            }
            norm = NumOps.Sqrt(norm);

            for (int j = 0; j < k; j++)
            {
                embedding[i, j] = NumOps.Divide(embedding[i, j], norm);
            }
        }
    }

    private Vector<T> ClusterEmbedding(T[,] embedding, int n, int k)
    {
        // Convert embedding to Matrix<T>
        var embeddingMatrix = new Matrix<T>(n, k);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                embeddingMatrix[i, j] = embedding[i, j];
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
            return kmeans.Labels ?? new Vector<T>(0);
        }
        else
        {
            // Discretization method
            return DiscretizeEmbedding(embedding, n, k);
        }
    }

    private Vector<T> DiscretizeEmbedding(T[,] embedding, int n, int k)
    {
        // Simple discretization: assign each point to the cluster
        // corresponding to its largest eigenvector component
        var labels = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            int bestCluster = 0;
            T maxVal = NumOps.Abs(embedding[i, 0]);

            for (int j = 1; j < k; j++)
            {
                T absVal = NumOps.Abs(embedding[i, j]);
                if (NumOps.GreaterThan(absVal, maxVal))
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
        int dims = x.Columns;
        var pointArr = new T[dims];
        var centerArr = new T[dims];

        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < dims; j++) pointArr[j] = x[i, j];
            T minDist = NumOps.MaxValue;
            int nearestCluster = 0;

            if (ClusterCenters is not null)
            {
                for (int k = 0; k < NumClusters; k++)
                {
                    for (int j = 0; j < dims; j++) centerArr[j] = ClusterCenters[k, j];
                    T dist = metric.ComputeInline(pointArr, centerArr, dims);

                    if (NumOps.LessThan(dist, minDist))
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
        return Labels ?? new Vector<T>(0);
    }

    /// <summary>
    /// Gets the spectral embedding of the data.
    /// </summary>
    /// <returns>Matrix of embedded points [n_samples x n_clusters].</returns>
    public Matrix<T> GetSpectralEmbedding()
    {
        ValidateIsTrained();

        if (_embedding is null) return new Matrix<T>(0, 0);

        int n = _embedding.GetLength(0);
        int k = _embedding.GetLength(1);
        var result = new Matrix<T>(n, k);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                result[i, j] = _embedding[i, j];
            }
        }

        return result;
    }
}
