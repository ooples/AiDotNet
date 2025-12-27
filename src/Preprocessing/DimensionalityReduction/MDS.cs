using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Multidimensional Scaling for dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// MDS finds a low-dimensional embedding that preserves pairwise distances between points.
/// Classical MDS uses eigendecomposition of the double-centered distance matrix.
/// Non-metric MDS uses iterative optimization to preserve distance rankings.
/// </para>
/// <para>
/// MDS is useful for:
/// - Visualizing similarity/dissimilarity data
/// - Preserving pairwise relationships
/// - When you have a distance matrix rather than feature vectors
/// </para>
/// <para><b>For Beginners:</b> MDS tries to place points in 2D/3D such that:
/// - Points that were close in high-D stay close in low-D
/// - Points that were far apart stay far apart
/// - Unlike t-SNE/UMAP, MDS tries to preserve actual distances, not just neighborhoods
///
/// Classical MDS: Preserves exact distances (works well when data is linear)
/// Non-metric MDS: Preserves distance rankings (more flexible, works better for complex data)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class MDS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly MDSType _mdsType;
    private readonly MDSMetric _metric;
    private readonly int _maxIter;
    private readonly double _eps;
    private readonly int? _randomState;
    private readonly bool _normalized;

    // Fitted parameters
    private double[,]? _embedding;
    private double _stress;
    private int _nSamples;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the MDS type (classical or non-metric).
    /// </summary>
    public MDSType MdsType => _mdsType;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public double[,]? Embedding => _embedding;

    /// <summary>
    /// Gets the final stress value (goodness of fit).
    /// </summary>
    public double Stress => _stress;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="MDS{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="mdsType">Type of MDS algorithm. Defaults to Classical.</param>
    /// <param name="metric">Distance metric. Defaults to Euclidean.</param>
    /// <param name="maxIter">Maximum iterations for non-metric MDS. Defaults to 300.</param>
    /// <param name="eps">Convergence tolerance. Defaults to 1e-6.</param>
    /// <param name="normalized">Whether to normalize stress. Defaults to false.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public MDS(
        int nComponents = 2,
        MDSType mdsType = MDSType.Classical,
        MDSMetric metric = MDSMetric.Euclidean,
        int maxIter = 300,
        double eps = 1e-6,
        bool normalized = false,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _mdsType = mdsType;
        _metric = metric;
        _maxIter = maxIter;
        _eps = eps;
        _normalized = normalized;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits MDS and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        int n = data.Rows;
        int p = data.Columns;

        // Compute pairwise distance matrix
        var distances = ComputeDistanceMatrix(data, n, p);

        if (_mdsType == MDSType.Classical)
        {
            _embedding = ClassicalMDS(distances, n);
        }
        else
        {
            var random = _randomState.HasValue
                ? RandomHelper.CreateSeededRandom(_randomState.Value)
                : RandomHelper.CreateSeededRandom(42);

            _embedding = NonMetricMDS(distances, n, random);
        }
    }

    private double[,] ComputeDistanceMatrix(Matrix<T> data, int n, int p)
    {
        var distances = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;

                switch (_metric)
                {
                    case MDSMetric.Euclidean:
                        for (int k = 0; k < p; k++)
                        {
                            double diff = NumOps.ToDouble(data[i, k]) - NumOps.ToDouble(data[j, k]);
                            dist += diff * diff;
                        }
                        dist = Math.Sqrt(dist);
                        break;

                    case MDSMetric.SquaredEuclidean:
                        for (int k = 0; k < p; k++)
                        {
                            double diff = NumOps.ToDouble(data[i, k]) - NumOps.ToDouble(data[j, k]);
                            dist += diff * diff;
                        }
                        break;

                    case MDSMetric.Manhattan:
                        for (int k = 0; k < p; k++)
                        {
                            dist += Math.Abs(NumOps.ToDouble(data[i, k]) - NumOps.ToDouble(data[j, k]));
                        }
                        break;
                }

                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        return distances;
    }

    private double[,] ClassicalMDS(double[,] distances, int n)
    {
        // Step 1: Square the distance matrix
        var D2 = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                D2[i, j] = distances[i, j] * distances[i, j];
            }
        }

        // Step 2: Double centering: B = -0.5 * J * D^2 * J
        // where J = I - (1/n) * 1 * 1^T (centering matrix)
        var B = new double[n, n];

        // Compute row means and grand mean
        var rowMeans = new double[n];
        double grandMean = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                rowMeans[i] += D2[i, j];
            }
            rowMeans[i] /= n;
            grandMean += rowMeans[i];
        }
        grandMean /= n;

        // Apply double centering
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                B[i, j] = -0.5 * (D2[i, j] - rowMeans[i] - rowMeans[j] + grandMean);
            }
        }

        // Step 3: Eigendecomposition of B
        var (eigenvalues, eigenvectors) = ComputeEigen(B, n);

        // Step 4: Sort eigenvalues and select top k
        var indices = Enumerable.Range(0, n)
            .OrderByDescending(i => eigenvalues[i])
            .ToArray();

        // Step 5: Compute embedding
        var embedding = new double[n, _nComponents];

        for (int k = 0; k < _nComponents; k++)
        {
            int idx = indices[k];
            double scale = eigenvalues[idx] > 0 ? Math.Sqrt(eigenvalues[idx]) : 0;

            for (int i = 0; i < n; i++)
            {
                embedding[i, k] = eigenvectors[idx, i] * scale;
            }
        }

        // Compute stress
        _stress = ComputeStress(distances, embedding, n);

        return embedding;
    }

    private double[,] NonMetricMDS(double[,] distances, int n, Random random)
    {
        // Initialize with classical MDS
        var embedding = ClassicalMDS(distances, n);

        // SMACOF algorithm for non-metric MDS
        double prevStress = double.MaxValue;

        for (int iter = 0; iter < _maxIter; iter++)
        {
            // Compute current distances in embedding
            var embeddingDist = ComputeEmbeddingDistances(embedding, n);

            // Monotonic regression (isotonic regression on distances)
            var disparities = IsotonicRegression(distances, embeddingDist, n);

            // Compute Guttman transform
            var newEmbedding = GuttmanTransform(embedding, disparities, embeddingDist, n);

            // Compute stress
            double stress = ComputeStressFromDisparities(disparities, newEmbedding, n);

            // Check convergence
            if (Math.Abs(prevStress - stress) < _eps)
            {
                embedding = newEmbedding;
                _stress = stress;
                break;
            }

            embedding = newEmbedding;
            prevStress = stress;
            _stress = stress;
        }

        return embedding;
    }

    private double[,] ComputeEmbeddingDistances(double[,] embedding, int n)
    {
        var dist = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double d = 0;
                for (int k = 0; k < _nComponents; k++)
                {
                    double diff = embedding[i, k] - embedding[j, k];
                    d += diff * diff;
                }
                d = Math.Sqrt(d);
                dist[i, j] = d;
                dist[j, i] = d;
            }
        }

        return dist;
    }

    private double[,] IsotonicRegression(double[,] originalDist, double[,] embeddingDist, int n)
    {
        // Pool Adjacent Violators (PAV) algorithm for isotonic regression
        var disparities = new double[n, n];

        // Create sorted list of original distances
        var pairs = new List<(int i, int j, double orig, double emb)>();
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                pairs.Add((i, j, originalDist[i, j], embeddingDist[i, j]));
            }
        }

        // Sort by original distance
        pairs.Sort((a, b) => a.orig.CompareTo(b.orig));

        // Apply PAV
        var fitted = pairs.Select(p => p.emb).ToArray();
        int m = fitted.Length;

        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int i = 0; i < m - 1; i++)
            {
                if (fitted[i] > fitted[i + 1])
                {
                    double avg = (fitted[i] + fitted[i + 1]) / 2;
                    fitted[i] = avg;
                    fitted[i + 1] = avg;
                    changed = true;
                }
            }
        }

        // Map back to matrix
        for (int k = 0; k < m; k++)
        {
            var (i, j, _, _) = pairs[k];
            disparities[i, j] = fitted[k];
            disparities[j, i] = fitted[k];
        }

        return disparities;
    }

    private double[,] GuttmanTransform(double[,] embedding, double[,] disparities, double[,] embeddingDist, int n)
    {
        var newEmbedding = new double[n, _nComponents];

        // Compute B matrix
        var B = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            double rowSum = 0;
            for (int j = 0; j < n; j++)
            {
                if (i != j && embeddingDist[i, j] > 1e-10)
                {
                    B[i, j] = -disparities[i, j] / embeddingDist[i, j];
                    rowSum += B[i, j];
                }
            }
            B[i, i] = -rowSum;
        }

        // New embedding: X_new = (1/n) * B * X_old
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < _nComponents; k++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++)
                {
                    sum += B[i, j] * embedding[j, k];
                }
                newEmbedding[i, k] = sum / n;
            }
        }

        return newEmbedding;
    }

    private double ComputeStress(double[,] originalDist, double[,] embedding, int n)
    {
        double stress = 0;
        double normalization = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double embDist = 0;
                for (int k = 0; k < _nComponents; k++)
                {
                    double diff = embedding[i, k] - embedding[j, k];
                    embDist += diff * diff;
                }
                embDist = Math.Sqrt(embDist);

                double diff2 = originalDist[i, j] - embDist;
                stress += diff2 * diff2;
                normalization += originalDist[i, j] * originalDist[i, j];
            }
        }

        return _normalized && normalization > 0
            ? Math.Sqrt(stress / normalization)
            : Math.Sqrt(stress);
    }

    private double ComputeStressFromDisparities(double[,] disparities, double[,] embedding, int n)
    {
        var embDist = ComputeEmbeddingDistances(embedding, n);
        double stress = 0;
        double normalization = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double diff = disparities[i, j] - embDist[i, j];
                stress += diff * diff;
                normalization += disparities[i, j] * disparities[i, j];
            }
        }

        return _normalized && normalization > 0
            ? Math.Sqrt(stress / normalization)
            : Math.Sqrt(stress);
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeEigen(double[,] matrix, int n)
    {
        // Power iteration with deflation
        var eigenvalues = new double[n];
        var eigenvectors = new double[n, n];
        var A = (double[,])matrix.Clone();

        for (int k = 0; k < Math.Min(n, _nComponents + 5); k++)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = 1.0 / Math.Sqrt(n);
            }

            for (int iter = 0; iter < 100; iter++)
            {
                var Av = new double[n];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        Av[i] += A[i, j] * v[j];
                    }
                }

                double norm = 0;
                for (int i = 0; i < n; i++)
                {
                    norm += Av[i] * Av[i];
                }
                norm = Math.Sqrt(norm);

                if (norm < 1e-10) break;

                for (int i = 0; i < n; i++)
                {
                    v[i] = Av[i] / norm;
                }
            }

            var Av2 = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Av2[i] += A[i, j] * v[j];
                }
            }

            double eigenvalue = 0;
            for (int i = 0; i < n; i++)
            {
                eigenvalue += v[i] * Av2[i];
            }

            eigenvalues[k] = eigenvalue;
            for (int i = 0; i < n; i++)
            {
                eigenvectors[k, i] = v[i];
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    A[i, j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        return (eigenvalues, eigenvectors);
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("MDS has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "MDS does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        int n = _embedding.GetLength(0);
        int d = _embedding.GetLength(1);
        var result = new T[n, d];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                result[i, j] = NumOps.FromDouble(_embedding[i, j]);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MDS does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"MDS{i + 1}";
        }
        return names;
    }
}
