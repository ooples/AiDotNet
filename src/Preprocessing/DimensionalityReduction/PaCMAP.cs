using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// PaCMAP: Pairwise Controlled Manifold Approximation.
/// </summary>
/// <remarks>
/// <para>
/// PaCMAP is a dimensionality reduction method designed to preserve both local and global
/// structure. It uses three types of point pairs with carefully controlled weights during
/// optimization to achieve better structure preservation than t-SNE or UMAP.
/// </para>
/// <para>
/// The algorithm:
/// 1. Create three types of pairs: nearby (local), mid-near (intermediate), further (global)
/// 2. Initialize embedding using PCA
/// 3. Optimize using attractive/repulsive forces with dynamic weighting
/// 4. Gradually shift from global to local focus during optimization
/// </para>
/// <para><b>For Beginners:</b> PaCMAP improves on UMAP/t-SNE by:
/// - Preserving global structure better through mid-near and further pairs
/// - Using controlled pair selection instead of random sampling
/// - Dynamically adjusting focus from global to local structure
/// - Being more robust to hyperparameter choices
///
/// Use cases:
/// - When you need faithful global structure
/// - When t-SNE/UMAP produces fragmented clusters
/// - When relative distances between clusters matter
/// - Biological data, image embeddings, document visualization
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class PaCMAP<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly double _mnRatio;
    private readonly double _fpRatio;
    private readonly int _nIter;
    private readonly double _learningRate;
    private readonly int? _randomState;

    // Fitted parameters
    private double[,]? _embedding;
    private int _nSamples;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the number of neighbors.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public double[,]? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="PaCMAP{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors for nearby pairs. Defaults to 10.</param>
    /// <param name="mnRatio">Ratio of mid-near pairs to neighbors. Defaults to 0.5.</param>
    /// <param name="fpRatio">Ratio of further pairs to neighbors. Defaults to 2.0.</param>
    /// <param name="nIter">Number of optimization iterations. Defaults to 450.</param>
    /// <param name="learningRate">Learning rate for optimization. Defaults to 1.0.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public PaCMAP(
        int nComponents = 2,
        int nNeighbors = 10,
        double mnRatio = 0.5,
        double fpRatio = 2.0,
        int nIter = 450,
        double learningRate = 1.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (nNeighbors < 1)
        {
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        }

        _nComponents = nComponents;
        _nNeighbors = nNeighbors;
        _mnRatio = mnRatio;
        _fpRatio = fpRatio;
        _nIter = nIter;
        _learningRate = learningRate;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits PaCMAP and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Convert to double array
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Step 1: Compute pairwise distances
        var distances = ComputePairwiseDistances(X, n, p);

        // Step 2: Generate pairs
        var (nearbyPairs, midNearPairs, furtherPairs) = GeneratePairs(distances, n, random);

        // Step 3: Initialize embedding with PCA
        var Y = InitializeWithPCA(X, n, p, random);

        // Step 4: Optimize using PaCMAP loss
        OptimizeEmbedding(Y, nearbyPairs, midNearPairs, furtherPairs, distances, n, random);

        _embedding = Y;
    }

    private double[,] ComputePairwiseDistances(double[,] X, int n, int p)
    {
        var distances = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;
                for (int k = 0; k < p; k++)
                {
                    double diff = X[i, k] - X[j, k];
                    dist += diff * diff;
                }
                distances[i, j] = dist; // Keep squared distances
                distances[j, i] = dist;
            }
        }

        return distances;
    }

    private (List<(int, int)> nearby, List<(int, int)> midNear, List<(int, int)> further)
        GeneratePairs(double[,] distances, int n, Random random)
    {
        int k = Math.Min(_nNeighbors, n - 1);
        int nMidNear = Math.Max(1, (int)(k * _mnRatio));
        int nFurther = Math.Max(1, (int)(k * _fpRatio));

        var nearbyPairs = new List<(int, int)>();
        var midNearPairs = new List<(int, int)>();
        var furtherPairs = new List<(int, int)>();

        // For each point, find neighbors at different distance scales
        for (int i = 0; i < n; i++)
        {
            var sortedDistances = new (double dist, int idx)[n];
            for (int j = 0; j < n; j++)
            {
                sortedDistances[j] = (j == i ? double.MaxValue : distances[i, j], j);
            }
            Array.Sort(sortedDistances, (a, b) => a.dist.CompareTo(b.dist));

            // Nearby pairs (k nearest neighbors)
            for (int j = 0; j < k && j < n - 1; j++)
            {
                nearbyPairs.Add((i, sortedDistances[j].idx));
            }

            // Mid-near pairs (neighbors at moderate distance)
            int midStart = k;
            int midEnd = Math.Min(6 * k, n - 1);
            for (int m = 0; m < nMidNear; m++)
            {
                if (midStart < midEnd)
                {
                    int idx = random.Next(midStart, midEnd);
                    if (idx < n - 1)
                    {
                        midNearPairs.Add((i, sortedDistances[idx].idx));
                    }
                }
            }

            // Further pairs (random sampling from distant points)
            // Guard against n == 1 which would cause infinite loop
            for (int f = 0; f < nFurther && n > 1; f++)
            {
                int idx = random.Next(n);
                while (idx == i)
                {
                    idx = random.Next(n);
                }
                furtherPairs.Add((i, idx));
            }
        }

        return (nearbyPairs, midNearPairs, furtherPairs);
    }

    private double[,] InitializeWithPCA(double[,] X, int n, int p, Random random)
    {
        // Center data
        var mean = new double[p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                mean[j] += X[i, j];
            }
        }
        for (int j = 0; j < p; j++) mean[j] /= n;

        var Xc = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                Xc[i, j] = X[i, j] - mean[j];
            }
        }

        // Compute covariance matrix
        var C = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    C[i, j] += Xc[k, i] * Xc[k, j];
                }
                C[i, j] /= n;
            }
        }

        // Power iteration to get top eigenvectors
        var V = new double[p, _nComponents];
        var A = (double[,])C.Clone();

        for (int d = 0; d < _nComponents; d++)
        {
            var v = new double[p];
            for (int i = 0; i < p; i++) v[i] = random.NextDouble() - 0.5;

            for (int iter = 0; iter < 50; iter++)
            {
                var Av = new double[p];
                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++) Av[i] += A[i, j] * v[j];
                }

                double norm = 0;
                for (int i = 0; i < p; i++) norm += Av[i] * Av[i];
                norm = Math.Sqrt(norm);
                if (norm < 1e-10) break;

                for (int i = 0; i < p; i++) v[i] = Av[i] / norm;
            }

            for (int i = 0; i < p; i++) V[i, d] = v[i];

            // Deflate
            var Av2 = new double[p];
            double lambda = 0;
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < p; j++) Av2[i] += A[i, j] * v[j];
                lambda += v[i] * Av2[i];
            }

            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    A[i, j] -= lambda * v[i] * v[j];
                }
            }
        }

        // Project data
        var Y = new double[n, _nComponents];
        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < _nComponents; d++)
            {
                for (int j = 0; j < p; j++)
                {
                    Y[i, d] += Xc[i, j] * V[j, d];
                }
            }
        }

        // Scale initial embedding
        double scale = 0.0001;
        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < _nComponents; d++)
            {
                Y[i, d] *= scale;
            }
        }

        return Y;
    }

    private void OptimizeEmbedding(
        double[,] Y,
        List<(int, int)> nearbyPairs,
        List<(int, int)> midNearPairs,
        List<(int, int)> furtherPairs,
        double[,] inputDistances,
        int n,
        Random random)
    {
        double lr = _learningRate;

        // PaCMAP uses three phases with different weight schedules
        // Phase 1: Focus on global structure (further pairs)
        // Phase 2: Balance all pairs
        // Phase 3: Focus on local structure (nearby pairs)

        for (int iter = 0; iter < _nIter; iter++)
        {
            // Compute dynamic weights based on iteration
            double t = (double)iter / _nIter;
            double wNearby, wMidNear, wFurther;

            if (t < 0.1)
            {
                // Early phase: emphasize global structure
                wNearby = 2.0;
                wMidNear = 3.0;
                wFurther = 1.0;
            }
            else if (t < 0.3)
            {
                // Transition phase
                wNearby = 3.0;
                wMidNear = 3.0;
                wFurther = 0.5;
            }
            else
            {
                // Late phase: emphasize local structure
                wNearby = 1.0;
                wMidNear = 0.5;
                wFurther = 0.1;
            }

            var gradients = new double[n, _nComponents];

            // Nearby pairs (attractive force)
            foreach (var (i, j) in nearbyPairs)
            {
                double dY = 0;
                for (int d = 0; d < _nComponents; d++)
                {
                    double diff = Y[i, d] - Y[j, d];
                    dY += diff * diff;
                }

                // PaCMAP attraction
                double denom = 10.0 + dY;
                double grad = 2.0 * wNearby / (denom * denom);

                for (int d = 0; d < _nComponents; d++)
                {
                    double diff = Y[i, d] - Y[j, d];
                    gradients[i, d] -= grad * diff;
                    gradients[j, d] += grad * diff;
                }
            }

            // Mid-near pairs (moderate attraction)
            foreach (var (i, j) in midNearPairs)
            {
                double dY = 0;
                for (int d = 0; d < _nComponents; d++)
                {
                    double diff = Y[i, d] - Y[j, d];
                    dY += diff * diff;
                }

                double denom = 10000.0 + dY;
                double grad = 2.0 * wMidNear * 10000.0 / (denom * denom);

                for (int d = 0; d < _nComponents; d++)
                {
                    double diff = Y[i, d] - Y[j, d];
                    gradients[i, d] -= grad * diff;
                    gradients[j, d] += grad * diff;
                }
            }

            // Further pairs (repulsive force)
            foreach (var (i, j) in furtherPairs)
            {
                double dY = 0;
                for (int d = 0; d < _nComponents; d++)
                {
                    double diff = Y[i, d] - Y[j, d];
                    dY += diff * diff;
                }

                // PaCMAP repulsion
                double denom = 1.0 + dY;
                double grad = 2.0 * wFurther / (denom * denom);

                for (int d = 0; d < _nComponents; d++)
                {
                    double diff = Y[i, d] - Y[j, d];
                    gradients[i, d] += grad * diff;
                    gradients[j, d] -= grad * diff;
                }
            }

            // Update embedding
            for (int i = 0; i < n; i++)
            {
                for (int d = 0; d < _nComponents; d++)
                {
                    Y[i, d] += lr * gradients[i, d];
                }
            }

            // Decay learning rate
            if (iter > 0 && iter % 50 == 0)
            {
                lr *= 0.95;
            }
        }

        // Center the final embedding
        var mean = new double[_nComponents];
        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < _nComponents; d++)
            {
                mean[d] += Y[i, d];
            }
        }
        for (int d = 0; d < _nComponents; d++) mean[d] /= n;

        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < _nComponents; d++)
            {
                Y[i, d] -= mean[d];
            }
        }
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("PaCMAP has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "PaCMAP does not support out-of-sample transformation. " +
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
        throw new NotSupportedException("PaCMAP does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"PaCMAP{i + 1}";
        }
        return names;
    }
}
