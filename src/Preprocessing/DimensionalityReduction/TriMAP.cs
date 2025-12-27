using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// TriMAP: Large-scale Dimensionality Reduction Using Triplets.
/// </summary>
/// <remarks>
/// <para>
/// TriMAP is a dimensionality reduction method that uses triplet constraints to preserve
/// both local and global structure. It outperforms t-SNE and UMAP on many datasets while
/// being more robust to hyperparameter choices.
/// </para>
/// <para>
/// The algorithm:
/// 1. Generate triplets (anchor, positive, negative) based on distance relationships
/// 2. Initialize embedding using PCA
/// 3. Optimize using triplet loss to preserve distance rankings
/// 4. Weight triplets to balance local and global structure
/// </para>
/// <para><b>For Beginners:</b> TriMAP preserves distance relationships using triplets:
/// - A triplet (i, j, k) means point i is closer to j than to k
/// - Optimization ensures these relationships hold in the embedding
/// - More accurate than t-SNE/UMAP for many datasets
/// - Less sensitive to parameter tuning
///
/// Use cases:
/// - Large-scale visualization (millions of points)
/// - When t-SNE/UMAP produces poor results
/// - When you need faithful global structure
/// - Complex datasets with hierarchical structure
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class TriMAP<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nInliers;
    private readonly int _nOutliers;
    private readonly int _nRandom;
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
    /// Gets the embedding result.
    /// </summary>
    public double[,]? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="TriMAP{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nInliers">Number of nearest neighbors for triplet generation. Defaults to 10.</param>
    /// <param name="nOutliers">Number of outliers per point. Defaults to 5.</param>
    /// <param name="nRandom">Number of random triplets per point. Defaults to 3.</param>
    /// <param name="nIter">Number of optimization iterations. Defaults to 400.</param>
    /// <param name="learningRate">Learning rate for optimization. Defaults to 100. Higher values may cause instability.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public TriMAP(
        int nComponents = 2,
        int nInliers = 10,
        int nOutliers = 5,
        int nRandom = 3,
        int nIter = 400,
        double learningRate = 100.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (nInliers < 1)
        {
            throw new ArgumentException("Number of inliers must be at least 1.", nameof(nInliers));
        }

        if (learningRate <= 0 || learningRate > 10000)
        {
            throw new ArgumentException(
                "Learning rate must be positive and not exceed 10000 to avoid numerical instability.",
                nameof(learningRate));
        }

        if (nIter < 1)
        {
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIter));
        }

        if (nOutliers < 0)
        {
            throw new ArgumentException("Number of outliers cannot be negative.", nameof(nOutliers));
        }

        if (nRandom < 0)
        {
            throw new ArgumentException("Number of random triplets cannot be negative.", nameof(nRandom));
        }

        _nComponents = nComponents;
        _nInliers = nInliers;
        _nOutliers = nOutliers;
        _nRandom = nRandom;
        _nIter = nIter;
        _learningRate = learningRate;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits TriMAP and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        int n = data.Rows;
        int p = data.Columns;

        // TriMAP requires at least 3 samples to form triplets (anchor, positive, negative)
        if (n < 3)
        {
            throw new ArgumentException("TriMAP requires at least 3 samples to form triplets.", nameof(data));
        }

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

        // Step 2: Generate triplets
        var triplets = GenerateTriplets(distances, n, random);

        // Step 3: Compute triplet weights
        var weights = ComputeTripletWeights(triplets, distances, n);

        // Step 4: Initialize embedding with PCA
        var Y = InitializeWithPCA(X, n, p, random);

        // Step 5: Optimize using triplet loss
        OptimizeEmbedding(Y, triplets, weights, n, random);

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
                dist = Math.Sqrt(dist);
                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        return distances;
    }

    private List<(int anchor, int positive, int negative)> GenerateTriplets(
        double[,] distances, int n, Random random)
    {
        var triplets = new List<(int, int, int)>();
        int k = Math.Min(_nInliers, n - 1);
        int nOut = Math.Min(_nOutliers, n - 1);

        // Find k-nearest neighbors for each point
        var neighbors = new int[n, k];
        for (int i = 0; i < n; i++)
        {
            var dists = new (double dist, int idx)[n];
            for (int j = 0; j < n; j++)
            {
                dists[j] = (j == i ? double.MaxValue : distances[i, j], j);
            }
            Array.Sort(dists, (a, b) => a.dist.CompareTo(b.dist));

            for (int j = 0; j < k; j++)
            {
                neighbors[i, j] = dists[j].idx;
            }
        }

        // Generate triplets
        for (int i = 0; i < n; i++)
        {
            // Sort distances once per point for mid-range negative selection
            var sortedDists = new (double dist, int idx)[n];
            for (int jj = 0; jj < n; jj++)
            {
                sortedDists[jj] = (jj == i ? double.MaxValue : distances[i, jj], jj);
            }
            Array.Sort(sortedDists, (a, b) => a.dist.CompareTo(b.dist));

            // Inlier triplets (nearest neighbors as positives)
            for (int j = 0; j < k; j++)
            {
                int positive = neighbors[i, j];

                // Mid-range negatives
                for (int m = 0; m < Math.Min(nOut, k); m++)
                {
                    int negIdx = Math.Min(k + m, n - 1);

                    if (negIdx < n)
                    {
                        int negative = sortedDists[negIdx].idx;
                        if (positive != negative)
                        {
                            triplets.Add((i, positive, negative));
                        }
                    }
                }
            }

            // Random triplets for global structure
            for (int r = 0; r < _nRandom; r++)
            {
                if (k > 0)
                {
                    int posIdx = random.Next(k);
                    int positive = neighbors[i, posIdx];
                    int negative = random.Next(n);
                    while (negative == i || negative == positive)
                    {
                        negative = random.Next(n);
                    }

                    if (distances[i, positive] < distances[i, negative])
                    {
                        triplets.Add((i, positive, negative));
                    }
                }
            }
        }

        return triplets;
    }

    private double[] ComputeTripletWeights(
        List<(int anchor, int positive, int negative)> triplets,
        double[,] distances,
        int n)
    {
        var weights = new double[triplets.Count];

        // Compute distance scale
        double maxDist = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                maxDist = Math.Max(maxDist, distances[i, j]);
            }
        }

        double scale = maxDist > 0 ? maxDist : 1.0;

        for (int t = 0; t < triplets.Count; t++)
        {
            var (anchor, positive, negative) = triplets[t];

            // Weight based on distance ratio
            double dPos = distances[anchor, positive];
            double dNeg = distances[anchor, negative];

            // Higher weight for triplets where positive is much closer than negative
            double ratio = dNeg / (dPos + 1e-10);
            weights[t] = Math.Log(1 + ratio);

            // Normalize by scale
            weights[t] *= scale / (dNeg + 1e-10);
            weights[t] = Math.Min(weights[t], 10.0); // Clip weights
        }

        // Normalize weights
        double sumWeights = 0;
        for (int t = 0; t < weights.Length; t++) sumWeights += weights[t];
        if (sumWeights > 0)
        {
            for (int t = 0; t < weights.Length; t++) weights[t] /= sumWeights;
        }

        return weights;
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

        // Scale to unit variance
        for (int d = 0; d < _nComponents; d++)
        {
            double meanD = 0, varD = 0;
            for (int i = 0; i < n; i++) meanD += Y[i, d];
            meanD /= n;
            for (int i = 0; i < n; i++)
            {
                Y[i, d] -= meanD;
                varD += Y[i, d] * Y[i, d];
            }
            varD /= n;
            double stdD = Math.Sqrt(varD);
            if (stdD > 1e-10)
            {
                for (int i = 0; i < n; i++) Y[i, d] /= stdD;
            }
        }

        return Y;
    }

    private void OptimizeEmbedding(
        double[,] Y,
        List<(int anchor, int positive, int negative)> triplets,
        double[] weights,
        int n,
        Random random)
    {
        double lr = _learningRate;

        // Momentum
        var velocity = new double[n, _nComponents];
        double momentum = 0.5;

        for (int iter = 0; iter < _nIter; iter++)
        {
            // Shuffle triplets using Fisher-Yates shuffle
            for (int i = triplets.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);

                // Swap triplets
                var tmpTriplet = triplets[i];
                triplets[i] = triplets[j];
                triplets[j] = tmpTriplet;

                // Swap corresponding weights
                double tmpWeight = weights[i];
                weights[i] = weights[j];
                weights[j] = tmpWeight;
            }

            var gradients = new double[n, _nComponents];

            // Compute gradients
            for (int t = 0; t < triplets.Count; t++)
            {
                var (anchor, positive, negative) = triplets[t];
                double w = weights[t];

                // Compute distances in embedding space
                double dPos = 0, dNeg = 0;
                for (int d = 0; d < _nComponents; d++)
                {
                    double diffPos = Y[anchor, d] - Y[positive, d];
                    double diffNeg = Y[anchor, d] - Y[negative, d];
                    dPos += diffPos * diffPos;
                    dNeg += diffNeg * diffNeg;
                }
                dPos = Math.Sqrt(dPos + 1e-10);
                dNeg = Math.Sqrt(dNeg + 1e-10);

                // Triplet loss gradient: want dPos < dNeg
                // Loss = max(0, dPos - dNeg + margin)
                double margin = 1.0;
                double loss = dPos - dNeg + margin;

                if (loss > 0)
                {
                    // Gradient for anchor
                    for (int d = 0; d < _nComponents; d++)
                    {
                        double diffPos = Y[anchor, d] - Y[positive, d];
                        double diffNeg = Y[anchor, d] - Y[negative, d];

                        double gradPos = diffPos / (dPos + 1e-10);
                        double gradNeg = diffNeg / (dNeg + 1e-10);

                        gradients[anchor, d] += w * (gradPos - gradNeg);
                        gradients[positive, d] += w * (-gradPos);
                        gradients[negative, d] += w * gradNeg;
                    }
                }
            }

            // Update with momentum
            if (iter > 100) momentum = 0.8;

            for (int i = 0; i < n; i++)
            {
                for (int d = 0; d < _nComponents; d++)
                {
                    velocity[i, d] = momentum * velocity[i, d] - lr * gradients[i, d];
                    Y[i, d] += velocity[i, d];
                }
            }

            // Decay learning rate
            if (iter > 0 && iter % 100 == 0)
            {
                lr *= 0.9;
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
            throw new InvalidOperationException("TriMAP has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "TriMAP does not support out-of-sample transformation. " +
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
        throw new NotSupportedException("TriMAP does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"TriMAP{i + 1}";
        }
        return names;
    }
}
