using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// PHATE: Potential of Heat-diffusion for Affinity-based Transition Embedding.
/// </summary>
/// <remarks>
/// <para>
/// PHATE is a dimensionality reduction method designed for visualizing high-dimensional
/// biological data, particularly single-cell data. It captures both local and global
/// structure by using diffusion-based distances and a special potential distance metric.
/// </para>
/// <para>
/// The algorithm:
/// 1. Compute local affinities using adaptive bandwidth Gaussian kernel
/// 2. Construct diffusion operator from affinities
/// 3. Diffuse for t steps to capture multi-scale structure
/// 4. Compute potential distance using log transform
/// 5. Embed using MDS on potential distances
/// </para>
/// <para><b>For Beginners:</b> PHATE excels at revealing data trajectories:
/// - Captures smooth progression paths in data (differentiation, time courses)
/// - Preserves both local clusters and global connectivity
/// - The potential distance emphasizes transitions between states
/// - Particularly effective for biological data
///
/// Use cases:
/// - Single-cell RNA sequencing visualization
/// - Developmental trajectories and cell differentiation
/// - Time-series data with smooth transitions
/// - Any data with underlying continuous processes
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class PHATE<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly int _diffusionTime;
    private readonly double _decay;
    private readonly double _gamma;
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
    /// Gets the diffusion time parameter.
    /// </summary>
    public int DiffusionTime => _diffusionTime;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public double[,]? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="PHATE{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors for affinity computation. Defaults to 5.</param>
    /// <param name="diffusionTime">Number of diffusion steps. Defaults to auto (-1) or use specific value.</param>
    /// <param name="decay">Alpha decay for kernel. Defaults to 40.</param>
    /// <param name="gamma">Informational distance parameter. Defaults to 1 (log potential).</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public PHATE(
        int nComponents = 2,
        int nNeighbors = 5,
        int diffusionTime = -1,
        double decay = 40,
        double gamma = 1,
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
        _diffusionTime = diffusionTime;
        _decay = decay;
        _gamma = gamma;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits PHATE and computes the embedding.
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

        // Step 2: Compute adaptive bandwidth kernel
        var K = ComputeAdaptiveKernel(distances, n);

        // Step 3: Normalize to get diffusion operator P
        var P = NormalizeToDiffusionOperator(K, n);

        // Step 4: Compute diffused operator P^t
        int t = _diffusionTime > 0 ? _diffusionTime : EstimateDiffusionTime(P, n);
        var Pt = DiffuseOperator(P, n, t);

        // Step 5: Compute potential distance
        var potentialDistance = ComputePotentialDistance(Pt, n);

        // Step 6: Embed using MDS
        _embedding = ComputeMDSEmbedding(potentialDistance, n, random);
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

    private double[,] ComputeAdaptiveKernel(double[,] distances, int n)
    {
        var K = new double[n, n];
        int k = Math.Min(_nNeighbors, n - 1);

        // Compute local bandwidth for each point (k-th nearest neighbor distance)
        var epsilon = new double[n];
        for (int i = 0; i < n; i++)
        {
            var sortedDists = new double[n];
            for (int j = 0; j < n; j++)
            {
                sortedDists[j] = j == i ? double.MaxValue : distances[i, j];
            }
            Array.Sort(sortedDists);
            epsilon[i] = sortedDists[k - 1];
            if (epsilon[i] < 1e-10) epsilon[i] = 1e-10;
        }

        // Compute adaptive Gaussian kernel
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    K[i, j] = 0;
                    continue;
                }

                // Adaptive bandwidth: use geometric mean of bandwidths
                double bandwidth = Math.Sqrt(epsilon[i] * epsilon[j]);

                // Alpha-decaying kernel
                double d = distances[i, j] / bandwidth;
                K[i, j] = Math.Exp(-Math.Pow(d, _decay / 10.0));
            }
        }

        // Make symmetric
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double sym = (K[i, j] + K[j, i]) / 2;
                K[i, j] = sym;
                K[j, i] = sym;
            }
        }

        return K;
    }

    private double[,] NormalizeToDiffusionOperator(double[,] K, int n)
    {
        var P = new double[n, n];

        // Row normalization to get Markov matrix
        for (int i = 0; i < n; i++)
        {
            double rowSum = 0;
            for (int j = 0; j < n; j++)
            {
                rowSum += K[i, j];
            }

            if (rowSum > 1e-10)
            {
                for (int j = 0; j < n; j++)
                {
                    P[i, j] = K[i, j] / rowSum;
                }
            }
            else
            {
                P[i, i] = 1.0; // Self-loop for isolated points
            }
        }

        return P;
    }

    private int EstimateDiffusionTime(double[,] P, int n)
    {
        // Estimate optimal diffusion time based on Von Neumann entropy
        // Start with a reasonable default
        int maxT = 100;
        int optimalT = 5;

        // Simple heuristic: use sqrt(n)
        optimalT = Math.Max(1, (int)Math.Sqrt(n));
        optimalT = Math.Min(optimalT, maxT);

        return optimalT;
    }

    private double[,] DiffuseOperator(double[,] P, int n, int t)
    {
        var Pt = (double[,])P.Clone();

        // Matrix power: P^t
        for (int step = 1; step < t; step++)
        {
            var Pnew = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int k = 0; k < n; k++)
                    {
                        Pnew[i, j] += Pt[i, k] * P[k, j];
                    }
                }
            }

            Pt = Pnew;
        }

        return Pt;
    }

    private double[,] ComputePotentialDistance(double[,] Pt, int n)
    {
        var potentialDistance = new double[n, n];

        if (_gamma == 1)
        {
            // Log potential distance
            // d(i,j) = || log(P^t[i,:]) - log(P^t[j,:]) ||

            // Compute log of Pt (with small epsilon for numerical stability)
            var logPt = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    logPt[i, j] = Math.Log(Pt[i, j] + 1e-10);
                }
            }

            // Compute Euclidean distance in log space
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double dist = 0;
                    for (int k = 0; k < n; k++)
                    {
                        double diff = logPt[i, k] - logPt[j, k];
                        dist += diff * diff;
                    }
                    dist = Math.Sqrt(dist);
                    potentialDistance[i, j] = dist;
                    potentialDistance[j, i] = dist;
                }
            }
        }
        else
        {
            // Gamma potential distance
            // d(i,j) = || P^t[i,:]^(1/gamma) - P^t[j,:]^(1/gamma) ||

            var powPt = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    powPt[i, j] = Math.Pow(Pt[i, j] + 1e-10, 1.0 / _gamma);
                }
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double dist = 0;
                    for (int k = 0; k < n; k++)
                    {
                        double diff = powPt[i, k] - powPt[j, k];
                        dist += diff * diff;
                    }
                    dist = Math.Sqrt(dist);
                    potentialDistance[i, j] = dist;
                    potentialDistance[j, i] = dist;
                }
            }
        }

        return potentialDistance;
    }

    private double[,] ComputeMDSEmbedding(double[,] distances, int n, Random random)
    {
        // Classical MDS (metric MDS)

        // Compute squared distance matrix
        var D2 = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                D2[i, j] = distances[i, j] * distances[i, j];
            }
        }

        // Double centering: B = -0.5 * H * D^2 * H
        // where H = I - (1/n) * 1 * 1^T
        var B = new double[n, n];

        // Compute row and column means
        var rowMeans = new double[n];
        var colMeans = new double[n];
        double grandMean = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                rowMeans[i] += D2[i, j];
                colMeans[j] += D2[i, j];
            }
        }

        for (int i = 0; i < n; i++)
        {
            rowMeans[i] /= n;
            grandMean += rowMeans[i];
        }
        grandMean /= n;

        for (int j = 0; j < n; j++) colMeans[j] /= n;

        // Apply double centering
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                B[i, j] = -0.5 * (D2[i, j] - rowMeans[i] - colMeans[j] + grandMean);
            }
        }

        // Eigendecomposition of B
        var eigenvalues = new double[n];
        var eigenvectors = new double[n, n];
        ComputeEigendecomposition(B, eigenvalues, eigenvectors, n, random);

        // Create embedding from top eigenvectors
        var embedding = new double[n, _nComponents];
        for (int d = 0; d < _nComponents && d < n; d++)
        {
            double scale = Math.Sqrt(Math.Max(eigenvalues[d], 0));
            for (int i = 0; i < n; i++)
            {
                embedding[i, d] = eigenvectors[i, d] * scale;
            }
        }

        return embedding;
    }

    private void ComputeEigendecomposition(
        double[,] B, double[] eigenvalues, double[,] eigenvectors, int n, Random random)
    {
        // Power iteration with deflation
        var A = (double[,])B.Clone();

        for (int d = 0; d < Math.Min(n, _nComponents + 2); d++)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++) v[i] = random.NextDouble() - 0.5;

            // Power iteration
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
                for (int i = 0; i < n; i++) norm += Av[i] * Av[i];
                norm = Math.Sqrt(norm);

                if (norm < 1e-10) break;

                for (int i = 0; i < n; i++) v[i] = Av[i] / norm;
            }

            // Store eigenvector
            for (int i = 0; i < n; i++)
            {
                eigenvectors[i, d] = v[i];
            }

            // Compute eigenvalue
            var Av2 = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Av2[i] += A[i, j] * v[j];
                }
            }

            double lambda = 0;
            for (int i = 0; i < n; i++)
            {
                lambda += v[i] * Av2[i];
            }
            eigenvalues[d] = lambda;

            // Deflate
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    A[i, j] -= lambda * v[i] * v[j];
                }
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
            throw new InvalidOperationException("PHATE has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "PHATE does not support out-of-sample transformation. " +
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
        throw new NotSupportedException("PHATE does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"PHATE{i + 1}";
        }
        return names;
    }
}
