using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// LargeVis for large-scale visualization and dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// LargeVis is designed for visualizing large-scale high-dimensional data. It builds a
/// k-NN graph, computes edge weights based on shared neighbors, and uses asynchronous
/// stochastic gradient descent with negative sampling for optimization.
/// </para>
/// <para>
/// The algorithm:
/// 1. Construct approximate k-NN graph using random projection trees
/// 2. Compute edge weights using shared neighbor similarity
/// 3. Initialize embedding with random projection or PCA
/// 4. Optimize using negative sampling SGD (similar to word2vec)
/// </para>
/// <para><b>For Beginners:</b> LargeVis is efficient for large datasets because:
/// - Uses approximate nearest neighbors for scalability
/// - Negative sampling reduces computation vs full graph
/// - Asynchronous updates enable parallel processing
/// - Layout preserves local neighborhood relationships
///
/// Use cases:
/// - Datasets with millions of points
/// - When t-SNE is too slow
/// - Interactive visualization systems
/// - Document and image embedding visualization
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LargeVis<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly int _nNegatives;
    private readonly int _nIter;
    private readonly double _learningRate;
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
    /// Gets the embedding result.
    /// </summary>
    public double[,]? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="LargeVis{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors for graph construction. Defaults to 150.</param>
    /// <param name="nNegatives">Number of negative samples per positive sample. Defaults to 5.</param>
    /// <param name="nIter">Number of optimization iterations. Defaults to 200.</param>
    /// <param name="learningRate">Initial learning rate. Defaults to 1.0.</param>
    /// <param name="gamma">Repulsion strength for negative samples. Defaults to 7.0.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public LargeVis(
        int nComponents = 2,
        int nNeighbors = 150,
        int nNegatives = 5,
        int nIter = 200,
        double learningRate = 1.0,
        double gamma = 7.0,
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
        _nNegatives = nNegatives;
        _nIter = nIter;
        _learningRate = learningRate;
        _gamma = gamma;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits LargeVis and computes the embedding.
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

        // Step 1: Build k-NN graph
        var (neighbors, weights) = BuildKNNGraph(X, n, p, random);

        // Step 2: Initialize embedding
        var Y = InitializeEmbedding(X, n, p, random);

        // Step 3: Optimize using negative sampling
        OptimizeWithNegativeSampling(Y, neighbors, weights, n, random);

        _embedding = Y;
    }

    private (List<int>[] neighbors, List<double>[] weights) BuildKNNGraph(
        double[,] X, int n, int p, Random random)
    {
        int k = Math.Min(_nNeighbors, n - 1);

        // Compute all pairwise distances for exact k-NN
        // (In production, would use approximate methods like random projection trees)
        var neighbors = new List<int>[n];
        var weights = new List<double>[n];

        for (int i = 0; i < n; i++)
        {
            neighbors[i] = new List<int>();
            weights[i] = new List<double>();
        }

        // Find k-nearest neighbors for each point
        for (int i = 0; i < n; i++)
        {
            var distances = new (double dist, int idx)[n];
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    distances[j] = (double.MaxValue, j);
                    continue;
                }

                double dist = 0;
                for (int d = 0; d < p; d++)
                {
                    double diff = X[i, d] - X[j, d];
                    dist += diff * diff;
                }
                distances[j] = (Math.Sqrt(dist), j);
            }

            Array.Sort(distances, (a, b) => a.dist.CompareTo(b.dist));

            // Store k nearest neighbors
            for (int j = 0; j < k; j++)
            {
                neighbors[i].Add(distances[j].idx);
            }
        }

        // Compute edge weights based on shared neighbors
        for (int i = 0; i < n; i++)
        {
            var neighborSet = new HashSet<int>(neighbors[i]);

            foreach (int j in neighbors[i])
            {
                // Count shared neighbors
                int shared = 0;
                foreach (int jNeighbor in neighbors[j])
                {
                    if (neighborSet.Contains(jNeighbor))
                    {
                        shared++;
                    }
                }

                // Weight based on shared neighbor similarity (Jaccard-like)
                double weight = (double)(shared + 1) / (neighbors[i].Count + neighbors[j].Count - shared + 1);
                weights[i].Add(weight);
            }
        }

        return (neighbors, weights);
    }

    private double[,] InitializeEmbedding(double[,] X, int n, int p, Random random)
    {
        var Y = new double[n, _nComponents];

        // Initialize with PCA for better starting point
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

        // Compute covariance
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

        // Power iteration for top eigenvectors
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

        // Project and scale
        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < _nComponents; d++)
            {
                for (int j = 0; j < p; j++)
                {
                    Y[i, d] += Xc[i, j] * V[j, d];
                }
                // Add small noise
                Y[i, d] = Y[i, d] * 0.0001 + (random.NextDouble() - 0.5) * 0.0001;
            }
        }

        return Y;
    }

    private void OptimizeWithNegativeSampling(
        double[,] Y,
        List<int>[] neighbors,
        List<double>[] weights,
        int n,
        Random random)
    {
        double lr = _learningRate;

        // Build edge list for sampling
        var edges = new List<(int i, int j, double w)>();
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < neighbors[i].Count; k++)
            {
                edges.Add((i, neighbors[i][k], weights[i][k]));
            }
        }

        // Build negative sampling distribution (proportional to degree^0.75)
        var negativeProbs = new double[n];
        for (int i = 0; i < n; i++)
        {
            negativeProbs[i] = Math.Pow(neighbors[i].Count + 1, 0.75);
        }
        double sumProbs = 0;
        for (int i = 0; i < n; i++) sumProbs += negativeProbs[i];
        for (int i = 0; i < n; i++) negativeProbs[i] /= sumProbs;

        // Create alias table for efficient sampling
        var aliasTable = BuildAliasTable(negativeProbs);

        for (int iter = 0; iter < _nIter; iter++)
        {
            // Shuffle edges
            for (int e = edges.Count - 1; e > 0; e--)
            {
                int swap = random.Next(e + 1);
                (edges[e], edges[swap]) = (edges[swap], edges[e]);
            }

            // Process edges with negative sampling
            foreach (var (i, j, w) in edges)
            {
                // Positive sample: attract i and j
                double dY = 0;
                for (int d = 0; d < _nComponents; d++)
                {
                    double diff = Y[i, d] - Y[j, d];
                    dY += diff * diff;
                }

                double f = w / (1.0 + dY);
                double g = 2.0 * f * f;

                for (int d = 0; d < _nComponents; d++)
                {
                    double diff = Y[i, d] - Y[j, d];
                    double grad = g * diff;
                    Y[i, d] += lr * grad;
                    Y[j, d] -= lr * grad;
                }

                // Negative samples: repel i from random points
                for (int neg = 0; neg < _nNegatives; neg++)
                {
                    int k = SampleFromAliasTable(aliasTable, random);
                    if (k == i || k == j) continue;

                    dY = 0;
                    for (int d = 0; d < _nComponents; d++)
                    {
                        double diff = Y[i, d] - Y[k, d];
                        dY += diff * diff;
                    }

                    // Repulsive force
                    f = _gamma / (0.1 + dY);
                    g = 2.0 * f * f;
                    g = Math.Min(g, 5.0); // Clip gradient

                    for (int d = 0; d < _nComponents; d++)
                    {
                        double diff = Y[i, d] - Y[k, d];
                        double grad = g * diff;
                        Y[i, d] += lr * grad;
                    }
                }
            }

            // Decay learning rate
            lr = _learningRate * (1.0 - (double)(iter + 1) / _nIter);
            lr = Math.Max(lr, 0.0001);
        }
    }

    private (int[] alias, double[] prob) BuildAliasTable(double[] probabilities)
    {
        int n = probabilities.Length;
        var alias = new int[n];
        var prob = new double[n];

        var small = new List<int>();
        var large = new List<int>();

        var scaledProbs = new double[n];
        for (int i = 0; i < n; i++)
        {
            scaledProbs[i] = probabilities[i] * n;
            if (scaledProbs[i] < 1.0)
                small.Add(i);
            else
                large.Add(i);
        }

        while (small.Count > 0 && large.Count > 0)
        {
            int l = small[small.Count - 1];
            small.RemoveAt(small.Count - 1);
            int g = large[large.Count - 1];
            large.RemoveAt(large.Count - 1);

            prob[l] = scaledProbs[l];
            alias[l] = g;

            scaledProbs[g] = (scaledProbs[g] + scaledProbs[l]) - 1.0;

            if (scaledProbs[g] < 1.0)
                small.Add(g);
            else
                large.Add(g);
        }

        while (large.Count > 0)
        {
            int g = large[large.Count - 1];
            large.RemoveAt(large.Count - 1);
            prob[g] = 1.0;
        }

        while (small.Count > 0)
        {
            int l = small[small.Count - 1];
            small.RemoveAt(small.Count - 1);
            prob[l] = 1.0;
        }

        return (alias, prob);
    }

    private int SampleFromAliasTable((int[] alias, double[] prob) table, Random random)
    {
        int n = table.prob.Length;
        int i = random.Next(n);
        return random.NextDouble() < table.prob[i] ? i : table.alias[i];
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("LargeVis has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "LargeVis does not support out-of-sample transformation. " +
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
        throw new NotSupportedException("LargeVis does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"LV{i + 1}";
        }
        return names;
    }
}
