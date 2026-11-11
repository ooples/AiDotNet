using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
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
    private Matrix<T>? _embedding;
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
    public Matrix<T>? Embedding => _embedding;

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

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        var x = Tensor<T>.FromMatrix(data);

        // Step 1: Build k-NN graph
        var (neighborTensor, weightTensor) = BuildKNNGraph(x, n);

        // Step 2: Initialize embedding
        var initial = InitializeEmbedding(x, n, random);

        // Step 3: Optimize using negative sampling (edge-sequential SGD, see OptimizeWithNegativeSampling)
        int k = neighborTensor.Shape[1];
        var flatNeighbors = neighborTensor.AsSpan().ToArray();
        var flatWeights = weightTensor.Cast<double>().AsSpan().ToArray();
        var neighbors = new List<int>[n];
        var weights = new List<double>[n];
        for (int i = 0; i < n; i++)
        {
            neighbors[i] = new List<int>(flatNeighbors.Skip(i * k).Take(k));
            weights[i] = new List<double>(flatWeights.Skip(i * k).Take(k));
        }

        var flat = initial.Cast<double>().AsSpan().ToArray();
        var y = new double[n, _nComponents];
        Buffer.BlockCopy(flat, 0, y, 0, flat.Length * sizeof(double));
        OptimizeWithNegativeSampling(y, neighbors, weights, n, random);

        Buffer.BlockCopy(y, 0, flat, 0, flat.Length * sizeof(double));
        _embedding = new Tensor<double>(flat, new[] { n, _nComponents }).Cast<T>().ToMatrix();
    }

    /// <summary>
    /// k nearest neighbours (ties in index order) and shared-neighbour weights
    /// (s_ij + 1) / (|N(i)| + |N(j)| - s_ij + 1), s_ij = |N(i) and N(j)|, for every j in N(i).
    /// </summary>
    private (Tensor<int> Neighbors, Tensor<T> Weights) BuildKNNGraph(Tensor<T> x, int n)
    {
        var engine = Engine;
        int k = Math.Min(_nNeighbors, n - 1);

        // Compute all pairwise distances for exact k-NN
        var neighbors = DimensionalityReductionOps.NearestNeighbors(DimensionalityReductionOps.EuclideanDistances(x, x), k);

        // Shared neighbour counts S = A A^T for the 0/1 adjacency A[i, j] = [j in N(i)]
        var adjacency = DimensionalityReductionOps.ScatterRows(
            neighbors, DimensionalityReductionOps.Full(new[] { n, k }, NumOps.One), n, NumOps.Zero);
        var shared = DimensionalityReductionOps.TakeRows(
            DimensionalityReductionOps.MatMulTranspose(adjacency, adjacency), neighbors);

        // Every neighbourhood has k members
        var weights = engine.TensorDivide(
            engine.TensorAddScalar(shared, NumOps.One),
            engine.TensorSubtract(DimensionalityReductionOps.Full(new[] { n, k }, NumOps.FromDouble(2 * k + 1)), shared));
        return (neighbors, weights);
    }

    /// <summary>
    /// PCA initialization: centered data projected on the top principal axes, scaled by 1e-4, plus
    /// 1e-4 (u - 0.5) noise drawn row-major.
    /// </summary>
    private Tensor<T> InitializeEmbedding(Tensor<T> x, int n, Random random)
    {
        var engine = Engine;
        int p = x.Shape[1];
        var centered = DimensionalityReductionOps.SubtractRow(x, DimensionalityReductionOps.ColumnMeans(x));
        var covariance = engine.TensorDivideScalar(
            DimensionalityReductionOps.TransposeMatMul(centered, centered), NumOps.FromDouble(n));

        // Top eigenvectors (zero columns past the feature count)
        var (_, vectors) = DimensionalityReductionOps.SymmetricEigen(covariance, descending: true);
        // Signs as the original deflated power iteration converges to them from its seeded start vectors
        // (their draws come before the noise below, so the stream position matches)
        vectors = DimensionalityReductionOps.MatchStartSigns(vectors,
            DimensionalityReductionOps.PowerIterationStarts<T>(random, _nComponents, p, 0.5));
        int available = Math.Min(_nComponents, p);
        var projected = engine.TensorMatMul(centered, DimensionalityReductionOps.FirstColumns(vectors, available));
        if (available < _nComponents)
        {
            projected = engine.TensorConcatenate(new[] { projected, new Tensor<T>(new[] { n, _nComponents - available }) }, 1);
        }

        var scale = NumOps.FromDouble(0.0001);
        var noise = engine.TensorMultiplyScalar(
            engine.TensorSubtractScalar(DimensionalityReductionOps.Uniform<T>(random, n, _nComponents), NumOps.FromDouble(0.5)), scale);
        return engine.TensorAdd(engine.TensorMultiplyScalar(projected, scale), noise);
    }

    /// <summary>
    /// Edge-sequential SGD with alias-table negative sampling. Each update reads the positions written by the
    /// previous one, so this stays the original scalar loop (in double) until the engine has an edge-sequential
    /// layout kernel; see missing-ops.txt.
    /// </summary>
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
                double g = 2.0 * w * f;

                for (int d = 0; d < _nComponents; d++)
                {
                    double diff = Y[i, d] - Y[j, d];
                    double grad = g * diff;
                    // Attractive force: subtract gradient to pull points together
                    Y[i, d] -= lr * grad;
                    Y[j, d] += lr * grad;
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
                        // Repulsive force: add gradient to push points apart (symmetric update)
                        Y[i, d] += lr * grad;
                        Y[k, d] -= lr * grad;
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

        return _embedding.Clone();
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
