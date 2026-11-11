using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Uniform Manifold Approximation and Projection for dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// UMAP is a nonlinear dimensionality reduction technique that constructs a high-dimensional
/// graph representation and optimizes a low-dimensional graph to be as structurally similar
/// as possible. It is based on Riemannian geometry and algebraic topology.
/// </para>
/// <para>
/// Key advantages over t-SNE:
/// - Much faster (scales better to large datasets)
/// - Preserves more global structure
/// - Supports out-of-sample transformation
/// - More deterministic results
/// </para>
/// <para><b>For Beginners:</b> UMAP creates visualizations similar to t-SNE but:
/// - It's faster, especially for large datasets
/// - Distances between clusters are more meaningful
/// - You can transform new data points without refitting
/// - Great for both visualization AND as a preprocessing step for ML
///
/// Example use cases:
/// - Visualizing high-dimensional data (gene expression, embeddings)
/// - Preprocessing features for classification
/// - Clustering analysis
/// - Anomaly detection
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class UMAP<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private const double DistanceEpsilon = 1e-10;
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly double _minDist;
    private readonly double _spread;
    private readonly UMAPMetric _metric;
    private readonly int _nEpochs;
    private readonly double _learningRate;
    private readonly double _negativeSampleRate;
    private readonly int? _randomState;
    private readonly double _localConnectivity;
    private readonly double _repulsionStrength;

    // Fitted parameters
    private Matrix<T>? _embedding;
    private Tensor<T>? _trainingData;
    private int _nSamples;
    private int _nFeatures;

    // UMAP-specific constants
    private double _a;
    private double _b;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the number of neighbors.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the minimum distance parameter.
    /// </summary>
    public double MinDist => _minDist;

    /// <summary>
    /// Gets the distance metric.
    /// </summary>
    public UMAPMetric Metric => _metric;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="UMAP{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality (usually 2 or 3). Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors for manifold approximation. Defaults to 15.</param>
    /// <param name="minDist">Minimum distance between points in embedding. Defaults to 0.1.</param>
    /// <param name="spread">Effective scale of embedded points. Defaults to 1.0.</param>
    /// <param name="metric">Distance metric to use. Defaults to Euclidean.</param>
    /// <param name="nEpochs">Number of training epochs. Defaults to 200.</param>
    /// <param name="learningRate">Learning rate for SGD. Defaults to 1.0.</param>
    /// <param name="negativeSampleRate">Negative samples per positive. Defaults to 5.</param>
    /// <param name="localConnectivity">Local connectivity constraint. Defaults to 1.0.</param>
    /// <param name="repulsionStrength">Repulsion strength during optimization. Defaults to 1.0.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public UMAP(
        int nComponents = 2,
        int nNeighbors = 15,
        double minDist = 0.1,
        double spread = 1.0,
        UMAPMetric metric = UMAPMetric.Euclidean,
        int nEpochs = 200,
        double learningRate = 1.0,
        double negativeSampleRate = 5.0,
        double localConnectivity = 1.0,
        double repulsionStrength = 1.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (nNeighbors < 2)
        {
            throw new ArgumentException("Number of neighbors must be at least 2.", nameof(nNeighbors));
        }

        if (minDist < 0)
        {
            throw new ArgumentException("Minimum distance must be non-negative.", nameof(minDist));
        }

        if (spread <= 0)
        {
            throw new ArgumentException("Spread must be positive.", nameof(spread));
        }

        if (nEpochs < 1)
        {
            throw new ArgumentException("Number of epochs must be at least 1.", nameof(nEpochs));
        }

        if (learningRate <= 0)
        {
            throw new ArgumentException("Learning rate must be positive.", nameof(learningRate));
        }

        if (negativeSampleRate < 0)
        {
            throw new ArgumentException("Negative sample rate must be non-negative.", nameof(negativeSampleRate));
        }

        if (repulsionStrength <= 0)
        {
            throw new ArgumentException("Repulsion strength must be positive.", nameof(repulsionStrength));
        }

        _nComponents = nComponents;
        _nNeighbors = nNeighbors;
        _minDist = minDist;
        _spread = spread;
        _metric = metric;
        _nEpochs = nEpochs;
        _learningRate = learningRate;
        _negativeSampleRate = negativeSampleRate;
        _localConnectivity = localConnectivity;
        _repulsionStrength = repulsionStrength;
        _randomState = randomState;

        // Compute a and b parameters from min_dist and spread
        (_a, _b) = FindAbParams(spread, minDist);
    }

    /// <summary>
    /// Fits UMAP and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        _nFeatures = data.Columns;

        if (_nNeighbors >= _nSamples)
        {
            throw new ArgumentException(
                $"Number of neighbors ({_nNeighbors}) must be less than number of samples ({_nSamples}).");
        }

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Store training data for transform
        _trainingData = Tensor<T>.FromMatrix(data);

        // Step 1: Compute k-nearest neighbors
        var (knnIndices, knnDistances) = ComputeKNN(_trainingData);

        // Step 2: Compute fuzzy simplicial set (graph weights)
        var graph = ComputeFuzzySimplicialSet(knnIndices, knnDistances);

        // Step 3: Initialize embedding
        var embedding = InitializeEmbedding(_nSamples, random);

        // Step 4: Optimize embedding
        _embedding = OptimizeLayout(graph, embedding, random).ToMatrix();
    }

    /// <summary>
    /// The k nearest neighbours of every training point (excluding itself, nearest first, ties in index order)
    /// and their distances.
    /// </summary>
    private (Tensor<int> Indices, Tensor<T> Distances) ComputeKNN(Tensor<T> data)
    {
        var distances = ComputeDistances(data, data);
        var indices = DimensionalityReductionOps.NearestNeighbors(distances, _nNeighbors);
        return (indices, DimensionalityReductionOps.TakeRows(distances, indices));
    }

    /// <summary>
    /// Pairwise distances between the rows of <paramref name="a"/> and <paramref name="b"/> using the configured metric.
    /// </summary>
    private Tensor<T> ComputeDistances(Tensor<T> a, Tensor<T> b)
    {
        var engine = Engine;
        switch (_metric)
        {
            case UMAPMetric.Euclidean:
                return engine.TensorCDist(a, b, 2.0);

            case UMAPMetric.Manhattan:
                return engine.TensorCDist(a, b, 1.0);

            case UMAPMetric.Cosine:
                return OneMinusCosine(a, b);

            case UMAPMetric.Correlation:
                // Cosine distance of the row-centered vectors
                return OneMinusCosine(
                    engine.TensorSubtract(a, engine.ReduceMean(a, new[] { 1 }, true)),
                    engine.TensorSubtract(b, engine.ReduceMean(b, new[] { 1 }, true)));

            default:
                throw new NotSupportedException($"Metric {_metric} is not supported.");
        }
    }

    /// <summary>1 - a_i . b_j / (|a_i| |b_j|), or 1 where the norm product is at most the distance epsilon.</summary>
    private Tensor<T> OneMinusCosine(Tensor<T> a, Tensor<T> b)
    {
        var engine = Engine;
        var dot = DimensionalityReductionOps.MatMulTranspose(a, b);
        var normA = engine.TensorSqrt(DimensionalityReductionOps.RowSums(engine.TensorSquare(a)));
        var normB = engine.TensorSqrt(DimensionalityReductionOps.RowSums(engine.TensorSquare(b)));
        var denom = engine.TensorMatMul(DimensionalityReductionOps.AsColumn(normA), DimensionalityReductionOps.AsRow(normB));
        var valid = engine.TensorGreaterThan(denom, NumOps.FromDouble(DistanceEpsilon));
        var ones = engine.TensorOnesLike(dot);
        return engine.TensorWhere(
            valid,
            engine.TensorSubtract(ones, engine.TensorDivide(dot, engine.TensorWhere(valid, denom, ones))),
            ones);
    }

    /// <summary>
    /// Fuzzy simplicial set: rho_i from the local-connectivity neighbour distance, sigma_i by a 64-step bisection
    /// (all rows advanced together; a row stops once |sum - log2 k| &lt; 1e-5), weights exp(-(d - rho) / sigma)
    /// (1 where d &lt;= rho), then the fuzzy union a + b - a b.
    /// </summary>
    private Tensor<T> ComputeFuzzySimplicialSet(Tensor<int> knnIndices, Tensor<T> knnDistances)
    {
        var engine = Engine;
        int n = _nSamples;
        int k = _nNeighbors;
        var one = NumOps.One;
        var zero = NumOps.Zero;

        // rho: distance to the local_connectivity-th neighbour (interpolated for fractional values)
        int localConnIdx = (int)Math.Floor(_localConnectivity) - 1;
        localConnIdx = Math.Max(0, Math.Min(localConnIdx, k - 1));
        double localConnFrac = _localConnectivity - Math.Floor(_localConnectivity);
        var rho = engine.TensorNarrow(knnDistances, 1, localConnIdx, 1);   // [n, 1]
        if (!(localConnFrac < 1e-5 || localConnIdx >= k - 1))
        {
            rho = engine.TensorAdd(
                engine.TensorMultiplyScalar(rho, NumOps.FromDouble(1 - localConnFrac)),
                engine.TensorMultiplyScalar(engine.TensorNarrow(knnDistances, 1, localConnIdx + 1, 1), NumOps.FromDouble(localConnFrac)));
        }

        var shifted = engine.TensorSubtract(knnDistances, rho);
        var beyond = engine.TensorGreaterThan(shifted, zero);
        var neighborOnes = engine.TensorOnesLike(shifted);
        var columnOnes = DimensionalityReductionOps.Full(new[] { n, 1 }, one);

        // Binary search for sigma
        var target = NumOps.FromDouble(Math.Log(k) / Math.Log(2));
        var tolerance = NumOps.FromDouble(1e-5);
        var half = NumOps.FromDouble(0.5);
        var two = NumOps.FromDouble(2.0);
        var unbounded = DimensionalityReductionOps.Full(new[] { n, 1 }, NumOps.FromDouble(999));
        var lo = DimensionalityReductionOps.Full(new[] { n, 1 }, NumOps.FromDouble(1e-20));
        var hi = DimensionalityReductionOps.Full(new[] { n, 1 }, NumOps.FromDouble(1000.0));
        var mid = columnOnes;
        var active = columnOnes;

        for (int iter = 0; iter < 64; iter++)
        {
            var terms = engine.TensorWhere(beyond, engine.TensorExp(engine.TensorNegate(engine.TensorDivide(shifted, mid))), neighborOnes);
            var sum = engine.ReduceSum(terms, new[] { 1 }, true);

            var converged = engine.TensorLessThan(engine.TensorAbs(engine.TensorSubtractScalar(sum, target)), tolerance);
            active = engine.TensorMultiply(active, engine.TensorSubtract(columnOnes, converged));
            if (!DimensionalityReductionOps.Above(engine.TensorSum(active), zero))
            {
                break;
            }

            var tooLarge = engine.TensorGreaterThan(sum, target);
            var newHi = engine.TensorWhere(tooLarge, mid, hi);
            var newLo = engine.TensorWhere(tooLarge, lo, mid);
            var newMid = engine.TensorWhere(
                tooLarge,
                engine.TensorMultiplyScalar(engine.TensorAdd(lo, mid), half),
                engine.TensorWhere(
                    engine.TensorGreaterEqual(hi, unbounded),
                    engine.TensorMultiplyScalar(mid, two),
                    engine.TensorMultiplyScalar(engine.TensorAdd(mid, hi), half)));
            hi = engine.TensorWhere(active, newHi, hi);
            lo = engine.TensorWhere(active, newLo, lo);
            mid = engine.TensorWhere(active, newMid, mid);
        }

        // Compute graph weights
        var weights = engine.TensorWhere(beyond, engine.TensorExp(engine.TensorNegate(engine.TensorDivide(shifted, mid))), neighborOnes);
        var graph = DimensionalityReductionOps.ScatterRows(knnIndices, weights, n, zero);

        // Symmetrize using fuzzy set union: a + b - a*b
        var transposed = engine.TensorTranspose(graph);
        return engine.TensorSubtract(engine.TensorAdd(graph, transposed), engine.TensorMultiply(graph, transposed));
    }

    /// <summary>Random initialization 10 (u - 0.5), row-major.</summary>
    private Tensor<T> InitializeEmbedding(int n, Random random) =>
        Engine.TensorMultiplyScalar(
            Engine.TensorSubtractScalar(DimensionalityReductionOps.Uniform<T>(random, n, _nComponents), NumOps.FromDouble(0.5)),
            NumOps.FromDouble(10.0));

    /// <summary>
    /// Sequential edge-wise SGD with negative sampling. Every edge update reads the positions written by the
    /// previous one, so this stays the original scalar loop (in double) until the engine has an edge-sequential
    /// layout kernel; see missing-ops.txt.
    /// </summary>
    private Tensor<T> OptimizeLayout(Tensor<T> graphTensor, Tensor<T> initial, Random random)
    {
        int n = _nSamples;
        var graph = graphTensor.Cast<double>().ToMatrix();
        var embedding = initial.Cast<double>().ToMatrix();

        // Build edge list for efficient sampling
        var edges = new List<(int i, int j, double weight)>();
        var epochsPerSample = new List<double>();

        double maxWeight = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (graph[i, j] > 0)
                {
                    edges.Add((i, j, graph[i, j]));
                    maxWeight = Math.Max(maxWeight, graph[i, j]);
                }
            }
        }

        // Compute epochs per sample based on weight
        foreach (var (_, _, weight) in edges)
        {
            epochsPerSample.Add(maxWeight > 0 ? _nEpochs * weight / maxWeight : _nEpochs);
        }

        // Optimization
        double alpha = _learningRate;
        var epochOfNextSample = new double[edges.Count];

        for (int epoch = 0; epoch < _nEpochs; epoch++)
        {
            // Update learning rate
            alpha = _learningRate * (1.0 - (double)epoch / _nEpochs);
            alpha = Math.Max(alpha, 0.0001);

            // Process edges
            for (int e = 0; e < edges.Count; e++)
            {
                if (epochOfNextSample[e] > epoch)
                {
                    continue;
                }

                var (i, j, _) = edges[e];

                // Compute gradient for attractive force
                double distSq = 0;
                for (int d = 0; d < _nComponents; d++)
                {
                    double diff = embedding[i, d] - embedding[j, d];
                    distSq += diff * diff;
                }

                double gradCoeff;
                if (distSq > 0)
                {
                    gradCoeff = -2.0 * _a * _b * Math.Pow(distSq, _b - 1.0) /
                                (1.0 + _a * Math.Pow(distSq, _b));
                }
                else
                {
                    gradCoeff = 0;
                }

                for (int d = 0; d < _nComponents; d++)
                {
                    double grad = gradCoeff * (embedding[i, d] - embedding[j, d]);
                    grad = Math.Max(-4, Math.Min(4, grad)); // Clip gradient

                    embedding[i, d] -= alpha * grad;
                    embedding[j, d] += alpha * grad;
                }

                // Negative sampling for repulsive forces
                int nNegativeSamples = (int)Math.Round(_negativeSampleRate);
                for (int neg = 0; neg < nNegativeSamples; neg++)
                {
                    int k = random.Next(n);
                    if (k == i) continue;

                    distSq = 0;
                    for (int d = 0; d < _nComponents; d++)
                    {
                        double diff = embedding[i, d] - embedding[k, d];
                        distSq += diff * diff;
                    }

                    if (distSq > 0)
                    {
                        gradCoeff = 2.0 * _repulsionStrength * _b /
                                    ((0.001 + distSq) * (1.0 + _a * Math.Pow(distSq, _b)));
                    }
                    else
                    {
                        gradCoeff = 0;
                    }

                    for (int d = 0; d < _nComponents; d++)
                    {
                        double grad = gradCoeff * (embedding[i, d] - embedding[k, d]);
                        grad = Math.Max(-4, Math.Min(4, grad)); // Clip gradient

                        embedding[i, d] += alpha * grad;
                    }
                }

                epochOfNextSample[e] += epochsPerSample[e];
            }
        }

        return Tensor<double>.FromMatrix(embedding).Cast<T>();
    }

    private static (double a, double b) FindAbParams(double spread, double minDist)
    {
        // Find a and b parameters that give the desired spread and min_dist
        // Using curve fitting for: 1 / (1 + a * x^(2*b))

        double a = 1.929;
        double b = 0.7915;

        // Adjust based on spread and min_dist
        if (Math.Abs(spread - 1.0) > 0.001 || Math.Abs(minDist - 0.1) > 0.001)
        {
            // Simple approximation - for more accurate results, use optimization
            double scale = spread;
            a = 1.929 / (scale * scale);
            b = 0.7915;

            if (minDist < 0.001)
            {
                b = 0.9;
            }
            else if (minDist > 0.5)
            {
                b = 0.6;
            }
        }

        return (a, b);
    }

    /// <summary>
    /// Transforms data using the fitted UMAP embedding.
    /// </summary>
    /// <remarks>
    /// <para>This method always performs out-of-sample transformation using the learned
    /// embedding space. To get the original training embedding, use <see cref="GetEmbedding"/>.</para>
    /// </remarks>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null || _trainingData is null)
        {
            throw new InvalidOperationException("UMAP has not been fitted.");
        }

        // Always perform out-of-sample transformation for consistency
        // Checking row count is fragile - different datasets can have the same number of rows
        return TransformNewData(data);
    }

    /// <summary>
    /// Gets the embedding computed during Fit for the training data.
    /// </summary>
    /// <returns>The embedding matrix for the training data.</returns>
    public Matrix<T> GetEmbedding()
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("UMAP has not been fitted.");
        }

        return _embedding.Clone();
    }

    private Matrix<T> TransformNewData(Matrix<T> data)
    {
        var engine = Engine;
        int p = data.Columns;

        if (p != _nFeatures)
        {
            throw new ArgumentException(
                $"New data has {p} features but model was fitted with {_nFeatures} features.");
        }

        var trainingData = _trainingData ?? throw new InvalidOperationException("Training data not set.");
        var embedding = _embedding ?? throw new InvalidOperationException("Embedding not computed.");

        // For each new point, find the k nearest training points (same metric, ties in index order)
        // and compute the inverse-distance weighted average of their embeddings
        var distances = ComputeDistances(Tensor<T>.FromMatrix(data), trainingData);
        var order = engine.TensorNarrow(engine.TensorArgsort(distances, 1, false), 1, 0, _nNeighbors);
        var weights = engine.TensorReciprocal(
            engine.TensorAddScalar(DimensionalityReductionOps.TakeRows(distances, order), NumOps.FromDouble(DistanceEpsilon)));

        var weightMatrix = DimensionalityReductionOps.ScatterRows(order, weights, _nSamples, NumOps.Zero);
        var weighted = engine.TensorMatMul(weightMatrix, Tensor<T>.FromMatrix(embedding));
        return DimensionalityReductionOps.ScaleRows(weighted, engine.TensorReciprocal(DimensionalityReductionOps.RowSums(weights))).ToMatrix();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("UMAP does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"UMAP{i + 1}";
        }
        return names;
    }
}
