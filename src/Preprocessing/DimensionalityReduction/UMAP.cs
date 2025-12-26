using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
public class UMAP<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
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
    private double[,]? _embedding;
    private double[,]? _trainingData;
    private int[,]? _knnIndices;
    private double[,]? _knnDistances;
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
    public double[,]? Embedding => _embedding;

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

        // Convert to double array and store for transform
        _trainingData = new double[_nSamples, _nFeatures];
        for (int i = 0; i < _nSamples; i++)
        {
            for (int j = 0; j < _nFeatures; j++)
            {
                _trainingData[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Step 1: Compute k-nearest neighbors
        ComputeKNN(_trainingData);

        // Step 2: Compute fuzzy simplicial set (graph weights)
        var graph = ComputeFuzzySimplicialSet();

        // Step 3: Initialize embedding
        _embedding = InitializeEmbedding(_nSamples, random);

        // Step 4: Optimize embedding
        OptimizeLayout(graph, random);
    }

    private void ComputeKNN(double[,] data)
    {
        int n = data.GetLength(0);
        int p = data.GetLength(1);

        _knnIndices = new int[n, _nNeighbors];
        _knnDistances = new double[n, _nNeighbors];

        // Compute all pairwise distances and find k-nearest neighbors
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

                double dist = ComputeDistance(data, i, j, p);
                distances[j] = (dist, j);
            }

            // Sort by distance
            Array.Sort(distances, (a, b) => a.dist.CompareTo(b.dist));

            // Store k nearest
            for (int k = 0; k < _nNeighbors; k++)
            {
                _knnIndices[i, k] = distances[k].idx;
                _knnDistances[i, k] = distances[k].dist;
            }
        }
    }

    private double ComputeDistance(double[,] data, int i, int j, int p)
    {
        double dist = 0;
        switch (_metric)
        {
            case UMAPMetric.Euclidean:
                for (int k = 0; k < p; k++)
                {
                    double diff = data[i, k] - data[j, k];
                    dist += diff * diff;
                }
                return Math.Sqrt(dist);

            case UMAPMetric.Manhattan:
                for (int k = 0; k < p; k++)
                {
                    dist += Math.Abs(data[i, k] - data[j, k]);
                }
                return dist;

            case UMAPMetric.Cosine:
                double dot = 0, normI = 0, normJ = 0;
                for (int k = 0; k < p; k++)
                {
                    dot += data[i, k] * data[j, k];
                    normI += data[i, k] * data[i, k];
                    normJ += data[j, k] * data[j, k];
                }
                double denom = Math.Sqrt(normI * normJ);
                return denom > 1e-10 ? 1 - dot / denom : 1;

            case UMAPMetric.Correlation:
                double meanI = 0, meanJ = 0;
                for (int k = 0; k < p; k++)
                {
                    meanI += data[i, k];
                    meanJ += data[j, k];
                }
                meanI /= p;
                meanJ /= p;

                double cov = 0, varI = 0, varJ = 0;
                for (int k = 0; k < p; k++)
                {
                    double diffI = data[i, k] - meanI;
                    double diffJ = data[j, k] - meanJ;
                    cov += diffI * diffJ;
                    varI += diffI * diffI;
                    varJ += diffJ * diffJ;
                }
                double denomCorr = Math.Sqrt(varI * varJ);
                return denomCorr > 1e-10 ? 1 - cov / denomCorr : 1;

            default:
                throw new NotSupportedException($"Metric {_metric} is not supported.");
        }
    }

    /// <summary>
    /// Computes distance between two points using the configured metric.
    /// </summary>
    private double ComputeDistance(double[] pointA, double[] pointB)
    {
        int p = pointA.Length;
        double dist = 0;
        switch (_metric)
        {
            case UMAPMetric.Euclidean:
                for (int k = 0; k < p; k++)
                {
                    double diff = pointA[k] - pointB[k];
                    dist += diff * diff;
                }
                return Math.Sqrt(dist);

            case UMAPMetric.Manhattan:
                for (int k = 0; k < p; k++)
                {
                    dist += Math.Abs(pointA[k] - pointB[k]);
                }
                return dist;

            case UMAPMetric.Cosine:
                double dot = 0, normA = 0, normB = 0;
                for (int k = 0; k < p; k++)
                {
                    dot += pointA[k] * pointB[k];
                    normA += pointA[k] * pointA[k];
                    normB += pointB[k] * pointB[k];
                }
                double denom = Math.Sqrt(normA * normB);
                return denom > 1e-10 ? 1 - dot / denom : 1;

            case UMAPMetric.Correlation:
                double meanA = 0, meanB = 0;
                for (int k = 0; k < p; k++)
                {
                    meanA += pointA[k];
                    meanB += pointB[k];
                }
                meanA /= p;
                meanB /= p;

                double cov = 0, varA = 0, varB = 0;
                for (int k = 0; k < p; k++)
                {
                    double diffA = pointA[k] - meanA;
                    double diffB = pointB[k] - meanB;
                    cov += diffA * diffB;
                    varA += diffA * diffA;
                    varB += diffB * diffB;
                }
                double denomCorr = Math.Sqrt(varA * varB);
                return denomCorr > 1e-10 ? 1 - cov / denomCorr : 1;

            default:
                throw new NotSupportedException($"Metric {_metric} is not supported.");
        }
    }

    private double[,] ComputeFuzzySimplicialSet()
    {
        int n = _nSamples;
        var sigmas = new double[n];
        var rhos = new double[n];

        // Compute rho (distance to nearest neighbor) and sigma for each point
        for (int i = 0; i < n; i++)
        {
            rhos[i] = _knnDistances![i, 0];

            // Binary search for sigma
            double target = Math.Log(_nNeighbors) / Math.Log(2);
            double lo = 1e-20;
            double hi = 1000.0;
            double mid = 1.0;

            for (int iter = 0; iter < 64; iter++)
            {
                double sum = 0;
                for (int j = 0; j < _nNeighbors; j++)
                {
                    double d = _knnDistances[i, j] - rhos[i];
                    if (d > 0)
                    {
                        sum += Math.Exp(-d / mid);
                    }
                    else
                    {
                        sum += 1;
                    }
                }

                if (Math.Abs(sum - target) < 1e-5)
                {
                    break;
                }

                if (sum > target)
                {
                    hi = mid;
                    mid = (lo + hi) / 2;
                }
                else
                {
                    lo = mid;
                    if (hi >= 999)
                    {
                        mid *= 2;
                    }
                    else
                    {
                        mid = (lo + hi) / 2;
                    }
                }
            }

            sigmas[i] = mid;
        }

        // Compute graph weights
        var graph = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < _nNeighbors; k++)
            {
                int j = _knnIndices![i, k];
                double d = _knnDistances![i, k] - rhos[i];
                double weight = d > 0 ? Math.Exp(-d / sigmas[i]) : 1.0;
                graph[i, j] = weight;
            }
        }

        // Symmetrize using fuzzy set union
        var symmetricGraph = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double a = graph[i, j];
                double b = graph[j, i];
                // Fuzzy set union: a + b - a*b
                symmetricGraph[i, j] = a + b - a * b;
            }
        }

        return symmetricGraph;
    }

    private double[,] InitializeEmbedding(int n, Random random)
    {
        var embedding = new double[n, _nComponents];

        // Spectral initialization using graph Laplacian
        // For simplicity, use random initialization scaled appropriately
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < _nComponents; j++)
            {
                // Initialize with small random values centered at 0
                embedding[i, j] = 10 * (random.NextDouble() - 0.5);
            }
        }

        return embedding;
    }

    private void OptimizeLayout(double[,] graph, Random random)
    {
        int n = _nSamples;
        var embedding = _embedding!;

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
                int nNegativeSamples = (int)_negativeSampleRate;
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

        _embedding = embedding;
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

    private Matrix<T> TransformNewData(Matrix<T> data)
    {
        int nNew = data.Rows;
        int p = data.Columns;

        if (p != _nFeatures)
        {
            throw new ArgumentException(
                $"New data has {p} features but model was fitted with {_nFeatures} features.");
        }

        var newData = new double[nNew, p];
        for (int i = 0; i < nNew; i++)
        {
            for (int j = 0; j < p; j++)
            {
                newData[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        var result = new T[nNew, _nComponents];

        // For each new point, find k-nearest neighbors in training data
        // and compute weighted average of their embeddings
        for (int i = 0; i < nNew; i++)
        {
            // Extract the new point as an array
            var newPoint = new double[p];
            for (int k = 0; k < p; k++)
            {
                newPoint[k] = newData[i, k];
            }

            var distances = new (double dist, int idx)[_nSamples];

            for (int j = 0; j < _nSamples; j++)
            {
                // Extract the training point as an array
                var trainPoint = new double[p];
                for (int k = 0; k < p; k++)
                {
                    trainPoint[k] = _trainingData![j, k];
                }
                // Use the same distance metric as during training
                double dist = ComputeDistance(newPoint, trainPoint);
                distances[j] = (dist, j);
            }

            Array.Sort(distances, (a, b) => a.dist.CompareTo(b.dist));

            // Compute weighted average of k-nearest embeddings
            double totalWeight = 0;
            var embeddingSum = new double[_nComponents];

            for (int k = 0; k < _nNeighbors; k++)
            {
                double weight = 1.0 / (distances[k].dist + 1e-10);
                totalWeight += weight;
                int idx = distances[k].idx;

                for (int d = 0; d < _nComponents; d++)
                {
                    embeddingSum[d] += weight * _embedding![idx, d];
                }
            }

            for (int d = 0; d < _nComponents; d++)
            {
                result[i, d] = NumOps.FromDouble(embeddingSum[d] / totalWeight);
            }
        }

        return new Matrix<T>(result);
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
