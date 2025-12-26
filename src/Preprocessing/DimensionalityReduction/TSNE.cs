using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// t-Distributed Stochastic Neighbor Embedding for visualization.
/// </summary>
/// <remarks>
/// <para>
/// t-SNE is a nonlinear dimensionality reduction technique well-suited for
/// visualizing high-dimensional data in 2D or 3D space. It preserves local
/// structure by keeping similar points close together.
/// </para>
/// <para>
/// The algorithm converts similarities between data points to joint probabilities
/// and tries to minimize the divergence between probability distributions in
/// high and low dimensional spaces.
/// </para>
/// <para><b>For Beginners:</b> t-SNE creates beautiful 2D/3D visualizations:
/// - Points that are similar stay close together
/// - Points that are different move apart
/// - Great for exploring clusters in your data
/// - Warning: Not for preserving global distances, just local neighborhoods
/// - Warning: Results can vary with different random seeds
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class TSNE<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly double _perplexity;
    private readonly double _learningRate;
    private readonly int _nIter;
    private readonly double _earlyExaggeration;
    private readonly int? _randomState;
    private readonly TSNEMetric _metric;
    private readonly TSNEInitialization _initialization;

    // Fitted parameters
    private double[,]? _embedding;
    private int _nSamples;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the perplexity parameter.
    /// </summary>
    public double Perplexity => _perplexity;

    /// <summary>
    /// Gets the learning rate.
    /// </summary>
    public double LearningRate => _learningRate;

    /// <summary>
    /// Gets the distance metric.
    /// </summary>
    public TSNEMetric Metric => _metric;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public double[,]? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="TSNE{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality (usually 2 or 3). Defaults to 2.</param>
    /// <param name="perplexity">Balance between local and global structure. Defaults to 30.</param>
    /// <param name="learningRate">Learning rate for optimization. Defaults to 200.</param>
    /// <param name="nIter">Number of optimization iterations. Defaults to 1000.</param>
    /// <param name="earlyExaggeration">Exaggeration factor for early iterations. Defaults to 12.</param>
    /// <param name="metric">Distance metric to use. Defaults to Euclidean.</param>
    /// <param name="initialization">Initialization method. Defaults to Random.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public TSNE(
        int nComponents = 2,
        double perplexity = 30.0,
        double learningRate = 200.0,
        int nIter = 1000,
        double earlyExaggeration = 12.0,
        TSNEMetric metric = TSNEMetric.Euclidean,
        TSNEInitialization initialization = TSNEInitialization.Random,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (perplexity <= 0)
        {
            throw new ArgumentException("Perplexity must be positive.", nameof(perplexity));
        }

        _nComponents = nComponents;
        _perplexity = perplexity;
        _learningRate = learningRate;
        _nIter = nIter;
        _earlyExaggeration = earlyExaggeration;
        _metric = metric;
        _initialization = initialization;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits t-SNE and computes the embedding.
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

        // Compute pairwise distances
        var distances = ComputeDistances(X, n, p);

        // Compute joint probabilities P
        var P = ComputeJointProbabilities(distances, n);

        // Apply early exaggeration
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                P[i, j] *= _earlyExaggeration;
            }
        }

        // Initialize embedding Y
        _embedding = InitializeEmbedding(X, n, p, random);

        // Gradient descent
        var Y = _embedding;
        var gains = new double[n, _nComponents];
        var velocity = new double[n, _nComponents];

        // Initialize gains to 1
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < _nComponents; j++)
            {
                gains[i, j] = 1.0;
            }
        }

        double momentum = 0.5;
        int earlyExaggerationEnd = 250;

        for (int iter = 0; iter < _nIter; iter++)
        {
            // Update momentum
            if (iter == 250)
            {
                momentum = 0.8;
            }

            // Remove early exaggeration
            if (iter == earlyExaggerationEnd)
            {
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        P[i, j] /= _earlyExaggeration;
                    }
                }
            }

            // Compute Q (student-t distribution)
            var Q = ComputeQ(Y, n);

            // Compute gradients
            var gradient = ComputeGradient(P, Q, Y, n);

            // Update with momentum and adaptive learning rate
            for (int i = 0; i < n; i++)
            {
                for (int d = 0; d < _nComponents; d++)
                {
                    // Adaptive gains
                    bool sameSign = (gradient[i, d] > 0) == (velocity[i, d] > 0);
                    if (sameSign)
                    {
                        gains[i, d] = Math.Max(gains[i, d] * 0.8, 0.01);
                    }
                    else
                    {
                        gains[i, d] += 0.2;
                    }

                    // Update velocity
                    velocity[i, d] = momentum * velocity[i, d] - _learningRate * gains[i, d] * gradient[i, d];

                    // Update position
                    Y[i, d] += velocity[i, d];
                }
            }

            // Re-center
            for (int d = 0; d < _nComponents; d++)
            {
                double mean = 0;
                for (int i = 0; i < n; i++)
                {
                    mean += Y[i, d];
                }
                mean /= n;

                for (int i = 0; i < n; i++)
                {
                    Y[i, d] -= mean;
                }
            }
        }

        _embedding = Y;
    }

    private double[,] ComputeDistances(double[,] X, int n, int p)
    {
        var distances = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;
                switch (_metric)
                {
                    case TSNEMetric.Euclidean:
                        for (int k = 0; k < p; k++)
                        {
                            double diff = X[i, k] - X[j, k];
                            dist += diff * diff;
                        }
                        dist = Math.Sqrt(dist);
                        break;

                    case TSNEMetric.Manhattan:
                        for (int k = 0; k < p; k++)
                        {
                            dist += Math.Abs(X[i, k] - X[j, k]);
                        }
                        break;

                    case TSNEMetric.Cosine:
                        double dot = 0, normI = 0, normJ = 0;
                        for (int k = 0; k < p; k++)
                        {
                            dot += X[i, k] * X[j, k];
                            normI += X[i, k] * X[i, k];
                            normJ += X[j, k] * X[j, k];
                        }
                        double denom = Math.Sqrt(normI * normJ);
                        dist = denom > 1e-10 ? 1 - dot / denom : 1;
                        break;
                }

                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        return distances;
    }

    private double[,] ComputeJointProbabilities(double[,] distances, int n)
    {
        // Compute conditional probabilities using binary search for sigma
        var P = new double[n, n];
        double logPerplexity = Math.Log(_perplexity);

        for (int i = 0; i < n; i++)
        {
            // Binary search for sigma
            double sigmaLow = 1e-20;
            double sigmaHigh = 1e20;
            double sigma = 1.0;

            for (int iter = 0; iter < 50; iter++)
            {
                // Compute P_i|j with current sigma
                double sumP = 0;
                double entropy = 0;

                for (int j = 0; j < n; j++)
                {
                    if (j != i)
                    {
                        double pij = Math.Exp(-distances[i, j] * distances[i, j] / (2 * sigma * sigma));
                        P[i, j] = pij;
                        sumP += pij;
                    }
                }

                if (sumP > 1e-10)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (j != i)
                        {
                            P[i, j] /= sumP;
                            if (P[i, j] > 1e-10)
                            {
                                entropy -= P[i, j] * Math.Log(P[i, j]);
                            }
                        }
                    }
                }

                // Adjust sigma
                double perpDiff = entropy - logPerplexity;
                if (Math.Abs(perpDiff) < 1e-5)
                {
                    break;
                }

                if (perpDiff > 0)
                {
                    sigmaHigh = sigma;
                    sigma = (sigmaLow + sigma) / 2;
                }
                else
                {
                    sigmaLow = sigma;
                    if (sigmaHigh >= 1e19)
                    {
                        sigma *= 2;
                    }
                    else
                    {
                        sigma = (sigma + sigmaHigh) / 2;
                    }
                }
            }
        }

        // Symmetrize
        var Psym = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Psym[i, j] = (P[i, j] + P[j, i]) / (2 * n);
            }
        }

        // Ensure minimum probability
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Psym[i, j] = Math.Max(Psym[i, j], 1e-12);
            }
        }

        return Psym;
    }

    private double[,] InitializeEmbedding(double[,] X, int n, int p, Random random)
    {
        var Y = new double[n, _nComponents];

        switch (_initialization)
        {
            case TSNEInitialization.PCA:
                // Simple PCA initialization
                var mean = new double[p];
                for (int j = 0; j < p; j++)
                {
                    for (int i = 0; i < n; i++)
                    {
                        mean[j] += X[i, j];
                    }
                    mean[j] /= n;
                }

                // Use first few principal directions (simplified)
                for (int i = 0; i < n; i++)
                {
                    for (int d = 0; d < _nComponents; d++)
                    {
                        Y[i, d] = 0;
                        for (int j = 0; j < Math.Min(p, _nComponents); j++)
                        {
                            Y[i, d] += (X[i, j] - mean[j]) * (d == j ? 1 : 0);
                        }
                        Y[i, d] *= 0.0001;
                    }
                }
                break;

            case TSNEInitialization.Random:
            default:
                for (int i = 0; i < n; i++)
                {
                    for (int d = 0; d < _nComponents; d++)
                    {
                        // Initialize with small random values
                        Y[i, d] = 0.0001 * (random.NextDouble() - 0.5);
                    }
                }
                break;
        }

        return Y;
    }

    private double[,] ComputeQ(double[,] Y, int n)
    {
        var Q = new double[n, n];
        double sumQ = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;
                for (int d = 0; d < _nComponents; d++)
                {
                    double diff = Y[i, d] - Y[j, d];
                    dist += diff * diff;
                }

                // Student-t distribution with 1 degree of freedom
                double qij = 1.0 / (1.0 + dist);
                Q[i, j] = qij;
                Q[j, i] = qij;
                sumQ += 2 * qij;
            }
        }

        // Normalize
        if (sumQ > 0)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Q[i, j] = Math.Max(Q[i, j] / sumQ, 1e-12);
                }
            }
        }

        return Q;
    }

    private double[,] ComputeGradient(double[,] P, double[,] Q, double[,] Y, int n)
    {
        var gradient = new double[n, _nComponents];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;

                double dist = 0;
                for (int d = 0; d < _nComponents; d++)
                {
                    double diff = Y[i, d] - Y[j, d];
                    dist += diff * diff;
                }

                double mult = 4 * (P[i, j] - Q[i, j]) / (1 + dist);

                for (int d = 0; d < _nComponents; d++)
                {
                    gradient[i, d] += mult * (Y[i, d] - Y[j, d]);
                }
            }
        }

        return gradient;
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("TSNE has not been fitted.");
        }

        // t-SNE doesn't support out-of-sample transformation
        // Return the fitted embedding if it's the same data
        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "t-SNE does not support out-of-sample transformation. " +
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
        throw new NotSupportedException("TSNE does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"TSNE{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the distance metric for t-SNE.
/// </summary>
public enum TSNEMetric
{
    /// <summary>
    /// Euclidean (L2) distance.
    /// </summary>
    Euclidean,

    /// <summary>
    /// Manhattan (L1) distance.
    /// </summary>
    Manhattan,

    /// <summary>
    /// Cosine distance (1 - cosine similarity).
    /// </summary>
    Cosine
}

/// <summary>
/// Specifies the initialization method for t-SNE.
/// </summary>
public enum TSNEInitialization
{
    /// <summary>
    /// Random initialization from a small normal distribution.
    /// </summary>
    Random,

    /// <summary>
    /// Initialize using PCA projection.
    /// </summary>
    PCA
}
