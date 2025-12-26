using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Neighborhood Components Analysis (NCA) for supervised dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// NCA is a supervised dimensionality reduction algorithm that learns a linear transformation
/// to maximize the expected leave-one-out classification accuracy in the transformed space.
/// It uses stochastic neighbor assignment for soft nearest-neighbor classification.
/// </para>
/// <para>
/// The algorithm:
/// 1. Define soft neighbor probabilities using softmax over distances
/// 2. Compute expected leave-one-out classification accuracy
/// 3. Optimize transformation matrix using gradient descent
/// 4. Project data using learned transformation
/// </para>
/// <para><b>For Beginners:</b> NCA learns a space where k-NN works well:
/// - Points with same label are pulled closer together
/// - Points with different labels are pushed apart
/// - The learned transformation is linear (a matrix)
/// - Can be used for feature extraction before classification
///
/// Use cases:
/// - Preprocessing for k-NN classifiers
/// - Metric learning for distance-based methods
/// - Visualization with class structure preserved
/// - Feature extraction for classification tasks
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class NCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _maxIter;
    private readonly double _learningRate;
    private readonly double _tol;
    private readonly int? _randomState;

    // Fitted parameters
    private double[,]? _transformationMatrix; // A: d x p (projects from p to d dimensions)
    private int _nFeatures;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the learned transformation matrix.
    /// </summary>
    public double[,]? TransformationMatrix => _transformationMatrix;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="NCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="maxIter">Maximum number of optimization iterations. Defaults to 100.</param>
    /// <param name="learningRate">Learning rate for gradient descent. Defaults to 0.01.</param>
    /// <param name="tol">Tolerance for convergence. Defaults to 1e-5.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public NCA(
        int nComponents = 2,
        int maxIter = 100,
        double learningRate = 0.01,
        double tol = 1e-5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _maxIter = maxIter;
        _learningRate = learningRate;
        _tol = tol;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits NCA using the provided data and labels.
    /// </summary>
    /// <remarks>
    /// This method is internal to maintain the transformer pattern.
    /// Use <see cref="FitTransformSupervised"/> for supervised fitting with labels.
    /// </remarks>
    /// <param name="data">The input data matrix.</param>
    /// <param name="labels">The class labels for each sample.</param>
    internal void Fit(Matrix<T> data, int[] labels)
    {
        _nFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (labels.Length != n)
        {
            throw new ArgumentException("Labels length must match number of samples.", nameof(labels));
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

        // Initialize transformation matrix A (d x p) with small random values
        var A = new double[_nComponents, p];
        for (int i = 0; i < _nComponents; i++)
        {
            for (int j = 0; j < p; j++)
            {
                A[i, j] = (random.NextDouble() - 0.5) * 0.01;
            }
        }

        // Gradient descent optimization
        double prevObjective = double.NegativeInfinity;
        double lr = _learningRate;

        for (int iter = 0; iter < _maxIter; iter++)
        {
            // Compute transformed data: Y = X * A^T (n x d)
            var Y = new double[n, _nComponents];
            for (int i = 0; i < n; i++)
            {
                for (int d = 0; d < _nComponents; d++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        Y[i, d] += X[i, j] * A[d, j];
                    }
                }
            }

            // Compute pairwise distances in transformed space
            var distances = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;
                    for (int d = 0; d < _nComponents; d++)
                    {
                        double diff = Y[i, d] - Y[j, d];
                        distances[i, j] += diff * diff;
                    }
                }
            }

            // Compute softmax probabilities p_ij
            var probabilities = new double[n, n];
            var sumExp = new double[n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;
                    probabilities[i, j] = Math.Exp(-distances[i, j]);
                    sumExp[i] += probabilities[i, j];
                }
            }

            // Normalize probabilities
            for (int i = 0; i < n; i++)
            {
                if (sumExp[i] > 1e-10)
                {
                    for (int j = 0; j < n; j++)
                    {
                        probabilities[i, j] /= sumExp[i];
                    }
                }
            }

            // Compute p_i (probability of correct classification for point i)
            var pi = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (labels[i] == labels[j])
                    {
                        pi[i] += probabilities[i, j];
                    }
                }
            }

            // Compute objective (sum of p_i)
            double objective = 0;
            for (int i = 0; i < n; i++)
            {
                objective += pi[i];
            }

            // Check convergence
            if (Math.Abs(objective - prevObjective) < _tol)
            {
                break;
            }
            prevObjective = objective;

            // Compute gradient
            var gradient = new double[_nComponents, p];

            for (int i = 0; i < n; i++)
            {
                // Compute weighted outer products
                // First term: sum over k where y_k = y_i of p_ik * (x_i - x_k)
                // Second term: p_i * sum over all k of p_ik * (x_i - x_k)

                for (int k = 0; k < n; k++)
                {
                    if (i == k) continue;

                    // Compute (x_i - x_k)
                    var xDiff = new double[p];
                    for (int j = 0; j < p; j++)
                    {
                        xDiff[j] = X[i, j] - X[k, j];
                    }

                    // Compute A * (x_i - x_k)
                    var aDiff = new double[_nComponents];
                    for (int d = 0; d < _nComponents; d++)
                    {
                        for (int j = 0; j < p; j++)
                        {
                            aDiff[d] += A[d, j] * xDiff[j];
                        }
                    }

                    double coeff = 0;
                    if (labels[i] == labels[k])
                    {
                        // Same class: attractive
                        coeff = probabilities[i, k] * (1 - pi[i]);
                    }
                    else
                    {
                        // Different class: repulsive (weighted by p_i)
                        coeff = -probabilities[i, k] * pi[i];
                    }

                    // Update gradient: 2 * coeff * A * (x_i - x_k) * (x_i - x_k)^T
                    for (int d = 0; d < _nComponents; d++)
                    {
                        for (int j = 0; j < p; j++)
                        {
                            gradient[d, j] += 2 * coeff * aDiff[d] * xDiff[j];
                        }
                    }
                }
            }

            // Update A using gradient ascent (maximizing objective)
            for (int d = 0; d < _nComponents; d++)
            {
                for (int j = 0; j < p; j++)
                {
                    A[d, j] += lr * gradient[d, j];
                }
            }

            // Decay learning rate
            if (iter > 0 && iter % 20 == 0)
            {
                lr *= 0.9;
            }
        }

        _transformationMatrix = A;
    }

    /// <summary>
    /// Fits NCA (without labels - uses unsupervised initialization).
    /// </summary>
    /// <remarks>
    /// This method provides a fallback when no labels are available.
    /// For best results, use the Fit(data, labels) overload with class labels.
    /// </remarks>
    protected override void FitCore(Matrix<T> data)
    {
        // Without labels, initialize with PCA-like transformation
        _nFeatures = data.Columns;
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

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i, j] -= mean[j];
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
                    C[i, j] += X[k, i] * X[k, j];
                }
                C[i, j] /= n;
            }
        }

        // Power iteration for top eigenvectors
        var A = new double[_nComponents, p];
        var Atemp = (double[,])C.Clone();

        for (int d = 0; d < _nComponents; d++)
        {
            var v = new double[p];
            for (int i = 0; i < p; i++) v[i] = random.NextDouble() - 0.5;

            for (int iter = 0; iter < 50; iter++)
            {
                var Av = new double[p];
                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++) Av[i] += Atemp[i, j] * v[j];
                }

                double norm = 0;
                for (int i = 0; i < p; i++) norm += Av[i] * Av[i];
                norm = Math.Sqrt(norm);
                if (norm < 1e-10) break;

                for (int i = 0; i < p; i++) v[i] = Av[i] / norm;
            }

            for (int j = 0; j < p; j++) A[d, j] = v[j];

            // Deflate
            var Av2 = new double[p];
            double lambda = 0;
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < p; j++) Av2[i] += Atemp[i, j] * v[j];
                lambda += v[i] * Av2[i];
            }

            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    Atemp[i, j] -= lambda * v[i] * v[j];
                }
            }
        }

        _transformationMatrix = A;
    }

    /// <summary>
    /// Fits NCA using the provided data and labels, then transforms the data.
    /// </summary>
    /// <param name="data">The input data matrix.</param>
    /// <param name="labels">The class labels for each sample.</param>
    /// <returns>The transformed data projected onto the learned space.</returns>
    public Matrix<T> FitTransformSupervised(Matrix<T> data, int[] labels)
    {
        Fit(data, labels);
        return TransformCore(data);
    }

    /// <summary>
    /// Transforms data using the learned transformation matrix.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_transformationMatrix is null)
        {
            throw new InvalidOperationException("NCA has not been fitted.");
        }

        int n = data.Rows;
        int p = data.Columns;

        if (p != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {p} features but NCA was fitted with {_nFeatures} features.",
                nameof(data));
        }

        var result = new T[n, _nComponents];

        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < _nComponents; d++)
            {
                double val = 0;
                for (int j = 0; j < p; j++)
                {
                    val += NumOps.ToDouble(data[i, j]) * _transformationMatrix[d, j];
                }
                result[i, d] = NumOps.FromDouble(val);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("NCA does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"NCA{i + 1}";
        }
        return names;
    }
}
