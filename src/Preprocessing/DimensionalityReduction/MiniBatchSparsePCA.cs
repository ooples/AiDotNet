using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Mini-batch Sparse PCA using online dictionary learning.
/// </summary>
/// <remarks>
/// <para>
/// MiniBatchSparsePCA is a faster, memory-efficient version of SparsePCA that
/// processes data in mini-batches instead of using the full dataset. This makes
/// it suitable for large datasets that don't fit in memory.
/// </para>
/// <para>
/// The algorithm uses online dictionary learning with mini-batches, updating
/// the components incrementally as it processes each batch.
/// </para>
/// <para><b>For Beginners:</b> Think of this as SparsePCA on a budget:
/// - SparsePCA looks at ALL your data at once (memory intensive)
/// - MiniBatchSparsePCA looks at small pieces at a time (memory efficient)
/// - Results are similar, but mini-batch is faster for large datasets
/// - Trade-off: Slightly less accurate but much more scalable
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class MiniBatchSparsePCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly double _alpha;
    private readonly double _ridge;
    private readonly int _batchSize;
    private readonly int _nIter;
    private readonly double _tol;
    private readonly int? _randomState;
    private readonly bool _shuffle;

    // Fitted parameters
    private double[]? _mean;
    private double[,]? _components;
    private int _nFeaturesIn;
    private int _nSamplesSeen;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the sparsity regularization parameter.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the ridge regularization parameter.
    /// </summary>
    public double Ridge => _ridge;

    /// <summary>
    /// Gets the batch size.
    /// </summary>
    public int BatchSize => _batchSize;

    /// <summary>
    /// Gets the mean of each feature.
    /// </summary>
    public double[]? Mean => _mean;

    /// <summary>
    /// Gets the sparse components (each row is a component).
    /// </summary>
    public double[,]? Components => _components;

    /// <summary>
    /// Gets the number of samples seen during fitting.
    /// </summary>
    public int NSamplesSeen => _nSamplesSeen;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="MiniBatchSparsePCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of sparse components. Defaults to 2.</param>
    /// <param name="alpha">Sparsity regularization parameter. Defaults to 1.0.</param>
    /// <param name="ridge">Ridge regularization for stability. Defaults to 0.01.</param>
    /// <param name="batchSize">Size of mini-batches. Defaults to 50.</param>
    /// <param name="nIter">Number of iterations over the full dataset. Defaults to 100.</param>
    /// <param name="tol">Convergence tolerance. Defaults to 1e-6.</param>
    /// <param name="shuffle">Whether to shuffle data before each iteration. Defaults to true.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public MiniBatchSparsePCA(
        int nComponents = 2,
        double alpha = 1.0,
        double ridge = 0.01,
        int batchSize = 50,
        int nIter = 100,
        double tol = 1e-6,
        bool shuffle = true,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (alpha < 0)
        {
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));
        }

        if (batchSize < 1)
        {
            throw new ArgumentException("Batch size must be at least 1.", nameof(batchSize));
        }

        _nComponents = nComponents;
        _alpha = alpha;
        _ridge = ridge;
        _batchSize = batchSize;
        _nIter = nIter;
        _tol = tol;
        _shuffle = shuffle;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Mini-batch Sparse PCA using online dictionary learning.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nComponents, Math.Min(n, p));

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Compute mean from entire dataset (or a large sample)
        _mean = new double[p];
        for (int j = 0; j < p; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += NumOps.ToDouble(data[i, j]);
            }
            _mean[j] = sum / n;
        }

        // Initialize components randomly
        _components = new double[k, p];
        for (int c = 0; c < k; c++)
        {
            double norm = 0;
            for (int j = 0; j < p; j++)
            {
                _components[c, j] = random.NextDouble() - 0.5;
                norm += _components[c, j] * _components[c, j];
            }
            norm = Math.Sqrt(norm);
            if (norm > 1e-10)
            {
                for (int j = 0; j < p; j++)
                {
                    _components[c, j] /= norm;
                }
            }
        }

        // Online dictionary learning
        var A = new double[k, k]; // Accumulator for code correlations
        var B = new double[k, p]; // Accumulator for data-code correlations

        // Initialize accumulators with small values for stability
        for (int i = 0; i < k; i++)
        {
            A[i, i] = _ridge;
        }

        _nSamplesSeen = 0;

        // Create index array for shuffling
        var indices = Enumerable.Range(0, n).ToArray();

        for (int iter = 0; iter < _nIter; iter++)
        {
            // Shuffle indices if requested
            if (_shuffle)
            {
                ShuffleArray(indices, random);
            }

            double[] prevComponents = new double[k * p];
            for (int c = 0; c < k; c++)
            {
                for (int j = 0; j < p; j++)
                {
                    prevComponents[c * p + j] = _components[c, j];
                }
            }

            // Process data in mini-batches
            for (int batchStart = 0; batchStart < n; batchStart += _batchSize)
            {
                int batchEnd = Math.Min(batchStart + _batchSize, n);
                int actualBatchSize = batchEnd - batchStart;

                // Extract and center batch
                var batchData = new double[actualBatchSize, p];
                for (int i = 0; i < actualBatchSize; i++)
                {
                    int idx = indices[batchStart + i];
                    for (int j = 0; j < p; j++)
                    {
                        batchData[i, j] = NumOps.ToDouble(data[idx, j]) - _mean[j];
                    }
                }

                // Compute codes for batch using current dictionary
                var batchCodes = ComputeCodes(batchData, actualBatchSize, p, k);

                // Update accumulators
                double weight = 1.0 / (_nSamplesSeen + actualBatchSize);

                for (int i = 0; i < k; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        double sum = 0;
                        for (int b = 0; b < actualBatchSize; b++)
                        {
                            sum += batchCodes[b, i] * batchCodes[b, j];
                        }
                        A[i, j] = (1 - weight) * A[i, j] + weight * sum;
                        if (i == j)
                        {
                            A[i, j] = Math.Max(A[i, j], _ridge);
                        }
                    }
                }

                for (int i = 0; i < k; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        double sum = 0;
                        for (int b = 0; b < actualBatchSize; b++)
                        {
                            sum += batchCodes[b, i] * batchData[b, j];
                        }
                        B[i, j] = (1 - weight) * B[i, j] + weight * sum;
                    }
                }

                _nSamplesSeen += actualBatchSize;

                // Update dictionary using block coordinate descent
                UpdateDictionaryFromAccumulators(A, B, k, p);
            }

            // Check convergence
            double maxChange = 0;
            for (int c = 0; c < k; c++)
            {
                for (int j = 0; j < p; j++)
                {
                    double change = Math.Abs(_components[c, j] - prevComponents[c * p + j]);
                    maxChange = Math.Max(maxChange, change);
                }
            }

            if (maxChange < _tol)
            {
                break;
            }
        }
    }

    private static void ShuffleArray(int[] array, Random random)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }

    private double[,] ComputeCodes(double[,] data, int n, int p, int k)
    {
        // Compute codes using ridge regression: codes = data * D^T * (D * D^T + ridge*I)^-1
        var gram = new double[k, k];
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < k; j++)
            {
                double sum = 0;
                for (int f = 0; f < p; f++)
                {
                    sum += _components![i, f] * _components[j, f];
                }
                gram[i, j] = sum;
                if (i == j)
                {
                    gram[i, j] += _ridge;
                }
            }
        }

        var gramInv = InvertMatrix(gram, k);

        var dataProj = new double[n, k];
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < k; c++)
            {
                double sum = 0;
                for (int j = 0; j < p; j++)
                {
                    sum += data[i, j] * _components![c, j];
                }
                dataProj[i, c] = sum;
            }
        }

        var codes = new double[n, k];
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < k; c++)
            {
                double sum = 0;
                for (int j = 0; j < k; j++)
                {
                    sum += dataProj[i, j] * gramInv[j, c];
                }
                codes[i, c] = sum;
            }
        }

        return codes;
    }

    private void UpdateDictionaryFromAccumulators(double[,] A, double[,] B, int k, int p)
    {
        // Update each component using coordinate descent with L1 regularization
        for (int c = 0; c < k; c++)
        {
            // Solve: A[c,c] * D[c,:] = B[c,:] - sum_{j!=c} A[c,j] * D[j,:]
            for (int j = 0; j < p; j++)
            {
                double rhs = B[c, j];
                for (int other = 0; other < k; other++)
                {
                    if (other != c)
                    {
                        rhs -= A[c, other] * _components![other, j];
                    }
                }

                double denominator = A[c, c];
                if (denominator < 1e-10)
                {
                    denominator = 1e-10;
                }

                // Soft thresholding
                double newValue = SoftThreshold(rhs / denominator, _alpha / denominator);
                _components![c, j] = newValue;
            }

            // Normalize component
            double norm = 0;
            for (int j = 0; j < p; j++)
            {
                norm += _components![c, j] * _components[c, j];
            }
            norm = Math.Sqrt(norm);

            if (norm > 1e-10)
            {
                for (int j = 0; j < p; j++)
                {
                    _components![c, j] /= norm;
                }
            }
        }
    }

    private static double SoftThreshold(double x, double lambda)
    {
        if (x > lambda)
        {
            return x - lambda;
        }
        else if (x < -lambda)
        {
            return x + lambda;
        }
        else
        {
            return 0;
        }
    }

    private static double[,] InvertMatrix(double[,] matrix, int n)
    {
        var result = new double[n, n];
        var temp = new double[n, 2 * n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp[i, j] = matrix[i, j];
                temp[i, j + n] = (i == j) ? 1.0 : 0.0;
            }
        }

        for (int i = 0; i < n; i++)
        {
            double maxVal = Math.Abs(temp[i, i]);
            int maxRow = i;
            for (int k = i + 1; k < n; k++)
            {
                if (Math.Abs(temp[k, i]) > maxVal)
                {
                    maxVal = Math.Abs(temp[k, i]);
                    maxRow = k;
                }
            }

            if (maxRow != i)
            {
                for (int j = 0; j < 2 * n; j++)
                {
                    (temp[i, j], temp[maxRow, j]) = (temp[maxRow, j], temp[i, j]);
                }
            }

            double pivot = temp[i, i];
            if (Math.Abs(pivot) < 1e-10)
            {
                pivot = 1e-10;
            }

            for (int j = 0; j < 2 * n; j++)
            {
                temp[i, j] /= pivot;
            }

            for (int k = 0; k < n; k++)
            {
                if (k != i)
                {
                    double factor = temp[k, i];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        temp[k, j] -= factor * temp[i, j];
                    }
                }
            }
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = temp[i, j + n];
            }
        }

        return result;
    }

    /// <summary>
    /// Transforms the data by projecting onto sparse components.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("MiniBatchSparsePCA has not been fitted.");
        }

        int n = data.Rows;
        int p = data.Columns;
        int k = _components.GetLength(0);
        var result = new T[n, k];

        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < k; c++)
            {
                double sum = 0;
                for (int j = 0; j < p; j++)
                {
                    double centered = NumOps.ToDouble(data[i, j]) - _mean[j];
                    sum += centered * _components[c, j];
                }
                result[i, c] = NumOps.FromDouble(sum);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Transforms data back to original space.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("MiniBatchSparsePCA has not been fitted.");
        }

        int n = data.Rows;
        int p = _nFeaturesIn;
        int k = _components.GetLength(0);
        var result = new T[n, p];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = _mean[j];
                for (int c = 0; c < k; c++)
                {
                    sum += NumOps.ToDouble(data[i, c]) * _components[c, j];
                }
                result[i, j] = NumOps.FromDouble(sum);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        int k = _components?.GetLength(0) ?? _nComponents;
        var names = new string[k];
        for (int i = 0; i < k; i++)
        {
            names[i] = $"MiniBatchSparsePC{i + 1}";
        }
        return names;
    }
}
