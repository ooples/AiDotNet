using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Incremental Principal Component Analysis for large datasets.
/// </summary>
/// <remarks>
/// <para>
/// IncrementalPCA processes data in batches, making it suitable for datasets
/// too large to fit in memory. It produces similar results to standard PCA
/// but with lower memory requirements.
/// </para>
/// <para>
/// The algorithm updates the covariance matrix incrementally as each batch
/// is processed, then computes principal components from the final estimate.
/// </para>
/// <para><b>For Beginners:</b> Regular PCA needs all data in memory at once.
/// Incremental PCA processes data in chunks:
/// - Feed data in batches (e.g., 1000 rows at a time)
/// - Updates its understanding of the data with each batch
/// - Produces similar principal components as regular PCA
/// - Uses much less memory for large datasets
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class IncrementalPCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _batchSize;
    private readonly bool _whiten;

    // Fitted parameters
    private double[]? _mean;
    private double[,]? _components;
    private double[]? _explainedVariance;
    private double[]? _explainedVarianceRatio;
    private double[]? _singularValues;
    private int _nSamplesSeen;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components to keep.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the batch size for incremental updates.
    /// </summary>
    public int BatchSize => _batchSize;

    /// <summary>
    /// Gets whether whitening is applied.
    /// </summary>
    public bool Whiten => _whiten;

    /// <summary>
    /// Gets the mean of each feature.
    /// </summary>
    public double[]? Mean => _mean;

    /// <summary>
    /// Gets the principal components.
    /// </summary>
    public double[,]? Components => _components;

    /// <summary>
    /// Gets the explained variance for each component.
    /// </summary>
    public double[]? ExplainedVariance => _explainedVariance;

    /// <summary>
    /// Gets the explained variance ratio for each component.
    /// </summary>
    public double[]? ExplainedVarianceRatio => _explainedVarianceRatio;

    /// <summary>
    /// Gets the number of samples seen during fitting.
    /// </summary>
    public int NSamplesSeen => _nSamplesSeen;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="IncrementalPCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of components to keep. Defaults to 2.</param>
    /// <param name="batchSize">Batch size for incremental updates. Defaults to 100.</param>
    /// <param name="whiten">If true, scale components to unit variance. Defaults to false.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public IncrementalPCA(
        int nComponents = 2,
        int batchSize = 100,
        bool whiten = false,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (batchSize < 1)
        {
            throw new ArgumentException("Batch size must be at least 1.", nameof(batchSize));
        }

        _nComponents = nComponents;
        _batchSize = batchSize;
        _whiten = whiten;
    }

    /// <summary>
    /// Fits IncrementalPCA by processing data in batches.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nComponents, Math.Min(n, p));

        // Initialize running statistics
        _mean = new double[p];
        _nSamplesSeen = 0;

        // Running sum of X and X^T * X for covariance estimation
        var runningSum = new double[p];
        var runningXTX = new double[p, p];

        // Process in batches
        for (int batchStart = 0; batchStart < n; batchStart += _batchSize)
        {
            int batchEnd = Math.Min(batchStart + _batchSize, n);
            int batchN = batchEnd - batchStart;

            // Extract batch
            var batchData = new double[batchN, p];
            for (int i = 0; i < batchN; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    batchData[i, j] = NumOps.ToDouble(data[batchStart + i, j]);
                }
            }

            // Update running sum
            for (int j = 0; j < p; j++)
            {
                for (int i = 0; i < batchN; i++)
                {
                    runningSum[j] += batchData[i, j];
                }
            }

            // Update running X^T * X
            for (int i = 0; i < p; i++)
            {
                for (int j = i; j < p; j++)
                {
                    double sum = 0;
                    for (int r = 0; r < batchN; r++)
                    {
                        sum += batchData[r, i] * batchData[r, j];
                    }
                    runningXTX[i, j] += sum;
                    if (i != j) runningXTX[j, i] = runningXTX[i, j];
                }
            }

            _nSamplesSeen += batchN;
        }

        // Compute mean
        for (int j = 0; j < p; j++)
        {
            _mean[j] = runningSum[j] / _nSamplesSeen;
        }

        // Compute covariance: (X^T * X) / (n-1) - n * mean * mean^T / (n-1)
        var covariance = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                covariance[i, j] = (runningXTX[i, j] - _nSamplesSeen * _mean[i] * _mean[j]) / (_nSamplesSeen - 1);
            }
        }

        // Compute eigenvalues and eigenvectors
        var (eigenvalues, eigenvectors) = ComputeEigen(covariance, p, k);

        // Sort by eigenvalue descending
        var indices = Enumerable.Range(0, k)
            .OrderByDescending(i => eigenvalues[i])
            .ToArray();

        _components = new double[k, p];
        _explainedVariance = new double[k];
        _singularValues = new double[k];

        double totalVariance = 0;
        for (int i = 0; i < p; i++)
        {
            totalVariance += covariance[i, i];
        }

        for (int i = 0; i < k; i++)
        {
            double eigenvalue = Math.Max(0, eigenvalues[indices[i]]);
            _explainedVariance[i] = eigenvalue;
            _singularValues[i] = Math.Sqrt(eigenvalue * (_nSamplesSeen - 1));

            for (int j = 0; j < p; j++)
            {
                _components[i, j] = eigenvectors[indices[i], j];
            }
        }

        // Compute explained variance ratio
        _explainedVarianceRatio = new double[k];
        if (totalVariance > 1e-10)
        {
            for (int i = 0; i < k; i++)
            {
                _explainedVarianceRatio[i] = _explainedVariance[i] / totalVariance;
            }
        }
    }

    /// <summary>
    /// Partially fits the model with a new batch of data.
    /// </summary>
    /// <param name="batch">A batch of training data.</param>
    public void PartialFit(Matrix<T> batch)
    {
        if (_mean is null)
        {
            // First call - initialize
            FitCore(batch);
            return;
        }

        // Incremental PCA update using SVD updating formula
        // Reference: Ross et al., "Incremental Learning for Robust Visual Tracking"

        int nBatch = batch.Rows;
        int p = batch.Columns;

        if (p != _nFeaturesIn)
        {
            throw new ArgumentException(
                $"Number of features ({p}) does not match fitted model ({_nFeaturesIn}).");
        }

        int k = _components!.GetLength(0);

        // Convert batch to double and compute batch statistics
        var batchData = new double[nBatch, p];
        var batchMean = new double[p];

        for (int i = 0; i < nBatch; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double val = NumOps.ToDouble(batch[i, j]);
                batchData[i, j] = val;
                batchMean[j] += val;
            }
        }

        for (int j = 0; j < p; j++)
        {
            batchMean[j] /= nBatch;
        }

        // Compute updated mean using weighted average
        int nOld = _nSamplesSeen;
        int nTotal = nOld + nBatch;
        var newMean = new double[p];

        for (int j = 0; j < p; j++)
        {
            newMean[j] = (nOld * _mean[j] + nBatch * batchMean[j]) / nTotal;
        }

        // Center the batch data with NEW mean
        for (int i = 0; i < nBatch; i++)
        {
            for (int j = 0; j < p; j++)
            {
                batchData[i, j] -= newMean[j];
            }
        }

        // Build the matrix for SVD update:
        // [sqrt(n_old/(n_total)) * S * V^T + sqrt(n_old*n_batch/n_total) * (old_mean - new_mean)]
        // [sqrt(1/n_total) * centered_batch]
        // where S * V^T represents the old principal components scaled by singular values

        // Compute mean correction vector scaled appropriately
        double meanCorrectionScale = Math.Sqrt((double)nOld * nBatch / nTotal);
        var meanCorrection = new double[p];
        for (int j = 0; j < p; j++)
        {
            meanCorrection[j] = meanCorrectionScale * (_mean[j] - newMean[j]);
        }

        // Build combined matrix for SVD
        // Row 0 to k-1: scaled old components (S_i * V_i^T)
        // Row k: mean correction
        // Rows k+1 to k+nBatch: scaled new batch
        int numCombinedRows = k + 1 + nBatch;
        var combinedMatrix = new double[numCombinedRows, p];

        // Add scaled old components
        double oldScale = Math.Sqrt((double)(nOld - 1) / (nTotal - 1));
        for (int i = 0; i < k; i++)
        {
            double singularValue = _singularValues != null ? _singularValues[i] : Math.Sqrt(_explainedVariance![i] * (nOld - 1));
            for (int j = 0; j < p; j++)
            {
                combinedMatrix[i, j] = oldScale * singularValue * _components[i, j];
            }
        }

        // Add mean correction row
        for (int j = 0; j < p; j++)
        {
            combinedMatrix[k, j] = meanCorrection[j];
        }

        // Add scaled new batch
        double batchScale = 1.0 / Math.Sqrt(nTotal - 1);
        for (int i = 0; i < nBatch; i++)
        {
            for (int j = 0; j < p; j++)
            {
                combinedMatrix[k + 1 + i, j] = batchScale * batchData[i, j];
            }
        }

        // Compute SVD of combined matrix (only need top k components)
        var (newSingularValues, newComponents) = ComputeTruncatedSVD(combinedMatrix, numCombinedRows, p, k);

        // Update state
        _mean = newMean;
        _nSamplesSeen = nTotal;
        _components = newComponents;
        _singularValues = newSingularValues;

        // Update explained variance
        _explainedVariance = new double[k];
        double totalVariance = 0;
        for (int i = 0; i < k; i++)
        {
            _explainedVariance[i] = newSingularValues[i] * newSingularValues[i] / (nTotal - 1);
            totalVariance += _explainedVariance[i];
        }

        // Compute explained variance ratio (approximation based on captured variance)
        _explainedVarianceRatio = new double[k];
        if (totalVariance > 1e-10)
        {
            for (int i = 0; i < k; i++)
            {
                _explainedVarianceRatio[i] = _explainedVariance[i] / totalVariance;
            }
        }
    }

    private (double[] SingularValues, double[,] RightVectors) ComputeTruncatedSVD(double[,] matrix, int m, int n, int k)
    {
        // Compute truncated SVD to get top k right singular vectors
        // Uses power iteration on A^T * A

        int numComponents = Math.Min(k, Math.Min(m, n));
        var singularValues = new double[numComponents];
        var rightVectors = new double[numComponents, n];

        // Compute A^T * A
        var AtA = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int r = 0; r < m; r++)
                {
                    sum += matrix[r, i] * matrix[r, j];
                }
                AtA[i, j] = sum;
            }
        }

        var A = (double[,])AtA.Clone();
        var random = new Random(42);

        for (int c = 0; c < numComponents; c++)
        {
            // Random initialization
            var v = new double[n];
            double norm = 0;
            for (int i = 0; i < n; i++)
            {
                v[i] = random.NextDouble() - 0.5;
                norm += v[i] * v[i];
            }
            norm = Math.Sqrt(norm);
            for (int i = 0; i < n; i++)
            {
                v[i] /= norm;
            }

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

                norm = 0;
                for (int i = 0; i < n; i++)
                {
                    norm += Av[i] * Av[i];
                }
                norm = Math.Sqrt(norm);

                if (norm < 1e-10) break;

                for (int i = 0; i < n; i++)
                {
                    v[i] = Av[i] / norm;
                }
            }

            // Compute eigenvalue (= singular value squared)
            var Av2 = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Av2[i] += A[i, j] * v[j];
                }
            }

            double eigenvalue = 0;
            for (int i = 0; i < n; i++)
            {
                eigenvalue += v[i] * Av2[i];
            }

            singularValues[c] = Math.Sqrt(Math.Max(0, eigenvalue));

            for (int i = 0; i < n; i++)
            {
                rightVectors[c, i] = v[i];
            }

            // Deflate
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    A[i, j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        return (singularValues, rightVectors);
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeEigen(double[,] matrix, int n, int k)
    {
        // Power iteration with deflation
        var eigenvalues = new double[k];
        var eigenvectors = new double[k, n];
        var A = (double[,])matrix.Clone();
        var random = new Random(42);

        for (int m = 0; m < k; m++)
        {
            var v = new double[n];
            double norm = 0;
            for (int i = 0; i < n; i++)
            {
                v[i] = random.NextDouble() - 0.5;
                norm += v[i] * v[i];
            }
            norm = Math.Sqrt(norm);
            for (int i = 0; i < n; i++)
            {
                v[i] /= norm;
            }

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

                norm = 0;
                for (int i = 0; i < n; i++)
                {
                    norm += Av[i] * Av[i];
                }
                norm = Math.Sqrt(norm);

                if (norm < 1e-10) break;

                for (int i = 0; i < n; i++)
                {
                    v[i] = Av[i] / norm;
                }
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

            double eigenvalue = 0;
            for (int i = 0; i < n; i++)
            {
                eigenvalue += v[i] * Av2[i];
            }

            eigenvalues[m] = Math.Max(0, eigenvalue);

            for (int i = 0; i < n; i++)
            {
                eigenvectors[m, i] = v[i];
            }

            // Deflate
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    A[i, j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        return (eigenvalues, eigenvectors);
    }

    /// <summary>
    /// Transforms the data by projecting onto principal components.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("IncrementalPCA has not been fitted.");
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

                if (_whiten && _singularValues is not null && _singularValues[c] > 1e-10)
                {
                    sum /= _singularValues[c];
                }

                result[i, c] = NumOps.FromDouble(sum);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Transforms data back to original space.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>Data in original feature space.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("IncrementalPCA has not been fitted.");
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
                    double val = NumOps.ToDouble(data[i, c]);
                    if (_whiten && _singularValues is not null && _singularValues[c] > 1e-10)
                    {
                        val *= _singularValues[c];
                    }
                    sum += val * _components[c, j];
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
            names[i] = $"PC{i + 1}";
        }
        return names;
    }
}
