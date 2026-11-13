using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class IncrementalPCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _batchSize;
    private readonly bool _whiten;

    // Fitted parameters
    private Vector<T>? _mean;
    private Matrix<T>? _components;
    private Vector<T>? _explainedVariance;
    private Vector<T>? _explainedVarianceRatio;
    private Vector<T>? _singularValues;
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
    public Vector<T>? Mean => _mean;

    /// <summary>
    /// Gets the principal components.
    /// </summary>
    public Matrix<T>? Components => _components;

    /// <summary>
    /// Gets the explained variance for each component.
    /// </summary>
    public Vector<T>? ExplainedVariance => _explainedVariance;

    /// <summary>
    /// Gets the explained variance ratio for each component.
    /// </summary>
    public Vector<T>? ExplainedVarianceRatio => _explainedVarianceRatio;

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
        var engine = Engine;
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < 2)
        {
            throw new ArgumentException("At least 2 samples are required to fit IncrementalPCA.", nameof(data));
        }

        int k = Math.Min(_nComponents, Math.Min(n, p));

        // Mean and sample covariance over every sample (the batched accumulation of sum and X^T X in one pass)
        var x = Tensor<T>.FromMatrix(data);
        var mean = DimensionalityReductionOps.ColumnMeans(x);
        var centered = DimensionalityReductionOps.SubtractRow(x, mean);
        var covariance = engine.TensorDivideScalar(
            DimensionalityReductionOps.TransposeMatMul(centered, centered), NumOps.FromDouble(n - 1));
        _nSamplesSeen = n;

        // Leading eigenpairs of the covariance
        var (eigenvalues, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(covariance, descending: true);
        var explainedVariance = engine.TensorClampMin(engine.TensorNarrow(eigenvalues, 0, 0, k), NumOps.Zero);

        T totalVariance = engine.TensorSum(engine.TensorDiagonal(covariance));

        _mean = mean.ToVector();
        _components = engine.TensorTranspose(DimensionalityReductionOps.FirstColumns(eigenvectors, k)).ToMatrix();
        _explainedVariance = explainedVariance.ToVector();
        _singularValues = engine.TensorSqrt(
            engine.TensorMultiplyScalar(explainedVariance, NumOps.FromDouble(_nSamplesSeen - 1))).ToVector();
        _explainedVarianceRatio = VarianceRatio(explainedVariance, totalVariance).ToVector();
    }

    /// <summary>
    /// Partially fits the model with a new batch of data.
    /// </summary>
    /// <param name="batch">A batch of training data.</param>
    public void PartialFit(Matrix<T> batch)
    {
        if (batch.Rows < 1)
        {
            throw new ArgumentException("Batch must contain at least 1 sample.", nameof(batch));
        }

        if (_mean is null)
        {
            // First call - initialize
            FitCore(batch);
            return;
        }

        // Incremental PCA update using SVD updating formula
        // Reference: Ross et al., "Incremental Learning for Robust Visual Tracking"

        var engine = Engine;
        int nBatch = batch.Rows;
        int p = batch.Columns;

        if (p != _nFeaturesIn)
        {
            throw new ArgumentException(
                $"Number of features ({p}) does not match fitted model ({_nFeaturesIn}).");
        }

        if (_components is null)
        {
            throw new InvalidOperationException(
                "Internal state is inconsistent: _components is null after initialization.");
        }

        var oldSingularValues = _singularValues is not null
            ? Tensor<T>.FromVector(_singularValues)
            : _explainedVariance is not null
                ? engine.TensorSqrt(engine.TensorMultiplyScalar(
                    Tensor<T>.FromVector(_explainedVariance), NumOps.FromDouble(_nSamplesSeen - 1)))
                : throw new InvalidOperationException(
                    "Internal state is inconsistent: neither _singularValues nor _explainedVariance is available after initialization.");

        int k = _components.Rows;
        int nOld = _nSamplesSeen;
        int nTotal = nOld + nBatch;

        // Updated mean: (nOld * oldMean + nBatch * batchMean) / nTotal
        var x = Tensor<T>.FromMatrix(batch);
        var oldMean = Tensor<T>.FromVector(_mean);
        var batchMean = DimensionalityReductionOps.ColumnMeans(x);
        var newMean = engine.TensorDivideScalar(
            engine.TensorAdd(
                engine.TensorMultiplyScalar(oldMean, NumOps.FromDouble(nOld)),
                engine.TensorMultiplyScalar(batchMean, NumOps.FromDouble(nBatch))),
            NumOps.FromDouble(nTotal));

        // Center the batch on the new mean
        var centeredBatch = DimensionalityReductionOps.SubtractRow(x, newMean);

        // Mean-correction row: sqrt(nOld * nBatch / nTotal) * (oldMean - newMean)
        var meanCorrection = engine.TensorMultiplyScalar(
            engine.TensorSubtract(oldMean, newMean), NumOps.FromDouble(Math.Sqrt((double)nOld * nBatch / nTotal)));

        // Stack [sqrt((nOld - 1) / (nTotal - 1)) * S * V^T; mean correction; X_centered / sqrt(nTotal - 1)]
        var oldRows = engine.TensorMultiplyScalar(
            DimensionalityReductionOps.ScaleRows(Tensor<T>.FromMatrix(_components), oldSingularValues),
            NumOps.FromDouble(Math.Sqrt((double)(nOld - 1) / (nTotal - 1))));
        var batchRows = engine.TensorMultiplyScalar(centeredBatch, NumOps.FromDouble(1.0 / Math.Sqrt(nTotal - 1)));
        var combined = engine.TensorConcatenate(
            new[] { oldRows, DimensionalityReductionOps.AsRow(meanCorrection), batchRows }, 0);

        // Truncated SVD of the stacked matrix
        var (newSingularValues, newComponents) = ComputeTruncatedSVD(combined, k);

        _mean = newMean.ToVector();
        _nSamplesSeen = nTotal;
        _components = newComponents.ToMatrix();
        _singularValues = newSingularValues.ToVector();

        // Explained variance: s^2 / (nTotal - 1), ratio over the kept components
        var explainedVariance = engine.TensorDivideScalar(
            engine.TensorSquare(newSingularValues), NumOps.FromDouble(nTotal - 1));
        _explainedVariance = explainedVariance.ToVector();
        _explainedVarianceRatio = VarianceRatio(explainedVariance, engine.TensorSum(explainedVariance)).ToVector();
    }

    /// <summary>
    /// Leading singular values and right singular vectors (as rows) of a matrix, from the Jacobi eigen solve
    /// of its Gram matrix A^T A.
    /// </summary>
    private (Tensor<T> SingularValues, Tensor<T> RightVectors) ComputeTruncatedSVD(Tensor<T> matrix, int k)
    {
        var engine = Engine;
        int numComponents = Math.Min(k, Math.Min(matrix.Shape[0], matrix.Shape[1]));
        var gram = DimensionalityReductionOps.TransposeMatMul(matrix, matrix);
        var (eigenvalues, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(gram, descending: true);
        var singularValues = engine.TensorSqrt(engine.TensorClampMin(engine.TensorNarrow(eigenvalues, 0, 0, numComponents), NumOps.Zero));
        var rightVectors = engine.TensorTranspose(DimensionalityReductionOps.FirstColumns(eigenvectors, numComponents));
        return (singularValues, rightVectors);
    }

    /// <summary>Explained variance divided by the total, or zeros when the total is (near) zero.</summary>
    private Tensor<T> VarianceRatio(Tensor<T> explainedVariance, T totalVariance)
    {
        return NumOps.GreaterThan(totalVariance, DimensionalityReductionOps.Tolerance<T>(1e-10))
            ? Engine.TensorDivideScalar(explainedVariance, totalVariance)
            : Engine.TensorZerosLike(explainedVariance);
    }

    /// <summary>
    /// Per-component whitening scale sqrt(explained variance) (unit-variance outputs, as scikit-learn's
    /// IncrementalPCA(whiten=True)); 1 for components with (near) zero variance.
    /// </summary>
    private Tensor<T> WhitenScale(Vector<T> explainedVariance)
    {
        var engine = Engine;
        var std = engine.TensorSqrt(Tensor<T>.FromVector(explainedVariance));
        return engine.TensorWhere(engine.TensorGreaterThan(std, DimensionalityReductionOps.Tolerance<T>(1e-10)), std, engine.TensorOnesLike(std));
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

        var centered = DimensionalityReductionOps.SubtractRow(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(_mean));
        var projected = DimensionalityReductionOps.MatMulTranspose(centered, Tensor<T>.FromMatrix(_components));

        if (_whiten && _explainedVariance is not null)
        {
            projected = Engine.TensorDivide(projected, DimensionalityReductionOps.AsRow(WhitenScale(_explainedVariance)));
        }

        return projected.ToMatrix();
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

        var scores = DimensionalityReductionOps.FirstColumns(Tensor<T>.FromMatrix(data), _components.Rows);

        if (_whiten && _explainedVariance is not null)
        {
            scores = DimensionalityReductionOps.ScaleColumns(scores, WhitenScale(_explainedVariance));
        }

        var reconstructed = Engine.TensorMatMul(scores, Tensor<T>.FromMatrix(_components));
        return DimensionalityReductionOps.AddRow(reconstructed, Tensor<T>.FromVector(_mean)).ToMatrix();
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        int k = _components?.Rows ?? _nComponents;
        var names = new string[k];
        for (int i = 0; i < k; i++)
        {
            names[i] = $"PC{i + 1}";
        }
        return names;
    }
}
