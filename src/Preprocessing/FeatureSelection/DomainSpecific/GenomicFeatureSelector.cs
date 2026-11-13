using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.DomainSpecific;

/// <summary>
/// Feature selection for genomic/bioinformatics data.
/// </summary>
/// <remarks>
/// <para>
/// GenomicFeatureSelector is designed for high-dimensional genomic data such as
/// gene expression profiles, SNPs, or methylation data. It addresses the "large p,
/// small n" problem common in genomics and can incorporate gene grouping.
/// </para>
/// <para><b>For Beginners:</b> Genomic data often has millions of features (genes)
/// but only a few hundred samples (patients). This selector uses techniques
/// specifically designed for this challenging scenario, finding the few genes
/// that truly matter for distinguishing between conditions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GenomicFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _variancePercentile;
    private readonly bool _useFoldChangeFilter;
    private readonly double _foldChangeThreshold;

    private Vector<T>? _featureScores;
    private Vector<T>? _foldChanges;
    private Vector<T>? _variances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double VariancePercentile => _variancePercentile;
    public bool UseFoldChangeFilter => _useFoldChangeFilter;
    public double FoldChangeThreshold => _foldChangeThreshold;
    public Vector<T>? FeatureScores => _featureScores;
    public Vector<T>? FoldChanges => _foldChanges;
    public Vector<T>? Variances => _variances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GenomicFeatureSelector(
        int nFeaturesToSelect = 100,
        double variancePercentile = 0.5,
        bool useFoldChangeFilter = true,
        double foldChangeThreshold = 1.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (variancePercentile < 0 || variancePercentile > 1)
            throw new ArgumentException("Variance percentile must be between 0 and 1.", nameof(variancePercentile));
        if (foldChangeThreshold < 0)
            throw new ArgumentException("Fold change threshold must be non-negative.", nameof(foldChangeThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _variancePercentile = variancePercentile;
        _useFoldChangeFilter = useFoldChangeFilter;
        _foldChangeThreshold = foldChangeThreshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GenomicFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");
        if (data.Rows == 0 || data.Columns == 0)
            throw new ArgumentException("Data must contain at least one row and one column.", nameof(data));

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        T epsilon = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var zeros = new Tensor<T>(new[] { p });

        // Split samples by class
        var class1Mask = PreprocessingTensorOps.Not(engine.TensorLessThan(Tensor<T>.FromVector(target), NumOps.FromDouble(0.5)));
        var class0Mask = PreprocessingTensorOps.Not(class1Mask);
        int class1Count = PreprocessingTensorOps.CountNonZero(class1Mask);
        int class0Count = n - class1Count;

        // Step 1: Compute variances and filter by percentile
        var variances = engine.TensorDivideScalar(FeatureSelectionTensorOps.ColumnSumSquaredDeviations(X), NumOps.FromDouble(n));
        _variances = variances.ToVector();

        int varianceIndex = (int)(p * _variancePercentile);
        if (varianceIndex >= p) varianceIndex = p - 1;
        if (varianceIndex < 0) varianceIndex = 0;
        var varianceThreshold = PreprocessingTensorOps.ColumnOrderStatistic(engine.Reshape(variances, new[] { p, 1 }), varianceIndex);
        var candidateMask = engine.TensorGreaterEqual(variances, engine.TensorBroadcastTo(varianceThreshold, new[] { p }));

        // Class means of every feature
        var mean0 = ClassMeans(X, class0Mask, class0Count, n, p);
        var mean1 = ClassMeans(X, class1Mask, class1Count, n, p);

        // Step 2: Compute fold changes if binary classification
        var foldChanges = zeros;
        if (class0Count > 0 && class1Count > 0 && _useFoldChangeFilter)
        {
            // Fold change (use log2 fold change); a high value of 10 when only one class mean is positive
            var positive0 = engine.TensorGreaterThan(mean0, epsilon);
            var positive1 = engine.TensorGreaterThan(mean1, epsilon);
            var logFold = engine.TensorAbs(engine.TensorDivideScalar(
                engine.TensorLog(engine.TensorDivide(mean1, mean0)), NumOps.FromDouble(Math.Log(2))));
            var onePositive = engine.TensorWhere(engine.TensorAdd(positive0, positive1),
                PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(10)), zeros);
            foldChanges = engine.TensorWhere(engine.TensorMultiply(positive0, positive1), logFold, onePositive);

            // Filter by fold change
            candidateMask = engine.TensorMultiply(candidateMask, engine.TensorGreaterEqual(foldChanges,
                PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(Math.Log(_foldChangeThreshold) / Math.Log(2)))));
        }

        _foldChanges = foldChanges.ToVector();

        // Step 3: Compute t-statistic (Welch) for remaining candidates
        Tensor<T> scores;
        if (class0Count >= 2 && class1Count >= 2)
        {
            var var0 = ClassSampleVariances(X, class0Mask, mean0, class0Count, n, p);
            var var1 = ClassSampleVariances(X, class1Mask, mean1, class1Count, n, p);
            var se = engine.TensorSqrt(engine.TensorAdd(
                engine.TensorDivideScalar(var0, NumOps.FromDouble(class0Count)),
                engine.TensorDivideScalar(var1, NumOps.FromDouble(class1Count))));
            scores = engine.TensorWhere(engine.TensorGreaterThan(se, epsilon),
                engine.TensorDivide(engine.TensorAbs(engine.TensorSubtract(mean0, mean1)), se), zeros);
        }
        else
        {
            scores = variances;
        }

        scores = engine.TensorWhere(candidateMask, scores, zeros);
        _featureScores = scores.ToVector();
        var candidates = PreprocessingTensorOps.MaskPositions(candidateMask);
        var candidateSet = new HashSet<int>(candidates);
        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, candidates.Length);
        _selectedIndices = PreprocessingTensorOps.StableOrder(scores, descending: true)
            .Where(j => candidateSet.Contains(j))
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        // If not enough candidates, add high-variance features
        if (_selectedIndices.Length < _nFeaturesToSelect)
        {
            var remaining = PreprocessingTensorOps.StableOrder(variances, descending: true)
                .Where(j => !candidateSet.Contains(j))
                .Take(_nFeaturesToSelect - _selectedIndices.Length);

            _selectedIndices = _selectedIndices.Concat(remaining).OrderBy(x => x).ToArray();
        }

        IsFitted = true;
    }

    private Tensor<T> ClassMeans(Tensor<T> X, Tensor<T> rowMask, int count, int n, int p)
    {
        if (count == 0)
            return new Tensor<T>(new[] { p });

        // Summed in double by GroupMeans: a Half class sum stalls past 2048 times the values' spacing and
        // overflows at 65504
        var engine = Engine;
        return engine.Reshape(FeatureSelectionTensorOps.GroupMeans(engine.Reshape(rowMask, new[] { n, 1 }), X, out _), new[] { p });
    }

    private Tensor<T> ClassSampleVariances(Tensor<T> X, Tensor<T> rowMask, Tensor<T> mean, int count, int n, int p)
    {
        var engine = Engine;
        var shape = new[] { n, p };
        // Other-class rows are selected away (not multiplied by 0, which turns an overflowed square into NaN) and
        // the squares are averaged in double by GroupMeans, then rescaled from 1/count to 1/(count - 1)
        var membership = engine.Reshape(rowMask, new[] { n, 1 });
        var deviations = engine.TensorWhere(engine.TensorBroadcastTo(membership, shape),
            engine.TensorSubtract(X, engine.TensorBroadcastTo(engine.Reshape(mean, new[] { 1, p }), shape)),
            new Tensor<T>(shape));
        var meanSquares = engine.Reshape(
            FeatureSelectionTensorOps.GroupMeans(membership, engine.TensorSquare(deviations), out _), new[] { p });
        return engine.TensorMultiplyScalar(meanSquares, NumOps.FromDouble(count / (count - 1.0)));
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GenomicFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GenomicFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GenomicFeatureSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Gene{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
