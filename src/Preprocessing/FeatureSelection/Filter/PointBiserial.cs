using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Point-Biserial Correlation for feature selection with binary target.
/// </summary>
/// <remarks>
/// <para>
/// Point-biserial correlation measures the relationship between a continuous feature
/// and a binary (dichotomous) target. It's mathematically equivalent to Pearson
/// correlation when one variable is binary.
/// </para>
/// <para><b>For Beginners:</b> When your target is binary (like pass/fail, yes/no),
/// point-biserial correlation tells you how much a continuous feature differs between
/// the two groups. If the feature values are very different for class 0 vs class 1,
/// the correlation is high and the feature is useful for classification.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PointBiserial<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minCorrelation;

    private Vector<T>? _correlations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? Correlations => _correlations;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PointBiserial(
        int nFeaturesToSelect = 10,
        double minCorrelation = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minCorrelation = minCorrelation;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PointBiserial requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        var engine = Engine;

        // Separate samples by class (binary)
        var class0 = FilterStatisticTensorOps.BelowMask(target, 0.5, out int n0);
        var class1 = FilterStatisticTensorOps.OneMinus(class0);
        int n1 = n - n0;

        if (n0 == 0 || n1 == 0)
            throw new ArgumentException("Both classes must have at least one sample.");

        double pProp = (double)n1 / n;
        double qProp = (double)n0 / n;

        // Overall (population) standard deviation and class means of every feature
        var x = FilterStatisticTensorOps.AsTensor(data);
        var overallStd = engine.TensorSqrt(engine.TensorDivideScalar(
            FilterStatisticTensorOps.ColumnSumOfSquaredDeviations(x), NumOps.FromDouble(n)));
        var (mean0, _) = FilterStatisticTensorOps.MaskedColumnMoments(x, class0, n0);
        var (mean1, _) = FilterStatisticTensorOps.MaskedColumnMoments(x, class1, n1);

        // Point-biserial correlation ((mean1 - mean0) / std) sqrt(p q); 0 for a constant feature
        var correlations = engine.TensorMultiplyScalar(
            FilterStatisticTensorOps.GuardedDivide(engine.TensorSubtract(mean1, mean0), overallStd, 1e-10, NumOps.Zero),
            NumOps.FromDouble(Math.Sqrt(pProp * qProp)));
        _correlations = correlations.ToVector();

        // Select features above threshold or top by absolute correlation
        _selectedIndices = FilterStatisticTensorOps.CandidatesOrTopK(
            engine.TensorAbs(correlations), NumOps.FromDouble(_minCorrelation), _nFeaturesToSelect);

        IsFitted = true;
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
            throw new InvalidOperationException("PointBiserial has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PointBiserial does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PointBiserial has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
