using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Spearman rank correlation for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Measures the monotonic relationship between features and target using ranks
/// instead of raw values. Robust to outliers and non-linear monotonic relationships.
/// </para>
/// <para><b>For Beginners:</b> While Pearson correlation only detects linear relationships,
/// Spearman detects any monotonic relationship (consistently increasing or decreasing).
/// It works by ranking values, so outliers have less impact. Perfect for ordinal data
/// or when relationships are curved but consistent.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SpearmanCorrelation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
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

    public SpearmanCorrelation(
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
            "SpearmanCorrelation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        // ranked and correlated in double, as master did: ranks pass 2048 and the rank sums of squares
        // (about n^3 / 12) pass 65504 in Half once n > 91
        var x = FeatureScoringTensorOps.ColumnAverageRanks(FilterStatisticTensorOps.AsTensor(data).Cast<double>());
        var y = FeatureScoringTensorOps.VectorAverageRanks(FilterStatisticTensorOps.AsTensor(target).Cast<double>());
        var (sxx, sxy, syy) = FilterStatisticTensorOps.CenteredCrossSums(x, y);
        var correlations = FilterStatisticTensorOps.GuardedDivide(
            sxy, Engine.TensorMultiplyScalar(Engine.TensorSqrt(sxx), Math.Sqrt(syy)), 1e-10, 0.0).Cast<T>();
        _correlations = correlations.ToVector();
        _selectedIndices = FilterStatisticTensorOps.CandidatesOrTopK(
            Engine.TensorAbs(correlations), NumOps.FromDouble(_minCorrelation), Math.Min(_nFeaturesToSelect, p));
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
            throw new InvalidOperationException("SpearmanCorrelation has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SpearmanCorrelation does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpearmanCorrelation has not been fitted.");

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
