using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// Wilcoxon Signed-Rank test for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Wilcoxon Signed-Rank test is a non-parametric test for paired samples.
/// It tests whether the median difference between pairs of observations is zero,
/// useful for before/after comparisons or matched samples.
/// </para>
/// <para><b>For Beginners:</b> When you have paired data (like measurements from
/// the same subjects at different times), this test checks if there's a consistent
/// difference. It doesn't assume normal distributions, making it more robust than
/// paired t-tests for real-world data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class WilcoxonSignedRank<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;

    private Vector<T>? _wStatistics;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public Vector<T>? WStatistics => _wStatistics;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public WilcoxonSignedRank(
        int nFeaturesToSelect = 10,
        double alpha = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "WilcoxonSignedRank requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var engine = Engine;

        // Signed-rank test of each feature against the target: differences within 1e-10 of 0 are dropped, the rest
        // ranked by magnitude (ties averaged), W = min(W+, W-), tie-corrected normal p-value (scipy.stats.wilcoxon)
        var (w, pValues, nonZero) = FeatureScoringTensorOps.WilcoxonSignedRank(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.AsTensor(target), 1e-10);

        // fewer than 2 non-zero differences: no evidence (W = 0, p = 1)
        var enough = engine.TensorGreaterThan(nonZero, NumOps.FromDouble(2 - 0.5));
        w = FeatureScoringTensorOps.KeepOrFill(w, enough, NumOps.Zero);
        pValues = FeatureScoringTensorOps.KeepOrFill(pValues, enough, NumOps.One);
        _wStatistics = w.ToVector();
        _pValues = pValues.ToVector();

        // Select features with smallest p-values
        _selectedIndices = PreprocessingTensorOps.BottomKIndicesSorted(pValues, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("WilcoxonSignedRank has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("WilcoxonSignedRank does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WilcoxonSignedRank has not been fitted.");

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
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
