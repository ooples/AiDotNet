using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Wilcoxon Signed-Rank test for paired non-parametric feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Wilcoxon Signed-Rank test compares paired samples without assuming normality.
/// Useful when comparing features across matched/paired observations.
/// </para>
/// <para><b>For Beginners:</b> This test works with paired data (before/after, matched pairs).
/// It ranks the absolute differences and compares positive vs negative ranks.
/// Good for non-normal paired comparisons.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class WilcoxonSignedRank<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _wStatistics;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? WStatistics => _wStatistics;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public WilcoxonSignedRank(int nFeaturesToSelect = 10, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
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

        // fewer than 1 non-zero differences: no evidence (W = 0, p = 1)
        var enough = engine.TensorGreaterThan(nonZero, NumOps.FromDouble(1 - 0.5));
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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
