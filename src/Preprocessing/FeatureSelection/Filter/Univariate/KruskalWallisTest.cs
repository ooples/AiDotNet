using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Kruskal-Wallis H test for non-parametric feature selection.
/// </summary>
/// <remarks>
/// <para>
/// A non-parametric alternative to one-way ANOVA that doesn't assume normal
/// distributions. Tests whether samples from different groups originate from
/// the same distribution based on ranks.
/// </para>
/// <para><b>For Beginners:</b> Unlike ANOVA which assumes data follows a normal
/// bell curve, Kruskal-Wallis works with any data distribution. It converts
/// values to ranks and tests whether the ranks differ between groups. This is
/// more robust when your data has outliers or non-normal distributions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KruskalWallisTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _pValueThreshold;

    private Vector<T>? _hStatistics;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? HStatistics => _hStatistics;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public KruskalWallisTest(
        int nFeaturesToSelect = 10,
        double pValueThreshold = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _pValueThreshold = pValueThreshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "KruskalWallisTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        int p = data.Columns;

        // Kruskal-Wallis H of every feature across the (integer) classes, tie-corrected, with chi-squared
        // p-values (scipy.stats.kruskal, which needs at least two groups: one class makes H = 0 and p = 1 for every
        // feature, a selection that carries no information)
        var classIndex = FilterStatisticTensorOps.ClassIndex(target, out int classCount);
        if (classCount < 2)
            throw new ArgumentException("KruskalWallisTest requires at least 2 classes.", nameof(target));

        _nInputFeatures = p;
        var (h, pValues) = FeatureSelectionTensorOps.KruskalWallis(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.ClassOneHot<T>(classIndex, classCount));
        _hStatistics = h.ToVector();
        _pValues = pValues.ToVector();

        // The significant features with the smallest p-values when there are enough, otherwise the largest H
        _selectedIndices = FilterStatisticTensorOps.SignificantOrTopK(
            pValues, NumOps.FromDouble(_pValueThreshold), h, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("KruskalWallisTest has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KruskalWallisTest does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KruskalWallisTest has not been fitted.");

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
