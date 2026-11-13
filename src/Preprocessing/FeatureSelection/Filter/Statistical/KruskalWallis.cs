using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// Kruskal-Wallis H test for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Kruskal-Wallis H test is a non-parametric alternative to one-way ANOVA.
/// It tests whether samples from different groups have the same distribution,
/// making it useful for multi-class classification problems.
/// </para>
/// <para><b>For Beginners:</b> When you have more than two groups (like multiple
/// categories), this test checks if the feature values differ across groups.
/// Unlike ANOVA, it doesn't assume the data follows a bell curve, making it
/// more robust for real-world data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KruskalWallis<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

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

    public KruskalWallis(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
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
            "KruskalWallis requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
            throw new ArgumentException("KruskalWallis requires at least 2 classes.", nameof(target));

        _nInputFeatures = p;
        var (h, pValues) = FeatureSelectionTensorOps.KruskalWallis(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.ClassOneHot<T>(classIndex, classCount));
        _hStatistics = h.ToVector();
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
            throw new InvalidOperationException("KruskalWallis has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KruskalWallis does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KruskalWallis has not been fitted.");

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
