using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Mann-Whitney U test for feature selection in binary classification.
/// </summary>
/// <remarks>
/// <para>
/// The Mann-Whitney U test is a non-parametric test that compares the rank distributions
/// of two groups. For binary classification, it tests whether one class tends to have
/// systematically higher values than the other for each feature.
/// </para>
/// <para><b>For Beginners:</b> Mann-Whitney U is the non-parametric version of the t-test
/// for comparing two groups. It ranks all values and checks if one class has consistently
/// higher or lower ranks than the other. It's robust to outliers and doesn't assume
/// your data follows any particular distribution.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MannWhitneyU<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _uStatistics;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? UStatistics => _uStatistics;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MannWhitneyU(
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
            "MannWhitneyU requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var engine = Engine;

        // Separate samples by class (binary): class 0 is the rounded label 0, every other label is class 1
        var labels = engine.TensorRound(FilterStatisticTensorOps.AsTensor(target));
        var class0 = engine.TensorEquals(labels, NumOps.Zero);
        int n1 = engine.TensorCountNonzero(class0);
        int n2 = data.Rows - n1;

        if (n1 == 0 || n2 == 0)
            throw new ArgumentException("Both classes must have at least one sample.");

        // U statistic (the smaller one) and tie-corrected normal p-value (scipy.stats.mannwhitneyu, asymptotic,
        // continuity corrected)
        var (u, pValues) = FeatureSelectionTensorOps.MannWhitney(FilterStatisticTensorOps.AsTensor(data), class0, n1, n2);
        _uStatistics = u.ToVector();
        _pValues = pValues.ToVector();

        // Select features with smallest p-values (most significant)
        _selectedIndices = FilterStatisticTensorOps.BottomKIndicesSorted(pValues, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("MannWhitneyU has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MannWhitneyU does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MannWhitneyU has not been fitted.");

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
