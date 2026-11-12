using AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.HighDimensional;

/// <summary>
/// Benjamini-Hochberg procedure for feature selection with FDR control.
/// </summary>
/// <remarks>
/// <para>
/// The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR) when
/// testing multiple hypotheses. It sorts p-values and finds the largest k such that
/// p_(k) &lt;= k/m * alpha, where m is the number of tests.
/// </para>
/// <para><b>For Beginners:</b> When you test many features, some will look significant
/// just by chance (false positives). This method controls how many of your "discoveries"
/// are actually false. If you set FDR to 0.1, you expect at most 10% of selected
/// features to be false positives.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BenjaminiHochberg<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;

    private Vector<T>? _pValues;
    private Vector<T>? _adjustedPValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public Vector<T>? PValues => _pValues;
    public Vector<T>? AdjustedPValues => _adjustedPValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BenjaminiHochberg(
        int nFeaturesToSelect = 10,
        double alpha = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha <= 0 || alpha > 1)
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BenjaminiHochberg requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // Step 1: p-values of the correlation t-test, t = r sqrt((n - 2) / (1 - r^2)) on n - 2 degrees of freedom
        // (scikit-learn f_regression, F = t^2); a constant feature has p = 1
        var (_, pValueVector) = StatisticalTestHelper.ComputeScoresWithPValues(data, target, SelectKBestScoreFunc.FRegression);
        var pValues = FilterStatisticTensorOps.AsTensor(pValueVector);
        _pValues = pValueVector;

        // Step 2: Benjamini-Hochberg adjusted p-values
        var adjusted = StatisticalTestHelper.BenjaminiHochbergCorrection(pValueVector);
        _adjustedPValues = adjusted;

        // Step 3: features with adjusted p-value <= alpha, by increasing p-value (ties by index)
        var order = FilterStatisticTensorOps.StableOrder(pValues, descending: false);
        var rejected = Engine.TensorGreaterThan(FilterStatisticTensorOps.AsTensor(adjusted), NumOps.FromDouble(_alpha))
            .Cast<int>().ToArray();
        var selected = order.Where(j => rejected[j] == 0).ToList();

        // If not enough selected, take top features by p-value
        _selectedIndices = selected.Count == 0
            ? order.Take(Math.Min(_nFeaturesToSelect, p)).OrderBy(x => x).ToArray()
            : selected.Take(_nFeaturesToSelect).OrderBy(x => x).ToArray();
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
            throw new InvalidOperationException("BenjaminiHochberg has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BenjaminiHochberg does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BenjaminiHochberg has not been fitted.");

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
