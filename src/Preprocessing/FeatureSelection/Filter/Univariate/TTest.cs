using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Student's t-test for binary classification feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Performs independent two-sample t-tests between feature values for each class.
/// Features with significant differences between class means are selected as they
/// help distinguish between classes.
/// </para>
/// <para><b>For Beginners:</b> The t-test asks: are the average values of this feature
/// really different between class 0 and class 1, or could the difference be due to chance?
/// A small p-value (high t-statistic) means the feature genuinely helps separate classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _pValueThreshold;

    private Vector<T>? _tStatistics;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? TStatistics => _tStatistics;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TTest(
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
            "TTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var class0 = FeatureSelectionTensorOps.BelowMask(target, 0.5, out int n0);
        var class1 = FeatureSelectionTensorOps.OneMinus(class0);
        int n1 = n - n0;

        if (n0 < 2 || n1 < 2)
            throw new ArgumentException("Both classes must have at least 2 samples for t-test.");

        int df = n0 + n1 - 2;

        // Means and sums of squared deviations of every feature within each class
        var x = FeatureSelectionTensorOps.AsTensor(data);
        var (mean0, ss0) = FeatureSelectionTensorOps.MaskedColumnMoments(x, class0, n0);
        var (mean1, ss1) = FeatureSelectionTensorOps.MaskedColumnMoments(x, class1, n1);

        // Pooled variance ((n0 - 1) var0 + (n1 - 1) var1) / df and its standard error
        var pooledStdErr = engine.TensorSqrt(engine.TensorMultiplyScalar(
            engine.TensorAdd(ss0, ss1), NumOps.FromDouble((1.0 / n0 + 1.0 / n1) / df)));

        // t-statistic; classes that do not overlap at all give +-inf
        var tStatistics = FeatureSelectionTensorOps.SignedSeparationAwareRatio(
            engine.TensorSubtract(mean0, mean1), pooledStdErr, 1e-10);
        _tStatistics = tStatistics.ToVector();

        // Two-tailed p-value from the t-distribution with df degrees of freedom (scipy.stats.ttest_ind)
        var pValues = FeatureSelectionTensorOps.StudentTTwoSided(tStatistics, df);
        _pValues = pValues.ToVector();

        // Select significant features or top by |t|
        _selectedIndices = FeatureSelectionTensorOps.SignificantOrTopK(
            pValues, NumOps.FromDouble(_pValueThreshold), engine.TensorAbs(tStatistics), _nFeaturesToSelect);

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
            throw new InvalidOperationException("TTest has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TTest does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TTest has not been fitted.");

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
