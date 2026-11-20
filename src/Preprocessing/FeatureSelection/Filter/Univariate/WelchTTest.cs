using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Welch's t-test for binary classification with unequal variances.
/// </summary>
/// <remarks>
/// <para>
/// Welch's t-test is an adaptation of Student's t-test that doesn't assume equal
/// variances between groups. It uses the Welch-Satterthwaite approximation for
/// degrees of freedom.
/// </para>
/// <para><b>For Beginners:</b> Regular t-test assumes both classes have similar
/// spread (variance). If one class has much more variation than the other, Welch's
/// t-test gives more accurate results. It's generally safer to use when you're
/// unsure about variance equality.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class WelchTTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _pValueThreshold;

    private Vector<T>? _tStatistics;
    private Vector<T>? _pValues;
    private Vector<T>? _degreesOfFreedom;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? TStatistics => _tStatistics;
    public Vector<T>? PValues => _pValues;
    public Vector<T>? DegreesOfFreedom => _degreesOfFreedom;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public WelchTTest(
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
            "WelchTTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var class1 = PreprocessingTensorOps.OneMinus(class0);
        int n1 = n - n0;

        if (n0 < 2 || n1 < 2)
            throw new ArgumentException("Both classes must have at least 2 samples for Welch's t-test.");

        // Means and sums of squared deviations of every feature within each class
        var x = FilterStatisticTensorOps.AsTensor(data);
        var (mean0, ss0) = FilterStatisticTensorOps.MaskedColumnMoments(x, class0, n0);
        var (mean1, ss1) = FilterStatisticTensorOps.MaskedColumnMoments(x, class1, n1);

        // Welch's t-statistic: s2_c = var_c / n_c = ss_c / ((n_c - 1) n_c); classes that do not overlap give +-inf
        var s20 = engine.TensorDivideScalar(ss0, NumOps.FromDouble((n0 - 1.0) * n0));
        var s21 = engine.TensorDivideScalar(ss1, NumOps.FromDouble((n1 - 1.0) * n1));
        var sum = engine.TensorAdd(s20, s21);
        var tStatistics = FilterStatisticTensorOps.SignedSeparationAwareRatio(
            engine.TensorSubtract(mean0, mean1), engine.TensorSqrt(sum), 1e-10);
        _tStatistics = tStatistics.ToVector();

        // Welch-Satterthwaite degrees of freedom (s2_0 + s2_1)^2 / (s2_0^2 / (n0 - 1) + s2_1^2 / (n1 - 1)),
        // 1 when both classes are constant
        var denominator = engine.TensorAdd(
            engine.TensorDivideScalar(engine.TensorSquare(s20), NumOps.FromDouble(n0 - 1)),
            engine.TensorDivideScalar(engine.TensorSquare(s21), NumOps.FromDouble(n1 - 1)));
        var degreesOfFreedom = FilterStatisticTensorOps.GuardedDivide(engine.TensorSquare(sum), denominator, 0.0, NumOps.One);
        _degreesOfFreedom = degreesOfFreedom.ToVector();

        // Two-tailed p-value from the t-distribution (scipy.stats.ttest_ind(equal_var=False))
        var pValues = FilterStatisticTensorOps.StudentTTwoSided(tStatistics, engine.TensorMax(degreesOfFreedom, NumOps.One));
        _pValues = pValues.ToVector();

        // Select significant features or top by |t|
        _selectedIndices = FilterStatisticTensorOps.SignificantOrTopK(
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
            throw new InvalidOperationException("WelchTTest has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("WelchTTest does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WelchTTest has not been fitted.");

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
