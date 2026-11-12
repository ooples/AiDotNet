using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// T-Test based feature selection for binary classification.
/// </summary>
/// <remarks>
/// <para>
/// Uses independent samples t-test to measure how well each feature separates
/// two classes. Features with higher t-statistics have significantly different
/// means between classes.
/// </para>
/// <para><b>For Beginners:</b> The t-test asks "are the average values of this
/// feature significantly different between the two groups?" A high t-statistic
/// means the feature's values are very different for each class, making it useful
/// for telling them apart.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TTestSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

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

    public TTestSelector(
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
            "TTestSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        var class0 = FilterStatisticTensorOps.BelowMask(target, 0.5, out int n0);
        var class1 = FilterStatisticTensorOps.OneMinus(class0);
        int n1 = n - n0;

        if (n0 < 2 || n1 < 2)
        {
            _tStatistics = new Vector<T>(p);
            _pValues = new Vector<T>(p);
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        // Class means and variances of every feature
        var x = FilterStatisticTensorOps.AsTensor(data);
        var (mean0, ss0) = FilterStatisticTensorOps.MaskedColumnMoments(x, class0, n0);
        var (mean1, ss1) = FilterStatisticTensorOps.MaskedColumnMoments(x, class1, n1);
        var s20 = engine.TensorDivideScalar(ss0, NumOps.FromDouble((n0 - 1.0) * n0));
        var s21 = engine.TensorDivideScalar(ss1, NumOps.FromDouble((n1 - 1.0) * n1));
        var sum = engine.TensorAdd(s20, s21);

        // |t| with the unpooled standard error sqrt(var0 / n0 + var1 / n1); classes that do not overlap give inf
        var tStatistics = FilterStatisticTensorOps.SeparationAwareRatio(
            engine.TensorAbs(engine.TensorSubtract(mean0, mean1)), engine.TensorSqrt(sum), 1e-10);
        _tStatistics = tStatistics.ToVector();

        // Two-sided p-value of that (Welch) statistic from the t-distribution with Welch-Satterthwaite degrees of
        // freedom (scipy.stats.ttest_ind(equal_var=False)), n0 + n1 - 2 when both classes are constant
        var den = engine.TensorAdd(
            engine.TensorDivideScalar(engine.TensorSquare(s20), NumOps.FromDouble(n0 - 1)),
            engine.TensorDivideScalar(engine.TensorSquare(s21), NumOps.FromDouble(n1 - 1)));
        var degreesOfFreedom = FilterStatisticTensorOps.GuardedDivide(
            engine.TensorSquare(sum), den, 0.0, NumOps.FromDouble(n0 + n1 - 2));
        _pValues = FilterStatisticTensorOps.StudentTTwoSided(tStatistics, degreesOfFreedom).ToVector();

        _selectedIndices = FilterStatisticTensorOps.TopKIndicesSorted(tStatistics, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("TTestSelector has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TTestSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TTestSelector has not been fitted.");

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
