using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// Student's t-test for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Student's t-test compares the means of two groups to determine if they are
/// statistically different. For feature selection, it tests whether each feature
/// has significantly different values between positive and negative classes.
/// </para>
/// <para><b>For Beginners:</b> This test asks: "Is the average value of this feature
/// different enough between the two classes that it's probably not due to random
/// chance?" Features with large differences are more likely to be useful for
/// distinguishing between classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class StudentTTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly bool _equalVariances;

    private Vector<T>? _tStatistics;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public bool EqualVariances => _equalVariances;
    public Vector<T>? TStatistics => _tStatistics;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public StudentTTest(
        int nFeaturesToSelect = 10,
        double alpha = 0.05,
        bool equalVariances = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _equalVariances = equalVariances;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "StudentTTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Split samples by class (binary classification): target == 1 is group 1, everything else group 0
        var group1 = engine.TensorEquals(FilterStatisticTensorOps.AsTensor(target), NumOps.One);
        var group0 = PreprocessingTensorOps.OneMinus(group1);
        int n1 = engine.TensorCountNonzero(group1);
        int n0 = n - n1;

        if (n0 < 2 || n1 < 2)
        {
            _tStatistics = new Vector<T>(p);
            _pValues = FilterStatisticTensorOps.Filled(new[] { p }, NumOps.One).ToVector();
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        // Means and sums of squared deviations of every feature within each group
        var x = FilterStatisticTensorOps.AsTensor(data);
        var (mean0, ss0) = FilterStatisticTensorOps.MaskedColumnMoments(x, group0, n0);
        var (mean1, ss1) = FilterStatisticTensorOps.MaskedColumnMoments(x, group1, n1);
        var var0 = engine.TensorDivideScalar(ss0, NumOps.FromDouble(n0 - 1));
        var var1 = engine.TensorDivideScalar(ss1, NumOps.FromDouble(n1 - 1));
        var meanDifference = engine.TensorSubtract(mean0, mean1);

        Tensor<T> tStatistics;
        Tensor<T> degreesOfFreedom;
        if (_equalVariances)
        {
            // Pooled variance (Student's t-test)
            var sp2 = engine.TensorDivideScalar(engine.TensorAdd(ss0, ss1), NumOps.FromDouble(n0 + n1 - 2));
            var se = engine.TensorSqrt(engine.TensorMultiplyScalar(sp2, NumOps.FromDouble(1.0 / n0 + 1.0 / n1)));
            tStatistics = FilterStatisticTensorOps.SignedSeparationAwareRatio(meanDifference, se, 1e-10);
            degreesOfFreedom = FilterStatisticTensorOps.Filled(new[] { p }, NumOps.FromDouble(n0 + n1 - 2));
        }
        else
        {
            // Welch's t-test (unequal variances)
            var s20 = engine.TensorDivideScalar(var0, NumOps.FromDouble(n0));
            var s21 = engine.TensorDivideScalar(var1, NumOps.FromDouble(n1));
            var sum = engine.TensorAdd(s20, s21);
            tStatistics = FilterStatisticTensorOps.SignedSeparationAwareRatio(meanDifference, engine.TensorSqrt(sum), 1e-10);

            // Welch-Satterthwaite degrees of freedom
            var den = engine.TensorAdd(
                engine.TensorDivideScalar(engine.TensorSquare(s20), NumOps.FromDouble(n0 - 1)),
                engine.TensorDivideScalar(engine.TensorSquare(s21), NumOps.FromDouble(n1 - 1)));
            degreesOfFreedom = FilterStatisticTensorOps.GuardedDivide(
                engine.TensorSquare(sum), den, 0.0, NumOps.FromDouble(n0 + n1 - 2));
        }

        _tStatistics = tStatistics.ToVector();

        // Two-tailed p-value from the t-distribution (scipy.stats.ttest_ind)
        var pValues = FilterStatisticTensorOps.StudentTTwoSided(tStatistics, degreesOfFreedom);
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
            throw new InvalidOperationException("StudentTTest has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("StudentTTest does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StudentTTest has not been fitted.");

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
