using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// Welch's t-test for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Welch's t-test is a variant of the Student's t-test that does not assume equal
/// variances between the two groups. It uses the Welch-Satterthwaite approximation
/// for degrees of freedom.
/// </para>
/// <para><b>For Beginners:</b> When comparing two groups that might have different
/// amounts of spread (variance), Welch's test is more reliable than the standard
/// t-test. It's generally recommended as the default choice for comparing two
/// group means.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class WelchTTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;

    private Vector<T>? _tStatistics;
    private Vector<T>? _pValues;
    private Vector<T>? _degreesOfFreedom;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public Vector<T>? TStatistics => _tStatistics;
    public Vector<T>? PValues => _pValues;
    public Vector<T>? DegreesOfFreedom => _degreesOfFreedom;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public WelchTTest(
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
            "WelchTTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Split samples by class
        var group0 = FeatureSelectionTensorOps.BelowMask(target, 0.5, out int n0);
        var group1 = FeatureSelectionTensorOps.OneMinus(group0);
        int n1 = n - n0;

        if (n0 < 2 || n1 < 2)
        {
            _tStatistics = new Vector<T>(p);
            _pValues = FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One).ToVector();
            _degreesOfFreedom = FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.FromDouble(n - 2)).ToVector();
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        // Means and sample variances of every feature within each group
        var x = FeatureSelectionTensorOps.AsTensor(data);
        var (mean0, ss0) = FeatureSelectionTensorOps.MaskedColumnMoments(x, group0, n0);
        var (mean1, ss1) = FeatureSelectionTensorOps.MaskedColumnMoments(x, group1, n1);
        var s20 = engine.TensorDivideScalar(ss0, NumOps.FromDouble((n0 - 1.0) * n0));
        var s21 = engine.TensorDivideScalar(ss1, NumOps.FromDouble((n1 - 1.0) * n1));
        var sum = engine.TensorAdd(s20, s21);

        // Welch's t-statistic; groups that do not overlap at all give +-inf
        var tStatistics = FeatureSelectionTensorOps.SignedSeparationAwareRatio(
            engine.TensorSubtract(mean0, mean1), engine.TensorSqrt(sum), 1e-10);
        _tStatistics = tStatistics.ToVector();

        // Welch-Satterthwaite degrees of freedom
        var den = engine.TensorAdd(
            engine.TensorDivideScalar(engine.TensorSquare(s20), NumOps.FromDouble(n0 - 1)),
            engine.TensorDivideScalar(engine.TensorSquare(s21), NumOps.FromDouble(n1 - 1)));
        var degreesOfFreedom = FeatureSelectionTensorOps.GuardedDivide(
            engine.TensorSquare(sum), den, 0.0, NumOps.FromDouble(n0 + n1 - 2));
        _degreesOfFreedom = degreesOfFreedom.ToVector();

        // Two-tailed p-value from the t-distribution (scipy.stats.ttest_ind(equal_var=False))
        var pValues = FeatureSelectionTensorOps.StudentTTwoSided(tStatistics, degreesOfFreedom);
        _pValues = pValues.ToVector();

        // Select features with smallest p-values
        _selectedIndices = FeatureSelectionTensorOps.BottomKIndicesSorted(pValues, Math.Min(_nFeaturesToSelect, p));

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

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
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
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
