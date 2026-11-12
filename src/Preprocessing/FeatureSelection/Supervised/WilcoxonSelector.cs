using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Supervised;

/// <summary>
/// Wilcoxon Rank-Sum Test Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Wilcoxon rank-sum test (Mann-Whitney U test) to select features
/// that show significant differences between classes without assuming
/// normal distributions.
/// </para>
/// <para><b>For Beginners:</b> The Wilcoxon test is a non-parametric test
/// that checks if one group tends to have larger values than another by
/// comparing ranks. It's robust and doesn't assume your data follows any
/// particular distribution.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class WilcoxonSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _testStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? TestStatistics => _testStatistics;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public WilcoxonSelector(
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
            "WilcoxonSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var positive = Engine.TensorGreaterThan(Tensor<T>.FromVector(target), NumOps.FromDouble(0.5)); // Binarize
        int n1 = PreprocessingTensorOps.CountNonZero(positive);
        int n0 = n - n1;

        if (n0 == 0 || n1 == 0)
            throw new InvalidOperationException("Target must have both classes present.");

        // Rank sum of class 1 over the average ranks of every feature
        var ranks = RankTestOps.AverageRanksWithTies(Tensor<T>.FromMatrix(data), out var tieTerm);
        var r1 = Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(positive, new[] { 1, n }), ranks), new[] { p });

        // Mann-Whitney U statistic
        var u1 = Engine.TensorSubtractScalar(r1, NumOps.FromDouble(n1 * (n1 + 1.0) / 2));
        var u0 = Engine.TensorAddScalar(Engine.TensorNegate(u1), NumOps.FromDouble((double)n0 * n1));
        var u = Engine.TensorMin(u1, u0);

        // Standardized statistic (normal approximation) with the tie-corrected variance of scipy.stats.mannwhitneyu:
        // sigma^2 = n0 n1 / 12 ((n + 1) - sum (t^3 - t) / (n (n - 1)))
        double meanU = (double)n0 * n1 / 2.0;
        var variance = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Engine.TensorNegate(
                Engine.TensorDivideScalar(tieTerm, NumOps.FromDouble((double)n * (n - 1)))), NumOps.FromDouble(n + 1)),
            NumOps.FromDouble((double)n0 * n1 / 12));
        var stdU = Engine.TensorAddScalar(Engine.TensorSqrt(Engine.TensorMax(variance, NumOps.Zero)), NumOps.FromDouble(1e-10));
        var scores = Engine.TensorDivide(Engine.TensorAbs(Engine.TensorSubtractScalar(u, NumOps.FromDouble(meanU))), stdU);
        _testStatistics = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));
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
            throw new InvalidOperationException("WilcoxonSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("WilcoxonSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WilcoxonSelector has not been fitted.");

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
