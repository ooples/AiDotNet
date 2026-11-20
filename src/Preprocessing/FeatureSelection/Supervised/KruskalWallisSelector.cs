using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Supervised;

/// <summary>
/// Kruskal-Wallis H-Test Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Kruskal-Wallis H-test to select features for multi-class problems.
/// This is a non-parametric alternative to one-way ANOVA.
/// </para>
/// <para><b>For Beginners:</b> The Kruskal-Wallis test extends the Wilcoxon test
/// to more than two groups. It checks if different classes have significantly
/// different distributions for a feature, using ranks instead of actual values.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KruskalWallisSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _hStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? HStatistics => _hStatistics;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public KruskalWallisSelector(
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
            "KruskalWallisSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;

        if (k < 2)
            throw new InvalidOperationException("At least 2 classes are required.");

        // Rank all values of every feature (average ranks for ties) and sum them per class
        var ranks = RankTestOps.AverageRanksWithTies(Tensor<T>.FromMatrix(data), out var tieTerm);
        var membership = PreprocessingTensorOps.ClassOneHot<T>(classIndex, k);
        // (class sizes counted as int: a 0/1 sum in T stops counting at 2048 for Half)
        var classCounts = Engine.Reshape(Engine.ReduceSum(
            membership.Cast<int>(), new[] { 0 }, false).Cast<T>(), new[] { k, 1 });

        // Kruskal-Wallis H = 12 / (n (n + 1)) sum_c R_c^2 / n_c - 3 (n + 1), divided by the tie correction
        // 1 - sum (t^3 - t) / (n^3 - n) as scipy.stats.kruskal (left as is when every value is tied). Formed as
        // H = 12 (n + 1) / n sum_c n_c (u_c - 1/2)^2 with u_c the class mean of rank / (n + 1), the same value:
        // R_c^2 / n_c reaches about n^3 / 4, past Half's 65504 from n = 64, while every term here stays below n.
        var scaledRanks = Engine.TensorDivideScalar(ranks, NumOps.FromDouble(n + 1.0));
        var classMeans = Engine.TensorDivide(Engine.TensorMatMul(Engine.TensorTranspose(membership), scaledRanks),
            Engine.TensorBroadcastTo(Engine.TensorMax(classCounts, NumOps.One), new[] { k, p }));
        var offset = Engine.TensorSubtractScalar(classMeans, NumOps.FromDouble(0.5));
        var spread = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorMultiply(offset, offset),
            Engine.TensorBroadcastTo(classCounts, new[] { k, p })), new[] { 0 }, false);
        var h = Engine.TensorMultiplyScalar(spread, NumOps.FromDouble(12.0 * (n + 1) / n));
        var correction = Engine.TensorSubtractScalar(
            Engine.TensorNegate(Engine.TensorDivideScalar(tieTerm, (double)n * n * n - n)), -1.0);
        var corrected = Engine.TensorGreaterThan(correction, 0.0);
        var divisor = Engine.TensorWhere(corrected, correction, PreprocessingTensorOps.Filled(new[] { p }, 1.0)).Cast<T>();
        var scores = Engine.TensorDivide(h, divisor);
        _hStatistics = scores.ToVector();
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
            throw new InvalidOperationException("KruskalWallisSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KruskalWallisSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KruskalWallisSelector has not been fitted.");

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
