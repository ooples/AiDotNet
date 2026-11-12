using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// Two-Sample T-Test based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their t-statistic from two-sample t-tests between
/// classes, identifying features with significantly different means.
/// </para>
/// <para><b>For Beginners:</b> The t-test checks if two groups have different
/// average values. This selector finds features where classes have significantly
/// different means, which helps in classification tasks.
/// </para>
/// </remarks>
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

        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        var zeros = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        var ones = PreprocessingTensorOps.Filled(new[] { p }, NumOps.One);
        if (classes.Length < 2)
        {
            // Can't do t-test with less than 2 classes
            _tStatistics = zeros.ToVector();
            _pValues = ones.ToVector();
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        int k = classes.Length;
        var sizes = new int[k];
        foreach (int c in classIndex)
            sizes[c]++;

        // Class means and within-class sums of squares, [k, p]
        var x = Tensor<T>.FromMatrix(data);
        var index = new Tensor<int>(classIndex, new[] { n });
        var membership = Engine.TensorOneHot<T>(index, k);
        var means = SelectorTensorOps.GroupMeans(membership, x, out _);
        var squares = Engine.TensorMatMul(Engine.TensorTranspose(membership),
            Engine.TensorSquare(Engine.TensorSubtract(x, Engine.TensorIndexSelect(means, index, 0))));

        T eps = NumOps.FromDouble(1e-10);
        var best = zeros;
        var bestP = ones;

        // For each feature, Welch's t-test between every pair of classes; keep the largest |t| (first pair on ties)
        for (int c1 = 0; c1 < k; c1++)
        {
            for (int c2 = c1 + 1; c2 < k; c2++)
            {
                int n1 = sizes[c1];
                int n2 = sizes[c2];
                if (n1 < 2 || n2 < 2)
                    continue;

                // var_c / n_c with the sample variance var_c = SS_c / (n_c - 1)
                var a = Engine.TensorDivideScalar(PreprocessingTensorOps.RowVector(squares, c1),
                    PreprocessingTensorOps.CountAsScalar<T>((n1 - 1) * n1));
                var b = Engine.TensorDivideScalar(PreprocessingTensorOps.RowVector(squares, c2),
                    PreprocessingTensorOps.CountAsScalar<T>((n2 - 1) * n2));
                var seSquared = Engine.TensorAdd(a, b);
                var usable = PreprocessingTensorOps.AtLeast(Engine.TensorSqrt(seSquared), eps);
                var safeSeSquared = Engine.TensorWhere(usable, seSquared, ones);
                var t = SelectorTensorOps.ZeroUnless(usable, Engine.TensorDivide(Engine.TensorSubtract(
                    PreprocessingTensorOps.RowVector(means, c1), PreprocessingTensorOps.RowVector(means, c2)),
                    Engine.TensorSqrt(safeSeSquared)));

                // Two-sided p-value from Student's t with the Welch-Satterthwaite degrees of freedom
                // (scipy.stats.ttest_ind(equal_var=False)): p = I_{df / (df + t^2)}(df / 2, 1 / 2)
                var df = Engine.TensorDivide(Engine.TensorMultiply(safeSeSquared, safeSeSquared), Engine.TensorAdd(
                    Engine.TensorDivideScalar(Engine.TensorMultiply(a, a), PreprocessingTensorOps.CountAsScalar<T>(n1 - 1)),
                    Engine.TensorDivideScalar(Engine.TensorMultiply(b, b), PreprocessingTensorOps.CountAsScalar<T>(n2 - 1))));
                df = Engine.TensorWhere(usable, df, ones);
                var pValue = Engine.TensorWhere(usable, FilterStatisticTensorOps.StudentTTwoSided(t, df), ones);

                var better = Engine.TensorGreaterThan(Engine.TensorAbs(t), Engine.TensorAbs(best));
                best = Engine.TensorWhere(better, t, best);
                bestP = Engine.TensorWhere(better, pValue, bestP);
            }
        }

        var scores = Engine.TensorAbs(best);
        _tStatistics = scores.ToVector();
        _pValues = bestP.ToVector();
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
            throw new InvalidOperationException("TTestSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
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
