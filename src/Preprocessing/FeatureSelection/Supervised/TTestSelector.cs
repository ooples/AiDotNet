using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Supervised;

/// <summary>
/// T-Test Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Student's t-test to select features that show statistically
/// significant differences between two classes.
/// </para>
/// <para><b>For Beginners:</b> The t-test checks whether two groups have
/// significantly different means. Features with large t-statistics have
/// means that are very different between classes, making them useful for
/// classification.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TTestSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _equalVariance;

    private Vector<T>? _tStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? TStatistics => _tStatistics;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TTestSelector(
        int nFeaturesToSelect = 10,
        bool equalVariance = false,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _equalVariance = equalVariance;
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

        var positive = Engine.TensorGreaterThan(Tensor<T>.FromVector(target), NumOps.FromDouble(0.5));
        int n1 = PreprocessingTensorOps.CountNonZero(positive);
        int n0 = n - n1;

        if (n0 < 2 || n1 < 2)
            throw new InvalidOperationException("Each class must have at least 2 samples.");

        // Means and sample variances of each class ([2, p], class 0 first)
        var x = Tensor<T>.FromMatrix(data);
        var membership = Engine.TensorStack(new[] { PreprocessingTensorOps.OneMinus(positive), positive }, 1);
        var means = SelectorTensorOps.GroupMeans(membership, x, out _);
        var classOfRow = Engine.TensorIndexSelect(means, positive.Cast<int>(), 0);
        var squares = Engine.TensorMatMul(Engine.TensorTranspose(membership),
            Engine.TensorSquare(Engine.TensorSubtract(x, classOfRow)));
        var ss0 = PreprocessingTensorOps.RowVector(squares, 0);
        var ss1 = PreprocessingTensorOps.RowVector(squares, 1);
        var meanDifference = Engine.TensorSubtract(
            PreprocessingTensorOps.RowVector(means, 1), PreprocessingTensorOps.RowVector(means, 0));

        Tensor<T> se;
        if (_equalVariance)
        {
            // Pooled variance t-test
            var pooledVar = Engine.TensorDivideScalar(Engine.TensorAdd(ss0, ss1), NumOps.FromDouble(n0 + n1 - 2));
            se = Engine.TensorSqrt(Engine.TensorMultiplyScalar(pooledVar, NumOps.FromDouble(1.0 / n0 + 1.0 / n1)));
        }
        else
        {
            // Welch's t-test (unequal variances)
            se = Engine.TensorSqrt(Engine.TensorAdd(
                Engine.TensorDivideScalar(ss0, NumOps.FromDouble((double)(n0 - 1) * n0)),
                Engine.TensorDivideScalar(ss1, NumOps.FromDouble((double)(n1 - 1) * n1))));
        }

        var scores = Engine.TensorAbs(Engine.TensorDivide(meanDifference, Engine.TensorAddScalar(se, PreprocessingTensorOps.Tolerance<T>(1e-10))));
        _tStatistics = scores.ToVector();
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
