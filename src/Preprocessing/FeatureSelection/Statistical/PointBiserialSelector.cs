using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// Point-Biserial Correlation Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses point-biserial correlation for feature selection when the target
/// is binary. This is the correlation between a continuous variable and
/// a binary variable.
/// </para>
/// <para><b>For Beginners:</b> Point-biserial correlation measures how well
/// a continuous feature separates two groups (like positive vs negative cases).
/// Features with high point-biserial correlation have very different values
/// for the two classes, making them useful for classification.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PointBiserialSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _correlationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? CorrelationScores => _correlationScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PointBiserialSelector(
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
            "PointBiserialSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var x = Tensor<T>.FromMatrix(data);
        var positive = Engine.TensorGreaterThan(Tensor<T>.FromVector(target), NumOps.FromDouble(0.5)); // Binarize

        // Count groups
        int n1 = PreprocessingTensorOps.CountNonZero(positive);
        int n0 = n - n1;

        if (n0 == 0 || n1 == 0)
            throw new InvalidOperationException("Target must have both classes present.");

        // Group means, population standard deviation and r_pb = (M1 - M0) / s * sqrt(n0 n1 / n^2)
        var membership = Engine.TensorStack(new[] { PreprocessingTensorOps.OneMinus(positive), positive }, 1);
        var groupMeans = SelectorTensorOps.GroupMeans(membership, x, out _);
        var meanDifference = Engine.TensorSubtract(
            PreprocessingTensorOps.RowVector(groupMeans, 1), PreprocessingTensorOps.RowVector(groupMeans, 0));
        var std = Engine.TensorAddScalar(
            Engine.TensorSqrt(SelectorTensorOps.ColumnVariance(x)), PreprocessingTensorOps.Tolerance<T>(1e-10));
        T share = NumOps.FromDouble(Math.Sqrt((double)n0 * n1 / ((double)n * n)));
        var scores = Engine.TensorAbs(Engine.TensorMultiplyScalar(Engine.TensorDivide(meanDifference, std), share));
        _correlationScores = scores.ToVector();
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
            throw new InvalidOperationException("PointBiserialSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PointBiserialSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PointBiserialSelector has not been fitted.");

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
