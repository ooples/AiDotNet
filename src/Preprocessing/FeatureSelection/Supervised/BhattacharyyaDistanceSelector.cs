using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Supervised;

/// <summary>
/// Bhattacharyya Distance Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the Bhattacharyya distance between class
/// distributions, which measures the separability between classes.
/// </para>
/// <para><b>For Beginners:</b> The Bhattacharyya distance measures how much
/// two probability distributions overlap. Features with high Bhattacharyya
/// distance have class distributions that don't overlap much, making them
/// good for distinguishing between classes.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BhattacharyyaDistanceSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _distanceScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? DistanceScores => _distanceScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BhattacharyyaDistanceSelector(
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
            "BhattacharyyaDistanceSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        int nClasses = classes.Length;
        var sizes = new int[nClasses];
        foreach (int c in classIndex)
            sizes[c]++;

        // Class means and sample variances (n_c - 1 + 1e-10 divisor, floored at 1e-10), [k, p]
        var x = Tensor<T>.FromMatrix(data);
        var index = new Tensor<int>(classIndex, new[] { n });
        var membership = Engine.TensorOneHot<T>(index, nClasses);
        var means = SelectorTensorOps.GroupMeans(membership, x, out var counts);
        var squares = Engine.TensorMatMul(Engine.TensorTranspose(membership),
            Engine.TensorSquare(Engine.TensorSubtract(x, Engine.TensorIndexSelect(means, index, 0))));
        var divisor = Engine.TensorAddScalar(Engine.TensorSubtractScalar(counts, NumOps.One), PreprocessingTensorOps.Tolerance<T>(1e-10));
        var variances = Engine.TensorMax(Engine.TensorDivide(squares,
            Engine.TensorBroadcastTo(Engine.Reshape(divisor, new[] { nClasses, 1 }), new[] { nClasses, p })), PreprocessingTensorOps.Tolerance<T>(1e-10));

        // Average pairwise Bhattacharyya distance between the class Gaussians:
        // D_B = 1/4 ln(1/4 (v1 / v2 + v2 / v1 + 2)) + 1/4 (m1 - m2)^2 / (v1 + v2)
        T quarter = NumOps.FromDouble(0.25);
        T two = NumOps.FromDouble(2);
        var totalDist = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        int pairCount = 0;
        for (int c1 = 0; c1 < nClasses; c1++)
        {
            for (int c2 = c1 + 1; c2 < nClasses; c2++)
            {
                if (sizes[c1] == 0 || sizes[c2] == 0) continue;

                var var1 = PreprocessingTensorOps.RowVector(variances, c1);
                var var2 = PreprocessingTensorOps.RowVector(variances, c2);
                var meanDifference = Engine.TensorSubtract(
                    PreprocessingTensorOps.RowVector(means, c1), PreprocessingTensorOps.RowVector(means, c2));
                var ratioSum = Engine.TensorAdd(Engine.TensorDivide(var1, var2), Engine.TensorDivide(var2, var1));
                var term1 = Engine.TensorMultiplyScalar(
                    Engine.TensorLog(Engine.TensorMultiplyScalar(Engine.TensorAddScalar(ratioSum, two), quarter)), quarter);
                var term2 = Engine.TensorMultiplyScalar(Engine.TensorDivide(
                    Engine.TensorMultiply(meanDifference, meanDifference), Engine.TensorAdd(var1, var2)), quarter);
                totalDist = Engine.TensorAdd(totalDist, Engine.TensorAdd(term1, term2));
                pairCount++;
            }
        }

        var scores = pairCount > 0
            ? Engine.TensorDivideScalar(totalDist, NumOps.FromDouble(pairCount))
            : totalDist;
        _distanceScores = scores.ToVector();
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
            throw new InvalidOperationException("BhattacharyyaDistanceSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BhattacharyyaDistanceSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BhattacharyyaDistanceSelector has not been fitted.");

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
