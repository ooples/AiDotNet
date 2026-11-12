using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Boundary;

/// <summary>
/// Margin based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the margin they create between classes,
/// preferring features with larger class separation margins.
/// </para>
/// <para><b>For Beginners:</b> The margin is the distance between classes.
/// Features with larger margins make it easier to separate classes with
/// more confidence, similar to how SVM maximizes margins.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MarginSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _marginScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? MarginScores => _marginScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MarginSelector(
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
            "MarginSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var X = Tensor<T>.FromMatrix(data);
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;
        var membership = engine.TensorTranspose(PreprocessingTensorOps.ClassOneHot<T>(classIndex, k)); // [k, n]

        // Standardize each feature (population std); near-constant features stay as they are
        T eps = NumOps.FromDouble(1e-10);
        var mean = engine.ReduceMean(X, new[] { 0 }, true);
        var centered = engine.TensorSubtract(X, mean);
        var std = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, true));
        var standardized = engine.TensorWhere(
            engine.TensorBroadcastTo(engine.TensorGreaterThan(std, eps), new[] { n, p }),
            engine.TensorDivide(centered, std),
            X);

        // Per-class range of every feature
        var classMin = new Tensor<T>[k];
        var classMax = new Tensor<T>[k];
        for (int c = 0; c < k; c++)
            (classMin[c], classMax[c]) = PreprocessingTensorOps.MaskedColumnRange(
                standardized, PreprocessingTensorOps.RowVector(membership, c));

        // For each pair of classes, compute the margin between their ranges
        var totalMargin = new Tensor<T>(new[] { p });
        var zeros = new Tensor<T>(new[] { p });
        int pairs = 0;
        T half = NumOps.FromDouble(0.5);
        for (int ci = 0; ci < k; ci++)
        {
            for (int cj = ci + 1; cj < k; cj++)
            {
                var min1 = classMin[ci];
                var max1 = classMax[ci];
                var min2 = classMin[cj];
                var max2 = classMax[cj];

                // Margin = gap between closest edges; overlapping ranges give the negative overlap
                var gap = engine.TensorMax(
                    engine.TensorMax(engine.TensorSubtract(min2, max1), engine.TensorSubtract(min1, max2)), zeros);
                var overlap = engine.TensorSubtract(engine.TensorMin(max1, max2), engine.TensorMax(min1, min2));
                var margin = engine.TensorWhere(engine.TensorEquals(gap, zeros), engine.TensorNegate(overlap), gap);

                // Normalize by average spread
                var avgSpread = engine.TensorMultiplyScalar(
                    engine.TensorAdd(engine.TensorSubtract(max1, min1), engine.TensorSubtract(max2, min2)), half);
                margin = engine.TensorWhere(engine.TensorGreaterThan(avgSpread, eps), engine.TensorDivide(margin, avgSpread), margin);

                totalMargin = engine.TensorAdd(totalMargin, margin);
                pairs++;
            }
        }

        var scores = pairs > 0 ? engine.TensorDivideScalar(totalMargin, NumOps.FromDouble(pairs)) : zeros;
        _marginScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

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
            throw new InvalidOperationException("MarginSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MarginSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MarginSelector has not been fitted.");

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
