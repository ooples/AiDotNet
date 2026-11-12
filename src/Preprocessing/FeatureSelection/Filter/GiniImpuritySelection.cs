using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Gini Impurity-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Measures how much each feature reduces impurity (Gini impurity) when used
/// to split the data. Features that create purer groups after splitting are
/// considered more important.
/// </para>
/// <para><b>For Beginners:</b> Gini impurity measures how "mixed" a group is.
/// If splitting data by a feature creates groups where samples are mostly
/// the same class (pure groups), that feature is valuable. This method finds
/// features that best separate different classes from each other.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GiniImpuritySelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nThresholds;

    private Vector<T>? _giniReductionScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? GiniReductionScores => _giniReductionScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GiniImpuritySelection(
        int nFeaturesToSelect = 10,
        int nThresholds = 20,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nThresholds < 1)
            throw new ArgumentException("Number of thresholds must be at least 1.", nameof(nThresholds));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nThresholds = nThresholds;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GiniImpuritySelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var shape = new[] { n, p };

        // Parent Gini over the rounded class labels
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;
        var classOneHot = PreprocessingTensorOps.ClassOneHot<T>(classIndex, k);
        T parentGini = FeatureSelectionTensorOps.GiniImpurity(classOneHot);
        var classTotals = engine.TensorBroadcastTo(
            engine.Reshape(engine.ReduceSum(classOneHot, new[] { 0 }, false), new[] { 1, k }), new[] { p, k });

        // Columns whose range is below 1e-10 score 0
        var minVal = PreprocessingTensorOps.ColumnMin(X);
        var range = engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(X), minVal);
        var varying = PreprocessingTensorOps.AtLeast(range, NumOps.FromDouble(1e-10));

        // Try evenly spaced thresholds min + range * t / (nThresholds + 1); a split leaving either side empty is skipped
        var zeros = new Tensor<T>(new[] { p });
        var bestReduction = zeros;
        T divisor = NumOps.FromDouble(_nThresholds + 1);
        T zero = NumOps.Zero;
        T sampleCount = NumOps.FromDouble(n);
        T negatedParent = NumOps.Negate(parentGini);
        for (int t = 1; t <= _nThresholds; t++)
        {
            var threshold = engine.TensorAdd(minVal,
                engine.TensorDivideScalar(engine.TensorMultiplyScalar(range, PreprocessingTensorOps.CountAsScalar<T>(t)), divisor));
            var left = PreprocessingTensorOps.Not(engine.TensorGreaterThan(X,
                engine.TensorBroadcastTo(engine.Reshape(threshold, new[] { 1, p }), shape)));
            var leftCounts = engine.TensorMatMul(engine.TensorTranspose(left), classOneHot);
            var sides = engine.TensorConcatenate(new[]
            {
                engine.Reshape(leftCounts, new[] { p, 1, k }),
                engine.Reshape(engine.TensorSubtract(classTotals, leftCounts), new[] { p, 1, k })
            }, 1);
            var weightedGini = FeatureSelectionTensorOps.WeightedBinGini(sides, n);
            var nLeft = engine.ReduceSum(left, new[] { 0 }, false);
            var bothSides = engine.TensorMultiply(engine.TensorGreaterThan(nLeft, zero),
                engine.TensorLessThan(nLeft, sampleCount));
            var reduction = engine.TensorWhere(bothSides,
                engine.TensorNegate(engine.TensorAddScalar(weightedGini, negatedParent)), zeros);
            bestReduction = engine.TensorMax(bestReduction, reduction);
        }

        var giniReductionScores = engine.TensorWhere(varying, bestReduction, zeros);
        _giniReductionScores = giniReductionScores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(giniReductionScores, numToSelect);

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
            throw new InvalidOperationException("GiniImpuritySelection has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GiniImpuritySelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GiniImpuritySelection has not been fitted.");

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
