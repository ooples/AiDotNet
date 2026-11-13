using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Boundary;

/// <summary>
/// Decision Boundary based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on how clearly they separate classes at the decision
/// boundary, focusing on features that create clean class separations.
/// </para>
/// <para><b>For Beginners:</b> A decision boundary is where the classifier switches
/// from predicting one class to another. Features that create clear, simple boundaries
/// are easier for classifiers to use effectively.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DecisionBoundarySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _boundaryClarityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? BoundaryClarityScores => _boundaryClarityScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DecisionBoundarySelector(
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
            "DecisionBoundarySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        // Counts and class sums are taken in double: a 0/1 mask summed in a narrow T (Half stops at 2048) saturates
        var X = Tensor<T>.FromMatrix(data).Cast<double>();
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;
        var membership = engine.TensorTranspose(PreprocessingTensorOps.ClassOneHot<double>(classIndex, k)); // [k, n]
        var classCounts = engine.ReduceSum(membership, new[] { 1 }, false);                             // [k]
        var classMeans = engine.TensorDivide(
            engine.TensorMatMul(membership, X), engine.Reshape(classCounts, new[] { k, 1 }));            // [k, p]

        // For each pair of classes, threshold every feature at the midpoint of the class means and
        // measure the accuracy of that boundary.
        var totalClarity = new Tensor<double>(new[] { p });
        int pairs = 0;
        double half = 0.5;
        for (int ci = 0; ci < k; ci++)
        {
            for (int cj = ci + 1; cj < k; cj++)
            {
                var mean1 = PreprocessingTensorOps.RowVector(classMeans, ci);
                var mean2 = PreprocessingTensorOps.RowVector(classMeans, cj);
                var threshold = engine.TensorMultiplyScalar(engine.TensorAdd(mean1, mean2), half);
                var below = engine.TensorLessThan(X, engine.TensorBroadcastTo(engine.Reshape(threshold, new[] { 1, p }), new[] { n, p }));

                var below1 = engine.Reshape(engine.TensorMatMul(PreprocessingTensorOps.Row(membership, ci), below), new[] { p });
                var below2 = engine.Reshape(engine.TensorMatMul(PreprocessingTensorOps.Row(membership, cj), below), new[] { p });
                var count1 = engine.TensorSlice(classCounts, new[] { ci }, new[] { 1 });
                var count2 = engine.TensorSlice(classCounts, new[] { cj }, new[] { 1 });

                // class 1 below / class 2 at-or-above when mean1 < mean2, the reverse otherwise
                var correctLow = engine.TensorAdd(below1, engine.TensorSubtract(count2, below2));
                var correctHigh = engine.TensorAdd(engine.TensorSubtract(count1, below1), below2);
                var correct = engine.TensorWhere(engine.TensorLessThan(mean1, mean2), correctLow, correctHigh);

                // Clarity score = accuracy at boundary
                totalClarity = engine.TensorAdd(totalClarity, engine.TensorDivide(correct, engine.TensorAdd(count1, count2)));
                pairs++;
            }
        }

        var scores = pairs > 0
            ? engine.TensorDivideScalar(totalClarity, (double)pairs)
            : PreprocessingTensorOps.Filled(new[] { p }, 0.5);
        _boundaryClarityScores = scores.Cast<T>().ToVector();

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
            throw new InvalidOperationException("DecisionBoundarySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DecisionBoundarySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DecisionBoundarySelector has not been fitted.");

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
