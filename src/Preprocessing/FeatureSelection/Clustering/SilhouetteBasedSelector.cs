using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Clustering;

/// <summary>
/// Silhouette Score based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that maximize the silhouette score of clusters, indicating
/// how well clusters are separated and cohesive.
/// </para>
/// <para><b>For Beginners:</b> The silhouette score measures how similar a point is
/// to its own cluster versus other clusters. Features that give high silhouette
/// scores create well-separated, tight clusters.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SilhouetteBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _silhouetteScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? SilhouetteScores => _silhouetteScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SilhouetteBasedSelector(
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
            "SilhouetteBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Class membership, class sizes and each sample's own-class size
        var Y = PreprocessingTensorOps.ClassOneHot<T>(classIndex, k);
        // Counted in int: a Half sum of the 0/1 membership stops at 2048
        var counts = engine.ReduceSum(Y.Cast<int>(), new[] { 0 }, false).Cast<T>();
        var ownCount = engine.Reshape(engine.TensorMatMul(Y, engine.Reshape(counts, new[] { k, 1 })), new[] { n });
        var inMultiMemberClass = engine.TensorGreaterThan(ownCount, NumOps.One);
        var ownDenominator = engine.TensorMax(engine.TensorSubtractScalar(ownCount, NumOps.One), NumOps.One);
        var classSizes = engine.TensorBroadcastTo(engine.Reshape(counts, new[] { 1, k }), new[] { n, k });
        T maxValue = NumOps.MaxValue;
        T zero = NumOps.Zero;
        var blocked = PreprocessingTensorOps.Filled(new[] { n, k }, maxValue);
        var zeros = new Tensor<T>(new[] { n });
        var pairShape = new[] { n, n };

        var perFeature = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            // Pairwise 1-D distances and their per-class sums
            var column = PreprocessingTensorOps.ColumnVector(X, j);
            var distances = engine.TensorAbs(engine.TensorSubtract(
                engine.TensorBroadcastTo(engine.Reshape(column, new[] { n, 1 }), pairShape),
                engine.TensorBroadcastTo(engine.Reshape(column, new[] { 1, n }), pairShape)));
            var classSums = engine.TensorMatMul(distances, Y);

            // a(i) - average distance to own cluster (self excluded)
            var a = engine.TensorWhere(inMultiMemberClass,
                engine.TensorDivide(engine.ReduceSum(engine.TensorMultiply(classSums, Y), new[] { 1 }, false), ownDenominator),
                zeros);

            // b(i) - min average distance to other clusters (0 when there is no other cluster)
            var otherAverages = engine.TensorWhere(Y, blocked, engine.TensorDivide(classSums, classSizes));
            var b = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(otherAverages), new[] { 1 }, false));
            b = engine.TensorWhere(engine.TensorEquals(b, maxValue), zeros, b);

            // s(i) = (b - a) / max(a, b); 0 for a sample alone in its cluster (Rousseeuw 1987)
            var largest = engine.TensorMax(a, b);
            var s = engine.TensorWhere(engine.TensorGreaterThan(largest, zero),
                engine.TensorDivide(engine.TensorSubtract(b, a), largest), zeros);
            s = engine.TensorWhere(inMultiMemberClass, s, zeros);

            // Averaged in double: a Half sum of n silhouettes stalls once it passes 2048 times their spacing
            perFeature[j] = engine.ReduceMean(s.Cast<double>(), new[] { 0 }, true).Cast<T>();
        }

        var scores = p == 0 ? new Tensor<T>(new[] { 0 }) : engine.Reshape(engine.TensorConcatenate(perFeature, 0), new[] { p });
        _silhouetteScores = scores.ToVector();

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
            throw new InvalidOperationException("SilhouetteBasedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SilhouetteBasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SilhouetteBasedSelector has not been fitted.");

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
