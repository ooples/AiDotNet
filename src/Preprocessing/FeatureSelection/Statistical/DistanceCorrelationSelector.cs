using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// Distance Correlation Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses distance correlation to detect both linear and non-linear dependencies
/// between features and the target. Distance correlation is zero if and only
/// if the variables are independent.
/// </para>
/// <para><b>For Beginners:</b> Distance correlation is like regular correlation
/// but can detect any type of relationship, not just linear ones. Unlike regular
/// correlation, a distance correlation of 0 truly means no relationship exists.
/// It computes distances between all pairs of points and correlates those distances.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DistanceCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _distanceCorrelations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? DistanceCorrelations => _distanceCorrelations;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DistanceCorrelationSelector(
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
            "DistanceCorrelationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Szekely et al. (2007): dCor = sqrt(dCov^2 / sqrt(dVar_x^2 dVar_y^2)) from double-centred distance matrices
        var x = Tensor<T>.FromMatrix(data);
        var dy = ComputeCenteredDistanceMatrix(Tensor<T>.FromVector(target));
        T count = NumOps.FromDouble((double)n * n);
        T dVarBSq = NumOps.Divide(Engine.TensorSumOfSquares(dy), count);

        var crossSums = new Tensor<T>[p];
        var squareSums = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            var dx = ComputeCenteredDistanceMatrix(Engine.TensorSliceAxis(x, 1, j));
            crossSums[j] = TotalSum(Engine.TensorMultiply(dx, dy));
            squareSums[j] = TotalSum(Engine.TensorMultiply(dx, dx));
        }

        var scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        T eps = NumOps.FromDouble(1e-10);
        if (p > 0 && !NumOps.LessThan(dVarBSq, eps))
        {
            var dCovSq = Engine.TensorDivideScalar(Engine.TensorConcatenate(crossSums, 0), count);
            var dVarASq = Engine.TensorDivideScalar(Engine.TensorConcatenate(squareSums, 0), count);
            var dCor = Engine.TensorSqrt(Engine.TensorDivide(dCovSq,
                Engine.TensorSqrt(Engine.TensorMultiplyScalar(dVarASq, dVarBSq))));
            scores = SelectorTensorOps.ZeroUnless(PreprocessingTensorOps.AtLeast(dVarASq, eps), dCor);
        }

        _distanceCorrelations = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <summary>The double-centred pairwise distance matrix <c>|v_i - v_k|</c> of a 1-D tensor, as [n, n].</summary>
    private Tensor<T> ComputeCenteredDistanceMatrix(Tensor<T> values)
    {
        int n = values.Length;
        var shape = new[] { n, n };
        var distances = Engine.TensorAbs(Engine.TensorSubtract(
            Engine.TensorBroadcastTo(Engine.Reshape(values, new[] { n, 1 }), shape),
            Engine.TensorBroadcastTo(Engine.Reshape(values, new[] { 1, n }), shape)));
        var rowMeans = Engine.TensorBroadcastTo(Engine.ReduceMean(distances, new[] { 1 }, true), shape);
        var colMeans = Engine.TensorBroadcastTo(Engine.ReduceMean(distances, new[] { 0 }, true), shape);
        return Engine.TensorAddScalar(
            Engine.TensorSubtract(Engine.TensorSubtract(distances, rowMeans), colMeans), Engine.TensorMean(distances));
    }

    /// <summary>The sum of every element of a tensor, as a [1] tensor.</summary>
    private Tensor<T> TotalSum(Tensor<T> values) =>
        Engine.ReduceSum(Engine.Reshape(values, new[] { 1, values.Length }), new[] { 1 }, false);

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
            throw new InvalidOperationException("DistanceCorrelationSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DistanceCorrelationSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DistanceCorrelationSelector has not been fitted.");

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
