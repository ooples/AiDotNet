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

        // Szekely et al. (2007): dCor = sqrt(dCov^2 / sqrt(dVar_x^2 dVar_y^2)) from double-centred distance matrices.
        // dCor does not change when either matrix is scaled, so each is divided by its largest magnitude and the
        // averages are taken as row means then their mean: the n^2-term sums of squares and their product overflow
        // a narrow T such as Half (300 samples already pass 65504), while every term here stays at most 1.
        var x = Tensor<T>.FromMatrix(data);
        var dy = ScaledCenteredDistanceMatrix(Tensor<T>.FromVector(target), out T yScale);
        T dVarBSqScaled = TotalMean(Engine.TensorMultiply(dy, dy)).GetFlat(0);
        T dVarBSq = NumOps.Multiply(NumOps.Multiply(dVarBSqScaled, yScale), yScale);

        var crossMeans = new Tensor<T>[p];
        var squareMeans = new Tensor<T>[p];
        var scales = new T[p];
        for (int j = 0; j < p; j++)
        {
            var dx = ScaledCenteredDistanceMatrix(Engine.TensorSliceAxis(x, 1, j), out scales[j]);
            crossMeans[j] = TotalMean(Engine.TensorMultiply(dx, dy));
            squareMeans[j] = TotalMean(Engine.TensorMultiply(dx, dx));
        }

        var scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        T eps = PreprocessingTensorOps.Tolerance<T>(1e-10);
        if (p > 0 && !NumOps.LessThan(dVarBSq, eps))
        {
            var dCovSq = Engine.TensorConcatenate(crossMeans, 0);
            var dVarASq = Engine.TensorConcatenate(squareMeans, 0);
            var dCor = Engine.TensorSqrt(Engine.TensorDivide(dCovSq, Engine.TensorMultiplyScalar(
                Engine.TensorSqrt(dVarASq), NumOps.Sqrt(dVarBSqScaled))));

            // The guard compares the unscaled dVar_x^2 (an overflow to infinity still compares as above eps)
            var scale = new Tensor<T>(scales, new[] { p });
            var unscaledVarA = Engine.TensorMultiply(Engine.TensorMultiply(dVarASq, scale), scale);
            scores = SelectorTensorOps.ZeroUnless(PreprocessingTensorOps.AtLeast(unscaledVarA, eps), dCor);
        }

        _distanceCorrelations = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <summary>
    /// The double-centred distance matrix divided by its largest magnitude <paramref name="scale"/> (1 when it is all
    /// zero), as [n, n].
    /// </summary>
    private Tensor<T> ScaledCenteredDistanceMatrix(Tensor<T> values, out T scale)
    {
        var centered = ComputeCenteredDistanceMatrix(values);
        T largest = Engine.TensorMaxValue(Engine.TensorAbs(centered));
        scale = NumOps.GreaterThan(largest, NumOps.Zero) ? largest : NumOps.One;
        return Engine.TensorDivideScalar(centered, scale);
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

    /// <summary>The mean of every element of an [n, n] tensor (the mean of its row means), as a [1] tensor.</summary>
    private Tensor<T> TotalMean(Tensor<T> values) =>
        Engine.ReduceMean(Engine.ReduceMean(values, new[] { 1 }, false), new[] { 0 }, true);

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
