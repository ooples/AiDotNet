using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Nonlinear;

/// <summary>
/// Distance Correlation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on distance correlation with the target,
/// which detects all types of dependence including nonlinear.
/// </para>
/// <para><b>For Beginners:</b> Distance correlation measures dependence between
/// variables using pairwise distances. Unlike Pearson correlation which only
/// detects linear relationships, distance correlation equals zero if and only if
/// the variables are truly independent. It ranges from 0 to 1.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DistanceCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _dcorValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? DCorValues => _dcorValues;
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

        var engine = Engine;
        T pairCount = NumOps.FromDouble((double)n * n);

        // Double-centred distance matrix of the target and its distance variance
        var yCentered = FeatureSelectionTensorOps.DoubleCenteredDistances(engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }));
        T yVar = NumOps.Divide(engine.TensorSum(engine.TensorSquare(yCentered)), pairCount);

        // Per feature: sum A .* B (distance covariance) and sum A .* A (distance variance)
        var sums = FeatureSelectionTensorOps.KernelWeightedSums(
            Tensor<T>.FromMatrix(data),
            column => FeatureSelectionTensorOps.DoubleCenteredDistances(column),
            new[] { yCentered },
            new[] { PreprocessingTensorOps.Filled(new[] { n, n }, NumOps.One) });
        var dcovSquared = engine.TensorDivideScalar(engine.Reshape(engine.TensorSlice(sums, new[] { 0, 0 }, new[] { p, 1 }), new[] { p }), pairCount);
        var xVar = engine.TensorDivideScalar(engine.Reshape(engine.TensorSlice(sums, new[] { 0, 1 }, new[] { p, 1 }), new[] { p }), pairCount);

        // dCor = dCov / sqrt(sqrt(dVar(X) dVar(Y))); 0 when either distance variance is below 1e-10
        var scores = new Tensor<T>(new[] { p });
        if (!NumOps.LessThan(yVar, NumOps.FromDouble(1e-10)))
        {
            var defined = PreprocessingTensorOps.OneMinus(engine.TensorLessThan(xVar, NumOps.FromDouble(1e-10)));
            var safeXVar = engine.TensorWhere(defined, xVar, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
            var dcor = engine.TensorDivide(
                engine.TensorSqrt(dcovSquared), engine.TensorSqrt(engine.TensorSqrt(engine.TensorMultiplyScalar(safeXVar, yVar))));
            scores = engine.TensorWhere(defined, dcor, scores);
        }

        _dcorValues = scores.ToVector();

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
