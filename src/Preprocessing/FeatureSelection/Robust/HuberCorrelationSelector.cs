using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Robust;

/// <summary>
/// Huber Correlation-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Huber M-estimator for robust correlation computation, downweighting
/// the influence of outliers in the feature-target relationship.
/// </para>
/// <para><b>For Beginners:</b> Regular correlation can be strongly affected by
/// outliers. The Huber method treats moderate deviations normally but reduces
/// the influence of extreme values. This gives you feature-target correlations
/// that better reflect the typical relationship, not just outlier effects.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class HuberCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _huberDelta;

    private Vector<T>? _robustCorrelations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double HuberDelta => _huberDelta;
    public Vector<T>? RobustCorrelations => _robustCorrelations;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public HuberCorrelationSelector(
        int nFeaturesToSelect = 10,
        double huberDelta = 1.35,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _huberDelta = huberDelta;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "HuberCorrelationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        T delta = NumOps.FromDouble(_huberDelta);

        // Standardize x and y with median / (1.4826 * MAD) + 1e-10
        var xStd = FeatureSelectionTensorOps.RobustZScores(Tensor<T>.FromMatrix(data));
        var yStd = FeatureSelectionTensorOps.RobustZScores(engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }));

        // Huber-weighted correlation
        var w = engine.TensorMultiply(
            FeatureSelectionTensorOps.HuberWeights(xStd, delta), FeatureSelectionTensorOps.HuberWeights(yStd, delta));
        var wx = engine.TensorMultiply(w, xStd);
        var sumWxy = engine.ReduceSum(engine.TensorMultiply(wx, yStd), new[] { 0 }, false);
        var sumWx2 = engine.ReduceSum(engine.TensorMultiply(wx, xStd), new[] { 0 }, false);
        var sumWy2 = engine.ReduceSum(engine.TensorMultiply(w, engine.TensorMultiply(yStd, yStd)), new[] { 0 }, false);
        var denom = engine.TensorSqrt(engine.TensorMultiply(sumWx2, sumWy2));
        var correlations = FeatureSelectionTensorOps.ZeroUnless(
            engine.TensorGreaterThan(denom, NumOps.FromDouble(1e-10)), engine.TensorAbs(engine.TensorDivide(sumWxy, denom)));

        _robustCorrelations = correlations.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(correlations, Math.Min(_nFeaturesToSelect, p));
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
            throw new InvalidOperationException("HuberCorrelationSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("HuberCorrelationSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HuberCorrelationSelector has not been fitted.");

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
