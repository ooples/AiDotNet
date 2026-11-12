using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// Polynomial Kernel based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their importance in a polynomial kernel space,
/// capturing polynomial relationships with the target.
/// </para>
/// <para><b>For Beginners:</b> The polynomial kernel can detect relationships
/// that are polynomial in nature (like x^2, x*y, etc.). This selector finds
/// features that have strong polynomial relationships with the target.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PolynomialKernelSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _degree;
    private readonly double _coef0;

    private Vector<T>? _kernelScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int Degree => _degree;
    public double Coef0 => _coef0;
    public Vector<T>? KernelScores => _kernelScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PolynomialKernelSelector(
        int nFeaturesToSelect = 10,
        int degree = 3,
        double coef0 = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _degree = degree;
        _coef0 = coef0;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PolynomialKernelSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var standardized = FeatureSelectionTensorOps.StandardizeColumnsOrKeep(Tensor<T>.FromMatrix(data));

        // Target similarity of each pair i < j: 1 - |y_i - y_j| / (1 + |y_i - y_j|)
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
        var targetGap = engine.TensorAbs(engine.TensorSubtract(
            engine.TensorBroadcastTo(y, new[] { n, n }), engine.TensorBroadcastTo(engine.TensorTranspose(y), new[] { n, n })));
        var similarity = FeatureSelectionTensorOps.OneMinus(
            engine.TensorDivide(targetGap, engine.TensorAddScalar(targetGap, NumOps.One)));
        var upper = FeatureSelectionTensorOps.StrictUpperMask<T>(n);

        // Kernel-target alignment of each feature's polynomial kernel k(a, b) = (a b + coef0)^degree
        T coef0 = NumOps.FromDouble(_coef0);
        var sums = FeatureSelectionTensorOps.KernelWeightedSums(
            standardized,
            column => FeatureSelectionTensorOps.IntegerPower(
                engine.TensorAddScalar(engine.TensorMatMul(column, engine.TensorTranspose(column)), coef0), _degree),
            new[] { engine.TensorMultiply(upper, similarity) },
            new[] { upper });
        var numerator = engine.Reshape(engine.TensorSlice(sums, new[] { 0, 0 }, new[] { p, 1 }), new[] { p });
        var denominator = engine.Reshape(engine.TensorSlice(sums, new[] { 0, 1 }, new[] { p, 1 }), new[] { p });
        var defined = engine.TensorGreaterThan(denominator, NumOps.FromDouble(1e-10));
        var safeDenominator = engine.TensorWhere(defined, denominator, FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One));
        var scores = engine.TensorWhere(defined, engine.TensorDivide(numerator, engine.TensorSqrt(safeDenominator)), new Tensor<T>(new[] { p }));
        _kernelScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(scores, numToSelect);

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
            throw new InvalidOperationException("PolynomialKernelSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PolynomialKernelSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PolynomialKernelSelector has not been fitted.");

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
