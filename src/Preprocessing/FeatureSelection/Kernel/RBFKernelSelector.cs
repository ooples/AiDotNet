using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// RBF Kernel based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their importance in an RBF (Radial Basis Function) kernel
/// space, measuring nonlinear relationships with the target.
/// </para>
/// <para><b>For Beginners:</b> The RBF kernel measures similarity based on distance.
/// Points close together have high similarity. This selector finds features that
/// create meaningful similarity patterns with respect to the target.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RBFKernelSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _gamma;

    private Vector<T>? _kernelScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Gamma => _gamma;
    public Vector<T>? KernelScores => _kernelScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public RBFKernelSelector(
        int nFeaturesToSelect = 10,
        double gamma = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _gamma = gamma;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RBFKernelSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Target similarity of each pair i < j: +1 when |y_i - y_j| < 0.5, otherwise -1
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
        var targetGap = engine.TensorAbs(engine.TensorSubtract(
            engine.TensorBroadcastTo(y, new[] { n, n }), engine.TensorBroadcastTo(engine.TensorTranspose(y), new[] { n, n })));
        var similarity = engine.TensorSubtractScalar(
            engine.TensorMultiplyScalar(engine.TensorLessThan(targetGap, NumOps.FromDouble(0.5)), NumOps.FromDouble(2)), NumOps.One);
        var upper = FeatureSelectionTensorOps.StrictUpperMask<T>(n);

        // Kernel-target alignment of each feature's RBF kernel: sum K S / sqrt(sum K^2) over the pairs i < j
        T negativeGamma = NumOps.FromDouble(-_gamma);
        var sums = FeatureSelectionTensorOps.KernelWeightedSums(
            standardized,
            column => FeatureSelectionTensorOps.RbfColumnKernel(column, negativeGamma),
            new[] { engine.TensorMultiply(upper, similarity) },
            new[] { upper });
        var numerator = engine.Reshape(engine.TensorSlice(sums, new[] { 0, 0 }, new[] { p, 1 }), new[] { p });
        var denominator = engine.Reshape(engine.TensorSlice(sums, new[] { 0, 1 }, new[] { p, 1 }), new[] { p });
        var defined = engine.TensorGreaterThan(denominator, NumOps.FromDouble(1e-10));
        var safeDenominator = engine.TensorWhere(defined, denominator, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        var scores = engine.TensorWhere(defined, engine.TensorDivide(numerator, engine.TensorSqrt(safeDenominator)), new Tensor<T>(new[] { p }));
        _kernelScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(engine.TensorAbs(scores), numToSelect);

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
            throw new InvalidOperationException("RBFKernelSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RBFKernelSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RBFKernelSelector has not been fitted.");

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
