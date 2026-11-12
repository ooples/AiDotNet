using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;

/// <summary>
/// Distance Correlation-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Distance correlation measures the dependence between random variables, detecting
/// both linear and nonlinear relationships. Unlike Pearson correlation, it equals
/// zero if and only if the variables are independent.
/// </para>
/// <para><b>For Beginners:</b> Regular correlation only catches straight-line
/// relationships. Distance correlation can detect curved, U-shaped, or any type
/// of pattern between variables. A distance correlation of 0 means truly no
/// relationship at all.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DistanceCorrelation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _distCorrelations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? DistCorrelations => _distCorrelations;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DistanceCorrelation(
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
            "DistanceCorrelation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Distance correlation dCor = sqrt(dCov^2 / sqrt(dVar^2(x) dVar^2(y))), 0 unless both variances exceed 1e-10
        var dcov2 = FeatureSelectionTensorOps.SquaredDistanceCovariances(
            Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target), out var xVariances, out T yVariance);
        var zeros = new Tensor<T>(new[] { p });
        T epsilon = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var usable = NumOps.GreaterThan(yVariance, epsilon)
            ? engine.TensorGreaterThan(xVariances, epsilon)
            : zeros;
        var denominators = engine.TensorSqrt(engine.TensorMultiplyScalar(
            engine.TensorWhere(usable, xVariances, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One)), yVariance));
        var distCorrelations = engine.TensorWhere(usable,
            engine.TensorSqrt(engine.TensorDivide(dcov2, denominators)), zeros);
        _distCorrelations = distCorrelations.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(distCorrelations, numToSelect);

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
            throw new InvalidOperationException("DistanceCorrelation has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DistanceCorrelation does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DistanceCorrelation has not been fitted.");

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
