using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Quantile;

/// <summary>
/// Quantile Dispersion (QCD) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the Quartile Coefficient of Dispersion,
/// a robust measure of relative variability: QCD = (Q3 - Q1) / (Q3 + Q1)
/// </para>
/// <para><b>For Beginners:</b> QCD measures spread relative to the central location,
/// similar to coefficient of variation but using quartiles instead of mean/std.
/// It's robust to outliers and works well when data contains extreme values.
/// Values range from 0 (no spread) to 1 (maximum relative spread).
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class QuantileDispersionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _qcdValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? QCDValues => _qcdValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public QuantileDispersionSelector(
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
        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);
        var q1 = FilterStatisticTensorOps.ColumnQuantile(x, 0.25);
        var q3 = FilterStatisticTensorOps.ColumnQuantile(x, 0.75);

        // QCD = (Q3 - Q1) / |Q3 + Q1|; where Q1 and Q3 cancel out (|Q3 + Q1| < 1e-10), the IQR itself
        var iqr = engine.TensorSubtract(q3, q1);
        var absSum = engine.TensorAbs(engine.TensorAdd(q3, q1));
        var cancels = engine.TensorLessThan(absSum, NumOps.FromDouble(1e-10));
        var safeSum = engine.TensorWhere(cancels, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One), absSum);
        var qcd = engine.TensorWhere(cancels, iqr, engine.TensorDivide(iqr, safeSum));
        _qcdValues = qcd.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(qcd, numToSelect);

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("QuantileDispersionSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("QuantileDispersionSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("QuantileDispersionSelector has not been fitted.");

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
