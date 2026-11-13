using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Moment;

/// <summary>
/// Kurtosis based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their kurtosis values, identifying features
/// with unusual tail behavior in their distributions.
/// </para>
/// <para><b>For Beginners:</b> Kurtosis measures how heavy the tails of a distribution
/// are. High kurtosis means more extreme values (outliers), while low kurtosis means
/// fewer extremes. Features with interesting kurtosis often capture important patterns.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KurtosisSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _preferHighKurtosis;

    private Vector<T>? _kurtosisValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool PreferHighKurtosis => _preferHighKurtosis;
    public Vector<T>? KurtosisValues => _kurtosisValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public KurtosisSelector(
        int nFeaturesToSelect = 10,
        bool preferHighKurtosis = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _preferHighKurtosis = preferHighKurtosis;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n == 0)
            throw new InvalidOperationException("At least one sample is required to compute kurtosis.");

        var engine = Engine;
        var centered = FilterStatisticTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var std = FeatureSelectionTensorOps.CenteredColumnStd(centered);

        // Constant columns (std < 1e-10) get 0; the others get the excess kurtosis E[(X-μ)⁴] / σ⁴ - 3
        // (Fisher's definition, scipy.stats.kurtosis(fisher=True, bias=True)).
        var constant = engine.TensorLessThan(std, NumOps.FromDouble(1e-10));
        var safeStd = engine.TensorWhere(constant, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One), std);
        var z = engine.TensorDivide(centered, engine.Reshape(safeStd, new[] { 1, p }));
        var moment4 = engine.ReduceMean(engine.TensorSquare(engine.TensorSquare(z)), new[] { 0 }, false);
        var excess = engine.TensorSubtractScalar(moment4, NumOps.FromDouble(3));
        var kurtosis = engine.TensorWhere(constant, new Tensor<T>(new[] { p }), excess);
        _kurtosisValues = kurtosis.ToVector();

        // Select features based on preference: high absolute excess kurtosis, or kurtosis close to normal
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        var magnitude = engine.TensorAbs(kurtosis);
        _selectedIndices = _preferHighKurtosis
            ? PreprocessingTensorOps.TopKIndicesSorted(magnitude, numToSelect)
            : PreprocessingTensorOps.BottomKIndicesSorted(magnitude, numToSelect);

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
            throw new InvalidOperationException("KurtosisSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KurtosisSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KurtosisSelector has not been fitted.");

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
