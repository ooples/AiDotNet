using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Moment;

/// <summary>
/// Skewness based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their skewness values, identifying features
/// with asymmetric distributions that may capture important patterns.
/// </para>
/// <para><b>For Beginners:</b> Skewness measures how lopsided a distribution is.
/// Positive skewness means a long right tail, negative skewness means a long
/// left tail. Features with moderate skewness often contain useful information.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SkewnessSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _preferHighSkewness;

    private Vector<T>? _skewnessValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool PreferHighSkewness => _preferHighSkewness;
    public Vector<T>? SkewnessValues => _skewnessValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SkewnessSelector(
        int nFeaturesToSelect = 10,
        bool preferHighSkewness = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _preferHighSkewness = preferHighSkewness;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n == 0)
            throw new InvalidOperationException("At least one sample is required to compute skewness.");

        var engine = Engine;
        var centered = FeatureSelectionTensorOps.CenterColumns(Tensor<T>.FromMatrix(data), out _, out var std);

        // Constant columns (std < 1e-10) get 0; the others get the skewness E[(X-μ)³] / σ³
        // (scipy.stats.skew(bias=True)).
        var constant = engine.TensorLessThan(std, NumOps.FromDouble(1e-10));
        var safeStd = engine.TensorWhere(constant, FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One), std);
        var z = engine.TensorDivide(centered, engine.Reshape(safeStd, new[] { 1, p }));
        var moment3 = engine.ReduceMean(engine.TensorMultiply(engine.TensorSquare(z), z), new[] { 0 }, false);
        var skewness = engine.TensorWhere(constant, new Tensor<T>(new[] { p }), moment3);
        _skewnessValues = skewness.ToVector();

        // Select features based on preference: high absolute skewness, or low skewness (more symmetric)
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        var magnitude = engine.TensorAbs(skewness);
        _selectedIndices = _preferHighSkewness
            ? FeatureSelectionTensorOps.TopKIndicesSorted(magnitude, numToSelect)
            : FeatureSelectionTensorOps.BottomKIndicesSorted(magnitude, numToSelect);

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
            throw new InvalidOperationException("SkewnessSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SkewnessSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SkewnessSelector has not been fitted.");

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
