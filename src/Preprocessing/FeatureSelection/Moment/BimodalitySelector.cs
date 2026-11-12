using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Moment;

/// <summary>
/// Bimodality Coefficient based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their bimodality coefficient, which indicates
/// whether a distribution has two distinct modes (peaks).
/// </para>
/// <para><b>For Beginners:</b> Bimodality measures if data has two separate groups.
/// A high bimodality coefficient (>0.555 for uniform threshold) suggests the feature
/// naturally separates data into two clusters, which can be useful for classification.
/// The formula is: BC = (skewness² + 1) / (kurtosis + 3 × (n-1)²/((n-2)(n-3))), with the bias-adjusted/n/// sample skewness and excess kurtosis (as in SAS and Pfister et al. 2013).
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BimodalitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _preferHighBimodality;

    private Vector<T>? _bimodalityValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool PreferHighBimodality => _preferHighBimodality;
    public Vector<T>? BimodalityValues => _bimodalityValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BimodalitySelector(
        int nFeaturesToSelect = 10,
        bool preferHighBimodality = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _preferHighBimodality = preferHighBimodality;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < 4)
            throw new ArgumentException("At least 4 samples are required for bimodality calculation.");

        var engine = Engine;
        var centered = FeatureSelectionTensorOps.CenterColumns(Tensor<T>.FromMatrix(data), out _, out var std);

        // Constant columns (std < 1e-10) get 0
        var constant = engine.TensorLessThan(std, NumOps.FromDouble(1e-10));
        var safeStd = engine.TensorWhere(constant, FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One), std);
        var z = engine.TensorDivide(centered, engine.Reshape(safeStd, new[] { 1, p }));
        var z2 = engine.TensorSquare(z);

        // Population (biased) skewness g1 and excess kurtosis g2
        var g1 = engine.ReduceMean(engine.TensorMultiply(z2, z), new[] { 0 }, false);
        var g2 = engine.TensorSubtractScalar(engine.ReduceMean(engine.TensorSquare(z2), new[] { 0 }, false), NumOps.FromDouble(3));

        // Sarle's bimodality coefficient uses the bias-adjusted sample skewness and excess kurtosis
        // (SAS; Pfister et al. 2013; scipy.stats.skew/kurtosis with bias=False):
        //   G1 = g1 * sqrt(n(n-1)) / (n-2),  G2 = (n-1) / ((n-2)(n-3)) * ((n+1) g2 + 6)
        //   BC = (G1² + 1) / (G2 + 3(n-1)² / ((n-2)(n-3)))
        double nd = n;
        var skewness = engine.TensorMultiplyScalar(g1, NumOps.FromDouble(Math.Sqrt(nd * (nd - 1)) / (nd - 2)));
        var kurtosis = engine.TensorMultiplyScalar(
            engine.TensorAddScalar(engine.TensorMultiplyScalar(g2, NumOps.FromDouble(nd + 1)), NumOps.FromDouble(6)),
            NumOps.FromDouble((nd - 1) / ((nd - 2) * (nd - 3))));
        double sampleCorrection = 3.0 * (nd - 1) * (nd - 1) / ((nd - 2) * (nd - 3));
        var denominator = engine.TensorAddScalar(kurtosis, NumOps.FromDouble(sampleCorrection));

        // Avoid division by zero or negative
        var nonPositive = FeatureSelectionTensorOps.OneMinus(engine.TensorGreaterThan(denominator, NumOps.Zero));
        denominator = engine.TensorWhere(nonPositive, FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.FromDouble(1e-10)), denominator);
        var coefficient = engine.TensorDivide(engine.TensorAddScalar(engine.TensorSquare(skewness), NumOps.One), denominator);
        var bimodality = engine.TensorWhere(constant, new Tensor<T>(new[] { p }), coefficient);
        _bimodalityValues = bimodality.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // High bimodality suggests two-group structure; low bimodality means more unimodal
        _selectedIndices = _preferHighBimodality
            ? FeatureSelectionTensorOps.TopKIndicesSorted(bimodality, numToSelect)
            : FeatureSelectionTensorOps.BottomKIndicesSorted(bimodality, numToSelect);

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
            throw new InvalidOperationException("BimodalitySelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BimodalitySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BimodalitySelector has not been fitted.");

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
