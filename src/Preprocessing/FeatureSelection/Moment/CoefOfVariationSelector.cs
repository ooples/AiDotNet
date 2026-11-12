using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Moment;

/// <summary>
/// Coefficient of Variation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their coefficient of variation (CV), which measures
/// the relative variability compared to the mean.
/// </para>
/// <para><b>For Beginners:</b> CV is the ratio of standard deviation to the mean.
/// Features with higher CV have more relative variability, which often indicates
/// they carry useful information for distinguishing between samples.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CoefOfVariationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _preferHighCV;

    private Vector<T>? _cvValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool PreferHighCV => _preferHighCV;
    public Vector<T>? CVValues => _cvValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public CoefOfVariationSelector(
        int nFeaturesToSelect = 10,
        bool preferHighCV = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _preferHighCV = preferHighCV;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n == 0)
            throw new InvalidOperationException("At least one sample is required to compute the coefficient of variation.");

        var engine = Engine;
        FeatureSelectionTensorOps.CenterColumns(Tensor<T>.FromMatrix(data), out var mean, out var std);

        // CV = std / |mean| (absolute mean handles negative values); 0 where |mean| <= 1e-10
        var absMean = engine.TensorAbs(engine.Reshape(mean, new[] { p }));
        var defined = engine.TensorGreaterThan(absMean, NumOps.FromDouble(1e-10));
        var safeMean = engine.TensorWhere(defined, absMean, FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One));
        var cv = engine.TensorWhere(defined, engine.TensorDivide(std, safeMean), new Tensor<T>(new[] { p }));
        _cvValues = cv.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        if (_preferHighCV)
        {
            _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(cv, numToSelect);
        }
        else
        {
            // Lowest CV among the features whose CV is above 1e-10; the others rank last and are never taken
            var candidates = engine.TensorGreaterThan(cv, NumOps.FromDouble(1e-10));
            int candidateCount = FeatureSelectionTensorOps.CountNonZero(candidates);
            T beyond = p == 0 ? NumOps.One : NumOps.Add(engine.TensorMaxValue(cv), NumOps.One);
            var ranked = engine.TensorWhere(candidates, cv, FeatureSelectionTensorOps.Filled(new[] { p }, beyond));
            _selectedIndices = FeatureSelectionTensorOps.BottomKIndicesSorted(ranked, Math.Min(numToSelect, candidateCount));
        }

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
            throw new InvalidOperationException("CoefOfVariationSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CoefOfVariationSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CoefOfVariationSelector has not been fitted.");

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
