using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Coefficient of Variation (CV) for unsupervised feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The coefficient of variation is the ratio of standard deviation to mean,
/// expressed as a percentage. Features with low CV have relatively little
/// variation and may not be informative.
/// </para>
/// <para><b>For Beginners:</b> Some features barely change across samples (low CV),
/// while others vary a lot (high CV). Features that don't change much probably
/// can't distinguish between different outcomes. CV is useful when features are
/// on different scales, because it measures relative variation.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CoefficientOfVariation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minCV;

    private Vector<T>? _cvValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? CVValues => _cvValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public CoefficientOfVariation(
        int nFeaturesToSelect = 10,
        double minCV = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minCV = minCV;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Coefficient of variation std / |mean| (population std; 0 for a near-zero mean)
        var absMeans = engine.TensorAbs(engine.ReduceMean(X, new[] { 0 }, false));
        var usable = engine.TensorGreaterThan(absMeans, PreprocessingTensorOps.Tolerance<T>(1e-10));
        var cvValues = engine.TensorWhere(usable, engine.TensorDivide(FeatureSelectionTensorOps.ColumnStd(X),
            engine.TensorWhere(usable, absMeans, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One))),
            new Tensor<T>(new[] { p }));
        _cvValues = cvValues.ToVector();

        // Select features above threshold or top by CV
        var candidates = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
            PreprocessingTensorOps.AtLeast(cvValues, NumOps.FromDouble(_minCV))));

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = PreprocessingTensorOps.StableOrder(cvValues, descending: true)
                .Where(j => candidates.Contains(j))
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(cvValues, Math.Min(_nFeaturesToSelect, p));
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CoefficientOfVariation has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CoefficientOfVariation does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CoefficientOfVariation has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
