using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// Independent Component Analysis (ICA) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses ICA to find independent components and selects features that
/// contribute most to these statistically independent sources.
/// </para>
/// <para><b>For Beginners:</b> ICA finds hidden "sources" that combine to create
/// your observed data (like separating mixed audio signals). Features that
/// strongly contribute to these independent sources capture unique, non-redundant
/// information and are good candidates for selection.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ICABasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nComponents;
    private readonly int _nIterations;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ICABasedSelector(
        int nFeaturesToSelect = 10,
        int nComponents = 5,
        int nIterations = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nComponents = nComponents;
        _nIterations = nIterations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        // Center data
        var raw = Tensor<T>.FromMatrix(data);
        var x = engine.TensorSubtract(raw, engine.TensorBroadcastTo(engine.ReduceMean(raw, new[] { 0 }, true), new[] { n, p }));

        int k = Math.Min(_nComponents, Math.Min(n, p));
        var rand = RandomHelper.CreateSecureRandom();

        // Whitening: FastICA (Hyvarinen 1999) runs on PCA-whitened data
        var z = FastIcaOps.Whiten(x, k, out var whitening);

        // Deflationary FastICA; component c starts from uniform draws in [-0.5, 0.5)
        var initial = engine.TensorTranspose(PopulationSearchOps.Affine(PopulationSearchOps.Uniform<T>(rand, k, k), 1, -0.5));
        var unmixing = engine.TensorMatMul(whitening, FastIcaOps.Deflation(z, initial, _nIterations, 0));

        // Feature scores = sum of absolute unmixing weights (in the original feature space) across components
        var scores = engine.ReduceSum(engine.TensorAbs(unmixing), new[] { 1 }, false);
        _featureScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ICABasedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ICABasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ICABasedSelector has not been fitted.");

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
