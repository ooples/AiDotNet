using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// Independent Component Analysis based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their contribution to independent components,
/// identifying features that contribute to statistically independent signals.
/// </para>
/// <para><b>For Beginners:</b> ICA separates mixed signals into independent sources
/// (like separating individual voices from a recording). This selector finds
/// features that contribute most to these independent signals.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class IndependentComponentSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nComponents;
    private readonly int? _randomState;

    private Vector<T>? _independenceScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NComponents => _nComponents;
    public Vector<T>? IndependenceScores => _independenceScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public IndependentComponentSelector(
        int nFeaturesToSelect = 10,
        int nComponents = 5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nComponents = nComponents;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Center the data
        var raw = Tensor<T>.FromMatrix(data);
        var x = engine.TensorSubtract(raw, engine.TensorBroadcastTo(engine.ReduceMean(raw, new[] { 0 }, true), new[] { n, p }));

        // Whitening via PCA: FastICA (Hyvarinen 1999) runs on whitened data
        int numComponents = Math.Min(_nComponents, Math.Min(n, p));
        // (directions without variance are dropped, so the whitened dimension d can be below numComponents)
        var z = FastIcaOps.Whiten(x, numComponents, out var whitening);
        int d = z.Shape[1];

        // Initialize random mixing matrix [d, d] with draws in [-0.5, 0.5), then deflationary FastICA. Constant
        // data has no component, and every feature scores 0.
        Tensor<T> scores;
        if (d == 0)
        {
            scores = new Tensor<T>(new[] { p });
        }
        else
        {
            var initial = PopulationSearchOps.Affine(PopulationSearchOps.Uniform<T>(rand, d, d), 1, -0.5);
            var unmixing = engine.TensorMatMul(whitening, FastIcaOps.Deflation(z, initial, 100, 1e-6));

            // Score features by their total contribution to independent components
            scores = engine.ReduceSum(engine.TensorAbs(unmixing), new[] { 1 }, false);
        }
        _independenceScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

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
            throw new InvalidOperationException("IndependentComponentSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("IndependentComponentSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("IndependentComponentSelector has not been fitted.");

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
