using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Distribution;

/// <summary>
/// Uniformity based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on how uniformly distributed their values are,
/// using entropy as a measure of uniformity.
/// </para>
/// <para><b>For Beginners:</b> A uniform distribution has equal probability for
/// all values in a range. This selector measures how close each feature is to
/// being uniformly distributed. High uniformity means values are evenly spread out,
/// while low uniformity means some values are much more common than others.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class UniformitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly bool _preferUniform;

    private Vector<T>? _uniformityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public bool PreferUniform => _preferUniform;
    public Vector<T>? UniformityScores => _uniformityScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public UniformitySelector(
        int nFeaturesToSelect = 10,
        int nBins = 20,
        bool preferUniform = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _preferUniform = preferUniform;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Maximum entropy for uniform distribution with _nBins bins
        double maxEntropy = Math.Log(_nBins) / Math.Log(2);

        // Histogram entropy (bits) of every feature; constant features have zero uniformity
        var binCounts = FeatureScoringTensorOps.RangeBinCounts(X, _nBins, out var constant);
        var entropy = FeatureScoringTensorOps.EntropyBits(binCounts, n);

        // Uniformity score = entropy / max_entropy (normalized)
        var zeros = new Tensor<T>(new[] { p });
        var scores = maxEntropy > 0
            ? engine.TensorWhere(constant, zeros, engine.TensorDivideScalar(entropy, NumOps.FromDouble(maxEntropy)))
            : zeros;
        _uniformityScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        if (_preferUniform)
        {
            // Prefer features with high uniformity
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);
        }
        else
        {
            // Prefer features with low uniformity (more concentrated), among those with positive uniformity
            var positive = new HashSet<int>(PreprocessingTensorOps.MaskPositions(engine.TensorGreaterThan(scores, NumOps.Zero)));
            _selectedIndices = PreprocessingTensorOps.StableOrder(scores, descending: false)
                .Where(j => positive.Contains(j))
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
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
            throw new InvalidOperationException("UniformitySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("UniformitySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("UniformitySelector has not been fitted.");

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
