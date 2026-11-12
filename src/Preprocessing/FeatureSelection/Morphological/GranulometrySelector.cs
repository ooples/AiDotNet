using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Morphological;

/// <summary>
/// Granulometry based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on granulometric analysis, measuring how features respond
/// to morphological openings of increasing size.
/// </para>
/// <para><b>For Beginners:</b> Granulometry measures the size distribution of structures
/// in data by applying progressively larger opening operations. Features with interesting
/// size distributions (neither too smooth nor too noisy) are selected.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GranulometrySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxWindowSize;

    private Vector<T>? _granulometryScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxWindowSize => _maxWindowSize;
    public Vector<T>? GranulometryScores => _granulometryScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GranulometrySelector(
        int nFeaturesToSelect = 10,
        int maxWindowSize = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxWindowSize = maxWindowSize;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Columns as rows ([p, n]) so the structuring elements slide along the samples
        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);
        var columns = engine.TensorTranspose(x);

        // Normalize each column to [0, 1] (columns whose range is not above 1e-10 stay as they are)
        var min = engine.Reshape(FeatureSelectionTensorOps.ColumnMin(x), new[] { p, 1 });
        var max = engine.Reshape(FeatureSelectionTensorOps.ColumnMax(x), new[] { p, 1 });
        var range = engine.TensorSubtract(max, min);
        var varying = engine.TensorBroadcastTo(engine.TensorGreaterThan(range, NumOps.FromDouble(1e-10)), new[] { p, n });
        var safeRange = engine.TensorWhere(
            engine.TensorGreaterThan(range, NumOps.FromDouble(1e-10)), range, FeatureSelectionTensorOps.Filled(new[] { p, 1 }, NumOps.One));
        var normalized = engine.TensorWhere(varying, engine.TensorDivide(engine.TensorSubtract(columns, min), safeRange), columns);

        // Pattern spectrum: the mass each opening (erosion then dilation) of growing size removes
        var spectrum = new List<Tensor<T>>();
        var previousSum = engine.ReduceSum(normalized, new[] { 1 }, false);
        for (int ws = 1; ws <= _maxWindowSize; ws++)
        {
            int halfWindow = ws / 2;
            var opened = FeatureSelectionTensorOps.SlidingMax(FeatureSelectionTensorOps.SlidingMin(normalized, halfWindow), halfWindow);
            var currentSum = engine.ReduceSum(opened, new[] { 1 }, false);
            spectrum.Add(engine.TensorSubtract(previousSum, currentSum));
            previousSum = currentSum;
        }

        // Score: entropy (bits) of the pattern spectrum; a diverse size distribution is interesting
        var scores = new Tensor<T>(new[] { p });
        if (spectrum.Count > 0)
        {
            var patternSpectrum = engine.TensorStack(spectrum.ToArray(), 1); // [p, sizes]
            var total = engine.ReduceSum(patternSpectrum, new[] { 1 }, true);
            var hasMass = engine.TensorGreaterThan(total, NumOps.FromDouble(1e-10));
            var safeTotal = engine.TensorWhere(hasMass, total, FeatureSelectionTensorOps.Filled(new[] { p, 1 }, NumOps.One));
            var prob = engine.TensorDivide(patternSpectrum, safeTotal);
            var counted = engine.TensorGreaterThan(prob, NumOps.FromDouble(1e-10));
            var safeProb = engine.TensorWhere(counted, prob, FeatureSelectionTensorOps.Filled(prob.Shape.ToArray(), NumOps.One));
            var terms = engine.TensorWhere(
                counted,
                engine.TensorDivideScalar(engine.TensorMultiply(prob, engine.TensorLog(safeProb)), NumOps.FromDouble(-Math.Log(2))),
                new Tensor<T>(prob.Shape.ToArray()));
            var entropy = engine.ReduceSum(terms, new[] { 1 }, false);
            scores = engine.TensorWhere(engine.Reshape(hasMass, new[] { p }), entropy, scores);
        }

        _granulometryScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(scores, numToSelect);

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
            throw new InvalidOperationException("GranulometrySelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GranulometrySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GranulometrySelector has not been fitted.");

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
