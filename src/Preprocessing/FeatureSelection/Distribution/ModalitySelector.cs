using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Distribution;

/// <summary>
/// Modality based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the estimated number of modes (peaks) in their
/// distributions, using a histogram-based peak detection approach.
/// </para>
/// <para><b>For Beginners:</b> Modality refers to how many peaks a distribution has.
/// Unimodal = 1 peak, bimodal = 2 peaks, multimodal = many peaks. Features with
/// multiple modes often indicate natural groupings or distinct populations in the data.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ModalitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly int _targetModality;

    private int[]? _modalityCounts;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public int TargetModality => _targetModality;
    public int[]? ModalityCounts => _modalityCounts;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ModalitySelector(
        int nFeaturesToSelect = 10,
        int nBins = 30,
        int targetModality = 2,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 3)
            throw new ArgumentException("Number of bins must be at least 3.", nameof(nBins));
        if (targetModality < 1)
            throw new ArgumentException("Target modality must be at least 1.", nameof(targetModality));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _targetModality = targetModality;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Create histograms of every feature
        var binCounts = FeatureSelectionTensorOps.RangeBinCounts(X, _nBins, out var constant);

        // Smooth histogram with simple moving average (window of 3, shrunk at the edges)
        var zeroColumn = new Tensor<T>(new[] { p, 1 });
        var left = engine.TensorConcatenate(new[] { zeroColumn, engine.TensorSlice(binCounts, new[] { 0, 0 }, new[] { p, _nBins - 1 }) }, 1);
        var right = engine.TensorConcatenate(new[] { engine.TensorSlice(binCounts, new[] { 0, 1 }, new[] { p, _nBins - 1 }), zeroColumn }, 1);
        var window = Enumerable.Repeat(3, _nBins).ToArray();
        window[0] = 2;
        window[_nBins - 1] = 2;
        var smoothed = engine.TensorDivide(
            engine.TensorAdd(engine.TensorAdd(left, binCounts), right),
            engine.TensorBroadcastTo(new Tensor<int>(window, new[] { 1, _nBins }).Cast<T>(), new[] { p, _nBins }));

        // Count peaks (local maxima)
        int interior = _nBins - 2;
        var center = engine.TensorSlice(smoothed, new[] { 0, 1 }, new[] { p, interior });
        var isPeak = engine.TensorMultiply(engine.TensorMultiply(
            engine.TensorGreaterThan(center, engine.TensorSlice(smoothed, new[] { 0, 0 }, new[] { p, interior })),
            engine.TensorGreaterThan(center, engine.TensorSlice(smoothed, new[] { 0, 2 }, new[] { p, interior }))),
            engine.TensorGreaterThan(center, NumOps.Zero));
        var peaks = engine.ReduceSum(isPeak, new[] { 1 }, false);

        // Handle edge cases: no peak in a non-empty histogram counts as one; constant = unimodal
        var nonEmpty = engine.TensorGreaterThan(engine.ReduceMax(smoothed, new[] { 1 }, false), NumOps.Zero);
        var one = PreprocessingTensorOps.Filled(new[] { p }, NumOps.One);
        peaks = engine.TensorWhere(engine.TensorMultiply(engine.TensorEquals(peaks, NumOps.Zero), nonEmpty), one, peaks);
        peaks = engine.TensorWhere(constant, one, peaks);
        _modalityCounts = peaks.Cast<int>().AsSpan().ToArray();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // Select features closest to target modality
        _selectedIndices = Enumerable.Range(0, p)
            .OrderBy(j => Math.Abs(_modalityCounts[j] - _targetModality))
            .ThenByDescending(j => _modalityCounts[j]) // Tie-break: prefer higher modality
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

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
            throw new InvalidOperationException("ModalitySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ModalitySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ModalitySelector has not been fitted.");

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
