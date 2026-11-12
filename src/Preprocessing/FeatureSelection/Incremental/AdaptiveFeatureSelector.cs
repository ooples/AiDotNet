using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Incremental;

/// <summary>
/// Adaptive Feature Selector that adjusts selection based on performance.
/// </summary>
/// <remarks>
/// <para>
/// This selector adapts its feature selection over time based on observed
/// performance. Features that consistently perform well are retained, while
/// poor performers are replaced with alternatives.
/// </para>
/// <para><b>For Beginners:</b> Think of this like a sports team manager who
/// keeps track of how well each player performs. Players who consistently
/// do well stay on the team, while underperformers get replaced. Over time,
/// this builds an optimal team (set of features).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AdaptiveFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _adaptationRate;
    private readonly int _minSamplesForAdaptation;

    private Vector<T>? _performanceScores;
    private int _sampleCount;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double AdaptationRate => _adaptationRate;
    public Vector<T>? PerformanceScores => _performanceScores;
    public int SampleCount => _sampleCount;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public AdaptiveFeatureSelector(
        int nFeaturesToSelect = 10,
        double adaptationRate = 0.1,
        int minSamplesForAdaptation = 50,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (adaptationRate <= 0 || adaptationRate > 1)
            throw new ArgumentException("Adaptation rate must be between 0 and 1.", nameof(adaptationRate));

        _nFeaturesToSelect = nFeaturesToSelect;
        _adaptationRate = adaptationRate;
        _minSamplesForAdaptation = minSamplesForAdaptation;
        _sampleCount = 0;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AdaptiveFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        Reset();
        PartialFit(data, target);
    }

    public void PartialFit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        int n = data.Rows;
        int p = data.Columns;

        // Initialize on first call
        bool first = _performanceScores is null;
        if (first)
            _nInputFeatures = p;

        // Compute performance scores for this batch
        var batchScores = ComputeBatchScores(data, target, n, p);

        // Update performance scores with exponential moving average
        if (_sampleCount == 0 || _performanceScores is null)
        {
            _performanceScores = batchScores.ToVector();
        }
        else
        {
            var engine = Engine;
            var previous = Tensor<T>.FromVector(_performanceScores);
            _performanceScores = engine.TensorAdd(
                engine.TensorMultiplyScalar(previous, NumOps.FromDouble(1 - _adaptationRate)),
                engine.TensorMultiplyScalar(batchScores, NumOps.FromDouble(_adaptationRate))).ToVector();
        }

        _sampleCount += n;

        // Update selection
        UpdateSelection(p);
    }

    /// <summary>|Pearson correlation| of every feature with the target over one batch (0 when undefined).</summary>
    private Tensor<T> ComputeBatchScores(Matrix<T> data, Vector<T> target, int n, int p) =>
        FeatureSelectionInformationOps.AbsCorrelationWithTarget(
            Tensor<T>.FromMatrix(data), Engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }), NumOps.Zero);

    private void UpdateSelection(int p)
    {
        if (_performanceScores is null)
            return;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(Tensor<T>.FromVector(_performanceScores), numToSelect);

        IsFitted = true;
    }

    public void Reset()
    {
        _performanceScores = null;
        _sampleCount = 0;
        _selectedIndices = null;
        IsFitted = false;
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AdaptiveFeatureSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AdaptiveFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AdaptiveFeatureSelector has not been fitted.");

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
