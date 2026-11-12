using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Incremental;

/// <summary>
/// Streaming Feature Selector for online/incremental learning.
/// </summary>
/// <remarks>
/// <para>
/// Maintains running statistics to update feature scores as new data arrives.
/// This enables feature selection on streaming data without storing all
/// historical samples.
/// </para>
/// <para><b>For Beginners:</b> When you can't store all your data (it's too big
/// or comes continuously), you need to update feature scores incrementally.
/// This method keeps running averages that update with each new sample,
/// allowing feature selection on endless data streams.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class StreamingFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _decayFactor;

    private Vector<T>? _runningMeanX;
    private Vector<T>? _runningMeanY;
    private Vector<T>? _runningMeanXY;
    private Vector<T>? _runningMeanX2;
    private T _runningMeanY2;
    private int _sampleCount;
    private Vector<T>? _correlationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double DecayFactor => _decayFactor;
    public Vector<T>? CorrelationScores => _correlationScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    public int SampleCount => _sampleCount;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public StreamingFeatureSelector(
        int nFeaturesToSelect = 10,
        double decayFactor = 0.99,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (decayFactor <= 0 || decayFactor > 1)
            throw new ArgumentException("Decay factor must be between 0 and 1.", nameof(decayFactor));

        _nFeaturesToSelect = nFeaturesToSelect;
        _decayFactor = decayFactor;
        _sampleCount = 0;
        _runningMeanY2 = NumOps.Zero;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "StreamingFeatureSelector requires target values. Use PartialFit or Fit with target.");
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
        var engine = Engine;

        // Initialize on first call
        if (_runningMeanX is null || _runningMeanY is null || _runningMeanXY is null || _runningMeanX2 is null)
        {
            _nInputFeatures = p;
            _runningMeanX = new Vector<T>(p);
            _runningMeanY = new Vector<T>(p);
            _runningMeanXY = new Vector<T>(p);
            _runningMeanX2 = new Vector<T>(p);
            _runningMeanY2 = NumOps.Zero;
            _correlationScores = new Vector<T>(p);
        }

        // Update running statistics with exponential decay. Sample i of the batch ends up weighted
        // (1 - decay) * decay^(n-1-i), and the previous statistics by decay^n, exactly as the per-sample
        // recurrence m = decay * m + (1 - decay) * value.
        if (n > 0)
        {
            T decay = NumOps.FromDouble(_decayFactor);
            var age = new int[n];
            for (int i = 0; i < n; i++)
                age[i] = n - 1 - i;
            var decayPowers = engine.TensorPower(
                PreprocessingTensorOps.Filled(new[] { 1, n }, decay), new Tensor<int>(age, new[] { 1, n }).Cast<T>());
            var weights = engine.TensorMultiplyScalar(decayPowers, NumOps.FromDouble(1 - _decayFactor)); // [1, n]
            T carry = engine.TensorSum(engine.TensorPower(
                PreprocessingTensorOps.Filled(new[] { 1 }, decay), PreprocessingTensorOps.Filled(new[] { 1 }, FeatureSelectionTensorOps.Count<T>(n))));

            var X = Tensor<T>.FromMatrix(data);
            var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
            var yBroadcast = engine.TensorBroadcastTo(y, new[] { n, p });

            _runningMeanX = Decay(_runningMeanX, carry, weights, X, p);
            _runningMeanY = Decay(_runningMeanY, carry, weights, yBroadcast, p);
            _runningMeanXY = Decay(_runningMeanXY, carry, weights, engine.TensorMultiply(X, yBroadcast), p);
            _runningMeanX2 = Decay(_runningMeanX2, carry, weights, engine.TensorSquare(X), p);
            _runningMeanY2 = NumOps.Add(NumOps.Multiply(carry, _runningMeanY2), engine.TensorSum(engine.TensorMatMul(weights, engine.TensorSquare(y))));
        }

        _sampleCount += n;

        UpdateScores(p);
    }

    /// <summary>carry * previous + weights * values, per feature ([p]).</summary>
    private Vector<T> Decay(Vector<T> previous, T carry, Tensor<T> weights, Tensor<T> values, int p)
    {
        var engine = Engine;
        var decayed = engine.TensorMultiplyScalar(Tensor<T>.FromVector(previous), carry);
        return engine.TensorAdd(decayed, engine.Reshape(engine.TensorMatMul(weights, values), new[] { p })).ToVector();
    }

    private void UpdateScores(int p)
    {
        if (_runningMeanX is null || _runningMeanY is null ||
            _runningMeanXY is null || _runningMeanX2 is null ||
            _correlationScores is null)
            return;

        var engine = Engine;
        var meanX = Tensor<T>.FromVector(_runningMeanX);
        var meanY = Tensor<T>.FromVector(_runningMeanY);
        var varX = engine.TensorSubtract(Tensor<T>.FromVector(_runningMeanX2), engine.TensorSquare(meanX));
        var varY = engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { p }, _runningMeanY2), engine.TensorSquare(meanY));
        var covXY = engine.TensorSubtract(Tensor<T>.FromVector(_runningMeanXY), engine.TensorMultiply(meanX, meanY));

        T threshold = NumOps.FromDouble(1e-10);
        var defined = engine.TensorMultiply(engine.TensorGreaterThan(varX, threshold), engine.TensorGreaterThan(varY, threshold));
        var ones = PreprocessingTensorOps.Filled(new[] { p }, NumOps.One);
        var safeVariance = engine.TensorWhere(defined, engine.TensorMultiply(varX, varY), ones);
        var scores = engine.TensorWhere(defined, engine.TensorAbs(engine.TensorDivide(covXY, engine.TensorSqrt(safeVariance))), new Tensor<T>(new[] { p }));
        _correlationScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    public void Reset()
    {
        _runningMeanX = null;
        _runningMeanY = null;
        _runningMeanXY = null;
        _runningMeanX2 = null;
        _runningMeanY2 = NumOps.Zero;
        _sampleCount = 0;
        _correlationScores = null;
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
            throw new InvalidOperationException("StreamingFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("StreamingFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StreamingFeatureSelector has not been fitted.");

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
