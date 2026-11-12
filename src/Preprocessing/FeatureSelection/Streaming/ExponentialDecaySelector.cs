using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Streaming;

/// <summary>
/// Exponential Decay Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses exponentially weighted statistics for feature selection, giving more
/// importance to recent samples while gradually forgetting older ones.
/// </para>
/// <para><b>For Beginners:</b> Unlike a sharp sliding window, exponential decay
/// smoothly reduces the importance of older samples. Recent data has the most
/// influence, but older data still contributes a little. This creates a smooth
/// transition as the data distribution changes.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ExponentialDecaySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _decayFactor;

    // Rows of block-wise EWMA processing (the closed form builds a [BlockSize, BlockSize] decay matrix)
    private const int BlockSize = 256;

    private Tensor<T>? _ewmaMeans;
    private Tensor<T>? _ewmaVars;
    private Tensor<T>? _ewmaCovariances;
    private double _effectiveN;
    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double DecayFactor => _decayFactor;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ExponentialDecaySelector(
        int nFeaturesToSelect = 10,
        double decayFactor = 0.99,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (decayFactor <= 0 || decayFactor >= 1)
            throw new ArgumentException("Decay factor must be between 0 and 1.", nameof(decayFactor));

        _nFeaturesToSelect = nFeaturesToSelect;
        _decayFactor = decayFactor;
        _effectiveN = 0;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ExponentialDecaySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        int n = data.Rows;
        int p = data.Columns;
        _nInputFeatures = p;

        // Initialize if needed: EWMA means and variances of the features and (last column) the target, and the
        // feature-target covariances, all as [1, width] rows
        if (_ewmaMeans is null)
        {
            _ewmaMeans = new Tensor<T>(new[] { 1, p + 1 });
            _ewmaVars = new Tensor<T>(new[] { 1, p + 1 });
            _ewmaCovariances = new Tensor<T>(new[] { 1, p });
        }

        // Process samples with exponential decay, one block of rows at a time
        var ewmaMeans = _ewmaMeans ?? throw new InvalidOperationException("EWMA means have not been initialized.");
        var ewmaVars = _ewmaVars ?? throw new InvalidOperationException("EWMA variances have not been initialized.");
        var ewmaCovariances = _ewmaCovariances ?? throw new InvalidOperationException("EWMA covariances have not been initialized.");

        var columns = Engine.TensorConcatenate(new[]
        {
            Tensor<T>.FromMatrix(data), Engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 })
        }, 1);
        for (int start = 0; start < n; start += BlockSize)
        {
            int rows = Math.Min(BlockSize, n - start);
            var block = Engine.TensorSlice(columns, new[] { start, 0 }, new[] { rows, p + 1 });
            var updated = UpdateBlock(block, ewmaMeans, ewmaVars, ewmaCovariances);
            ewmaMeans = updated.Means;
            ewmaVars = updated.Vars;
            ewmaCovariances = updated.Covariances;
        }

        _ewmaMeans = ewmaMeans;
        _ewmaVars = ewmaVars;
        _ewmaCovariances = ewmaCovariances;

        // Update effective sample count: n steps of N <- decay N + 1
        double decayN = Math.Pow(_decayFactor, n);
        _effectiveN = decayN * _effectiveN + (1 - decayN) / (1 - _decayFactor);

        // Compute correlation scores, 0 unless both variances are above 1e-10
        T eps = NumOps.FromDouble(1e-10);
        var xVar = Engine.Reshape(Engine.TensorSlice(ewmaVars, new[] { 0, 0 }, new[] { 1, p }), new[] { p });
        T yVar = Engine.TensorSlice(ewmaVars, new[] { 0, p }, new[] { 1, 1 }).GetFlat(0);
        var scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (NumOps.GreaterThan(yVar, eps))
        {
            var usable = Engine.TensorGreaterThan(xVar, eps);
            var safeVar = Engine.TensorWhere(usable, xVar, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
            scores = SelectorTensorOps.ZeroUnless(usable, Engine.TensorAbs(Engine.TensorDivide(
                Engine.Reshape(ewmaCovariances, new[] { p }), Engine.TensorSqrt(Engine.TensorMultiplyScalar(safeVar, yVar)))));
        }

        _featureScores = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <summary>
    /// Applies the EWMA recurrences to a block of rows ([rows, p + 1], target last) in closed form:
    /// <c>m_i = d m_(i-1) + (1 - d) x_i</c>, <c>v_i = d v_(i-1) + (1 - d) (x_i - m_(i-1)) (x_i - m_i)</c> and
    /// <c>c_i = d c_(i-1) + (1 - d) (x_i - m_i) (y_i - my_i)</c>, using <c>x_i - m_i = d (x_i - m_(i-1))</c>.
    /// </summary>
    private (Tensor<T> Means, Tensor<T> Vars, Tensor<T> Covariances) UpdateBlock(
        Tensor<T> block, Tensor<T> means, Tensor<T> vars, Tensor<T> covariances)
    {
        int rows = block.Shape[0];
        int width = block.Shape[1];
        int p = width - 1;
        var shape = new[] { rows, width };
        T d = NumOps.FromDouble(_decayFactor);
        T oneMinusD = NumOps.FromDouble(1 - _decayFactor);
        var index = Engine.TensorArange<T>(0, rows, 1);

        // Running means: m_i = d^(i+1) m_(-1) + (1 - d) sum_(k <= i) d^(i-k) x_k
        var lag = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(Engine.Reshape(index, new[] { rows, 1 }), new[] { rows, rows }),
            Engine.TensorBroadcastTo(Engine.Reshape(index, new[] { 1, rows }), new[] { rows, rows }));
        var weights = Engine.TensorMultiplyScalar(
            Engine.TensorTril(DecayPowers(Engine.TensorMax(lag, NumOps.Zero)), 0), oneMinusD);
        var carry = DecayPowers(Engine.TensorAddScalar(Engine.Reshape(index, new[] { rows, 1 }), NumOps.One));
        var runningMeans = Engine.TensorAdd(Engine.TensorMatMul(weights, block), Engine.TensorMultiply(
            Engine.TensorBroadcastTo(carry, shape), Engine.TensorBroadcastTo(means, shape)));

        // Deviations from the mean before each row
        var previous = rows == 1
            ? means
            : Engine.TensorConcatenate(new[] { means, Engine.TensorSlice(runningMeans, new[] { 0, 0 }, new[] { rows - 1, width }) }, 0);
        var deviation = Engine.TensorSubtract(block, previous);

        // tail_i = d^(rows-1-i) folds the per-row increments into the end-of-block value
        var tail = Engine.Reshape(DecayPowers(Engine.TensorSubtract(
            PreprocessingTensorOps.Filled(new[] { rows }, NumOps.FromDouble(rows - 1)), index)), new[] { 1, rows });
        T decayRows = NumOps.FromDouble(Math.Pow(_decayFactor, rows));

        // v_end = d^rows v + (1 - d) d sum_i d^(rows-1-i) (x_i - m_(i-1))^2
        var newVars = Engine.TensorAdd(Engine.TensorMultiplyScalar(vars, decayRows), Engine.TensorMultiplyScalar(
            Engine.TensorMatMul(tail, Engine.TensorMultiply(deviation, deviation)), NumOps.Multiply(oneMinusD, d)));

        // c_end = d^rows c + (1 - d) d^2 sum_i d^(rows-1-i) (x_i - m_(i-1)) (y_i - my_(i-1))
        var featureDeviation = Engine.TensorSlice(deviation, new[] { 0, 0 }, new[] { rows, p });
        var targetDeviation = Engine.TensorSlice(deviation, new[] { 0, p }, new[] { rows, 1 });
        var newCovariances = Engine.TensorAdd(Engine.TensorMultiplyScalar(covariances, decayRows),
            Engine.TensorMultiplyScalar(Engine.TensorMatMul(
                Engine.TensorMultiply(tail, Engine.TensorTranspose(targetDeviation)), featureDeviation),
                NumOps.Multiply(oneMinusD, NumOps.Multiply(d, d))));

        var newMeans = Engine.TensorSlice(runningMeans, new[] { rows - 1, 0 }, new[] { 1, width });
        return (newMeans, newVars, newCovariances);
    }

    /// <summary>The decay factor raised to each exponent.</summary>
    private Tensor<T> DecayPowers(Tensor<T> exponents) =>
        Engine.TensorPower(PreprocessingTensorOps.Filled(exponents.Shape.ToArray(), NumOps.FromDouble(_decayFactor)), exponents);

    public void PartialFit(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
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
            throw new InvalidOperationException("ExponentialDecaySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ExponentialDecaySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ExponentialDecaySelector has not been fitted.");

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
