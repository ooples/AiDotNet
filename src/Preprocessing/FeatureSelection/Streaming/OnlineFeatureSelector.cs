using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Streaming;

/// <summary>
/// Online Feature Selection for Streaming Data.
/// </summary>
/// <remarks>
/// <para>
/// Performs incremental feature selection that can update as new data arrives,
/// suitable for streaming or online learning scenarios.
/// </para>
/// <para><b>For Beginners:</b> Traditional feature selection processes all data
/// at once. Online selection updates incrementally as new samples arrive, which
/// is useful when you can't store all data in memory or when data comes in
/// continuously.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class OnlineFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _learningRate;

    private Tensor<T>? _runningScores;
    private Vector<T>? _runningScoreValues;
    private Tensor<T>? _runningMeans;
    private Tensor<T>? _runningVars;
    private int _sampleCount;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double LearningRate => _learningRate;
    public Vector<T>? RunningScores => _runningScoreValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public OnlineFeatureSelector(
        int nFeaturesToSelect = 10,
        double learningRate = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _learningRate = learningRate;
        _sampleCount = 0;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "OnlineFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        // Initialize or update statistics
        int n = data.Rows;
        int p = data.Columns;
        _nInputFeatures = p;

        // Running means and sums of squared deviations of the features and (last column) the target,
        // feature-target co-moments and the running scores
        if (_runningScores is null)
        {
            _runningScores = new Tensor<T>(new[] { 1, p });
            _runningMeans = new Tensor<T>(new[] { 1, p + 1 });
            _runningVars = new Tensor<T>(new[] { 1, p + 1 });
            _covariances = new Tensor<T>(new[] { 1, p });
        }

        var runningMeans = _runningMeans ?? throw new InvalidOperationException("Running means have not been initialized.");
        var runningVars = _runningVars ?? throw new InvalidOperationException("Running variances have not been initialized.");
        var runningScores = _runningScores ?? throw new InvalidOperationException("Running scores have not been initialized.");
        var covariances = _covariances ?? throw new InvalidOperationException("Running covariances have not been initialized.");

        if (n > 0)
        {
            int width = p + 1;
            var shape = new[] { n, width };
            var columns = Engine.TensorConcatenate(new[]
            {
                Tensor<T>.FromMatrix(data), Engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 })
            }, 1);

            // Welford's statistics after every sample, from prefix sums of the deviations from a shift s (the
            // running mean, or the first sample when there is none): with N_k samples seen and A_k, Q_k the prefix
            // sums of (z - s) and (z - s)^2, mean_k = s + A_k / N_k and M2_k = M2_0 + Q_k - A_k^2 / N_k; the
            // co-moment is C_k = C_0 + sum (x - s_x)(y - s_y) - A_x A_y / N_k.
            var shift = _sampleCount > 0 ? runningMeans : PreprocessingTensorOps.Row(columns, 0);
            var deviation = Engine.TensorSubtract(columns, Engine.TensorBroadcastTo(shift, shape));
            var sums = Engine.TensorCumSum(deviation, 0);
            var squares = Engine.TensorCumSum(Engine.TensorMultiply(deviation, deviation), 0);
            var featureDeviation = Engine.TensorSlice(deviation, new[] { 0, 0 }, new[] { n, p });
            var targetDeviation = Engine.TensorBroadcastTo(Engine.TensorSlice(deviation, new[] { 0, p }, new[] { n, 1 }), new[] { n, p });
            var crossSums = Engine.TensorCumSum(Engine.TensorMultiply(featureDeviation, targetDeviation), 0);
            var counts = Engine.Reshape(Engine.TensorArange<T>(_sampleCount + 1, _sampleCount + n + 1, 1), new[] { n, 1 });

            var vars = Engine.TensorAdd(Engine.TensorBroadcastTo(runningVars, shape), Engine.TensorSubtract(squares,
                Engine.TensorDivide(Engine.TensorMultiply(sums, sums), Engine.TensorBroadcastTo(counts, shape))));
            var featureSums = Engine.TensorSlice(sums, new[] { 0, 0 }, new[] { n, p });
            var targetSums = Engine.TensorBroadcastTo(Engine.TensorSlice(sums, new[] { 0, p }, new[] { n, 1 }), new[] { n, p });
            var comoments = Engine.TensorAdd(Engine.TensorBroadcastTo(covariances, new[] { n, p }), Engine.TensorSubtract(crossSums,
                Engine.TensorDivide(Engine.TensorMultiply(featureSums, targetSums), Engine.TensorBroadcastTo(counts, new[] { n, p }))));

            // Correlation estimate after every sample, 0 unless both variances are above 1e-10
            T eps = PreprocessingTensorOps.Tolerance<T>(1e-10);
            var xVar = Engine.TensorSlice(vars, new[] { 0, 0 }, new[] { n, p });
            var yVar = Engine.TensorBroadcastTo(Engine.TensorSlice(vars, new[] { 0, p }, new[] { n, 1 }), new[] { n, p });
            var usable = Engine.TensorMultiply(Engine.TensorGreaterThan(xVar, eps), Engine.TensorGreaterThan(yVar, eps));
            var product = Engine.TensorWhere(usable, Engine.TensorMultiply(xVar, yVar), PreprocessingTensorOps.Filled(new[] { n, p }, NumOps.One));
            var correlations = SelectorTensorOps.ZeroUnless(usable, Engine.TensorDivide(comoments, Engine.TensorSqrt(product)));

            // Running scores: n steps of S <- (1 - rate) S + rate r
            T keep = NumOps.FromDouble(1 - _learningRate);
            var exponents = Engine.TensorSubtract(
                PreprocessingTensorOps.Filled(new[] { n }, NumOps.FromDouble(n - 1)), Engine.TensorArange<T>(0, n, 1));
            var decay = Engine.Reshape(Engine.TensorPower(PreprocessingTensorOps.Filled(new[] { n }, keep), exponents), new[] { 1, n });
            runningScores = Engine.TensorAdd(
                Engine.TensorMultiplyScalar(runningScores, NumOps.FromDouble(Math.Pow(1 - _learningRate, n))),
                Engine.TensorMultiplyScalar(Engine.TensorMatMul(decay, correlations), NumOps.FromDouble(_learningRate)));

            _sampleCount += n;
            runningMeans = Engine.TensorAdd(shift, Engine.TensorDivideScalar(
                Engine.TensorSlice(sums, new[] { n - 1, 0 }, new[] { 1, width }), NumOps.FromDouble(_sampleCount)));
            runningVars = Engine.TensorSlice(vars, new[] { n - 1, 0 }, new[] { 1, width });
            covariances = Engine.TensorSlice(comoments, new[] { n - 1, 0 }, new[] { 1, p });
        }

        _runningScores = Owned(runningScores);
        _runningMeans = Owned(runningMeans);
        _runningVars = Owned(runningVars);
        _covariances = Owned(covariances);
        _runningScoreValues = Engine.Reshape(runningScores, new[] { p }).ToVector();

        // Select top features
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(
            Engine.TensorAbs(Engine.Reshape(runningScores, new[] { p })), Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <summary>
    /// A garbage-collected copy of <paramref name="value"/> for state kept across fits: an engine result (or a
    /// slice view of one) may live in a tensor arena buffer that is handed out again after this fit returns.
    /// </summary>
    private static Tensor<T> Owned(Tensor<T> value) => new Tensor<T>(value.Shape.ToArray(), value.ToVector());

    private Tensor<T>? _covariances;

    public void PartialFit(Matrix<T> data, Vector<T> target)
    {
        // Same as Fit but designed for incremental updates
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
            throw new InvalidOperationException("OnlineFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("OnlineFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("OnlineFeatureSelector has not been fitted.");

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
