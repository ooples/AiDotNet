using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

using AiDotNet.Models.Parameters;

namespace AiDotNet.TimeSeries;

/// <summary>
/// DLinear — decomposition-linear forecaster (Zeng et al., AAAI 2023, "Are Transformers Effective for Time
/// Series Forecasting?"). The input window is split into a trend (moving average) and a seasonal remainder;
/// a separate linear map projects each to the forecast, and the two are summed. It is deliberately simple
/// yet a strong, current baseline that often matches or beats heavier transformers on long-horizon
/// benchmarks — the right "do we even need attention?" control in any SOTA panel.
/// </summary>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Are Transformers Effective for Time Series Forecasting?", "https://arxiv.org/abs/2205.13504", Year = 2023, Authors = "Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu")]
public partial class DLinearModel<T> : TimeSeriesModelBase<T>
{
    private readonly DLinearOptions<T> _options;
    private readonly Random _random;
    private readonly int _l;
    private readonly int _kernel;

    // Two linear maps from the length-L window to a scalar next-step forecast (ForecastHorizon=1 in the
    // supervised harness): seasonal and trend weights + biases.
    [TrainableParameter]
    private readonly Vector<T> _wSeasonal;
    [TrainableParameter]
    private readonly Vector<T> _wTrend;
    [TrainableParameter]
    private T _bSeasonal;
    [TrainableParameter]
    private T _bTrend;

    public DLinearModel(DLinearOptions<T>? options = null)
        : base(options ??= new DLinearOptions<T>())
    {
        _options = options;
        Options = _options;
        _random = RandomHelper.CreateSeededRandom(SeedOr(42));

        _l = Math.Max(2, _options.LookbackWindow);
        int k = Math.Max(1, Math.Min(_options.MovingAverageKernel, _l));
        _kernel = k % 2 == 0 ? k + 1 : k; // odd kernel for a centered moving average

        // Initialize both maps to a uniform 1/L (a moving-average-like start), the common DLinear init.
        T uniform = NumOps.FromDouble(1.0 / _l);
        _wSeasonal = Engine.Fill(_l, uniform);
        _wTrend = Engine.Fill(_l, uniform);
        _bSeasonal = NumOps.Zero;
        _bTrend = NumOps.Zero;
    }

    /// <summary>
    /// Centered moving-average trend (edge-padded by replication) and the seasonal remainder of every row of
    /// <paramref name="windows"/> ([batch, L]).
    /// </summary>
    private (Tensor<T> Trend, Tensor<T> Seasonal) Decompose(Tensor<T> windows)
    {
        int half = _kernel / 2;
        var padded = Engine.PadNd(windows, new[] { half, half }, AiDotNet.Tensors.Engines.PadMode.Replicate);

        // [batch, L, kernel] sliding windows summed over the kernel axis, in master's offset order.
        var sums = Engine.ReduceSum(Engine.TensorUnfold(padded, 1, _kernel, 1), new[] { 2 }, keepDims: false);
        var trend = Engine.TensorDivideScalar(sums, NumOps.FromDouble(_kernel));
        return (trend, Engine.TensorSubtract(windows, trend));
    }

    /// <summary>Forecast of every row: seasonal·wS + trend·wT + bS + bT, as a [batch, 1] tensor.</summary>
    private Tensor<T> Forecast(Tensor<T> trend, Tensor<T> seasonal)
    {
        var seasonalPart = Engine.TensorMatMul(seasonal, Tensor<T>.FromVector(_wSeasonal, new[] { _l, 1 }));
        var trendPart = Engine.TensorMatMul(trend, Tensor<T>.FromVector(_wTrend, new[] { _l, 1 }));
        return Engine.TensorAddScalar(Engine.TensorAdd(seasonalPart, trendPart), NumOps.Add(_bSeasonal, _bTrend));
    }

    /// <summary>
    /// The last <paramref name="l"/> columns of every row of <paramref name="rows"/> ([batch, cols]),
    /// zero-padded on the right when there are fewer columns, with non-finite values replaced by zero.
    /// </summary>
    private Tensor<T> LastWindow(Tensor<T> rows, int l)
    {
        int batch = rows.Shape[0];
        int cols = rows.Shape[1];
        int start = Math.Max(0, cols - l);
        int take = Math.Min(l, cols);
        var window = take > 0
            ? Engine.TensorSlice(rows, new[] { 0, start }, new[] { batch, take })
            : new Tensor<T>(new[] { batch, 0 });
        if (take < l)
        {
            window = Engine.TensorConcatenate(new[] { window, new Tensor<T>(new[] { batch, l - take }) }, 1);
        }

        return Engine.TensorNanToNum(window, 0.0, 0.0, 0.0);
    }

    /// <inheritdoc />
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        var rows = Tensor<T>.FromMatrix(x);
        var targets = Tensor<T>.FromVector(y, new[] { n, 1 });

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();
            var order = Enumerable.Range(0, n).OrderBy(_ => _random.Next()).ToArray();

            // Per-sample errors of the epoch, to report a mean squared error below.
            var epochErrors = new List<Tensor<T>>();

            for (int batchStart = 0; batchStart < n; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, n);
                var batch = new Tensor<int>(new[] { batchEnd - batchStart }, new Vector<int>(
                    new ArraySegment<int>(order, batchStart, batchEnd - batchStart)));
                epochErrors.Add(TrainBatch(rows, targets, batch));
            }

            // Report every epoch, including a fully diverged one (no finite sample): a non-finite loss counts
            // as non-improving, so the training callback and patience-based early stopping still fire for the
            // worst-case divergence this feature is meant to guard against.
            if (!ReportEpoch(epoch, _options.Epochs, EpochLoss(epochErrors)))
            {
                break;
            }
        }

    }

    /// <summary>One mini-batch gradient step on the rows at <paramref name="batch"/>; returns the batch errors.</summary>
    private Tensor<T> TrainBatch(Tensor<T> rows, Tensor<T> targets, Tensor<int> batch)
    {
        int bs = batch.Length;
        var window = LastWindow(Engine.TensorIndexSelect(rows, batch, 0), _l);
        var (trend, seasonal) = Decompose(window);
        var err = Engine.TensorSubtract(Forecast(trend, seasonal), Engine.TensorIndexSelect(targets, batch, 0));

        // dMSE/dpred ∝ error (linear gradients), averaged over the batch.
        T step = NumOps.FromDouble(bs > 0 ? _options.LearningRate / bs : 0.0);
        var gS = Engine.TensorMatMul(Engine.TensorTranspose(seasonal), err).ToVector();
        var gT = Engine.TensorMatMul(Engine.TensorTranspose(trend), err).ToVector();
        _wSeasonal.SubtractInPlace(Engine.Multiply(gS, step));
        _wTrend.SubtractInPlace(Engine.Multiply(gT, step));

        T biasStep = NumOps.Multiply(Engine.TensorSum(err), step);
        _bSeasonal = NumOps.Subtract(_bSeasonal, biasStep);
        _bTrend = NumOps.Subtract(_bTrend, biasStep);
        return err;
    }

    /// <summary>Mean squared error over the finite errors of an epoch; +∞ when none is finite.</summary>
    private T EpochLoss(List<Tensor<T>> epochErrors)
    {
        if (epochErrors.Count == 0)
        {
            return NumOps.FromDouble(double.PositiveInfinity);
        }

        var errors = Engine.TensorConcatenate(epochErrors.ToArray(), 0);
        var finite = Engine.TensorIsFinite(errors);
        var zeros = Engine.TensorZerosLike(errors);
        int samples = Engine.TensorCountNonzero(Engine.TensorWhere(finite, Engine.TensorOnesLike(errors), zeros));
        return samples > 0
            ? NumOps.Divide(Engine.TensorSumOfSquares(Engine.TensorWhere(finite, errors, zeros)), NumOps.FromDouble(samples))
            : NumOps.FromDouble(double.PositiveInfinity);
    }

    /// <inheritdoc />
    public override T PredictSingle(Vector<T> input)
    {
        var window = LastWindow(Tensor<T>.FromVector(input, new[] { 1, input.Length }), _l);
        var (trend, seasonal) = Decompose(window);
        T pred = Forecast(trend, seasonal)[0, 0];
        return NumOps.IsNaN(pred) || NumOps.IsInfinity(pred) ? NumOps.Zero : pred;
    }





    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DLinear",
            Description = "Decomposition-linear forecaster (trend + seasonal linear maps) — strong simple baseline (Zeng et al. 2023)",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "LookbackWindow", _options.LookbackWindow },
                { "ForecastHorizon", _options.ForecastHorizon },
                { "MovingAverageKernel", _kernel },
            },
        };
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
        => new DLinearModel<T>(new DLinearOptions<T>(_options));
}
