using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;

using AiDotNet.Models.Parameters;

namespace AiDotNet.TimeSeries;

/// <summary>
/// NLinear — normalization-linear forecaster (Zeng et al., AAAI 2023). Subtracts the last value of the input
/// window (a simple per-window normalization that absorbs level/distribution shift), applies one linear map,
/// then adds the last value back. With DLinear it forms the pair of strong, current linear baselines that
/// frequently rival transformers on long-horizon forecasting.
/// </summary>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Are Transformers Effective for Time Series Forecasting?", "https://arxiv.org/abs/2205.13504", Year = 2023, Authors = "Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu")]
public partial class NLinearModel<T> : TimeSeriesModelBase<T>
{
    private readonly NLinearOptions<T> _options;
    private readonly Random _random;
    private readonly int _l;
    [TrainableParameter]
    private readonly Vector<T> _w;
    [TrainableParameter]
    private T _b;
    private readonly IGradientBasedOptimizer<T, Matrix<T>, Vector<T>> _optimizer;

    // StandardScaler statistics (LTSF-Linear reference pipeline: the series is z-score normalized before
    // training/inference, then predictions are denormalized). Fit on the training data; identity until then.
    // Inputs and targets get SEPARATE scalers: for a real forecast x and y are the same series so the two
    // coincide and this reduces to plain NLinear, but normalizing in the model's own space is what makes
    // training converge independent of the series magnitude (Adam can't otherwise move the bias/weights far
    // enough on large-scale targets within the epoch budget).
    [Buffer]
    private T _xMean;
    [Buffer]
    private T _xStd;
    [Buffer]
    private T _yMean;
    [Buffer]
    private T _yStd;

    /// <param name="options">Model configuration (window, horizon, epochs, batch size, learning rate).</param>
    /// <param name="optimizer">
    /// Optimizer used to update the linear map. When <c>null</c>, defaults to <see cref="AdamOptimizer{T,TInput,TOutput}"/>
    /// — the optimizer the LTSF-Linear reference (Zeng et al., AAAI 2023) trains NLinear with — seeded from
    /// <see cref="NLinearOptions{T}.LearningRate"/>. Adam's per-parameter adaptive step keeps training scale-invariant,
    /// so it converges on large-magnitude series where a fixed-step SGD update diverges. Pass a fully configured
    /// optimizer to override the paper default; nothing here is hardcoded beyond that swappable default.
    /// </param>
    public NLinearModel(NLinearOptions<T>? options = null,
        IGradientBasedOptimizer<T, Matrix<T>, Vector<T>>? optimizer = null)
        : base(options ??= new NLinearOptions<T>())
    {
        _options = options;
        Options = _options;
        _random = RandomHelper.CreateSeededRandom(SeedOr(42));
        _l = Math.Max(2, _options.LookbackWindow);
        _w = Engine.Fill(_l, NumOps.FromDouble(1.0 / _l));
        _b = NumOps.Zero;
        _xMean = NumOps.Zero;
        _xStd = NumOps.One;
        _yMean = NumOps.Zero;
        _yStd = NumOps.One;
        _optimizer = optimizer ?? new AdamOptimizer<T, Matrix<T>, Vector<T>>(
            this, new AdamOptimizerOptions<T, Matrix<T>, Vector<T>> { InitialLearningRate = _options.LearningRate });
    }

    // The x-normalized last-L window of every row of a RAW [batch, cols] tensor. Each entry is normalized
    // BEFORE the non-finite clamp to 0.0, and short rows are zero-padded on the right in normalized space.
    //
    // Normalizing before the clamp is what keeps training and inference imputing in the SAME space: a
    // non-finite entry maps to 0.0 in normalized terms (the mean in raw terms) both while training and at
    // inference. Clamping raw values first would map it to (0 - _xMean) / _xStd instead, a gap that grows
    // with _xMean.
    private Tensor<T> NormalizedWindow(Tensor<T> raw)
    {
        int batch = raw.Shape[0];
        int count = raw.Shape[1];
        int start = Math.Max(0, count - _l);
        int take = Math.Min(_l, count);
        var window = take > 0
            ? Engine.TensorSlice(raw, new[] { 0, start }, new[] { batch, take })
            : new Tensor<T>(new[] { batch, 0 });
        window = Engine.TensorDivideScalar(Engine.TensorSubtractScalar(window, _xMean), _xStd);
        window = Engine.TensorNanToNum(window, 0.0, 0.0, 0.0);
        if (take < _l)
        {
            window = Engine.TensorConcatenate(new[] { window, new Tensor<T>(new[] { batch, _l - take }) }, 1);
        }

        return window;
    }

    // NLinear in the model's normalized space: subtract-last on the normalized window, linear map, add last
    // back. Returns the window minus its last value ([batch, L]) and the y-normalized prediction ([batch, 1]).
    private (Tensor<T> Centered, Tensor<T> Prediction) ForecastNormalized(Tensor<T> windowNorm)
    {
        int batch = windowNorm.Shape[0];
        var last = Engine.TensorSlice(windowNorm, new[] { 0, _l - 1 }, new[] { batch, 1 });
        var centered = Engine.TensorSubtract(windowNorm, Engine.TensorBroadcastTo(last, new[] { batch, _l }));
        var pred = Engine.TensorMatMul(centered, Tensor<T>.FromVector(_w, new[] { _l, 1 }));
        return (centered, Engine.TensorAddScalar(Engine.TensorAdd(pred, last), _b));
    }


    /// <inheritdoc />
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;

        // StandardScaler fit (LTSF-Linear reference pipeline): z-score the inputs and targets so the
        // regression is solved in a magnitude-free space. This is what lets training converge on any series
        // scale — the bias/weights only ever need to reach O(1) — and it makes the model exactly translation-
        // and scaling-equivariant in the target (shifting/scaling y shifts/scales the denormalized output by
        // the same amount, since the normalized problem is unchanged). Separate x/y scalers: for a genuine
        // forecast x and y are the same series so they coincide (plain NLinear), but keeping them independent
        // is what makes the equivariance exact when only the target is transformed.
        var rows = Tensor<T>.FromMatrix(x);
        var targets = Tensor<T>.FromVector(y, new[] { n, 1 });
        FitScalers(rows, targets);

        // x-normalized windows and y-normalized targets — the whole regression runs in scaler space.
        var windows = NormalizedWindow(rows);
        var targetsNorm = Engine.TensorDivideScalar(Engine.TensorSubtractScalar(targets, _yMean), _yStd);

        // Start each fit from clean optimizer state so repeated Train() calls are reproducible.
        _optimizer.Reset();

        // Parameter vector theta = [w_0 .. w_{l-1}, b]. The pluggable optimizer (Adam by default, per the
        // LTSF-Linear reference) owns the update rule.
        var theta = Vector<T>.Concatenate(_w, new Vector<T>(new[] { _b }));

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();
            var order = Enumerable.Range(0, n).OrderBy(_ => _random.Next()).ToArray();

            for (int batchStart = 0; batchStart < n; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, n);
                var batch = new Tensor<int>(new[] { batchEnd - batchStart }, new Vector<int>(
                    new ArraySegment<int>(order, batchStart, batchEnd - batchStart)));
                theta = TrainBatch(windows, targetsNorm, batch, theta);
            }
        }

    }

    // One optimizer step on the mean gradient of the 0.5*err^2 objective over the rows at `batch`
    // (dL/dw_j = err*(window_j - last), dL/db = err); mirrors the result into the working weights.
    private Vector<T> TrainBatch(Tensor<T> windows, Tensor<T> targetsNorm, Tensor<int> batch, Vector<T> theta)
    {
        int bs = batch.Length;
        var (centered, predNorm) = ForecastNormalized(Engine.TensorIndexSelect(windows, batch, 0));
        var err = Engine.TensorSubtract(predNorm, Engine.TensorIndexSelect(targetsNorm, batch, 0));

        T inv = NumOps.FromDouble(bs > 0 ? 1.0 / bs : 0.0);
        var g = Engine.TensorMatMul(Engine.TensorTranspose(centered), err).ToVector();
        var grad = Vector<T>.Concatenate(
            Engine.Multiply(g, inv), new Vector<T>(new[] { NumOps.Multiply(Engine.TensorSum(err), inv) }));

        var updated = _optimizer.UpdateParameters(theta, grad);

        // Mirror the updated parameters back into the working weights the forecast reads.
        updated.GetSubVector(0, _l).CopyTo(_w.AsWritableSpan());
        _b = updated[_l];
        return updated;
    }

    // Fit z-score scalers over the training inputs (all window values) and targets.
    private void FitScalers(Tensor<T> x, Tensor<T> y)
    {
        (_xMean, _xStd) = FitScaler(x);
        (_yMean, _yStd) = FitScaler(y);
    }

    // Mean and population std of the finite entries (identity scaler when there are none). A (near-)constant
    // series has zero variance; guard the std to 1 so normalization is a pure mean-shift and never divides
    // by zero.
    private (T Mean, T Std) FitScaler(Tensor<T> values)
    {
        // Accumulated in double: a Half sum of the training values overflows past 65504, and the
        // 1e-8 floor would round to zero in Half.
        var v = values.Cast<double>();
        var finite = Engine.TensorIsFinite(v);
        var zeros = Engine.TensorZerosLike(v);
        int count = Engine.TensorCountNonzero(Engine.TensorWhere(finite, Engine.TensorOnesLike(v), zeros));
        if (count == 0)
        {
            return (NumOps.Zero, NumOps.One);
        }

        double mean = Engine.TensorSum(Engine.TensorWhere(finite, v, zeros)) / count;
        var deviations = Engine.TensorWhere(finite, Engine.TensorSubtractScalar(v, mean), zeros);
        double std = Math.Sqrt(Engine.TensorSum(Engine.TensorMultiply(deviations, deviations)) / count);
        return (NumOps.FromDouble(mean), NumOps.FromDouble(std < 1e-8 ? 1.0 : std));
    }

    /// <inheritdoc />
    public override T PredictSingle(Vector<T> input)
    {
        var window = NormalizedWindow(Tensor<T>.FromVector(input, new[] { 1, input.Length }));
        T pred = NumOps.Add(NumOps.Multiply(ForecastNormalized(window).Prediction[0, 0], _yStd), _yMean);
        return NumOps.IsNaN(pred) || NumOps.IsInfinity(pred) ? NumOps.Zero : pred;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "NLinear",
            Description = "Normalization-linear forecaster (subtract-last + linear) — strong simple baseline (Zeng et al. 2023)",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "LookbackWindow", _options.LookbackWindow },
                { "ForecastHorizon", _options.ForecastHorizon },
            },
        };
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
        => new NLinearModel<T>(new NLinearOptions<T>(_options));
}
