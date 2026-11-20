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
    private readonly IGradientBasedOptimizer<T, Matrix<T>, Vector<T>>? _optimizer;

    // StandardScaler statistics (LTSF-Linear reference pipeline: the series is z-score normalized before
    // training/inference, then predictions are denormalized). Fit on the training data; identity until then.
    // Stored in double (with exact-double state persistence across Serialize/Deserialize and Clone/DeepCopy)
    // so training and inference apply the exact same affine transform even when subnormal Half statistics
    // (e.g. std ≈ 2.58e-8 on [0, 0, 0, Half.Epsilon]) would underflow to zero if rounded to T.
    [AiDotNet.Attributes.FittedParameter(InputSized = true)]
    private double _xMean;
    [AiDotNet.Attributes.FittedParameter(InputSized = true)]
    private double _xStd;
    [AiDotNet.Attributes.FittedParameter(InputSized = true)]
    private double _yMean;
    [AiDotNet.Attributes.FittedParameter(InputSized = true)]
    private double _yStd;

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
        _xMean = 0.0;
        _xStd = 1.0;
        _yMean = 0.0;
        _yStd = 1.0;
        _optimizer = optimizer;
    }

    // The x-normalized last-L window of every row of a RAW [batch, cols] tensor in double precision. Each entry is
    // normalized BEFORE the non-finite clamp to 0.0, and short rows are zero-padded on the right in normalized space.
    //
    // Normalizing before the clamp is what keeps training and inference imputing in the SAME space: a
    // non-finite entry maps to 0.0 in normalized terms (the mean in raw terms) both while training and at
    // inference. Clamping raw values first would map it to (0 - _xMean) / _xStd instead, a gap that grows
    // with _xMean.
    private Tensor<double> NormalizedWindow(Tensor<double> raw, double xMean, double xStd)
    {
        int batch = raw.Shape[0];
        int count = raw.Shape[1];
        int start = Math.Max(0, count - _l);
        int take = Math.Min(_l, count);
        var window = take > 0
            ? Engine.TensorSlice(raw, new[] { 0, start }, new[] { batch, take })
            : new Tensor<double>(new[] { batch, 0 });
        window = Engine.TensorDivideScalar(Engine.TensorSubtractScalar(window, xMean), xStd);
        window = AiDotNet.Helpers.NanToNumHelper.Apply(Engine, window, 0.0, 0.0, 0.0);
        if (take < _l)
        {
            window = Engine.TensorConcatenate(new[] { window, new Tensor<double>(new[] { batch, _l - take }) }, 1);
        }

        return window;
    }

    // NLinear in the model's normalized space: subtract-last on the normalized window, linear map, add last
    // back. Returns the window minus its last value ([batch, L]) and the y-normalized prediction ([batch, 1]).
    private (Tensor<double> Centered, Tensor<double> Prediction) ForecastNormalized(
        Tensor<double> windowNorm, Vector<double> weights, double bias)
    {
        int batch = windowNorm.Shape[0];
        var last = Engine.TensorSlice(windowNorm, new[] { 0, _l - 1 }, new[] { batch, 1 });
        var centered = Engine.TensorSubtract(windowNorm, Engine.TensorBroadcastTo(last, new[] { batch, _l }));
        var pred = Engine.TensorMatMul(centered, Tensor<double>.FromVector(weights, new[] { _l, 1 }));
        return (centered, Engine.TensorAddScalar(Engine.TensorAdd(pred, last), bias));
    }


    /// <inheritdoc />
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        using var _ = Tensors.Helpers.TensorArena.Suspend();

        // StandardScaler fit (LTSF-Linear reference pipeline): z-score the inputs and targets so the
        // regression is solved in a magnitude-free space. This is what lets training converge on any series
        // scale — the bias/weights only ever need to reach O(1) — and it makes the model exactly translation-
        // and scaling-equivariant in the target (shifting/scaling y shifts/scales the denormalized output by
        // the same amount, since the normalized problem is unchanged). Separate x/y scalers: for a genuine
        // forecast x and y are the same series so they coincide (plain NLinear), but keeping them independent
        // is what makes the equivariance exact when only the target is transformed.
        var rows = Tensor<T>.FromMatrix(x).Cast<double>();
        var targets = Tensor<T>.FromVector(y, new[] { n, 1 }).Cast<double>();
        var (xMean, xStd, yMean, yStd) = FitScalers(rows, targets);

        // x-normalized windows and y-normalized targets — the whole regression runs in double scaler space so
        // Half statistics, small moments, and Adam's 1e-8 epsilon never overflow or underflow to 0/0.
        var windows = NormalizedWindow(rows, xMean, xStd);
        var targetsNorm = Engine.TensorDivideScalar(Engine.TensorSubtractScalar(targets, yMean), yStd);
        targetsNorm = AiDotNet.Helpers.NanToNumHelper.Apply(Engine, targetsNorm, 0.0, 0.0, 0.0);

        if (_optimizer is not null)
        {
            _optimizer.Reset();
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
                    theta = TrainBatchCustomOptimizer(windows, targetsNorm, batch, theta, _optimizer);
                }
            }

            return;
        }

        // Default optimizer (Adam, per LTSF-Linear reference) runs in double precision so repeated Train() calls
        // start from clean state and Half models never divide 0 / (0 + (Half)1e-8) = NaN.
        var defaultOptimizer = AdamOptimizer<double, Matrix<double>, Vector<double>>.CreateForFunction(
            new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = _options.LearningRate
            });
        var thetaDouble = Vector<double>.Concatenate(
            _w.AsTensor().Cast<double>().ToVector(),
            new Vector<double>(new[] { NumOps.ToDouble(_b) }));

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();
            var order = Enumerable.Range(0, n).OrderBy(_ => _random.Next()).ToArray();

            for (int batchStart = 0; batchStart < n; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, n);
                var batch = new Tensor<int>(new[] { batchEnd - batchStart }, new Vector<int>(
                    new ArraySegment<int>(order, batchStart, batchEnd - batchStart)));
                thetaDouble = TrainBatchDefaultOptimizer(windows, targetsNorm, batch, thetaDouble, defaultOptimizer);
            }
        }

        StoreDoubleTheta(thetaDouble);
    }

    private Vector<T> TrainBatchCustomOptimizer(
        Tensor<double> windows,
        Tensor<double> targetsNorm,
        Tensor<int> batch,
        Vector<T> theta,
        IGradientBasedOptimizer<T, Matrix<T>, Vector<T>> optimizer)
    {
        var weights = theta.GetSubVector(0, _l).AsTensor().Cast<double>().ToVector();
        double bias = NumOps.ToDouble(theta.AsSpan()[_l]);
        var gradD = ComputeBatchGradient(windows, targetsNorm, batch, weights, bias);
        var updated = optimizer.UpdateParameters(theta, gradD.AsTensor().Cast<T>().AsVector());
        updated.GetSubVector(0, _l).CopyTo(_w.AsWritableSpan());
        _b = updated.AsSpan()[_l];
        return updated;
    }

    private Vector<double> TrainBatchDefaultOptimizer(
        Tensor<double> windows,
        Tensor<double> targetsNorm,
        Tensor<int> batch,
        Vector<double> thetaDouble,
        AdamOptimizer<double, Matrix<double>, Vector<double>> optimizer)
    {
        var gradDouble = ComputeBatchGradient(
            windows, targetsNorm, batch, thetaDouble.GetSubVector(0, _l), thetaDouble.AsSpan()[_l]);
        return optimizer.UpdateParameters(thetaDouble, gradDouble);
    }

    private void StoreDoubleTheta(Vector<double> thetaDouble)
    {
        thetaDouble.GetSubVector(0, _l).AsTensor().Cast<T>().AsVector().CopyTo(_w.AsWritableSpan());
        _b = NumOps.FromDouble(thetaDouble.AsSpan()[_l]);
    }

    // Mean gradient of the 0.5*err^2 objective over the rows at `batch`
    // (dL/dw_j = err*(window_j - last), dL/db = err).
    private Vector<double> ComputeBatchGradient(
        Tensor<double> windows, Tensor<double> targetsNorm, Tensor<int> batch, Vector<double> weights, double bias)
    {
        int bs = batch.Length;
        var (centered, predNorm) = ForecastNormalized(Engine.TensorIndexSelect(windows, batch, 0), weights, bias);
        var err = Engine.TensorSubtract(predNorm, Engine.TensorIndexSelect(targetsNorm, batch, 0));

        double inv = bs > 0 ? 1.0 / bs : 0.0;
        var g = Engine.TensorMatMul(Engine.TensorTranspose(centered), err).ToVector();
        return Vector<double>.Concatenate(
            Engine.Multiply(g, inv), new Vector<double>(new[] { Engine.TensorSum(err) * inv }));
    }

    // Fit z-score scalers over the training inputs (all window values) and targets.
    private (double XMean, double XStd, double YMean, double YStd) FitScalers(Tensor<double> x, Tensor<double> y)
    {
        var (xMean, xStd) = FitScaler(x);
        var (yMean, yStd) = FitScaler(y);
        _xMean = xMean;
        _xStd = xStd;
        _yMean = yMean;
        _yStd = yStd;
        return (xMean, xStd, yMean, yStd);
    }

    // Mean and population std of the finite entries (identity scaler when there are none). Only genuinely
    // constant series (max == min) use unit scale; non-degenerate small-amplitude series (including subnormal
    // Half inputs whose std is ~2.58e-8) keep their exact double std rather than collapsing to 1.0.
    private (double Mean, double Std) FitScaler(Tensor<double> v)
    {
        var finite = Engine.TensorIsFinite(v);
        var zeros = Engine.TensorZerosLike(v);
        int count = Engine.TensorCountNonzero(Engine.TensorWhere(finite, Engine.TensorOnesLike(v), zeros));
        if (count == 0)
        {
            return (0.0, 1.0);
        }

        var posInf = Engine.TensorFullLike(v, double.PositiveInfinity);
        var negInf = Engine.TensorFullLike(v, double.NegativeInfinity);
        double min = Engine.TensorMinValue(Engine.TensorWhere(finite, v, posInf));
        double max = Engine.TensorMaxValue(Engine.TensorWhere(finite, v, negInf));
        if (!(max > min))
        {
            return (min, 1.0);
        }

        double mean = Engine.TensorSum(Engine.TensorWhere(finite, v, zeros)) / count;
        var deviations = Engine.TensorWhere(finite, Engine.TensorSubtractScalar(v, mean), zeros);
        double std = Math.Sqrt(Engine.TensorSum(Engine.TensorMultiply(deviations, deviations)) / count);
        if (!(std > 0.0) || double.IsNaN(std) || double.IsInfinity(std))
        {
            return (mean, 1.0);
        }

        return (mean, std);
    }

    /// <inheritdoc />
    public override T PredictSingle(Vector<T> input)
    {
        if (!(_xStd > 0.0) || double.IsNaN(_xStd) || double.IsInfinity(_xStd)
            || !(_yStd > 0.0) || double.IsNaN(_yStd) || double.IsInfinity(_yStd))
        {
            throw new InvalidOperationException("NLinear scaler standard deviations must be finite and strictly positive.");
        }

        var raw = Tensor<T>.FromVector(input, new[] { 1, input.Length }).Cast<double>();
        var window = NormalizedWindow(raw, _xMean, _xStd);
        var weights = _w.AsTensor().Cast<double>().ToVector();
        double bias = NumOps.ToDouble(_b);
        double predDouble = ForecastNormalized(window, weights, bias).Prediction[0, 0] * _yStd + _yMean;
        if (double.IsNaN(predDouble) || double.IsInfinity(predDouble))
        {
            return NumOps.Zero;
        }

        T pred = NumOps.FromDouble(predDouble);
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
