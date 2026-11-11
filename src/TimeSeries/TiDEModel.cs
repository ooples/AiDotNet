using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

using AiDotNet.Models.Parameters;

namespace AiDotNet.TimeSeries;

/// <summary>
/// TiDE — Time-series Dense Encoder (Das et al., TMLR 2023). A pure-MLP forecaster: a ReLU encoder maps the
/// input window to a latent, a decoder projects it to the forecast, and a linear residual skips the window
/// straight to the output. Despite using no attention it matches or beats transformers on long-horizon
/// benchmarks at far lower cost — a strong, current member of the SOTA panel. Implemented with explicit
/// forward + manual backprop (a 1-hidden-layer ReLU MLP + linear skip), so every gradient is exact.
/// </summary>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Long-term Forecasting with TiDE: Time-series Dense Encoder", "https://arxiv.org/abs/2304.08424", Year = 2023, Authors = "Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan Mathur, Rajat Sen, Rose Yu")]
public partial class TiDEModel<T> : TimeSeriesModelBase<T>
{
    private readonly TiDEOptions<T> _options;
    private readonly Random _random;
    private readonly int _l;
    private readonly int _h;

    // Encoder: hidden = ReLU(W1·x + b1). Decoder: out = W2·hidden + b2. Linear skip: + Wr·x + br.
    [TrainableParameter]
    private readonly Tensor<T> _w1; // [H, L]
    [TrainableParameter]
    private readonly Tensor<T> _b1; // [H]
    [TrainableParameter]
    private readonly Tensor<T> _w2; // [H]
    [TrainableParameter]
    private T _b2;
    [TrainableParameter]
    private readonly Tensor<T> _wr; // [L]
    [TrainableParameter]
    private T _br;

    // TiDE's reference implementation optionally normalizes each series before
    // the dense encoder and restores its scale after decoding. This supervised
    // Matrix/Vector adaptation keeps equivalent training-set statistics so raw
    // time indices and large target offsets do not destabilize the MLP.
    [Buffer]
    private readonly Tensor<T> _inputMeans;
    [Buffer]
    private readonly Tensor<T> _inputStds;
    [Buffer]
    private T _targetMean;
    [Buffer]
    private T _targetStd;

    public TiDEModel(TiDEOptions<T>? options = null)
        : base(options ??= new TiDEOptions<T>())
    {
        _options = options;
        Options = _options;
        _random = RandomHelper.CreateSeededRandom(SeedOr(42));
        _l = Math.Max(2, _options.LookbackWindow);
        _h = Math.Max(1, _options.HiddenSize);

        _b1 = new Tensor<T>(new[] { _h });
        _b2 = NumOps.Zero;
        _br = NumOps.Zero;
        _inputMeans = new Tensor<T>(new[] { _l });
        _inputStds = Tensor<T>.FromVector(Engine.Fill(_l, NumOps.One));
        _targetMean = NumOps.Zero;
        _targetStd = NumOps.One;

        // He init for ReLU. Row i of the draws holds W1 row i followed by W2[i]: the seeded stream is consumed
        // in the same order as drawing each W1 row and then its decoder weight.
        T s1 = NumOps.FromDouble(Math.Sqrt(2.0 / _l));
        T s2 = NumOps.FromDouble(Math.Sqrt(1.0 / _h));
        var draws = Engine.TensorSubtractScalar(
            Engine.TensorMultiplyScalar(Tensor<T>.CreateRandom(_random, _h, _l + 1), NumOps.FromDouble(2.0)),
            NumOps.One);
        _w1 = Engine.TensorMultiplyScalar(Engine.TensorSlice(draws, new[] { 0, 0 }, new[] { _h, _l }), s1);
        _w2 = Engine.TensorMultiplyScalar(
            Engine.TensorSlice(draws, new[] { 0, _l }, new[] { _h, 1 }), s2).Reshape(new[] { _h });

        _wr = Tensor<T>.FromVector(Engine.Fill(_l, NumOps.FromDouble(1.0 / _l))); // skip starts as a moving average
    }

    private bool IsFiniteValue(T v) => !NumOps.IsNaN(v) && !NumOps.IsInfinity(v);

    /// <summary>
    /// The last L columns of every row of <paramref name="rows"/> ([batch, cols]), zero-padded on the right
    /// when there are fewer columns, with non-finite values replaced by zero.
    /// </summary>
    private Tensor<T> Window(Tensor<T> rows)
    {
        int batch = rows.Shape[0];
        int count = rows.Shape[1];
        int start = Math.Max(0, count - _l);
        int take = Math.Min(_l, count);
        var window = take > 0
            ? Engine.TensorSlice(rows, new[] { 0, start }, new[] { batch, take })
            : new Tensor<T>(new[] { batch, 0 });
        if (take < _l)
        {
            window = Engine.TensorConcatenate(new[] { window, new Tensor<T>(new[] { batch, _l - take }) }, 1);
        }

        return Engine.TensorNanToNum(window, 0.0, 0.0, 0.0);
    }

    /// <summary>ReLU that maps NaN to zero, like <c>z &gt; 0 ? z : 0</c>.</summary>
    private Tensor<T> PositivePart(Tensor<T> z)
        => Engine.TensorWhere(Engine.TensorGreaterThan(z, NumOps.Zero), z, Engine.TensorZerosLike(z));

    /// <summary>Hidden activations ([batch, H]) and normalized predictions ([batch, 1]) of normalized windows.</summary>
    private (Tensor<T> Hidden, Tensor<T> Pred) Forward(Tensor<T> x)
    {
        int batch = x.Shape[0];
        var z = Engine.TensorAdd(
            Engine.TensorMatMul(x, Engine.TensorTranspose(_w1)),
            Engine.TensorBroadcastTo(_b1.Reshape(new[] { 1, _h }), new[] { batch, _h }));
        var hidden = PositivePart(z);

        var pred = Engine.TensorAdd(
            Engine.TensorMatMul(hidden, _w2.Reshape(new[] { _h, 1 })),
            Engine.TensorMatMul(x, _wr.Reshape(new[] { _l, 1 })));
        return (hidden, Engine.TensorAddScalar(pred, NumOps.Add(_b2, _br)));
    }

    private Tensor<T> NormalizeWindow(Tensor<T> x)
    {
        int batch = x.Shape[0];
        var means = Engine.TensorBroadcastTo(_inputMeans.Reshape(new[] { 1, _l }), new[] { batch, _l });
        var stds = Engine.TensorBroadcastTo(_inputStds.Reshape(new[] { 1, _l }), new[] { batch, _l });
        return Engine.TensorDivide(Engine.TensorSubtract(x, means), stds);
    }

    /// <summary>Keeps <paramref name="std"/> where it is finite and above 1e-8, and uses 1 elsewhere.</summary>
    private Tensor<T> GuardStd(Tensor<T> std)
    {
        var ones = Engine.TensorOnesLike(std);
        return Engine.TensorWhere(Engine.TensorIsFinite(std),
            Engine.TensorWhere(Engine.TensorGreaterThan(std, NumOps.FromDouble(1e-8)), std, ones), ones);
    }

    private void FitNormalization(Tensor<T> windows, Tensor<T> targets)
    {
        int n = windows.Shape[0];

        Engine.TensorFill(_inputMeans, NumOps.Zero);
        Engine.TensorFill(_inputStds, NumOps.One);
        if (n > 0)
        {
            var means = Engine.ReduceMean(windows, new[] { 0 }, keepDims: false);
            Engine.TensorCopyInPlace(_inputMeans, means);
            var centered = Engine.TensorSubtract(
                windows, Engine.TensorBroadcastTo(means.Reshape(new[] { 1, _l }), new[] { n, _l }));
            var variances = Engine.ReduceSum(Engine.TensorMultiply(centered, centered), new[] { 0 }, keepDims: false);
            var stds = Engine.TensorSqrt(Engine.TensorDivideScalar(variances, NumOps.FromDouble(n)));
            Engine.TensorCopyInPlace(_inputStds, GuardStd(stds));
        }

        // Non-finite targets are EXCLUDED from the statistics, matching NLinearModel.FitScalers.
        // Window already clamps non-finite inputs, so _inputMeans and _inputStds were protected;
        // the target path was not. A single NaN label made _targetMean NaN, and while _targetStd
        // rescues _targetStd back to 1.0, nothing rescued the mean -- every normalized target,
        // every error and every gradient then went NaN, training completed without an exception,
        // and every later prediction collapsed to 0 through the output guard. Silent and total.
        var finite = Engine.TensorIsFinite(targets);
        var zeros = Engine.TensorZerosLike(targets);
        int finiteTargets = Engine.TensorCountNonzero(Engine.TensorWhere(finite, Engine.TensorOnesLike(targets), zeros));
        _targetMean = n > 0 && finiteTargets > 0
            ? NumOps.Divide(Engine.TensorSum(Engine.TensorWhere(finite, targets, zeros)), NumOps.FromDouble(finiteTargets))
            : NumOps.Zero;

        T targetStd = NumOps.One;
        if (finiteTargets > 0)
        {
            var centered = Engine.TensorWhere(finite, Engine.TensorSubtractScalar(targets, _targetMean), zeros);
            targetStd = NumOps.Sqrt(NumOps.Divide(Engine.TensorSumOfSquares(centered), NumOps.FromDouble(finiteTargets)));
        }

        _targetStd = NumOps.GreaterThan(targetStd, NumOps.FromDouble(1e-8)) && IsFiniteValue(targetStd)
            ? targetStd
            : NumOps.One;
    }

    /// <inheritdoc />
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        var windows = Window(Tensor<T>.FromMatrix(x));
        var targets = Tensor<T>.FromVector(y, new[] { n, 1 });

        FitNormalization(windows, targets);
        var normalizedWindows = NormalizeWindow(windows);

        // Filtering the statistics alone does not keep a NaN label out of the gradient: that row's error
        // would still be NaN and would poison every weight it touches. Such rows are skipped (zero error).
        var finiteTarget = Engine.TensorIsFinite(targets);
        var keep = Engine.TensorWhere(finiteTarget, Engine.TensorOnesLike(targets), Engine.TensorZerosLike(targets));
        var normalizedTargets = Engine.TensorDivideScalar(Engine.TensorSubtractScalar(targets, _targetMean), _targetStd);

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();
            var order = Enumerable.Range(0, n).OrderBy(_ => _random.Next()).ToArray();

            for (int batchStart = 0; batchStart < n; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, n);
                var batch = new Tensor<int>(new[] { batchEnd - batchStart }, new Vector<int>(
                    new ArraySegment<int>(order, batchStart, batchEnd - batchStart)));
                TrainBatch(normalizedWindows, normalizedTargets, Engine.TensorIndexSelect(keep, batch, 0), batch);
            }
        }

    }

    /// <summary>One SGD step on the mean MSE gradient of the rows at <paramref name="batch"/>.</summary>
    private void TrainBatch(Tensor<T> windows, Tensor<T> targets, Tensor<T> keep, Tensor<int> batch)
    {
        int bs = batch.Length;
        var xv = Engine.TensorIndexSelect(windows, batch, 0);
        var (hidden, pred) = Forward(xv);
        var raw = Engine.TensorSubtract(pred, Engine.TensorIndexSelect(targets, batch, 0)); // dMSE/dpred
        var err = Engine.TensorWhere(keep, raw, Engine.TensorZerosLike(raw));

        // ReLU derivative: dz = hidden > 0 ? err * w2 : 0.
        var errTimesW2 = Engine.TensorMatMul(err, _w2.Reshape(new[] { 1, _h }));
        var dz = Engine.TensorWhere(Engine.TensorGreaterThan(hidden, NumOps.Zero), errTimesW2, Engine.TensorZerosLike(errTimesW2));

        T inv = NumOps.FromDouble(bs > 0 ? _options.LearningRate / bs : 0.0);
        T errSum = Engine.TensorSum(err);
        _b2 = NumOps.Subtract(_b2, NumOps.Multiply(inv, errSum));
        _br = NumOps.Subtract(_br, NumOps.Multiply(inv, errSum));

        var gWr = Engine.TensorMatMul(Engine.TensorTranspose(xv), err).Reshape(new[] { _l });
        var gW2 = Engine.TensorMatMul(Engine.TensorTranspose(hidden), err).Reshape(new[] { _h });
        var gB1 = Engine.ReduceSum(dz, new[] { 0 }, keepDims: false);
        var gW1 = Engine.TensorMatMul(Engine.TensorTranspose(dz), xv);
        Engine.TensorSubtractInPlace(_wr, Engine.TensorMultiplyScalar(gWr, inv));
        Engine.TensorSubtractInPlace(_w2, Engine.TensorMultiplyScalar(gW2, inv));
        Engine.TensorSubtractInPlace(_b1, Engine.TensorMultiplyScalar(gB1, inv));
        Engine.TensorSubtractInPlace(_w1, Engine.TensorMultiplyScalar(gW1, inv));
    }

    /// <inheritdoc />
    public override T PredictSingle(Vector<T> input)
    {
        var xv = NormalizeWindow(Window(Tensor<T>.FromVector(input, new[] { 1, input.Length })));
        var (_, normalizedPrediction) = Forward(xv);
        T pred = NumOps.Add(NumOps.Multiply(normalizedPrediction[0, 0], _targetStd), _targetMean);
        // The finiteness check runs on the value in T: every double from about 3.4e38 up is finite yet
        // overflows to Infinity in float, so only a check after the arithmetic in T is sufficient.
        // GuardPrediction then still runs, as a second line of defence for the recursive-forecast path.
        return GuardPrediction(IsFiniteValue(pred) ? pred : NumOps.Zero);
    }





    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "TiDE",
            Description = "Time-series Dense Encoder — pure-MLP encoder/decoder + linear residual (Das et al. 2023)",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "LookbackWindow", _options.LookbackWindow },
                { "HiddenSize", _options.HiddenSize },
                { "ForecastHorizon", _options.ForecastHorizon },
            },
        };
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
        => new TiDEModel<T>(new TiDEOptions<T>(_options));
}
