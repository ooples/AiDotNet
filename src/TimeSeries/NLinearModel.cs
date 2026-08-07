using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;

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
public class NLinearModel<T> : TimeSeriesModelBase<T>
{
    private readonly NLinearOptions<T> _options;
    private readonly Random _random;
    private readonly int _l;
    private readonly double[] _w;
    private double _b;
    private readonly IGradientBasedOptimizer<T, Matrix<T>, Vector<T>> _optimizer;

    // StandardScaler statistics (LTSF-Linear reference pipeline: the series is z-score normalized before
    // training/inference, then predictions are denormalized). Fit on the training data; identity until then.
    // Inputs and targets get SEPARATE scalers: for a real forecast x and y are the same series so the two
    // coincide and this reduces to plain NLinear, but normalizing in the model's own space is what makes
    // training converge independent of the series magnitude (Adam can't otherwise move the bias/weights far
    // enough on large-scale targets within the epoch budget).
    private double _xMean;
    private double _xStd = 1.0;
    private double _yMean;
    private double _yStd = 1.0;

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
        : base(options ?? new NLinearOptions<T>())
    {
        _options = options ?? new NLinearOptions<T>();
        Options = _options;
        _random = RandomHelper.CreateSeededRandom(42);
        _l = Math.Max(2, _options.LookbackWindow);
        _w = new double[_l];
        for (int j = 0; j < _l; j++) { _w[j] = 1.0 / _l; }
        _optimizer = optimizer ?? new AdamOptimizer<T, Matrix<T>, Vector<T>>(
            this, new AdamOptimizerOptions<T, Matrix<T>, Vector<T>> { InitialLearningRate = _options.LearningRate });
    }

    private static bool IsFiniteValue(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    private static double[] LastWindow(int l, Func<int, double> get, int count)
    {
        var x = new double[l];
        int start = Math.Max(0, count - l);
        for (int j = 0; j < l; j++)
        {
            int idx = start + j;
            double v = idx < count ? get(idx) : 0.0;
            x[j] = IsFiniteValue(v) ? v : 0.0;
        }

        return x;
    }

    // NLinear in the model's normalized space: subtract-last on the normalized window, linear map, add last
    // back. Operates on an ALREADY x-normalized window and returns a y-normalized prediction.
    private double ForecastNormalized(double[] windowNorm)
    {
        double last = windowNorm[_l - 1];
        double pred = _b + last;
        for (int j = 0; j < _l; j++)
        {
            pred += _w[j] * (windowNorm[j] - last);
        }

        return pred;
    }

    // Full forecast on a RAW window: x-normalize, run the normalized NLinear map, then y-denormalize.
    private double Forecast(double[] window)
    {
        var windowNorm = new double[_l];
        for (int j = 0; j < _l; j++)
            windowNorm[j] = (window[j] - _xMean) / _xStd;

        return ForecastNormalized(windowNorm) * _yStd + _yMean;
    }

    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int cols = x.Columns;

        // StandardScaler fit (LTSF-Linear reference pipeline): z-score the inputs and targets so the
        // regression is solved in a magnitude-free space. This is what lets training converge on any series
        // scale — the bias/weights only ever need to reach O(1) — and it makes the model exactly translation-
        // and scaling-equivariant in the target (shifting/scaling y shifts/scales the denormalized output by
        // the same amount, since the normalized problem is unchanged). Separate x/y scalers: for a genuine
        // forecast x and y are the same series so they coincide (plain NLinear), but keeping them independent
        // is what makes the equivariance exact when only the target is transformed.
        FitScalers(x, y, n, cols);

        // Start each fit from clean optimizer state so repeated Train() calls are reproducible.
        _optimizer.Reset();

        // Parameter vector theta = [w_0 .. w_{l-1}, b]. The pluggable optimizer (Adam by default, per the
        // LTSF-Linear reference) owns the update rule.
        var theta = new Vector<T>(_l + 1);
        for (int j = 0; j < _l; j++) { theta[j] = NumOps.FromDouble(_w[j]); }
        theta[_l] = NumOps.FromDouble(_b);

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();
            var order = Enumerable.Range(0, n).OrderBy(_ => _random.Next()).ToList();

            for (int batchStart = 0; batchStart < n; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, n);
                int bs = batchEnd - batchStart;
                var g = new double[_l];
                double gb = 0;

                for (int bi = batchStart; bi < batchEnd; bi++)
                {
                    int i = order[bi];
                    // x-normalized window and y-normalized target — the whole regression runs in scaler space.
                    var window = LastWindow(_l, c => (Convert.ToDouble(x[i, c]) - _xMean) / _xStd, cols);
                    double last = window[_l - 1];
                    double predNorm = ForecastNormalized(window);
                    double targetNorm = (Convert.ToDouble(y[i]) - _yMean) / _yStd;
                    double err = predNorm - targetNorm;
                    for (int j = 0; j < _l; j++)
                    {
                        g[j] += err * (window[j] - last);
                    }

                    gb += err;
                }

                // Mean gradient of the 0.5*err^2 objective (dL/dw_j = err*(window_j - last), dL/db = err).
                double inv = bs > 0 ? 1.0 / bs : 0.0;
                var grad = new Vector<T>(_l + 1);
                for (int j = 0; j < _l; j++) { grad[j] = NumOps.FromDouble(g[j] * inv); }
                grad[_l] = NumOps.FromDouble(gb * inv);

                theta = _optimizer.UpdateParameters(theta, grad);

                // Mirror the updated parameters back into the double working weights Forecast reads.
                for (int j = 0; j < _l; j++) { _w[j] = Convert.ToDouble(theta[j]); }
                _b = Convert.ToDouble(theta[_l]);
            }
        }

        ModelParameters = FlattenParameters();
    }

    // Fit z-score scalers over the training inputs (all window values) and targets. A (near-)constant series
    // has zero variance; guard the std to 1 so normalization is a pure mean-shift and never divides by zero.
    private void FitScalers(Matrix<T> x, Vector<T> y, int n, int cols)
    {
        const double eps = 1e-8;

        double xSum = 0; long xCount = 0;
        for (int i = 0; i < n; i++)
            for (int c = 0; c < cols; c++)
            {
                double v = Convert.ToDouble(x[i, c]);
                if (IsFiniteValue(v)) { xSum += v; xCount++; }
            }
        _xMean = xCount > 0 ? xSum / xCount : 0.0;
        double xVar = 0;
        for (int i = 0; i < n; i++)
            for (int c = 0; c < cols; c++)
            {
                double v = Convert.ToDouble(x[i, c]);
                if (IsFiniteValue(v)) { double d = v - _xMean; xVar += d * d; }
            }
        _xStd = xCount > 0 ? Math.Sqrt(xVar / xCount) : 1.0;
        if (_xStd < eps) _xStd = 1.0;

        double ySum = 0; long yCount = 0;
        for (int i = 0; i < n; i++)
        {
            double v = Convert.ToDouble(y[i]);
            if (IsFiniteValue(v)) { ySum += v; yCount++; }
        }
        _yMean = yCount > 0 ? ySum / yCount : 0.0;
        double yVar = 0;
        for (int i = 0; i < n; i++)
        {
            double v = Convert.ToDouble(y[i]);
            if (IsFiniteValue(v)) { double d = v - _yMean; yVar += d * d; }
        }
        _yStd = yCount > 0 ? Math.Sqrt(yVar / yCount) : 1.0;
        if (_yStd < eps) _yStd = 1.0;
    }

    public override T PredictSingle(Vector<T> input)
    {
        var window = LastWindow(_l, j => Convert.ToDouble(input[j]), input.Length);
        double pred = Forecast(window);
        return NumOps.FromDouble(IsFiniteValue(pred) ? pred : 0.0);
    }

    private Vector<T> FlattenParameters()
    {
        var flat = new T[_l + 1];
        for (int j = 0; j < _l; j++) { flat[j] = NumOps.FromDouble(_w[j]); }
        flat[_l] = NumOps.FromDouble(_b);
        return new Vector<T>(flat);
    }

    public override long ParameterCount => _l + 1;

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_l);
        for (int j = 0; j < _l; j++) { writer.Write(_w[j]); }
        writer.Write(_b);
        // StandardScaler stats are model state Predict depends on (Clone() round-trips through here).
        writer.Write(_xMean);
        writer.Write(_xStd);
        writer.Write(_yMean);
        writer.Write(_yStd);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        reader.ReadInt32();
        for (int j = 0; j < _l; j++) { _w[j] = reader.ReadDouble(); }
        _b = reader.ReadDouble();
        _xMean = reader.ReadDouble();
        _xStd = reader.ReadDouble();
        _yMean = reader.ReadDouble();
        _yStd = reader.ReadDouble();
    }

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

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
        => new NLinearModel<T>(new NLinearOptions<T>(_options));
}
