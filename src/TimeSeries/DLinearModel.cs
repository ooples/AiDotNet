using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

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
public class DLinearModel<T> : TimeSeriesModelBase<T>
{
    private readonly DLinearOptions<T> _options;
    private readonly Random _random;
    private readonly int _l;
    private readonly int _kernel;

    // Two linear maps from the length-L window to a scalar next-step forecast (ForecastHorizon=1 in the
    // supervised harness): seasonal and trend weights + biases. Stored as double for the closed-form update.
    private readonly double[] _wSeasonal;
    private readonly double[] _wTrend;
    private double _bSeasonal;
    private double _bTrend;

    public DLinearModel(DLinearOptions<T>? options = null)
        : base(options ?? new DLinearOptions<T>())
    {
        _options = options ?? new DLinearOptions<T>();
        Options = _options;
        _random = RandomHelper.CreateSeededRandom(42);

        _l = Math.Max(2, _options.LookbackWindow);
        int k = Math.Max(1, Math.Min(_options.MovingAverageKernel, _l));
        _kernel = k % 2 == 0 ? k + 1 : k; // odd kernel for a centered moving average

        // Initialize both maps to a uniform 1/L (a moving-average-like start), the common DLinear init.
        _wSeasonal = new double[_l];
        _wTrend = new double[_l];
        for (int j = 0; j < _l; j++)
        {
            _wSeasonal[j] = 1.0 / _l;
            _wTrend[j] = 1.0 / _l;
        }
    }

    /// <summary>Centered moving-average trend (edge-padded by replication) and the seasonal remainder.</summary>
    private (double[] Trend, double[] Seasonal) Decompose(double[] x)
    {
        int n = x.Length;
        var trend = new double[n];
        int half = _kernel / 2;
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int o = -half; o <= half; o++)
            {
                int idx = Math.Max(0, Math.Min(n - 1, i + o)); // replicate edges (net471: no Math.Clamp)
                sum += x[idx];
            }

            trend[i] = sum / _kernel;
        }

        var seasonal = new double[n];
        for (int i = 0; i < n; i++)
        {
            seasonal[i] = x[i] - trend[i];
        }

        return (trend, seasonal);
    }

    private double Forecast(double[] trend, double[] seasonal)
    {
        double pred = _bSeasonal + _bTrend;
        for (int j = 0; j < _l; j++)
        {
            pred += _wSeasonal[j] * seasonal[j] + _wTrend[j] * trend[j];
        }

        return pred;
    }

    private static bool IsFiniteValue(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    private static double[] LastWindow(Vector<T> input, int l)
    {
        var x = new double[l];
        int start = Math.Max(0, input.Length - l);
        for (int j = 0; j < l; j++)
        {
            int srcIdx = start + j;
            double v = srcIdx < input.Length ? Convert.ToDouble(input[srcIdx]) : 0.0;
            x[j] = IsFiniteValue(v) ? v : 0.0;
        }

        return x;
    }

    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        double lr = _options.LearningRate;

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();
            var order = Enumerable.Range(0, n).OrderBy(_ => _random.Next()).ToList();

            for (int batchStart = 0; batchStart < n; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, n);
                int bs = batchEnd - batchStart;

                var gS = new double[_l];
                var gT = new double[_l];
                double gbS = 0, gbT = 0;

                for (int bi = batchStart; bi < batchEnd; bi++)
                {
                    int i = order[bi];
                    var window = new double[_l];
                    int cols = x.Columns;
                    int start = Math.Max(0, cols - _l);
                    for (int j = 0; j < _l; j++)
                    {
                        int c = start + j;
                        double v = c < cols ? Convert.ToDouble(x[i, c]) : 0.0;
                        window[j] = IsFiniteValue(v) ? v : 0.0;
                    }

                    var (trend, seasonal) = Decompose(window);
                    double pred = Forecast(trend, seasonal);
                    double err = pred - Convert.ToDouble(y[i]); // dMSE/dpred ∝ error (linear gradients)

                    for (int j = 0; j < _l; j++)
                    {
                        gS[j] += err * seasonal[j];
                        gT[j] += err * trend[j];
                    }

                    gbS += err;
                    gbT += err;
                }

                double inv = bs > 0 ? 1.0 / bs : 0.0;
                for (int j = 0; j < _l; j++)
                {
                    _wSeasonal[j] -= lr * gS[j] * inv;
                    _wTrend[j] -= lr * gT[j] * inv;
                }

                _bSeasonal -= lr * gbS * inv;
                _bTrend -= lr * gbT * inv;
            }
        }

        ModelParameters = FlattenParameters();
    }

    public override T PredictSingle(Vector<T> input)
    {
        var window = LastWindow(input, _l);
        var (trend, seasonal) = Decompose(window);
        double pred = Forecast(trend, seasonal);
        return NumOps.FromDouble(IsFiniteValue(pred) ? pred : 0.0);
    }

    private Vector<T> FlattenParameters()
    {
        var flat = new T[2 * _l + 2];
        int k = 0;
        for (int j = 0; j < _l; j++) { flat[k++] = NumOps.FromDouble(_wSeasonal[j]); }
        for (int j = 0; j < _l; j++) { flat[k++] = NumOps.FromDouble(_wTrend[j]); }
        flat[k++] = NumOps.FromDouble(_bSeasonal);
        flat[k] = NumOps.FromDouble(_bTrend);
        return new Vector<T>(flat);
    }

    public override long ParameterCount => 2L * _l + 2;

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_l);
        writer.Write(_kernel);
        for (int j = 0; j < _l; j++) { writer.Write(_wSeasonal[j]); }
        for (int j = 0; j < _l; j++) { writer.Write(_wTrend[j]); }
        writer.Write(_bSeasonal);
        writer.Write(_bTrend);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        reader.ReadInt32(); // _l (fixed by ctor/options)
        reader.ReadInt32(); // _kernel
        for (int j = 0; j < _l; j++) { _wSeasonal[j] = reader.ReadDouble(); }
        for (int j = 0; j < _l; j++) { _wTrend[j] = reader.ReadDouble(); }
        _bSeasonal = reader.ReadDouble();
        _bTrend = reader.ReadDouble();
    }

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

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
        => new DLinearModel<T>(new DLinearOptions<T>(_options));
}
