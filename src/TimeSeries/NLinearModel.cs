using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

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

    public NLinearModel(NLinearOptions<T>? options = null)
        : base(options ?? new NLinearOptions<T>())
    {
        _options = options ?? new NLinearOptions<T>();
        Options = _options;
        _random = RandomHelper.CreateSeededRandom(42);
        _l = Math.Max(2, _options.LookbackWindow);
        _w = new double[_l];
        for (int j = 0; j < _l; j++) { _w[j] = 1.0 / _l; }
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

    private double Forecast(double[] window)
    {
        double last = window[_l - 1];
        double pred = _b + last;
        for (int j = 0; j < _l; j++)
        {
            pred += _w[j] * (window[j] - last);
        }

        return pred;
    }

    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int cols = x.Columns;
        double lr = _options.LearningRate;

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
                    var window = LastWindow(_l, c => Convert.ToDouble(x[i, c]), cols);
                    double last = window[_l - 1];
                    double pred = Forecast(window);
                    double err = pred - Convert.ToDouble(y[i]);
                    for (int j = 0; j < _l; j++)
                    {
                        g[j] += err * (window[j] - last);
                    }

                    gb += err;
                }

                double inv = bs > 0 ? 1.0 / bs : 0.0;
                for (int j = 0; j < _l; j++) { _w[j] -= lr * g[j] * inv; }
                _b -= lr * gb * inv;
            }
        }

        ModelParameters = FlattenParameters();
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
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        reader.ReadInt32();
        for (int j = 0; j < _l; j++) { _w[j] = reader.ReadDouble(); }
        _b = reader.ReadDouble();
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
