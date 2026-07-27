using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

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
public class TiDEModel<T> : TimeSeriesModelBase<T>
{
    private readonly TiDEOptions<T> _options;
    private readonly Random _random;
    private readonly int _l;
    private readonly int _h;

    // Encoder: hidden = ReLU(W1·x + b1). Decoder: out = W2·hidden + b2. Linear skip: + Wr·x + br.
    private readonly double[][] _w1; // [H][L]
    private readonly double[] _b1;   // [H]
    private readonly double[] _w2;   // [H]
    private double _b2;
    private readonly double[] _wr;   // [L]
    private double _br;

    public TiDEModel(TiDEOptions<T>? options = null)
        : base(options ?? new TiDEOptions<T>())
    {
        _options = options ?? new TiDEOptions<T>();
        Options = _options;
        _random = RandomHelper.CreateSeededRandom(42);
        _l = Math.Max(2, _options.LookbackWindow);
        _h = Math.Max(1, _options.HiddenSize);

        _w1 = new double[_h][];
        _b1 = new double[_h];
        _w2 = new double[_h];
        _wr = new double[_l];
        double s1 = Math.Sqrt(2.0 / _l); // He init for ReLU
        double s2 = Math.Sqrt(1.0 / _h);
        for (int i = 0; i < _h; i++)
        {
            _w1[i] = new double[_l];
            for (int j = 0; j < _l; j++) { _w1[i][j] = (_random.NextDouble() * 2 - 1) * s1; }
            _w2[i] = (_random.NextDouble() * 2 - 1) * s2;
        }

        for (int j = 0; j < _l; j++) { _wr[j] = 1.0 / _l; } // skip starts as a moving average
    }

    private static bool IsFiniteValue(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    /// <summary>
    /// Reversible instance normalization of the look-back window, which the TiDE paper applies so the
    /// network sees a scale-free series and distribution shift between windows cannot move the input
    /// magnitude. It was missing here, and without it the manual SGD below took steps proportional to
    /// the raw series scale and diverged outright — <c>Prediction[0]</c> came back Infinity.
    /// Returns the normalized window plus the mean and scale needed to invert the transform.
    /// </summary>
    private static (double[] Normalized, double Mean, double Scale) NormalizeWindow(double[] x)
    {
        double mean = 0.0;
        for (int j = 0; j < x.Length; j++) { mean += x[j]; }
        mean /= x.Length;

        double variance = 0.0;
        for (int j = 0; j < x.Length; j++) { double d = x[j] - mean; variance += d * d; }
        variance /= x.Length;

        double scale = Math.Sqrt(variance);
        // A constant window has zero spread; dividing by it would manufacture the very non-finite
        // values this normalization exists to prevent, so fall back to an identity scale.
        if (!IsFiniteValue(scale) || scale < 1e-8) { scale = 1.0; }

        var normalized = new double[x.Length];
        for (int j = 0; j < x.Length; j++) { normalized[j] = (x[j] - mean) / scale; }
        return (normalized, mean, scale);
    }

    /// <summary>
    /// Converts a forecast to <typeparamref name="T"/> and verifies finiteness AFTER the conversion.
    /// Checking only the double is not enough: every double from about 3.4e38 up is finite yet
    /// overflows to Infinity once narrowed to float, so a pre-conversion guard passes the value
    /// through and the float caller still receives Infinity.
    /// </summary>
    private T ToFiniteT(double value)
    {
        if (!IsFiniteValue(value)) { return NumOps.Zero; }
        T converted = NumOps.FromDouble(value);
        return IsFiniteValue(NumOps.ToDouble(converted)) ? converted : NumOps.Zero;
    }

    private static double[] Window(int l, Func<int, double> get, int count)
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

    private (double[] Hidden, double Pred) Forward(double[] x)
    {
        var hidden = new double[_h];
        for (int i = 0; i < _h; i++)
        {
            double z = _b1[i];
            var row = _w1[i];
            for (int j = 0; j < _l; j++) { z += row[j] * x[j]; }
            hidden[i] = z > 0 ? z : 0.0; // ReLU
        }

        double pred = _b2 + _br;
        for (int i = 0; i < _h; i++) { pred += _w2[i] * hidden[i]; }
        for (int j = 0; j < _l; j++) { pred += _wr[j] * x[j]; }
        return (hidden, pred);
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

                // Gradient accumulators.
                var gW1 = new double[_h][];
                for (int i = 0; i < _h; i++) { gW1[i] = new double[_l]; }
                var gB1 = new double[_h];
                var gW2 = new double[_h];
                double gB2 = 0;
                var gWr = new double[_l];
                double gBr = 0;

                for (int bi = batchStart; bi < batchEnd; bi++)
                {
                    int idx = order[bi];
                    var xvRaw = Window(_l, c => Convert.ToDouble(x[idx, c]), cols);
                    var (xv, wMean, wScale) = NormalizeWindow(xvRaw);
                    var (hidden, pred) = Forward(xv);
                    // The target has to enter the same normalized space as the window, otherwise the
                    // error — and every gradient derived from it — carries the raw series scale.
                    double target = (Convert.ToDouble(y[idx]) - wMean) / wScale;
                    double err = pred - target; // dMSE/dpred

                    gB2 += err;
                    gBr += err;
                    for (int j = 0; j < _l; j++) { gWr[j] += err * xv[j]; }
                    for (int i = 0; i < _h; i++)
                    {
                        gW2[i] += err * hidden[i];
                        double dz = hidden[i] > 0 ? err * _w2[i] : 0.0; // ReLU derivative
                        gB1[i] += dz;
                        var grow = gW1[i];
                        for (int j = 0; j < _l; j++) { grow[j] += dz * xv[j]; }
                    }
                }

                double inv = bs > 0 ? lr / bs : 0.0;
                _b2 -= inv * gB2;
                _br -= inv * gBr;
                for (int j = 0; j < _l; j++) { _wr[j] -= inv * gWr[j]; }
                for (int i = 0; i < _h; i++)
                {
                    _w2[i] -= inv * gW2[i];
                    _b1[i] -= inv * gB1[i];
                    var row = _w1[i];
                    var grow = gW1[i];
                    for (int j = 0; j < _l; j++) { row[j] -= inv * grow[j]; }
                }
            }
        }

        ModelParameters = FlattenParameters();
    }

    /// <summary>
    /// Flattens every learned weight (<c>_w1</c>, <c>_b1</c>, <c>_w2</c>, <c>_b2</c>, <c>_wr</c>,
    /// <c>_br</c>) into a single contiguous vector, in the same layout <see cref="ParameterCount"/>
    /// describes. This is the real parameter set the base-class contract expects for model inspection,
    /// checkpointing, and parameter transfer — not a one-element placeholder.
    /// </summary>
    private Vector<T> FlattenParameters()
    {
        var flat = new T[ParameterCount];
        int k = 0;
        for (int i = 0; i < _h; i++)
            for (int j = 0; j < _l; j++) { flat[k++] = NumOps.FromDouble(_w1[i][j]); }
        for (int i = 0; i < _h; i++) { flat[k++] = NumOps.FromDouble(_b1[i]); }
        for (int i = 0; i < _h; i++) { flat[k++] = NumOps.FromDouble(_w2[i]); }
        flat[k++] = NumOps.FromDouble(_b2);
        for (int j = 0; j < _l; j++) { flat[k++] = NumOps.FromDouble(_wr[j]); }
        flat[k] = NumOps.FromDouble(_br);
        return new Vector<T>(flat);
    }

    public override T PredictSingle(Vector<T> input)
    {
        var xvRaw = Window(_l, j => Convert.ToDouble(input[j]), input.Length);
        var (xv, wMean, wScale) = NormalizeWindow(xvRaw);
        var (_, predNormalized) = Forward(xv);
        // Reverse the instance normalization to return the forecast in the series' own units.
        return ToFiniteT(wMean + (wScale * predNormalized));
    }

    public override long ParameterCount => (long)_h * _l + _h + _h + 1 + _l + 1;

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_l);
        writer.Write(_h);
        for (int i = 0; i < _h; i++)
        {
            for (int j = 0; j < _l; j++) { writer.Write(_w1[i][j]); }
            writer.Write(_b1[i]);
            writer.Write(_w2[i]);
        }

        writer.Write(_b2);
        for (int j = 0; j < _l; j++) { writer.Write(_wr[j]); }
        writer.Write(_br);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        reader.ReadInt32();
        reader.ReadInt32();
        for (int i = 0; i < _h; i++)
        {
            for (int j = 0; j < _l; j++) { _w1[i][j] = reader.ReadDouble(); }
            _b1[i] = reader.ReadDouble();
            _w2[i] = reader.ReadDouble();
        }

        _b2 = reader.ReadDouble();
        for (int j = 0; j < _l; j++) { _wr[j] = reader.ReadDouble(); }
        _br = reader.ReadDouble();
    }

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

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
        => new TiDEModel<T>(new TiDEOptions<T>(_options));
}
