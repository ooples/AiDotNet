using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TimeSeries.AnomalyDetection;

/// <summary>
/// Implements DeepAnT (Munir et al. 2019), unsupervised anomaly detection in time series by forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// DeepAnT has two modules. The time-series predictor, a convolutional network, reads a history window of w values
/// and predicts the next p_w values; it is trained on the series' own overlapping windows with the mean absolute error
/// and stochastic gradient descent, on the series normalized with statistics of the training data. The anomaly
/// detector scores each observed value by its Euclidean distance from the prediction, and flags scores above a
/// threshold. No labels are used, and anomalies need not be removed from the training data.
/// </para>
/// <para>
/// The paper fixes the predictor's structure (two convolution and max-pooling blocks with 32 filters and ReLU, a
/// fully connected layer and a linear output), the loss, the optimizer and the defaults taken from its Table 1 (window
/// 45 and threshold 0.5, the A1 benchmark). It does not state the kernel, pooling and hidden widths, the dropout, the
/// normalization, the learning rate or the training length, and no author code was released. Those follow the PyTorch
/// implementation at github.com/datacubeR/DeepAnt (kernel 3, pooling 2, 40 hidden units, dropout 0.25, min-max
/// scaling, batches of 32, 30 epochs), except the optimizer, where the paper's SGD replaces that code's Adam; its
/// learning rate, which neither source gives, is Keras' SGD default of 0.01.
/// </para>
/// <para><b>For Beginners:</b> DeepAnT learns what "normal" looks like in your time series,
/// then flags anything unusual as an anomaly. It works by:
/// 1. Learning to predict the next value based on past values
/// 2. Comparing actual values to predictions
/// 3. Marking large prediction errors as anomalies
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var series = new Vector&lt;double&gt;(values);
/// var timeIndex = new Matrix&lt;double&gt;(values.Length, 1);
/// for (int i = 0; i &lt; values.Length; i++) timeIndex[i, 0] = i;
/// var model = new DeepANT&lt;double&gt;(new DeepANTOptions&lt;double&gt; { WindowSize = 45 });
/// model.Train(timeIndex, series);
/// bool[] anomalies = model.DetectAnomalies(series);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series", "https://doi.org/10.1109/ACCESS.2018.2886457", Year = 2019, Authors = "Mohsin Munir, Shoaib Ahmed Siddiqui, Andreas Dengel, Sheraz Ahmed")]
public partial class DeepANT<T> : TimeSeriesModelBase<T>
{
    private readonly DeepANTOptions<T> _options;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly List<NeuralNetworks.Layers.LayerBase<T>> _layers = new();
    private readonly DeepAntNetwork<T> _network;
    private readonly Random _random;
    [Buffer]
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    // Min-max scaling fitted on the training data (the reference's MinMaxScaler): x' = (x − min) / (max − min).
    private double _scaleMin;
    private double _scaleRange = 1.0;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the DeepAnT model.
    /// </summary>
    public DeepANT(DeepANTOptions<T>? options = null)
        : base(options ?? new DeepANTOptions<T>())
    {
        _options = (DeepANTOptions<T>)Options;
        var o = _options;
        if (o.Epochs <= 0 || o.BatchSize <= 0 || o.LearningRate <= 0)
            throw new ArgumentException("Epochs, BatchSize and LearningRate must be positive.", nameof(options));
        if (o.AnomalyThreshold < 0)
            throw new ArgumentException("AnomalyThreshold cannot be negative.", nameof(options));
        _random = RandomHelper.CreateSeededRandom(SeedOr(1042));
        _network = new DeepAntNetwork<T>(Engine, _layers, new DeepAntConfiguration(o.WindowSize, o.ForecastHorizon, o.Filters,
            o.KernelSize, o.PoolSize, o.HiddenUnits, o.DropoutRate));
        _network.InitializeLikeReference(RandomHelper.CreateSeededRandom(SeedOr(42)));
    }

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        for (int i = 0; i < _layers.Count; i++)
            RegisterParameterComponent($"deepant.layer{i}", _layers[i]);
    }

    private T Scale(T value) => _numOps.FromDouble((_numOps.ToDouble(value) - _scaleMin) / _scaleRange);

    private T Unscale(double value) => _numOps.FromDouble(value * _scaleRange + _scaleMin);

    /// <summary>
    /// Trains the predictor on windows of WindowSize values and the ForecastHorizon values after each. A one-column
    /// input is a time index: the windows are the training series' own (§IV-A). Otherwise each row of the input is a
    /// window and y holds the value after it (ForecastHorizon must then be 1). The data are min-max scaled with
    /// statistics of the training data, and SGD minimizes the mean absolute error over shuffled batches.
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        var o = _options;
        int w = o.WindowSize, horizon = o.ForecastHorizon;
        T[][] windows;
        T[][] targets;
        if (x.Columns == 1 && w > 1)
        {
            int samples = y.Length - w - horizon + 1;
            if (samples <= 0)
                throw new ArgumentException(
                    $"Training series must contain at least WindowSize + ForecastHorizon ({w + horizon}) values; got {y.Length}.", nameof(y));
            windows = new T[samples][];
            targets = new T[samples][];
            for (int s = 0; s < samples; s++)
            {
                windows[s] = new T[w];
                for (int t = 0; t < w; t++) windows[s][t] = y[s + t];
                targets[s] = new T[horizon];
                for (int h = 0; h < horizon; h++) targets[s][h] = y[s + w + h];
            }
        }
        else
        {
            if (x.Columns != w)
                throw new ArgumentException($"Each input row must be a window of WindowSize ({w}) values; got {x.Columns}.", nameof(x));
            if (horizon != 1)
                throw new ArgumentException("Rows of windows carry one target each; train on a time index for ForecastHorizon > 1.", nameof(x));
            windows = new T[x.Rows][];
            targets = new T[x.Rows][];
            for (int s = 0; s < x.Rows; s++)
            {
                windows[s] = new T[w];
                for (int t = 0; t < w; t++) windows[s][t] = x[s, t];
                targets[s] = new[] { y[s] };
            }
        }

        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++) _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = _numOps.FromDouble(y.Length);
        FitScaler(windows, targets);

        var optimizer = new StochasticGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>(null,
            new StochasticGradientDescentOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
            });
        var loss = new MeanAbsoluteErrorLoss<T>();
        var parameters = Training.TapeTrainingStep<T>.CollectParameters(_layers);
        int count = windows.Length;
        bool timeBounded = o.MaxTrainingTimeSeconds > 0;
        int epochs = timeBounded ? int.MaxValue : o.Epochs;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            if (timeBounded && TrainingCancellationToken.IsCancellationRequested) break;
            TrainingCancellationToken.ThrowIfCancellationRequested();
            var order = Enumerable.Range(0, count).OrderBy(_ => _random.Next()).ToArray();
            double lossSum = 0;
            int batches = 0;
            // DataLoader(shuffle=True) keeps the last, incomplete batch.
            for (int start = 0; start < count; start += o.BatchSize)
            {
                if (timeBounded && TrainingCancellationToken.IsCancellationRequested) break;
                TrainingCancellationToken.ThrowIfCancellationRequested();
                int size = Math.Min(o.BatchSize, count - start);
                var (history, target) = Batch(windows, targets, order, start, size);
                // One update on the shared eager step: the network is trained directly, outside the model's Layers chain.
                T lossValue = Training.TapeTrainingStepper<T>.EagerObjectiveStep(parameters,
                    () => loss.ComputeTapeLoss(_network.Forward(history, training: true, _random), target), optimizer);
                lossSum += _numOps.ToDouble(lossValue);
                batches++;
            }
            if (batches > 0 && !ReportEpoch(epoch, timeBounded ? 0 : o.Epochs, _numOps.FromDouble(lossSum / batches)))
                break;
        }
    }

    private void FitScaler(T[][] windows, T[][] targets)
    {
        double min = double.PositiveInfinity, max = double.NegativeInfinity;
        void Observe(T[] values)
        {
            foreach (var v in values)
            {
                double d = _numOps.ToDouble(v);
                if (d < min) min = d;
                if (d > max) max = d;
            }
        }
        foreach (var window in windows) Observe(window);
        foreach (var target in targets) Observe(target);
        _scaleMin = min;
        // sklearn's MinMaxScaler leaves a constant feature unscaled (range treated as 1).
        _scaleRange = max - min > 0 ? max - min : 1.0;
    }

    // The scaled windows [B, WindowSize, 1] and targets [B, ForecastHorizon] of the given samples.
    private (Tensor<T> history, Tensor<T> target) Batch(T[][] windows, T[][] targets, int[] order, int start, int count)
    {
        int w = _options.WindowSize, horizon = _options.ForecastHorizon;
        var history = new Tensor<T>(new[] { count, w, 1 });
        var target = new Tensor<T>(new[] { count, horizon });
        for (int b = 0; b < count; b++)
        {
            int s = order[start + b];
            for (int t = 0; t < w; t++) history[b, t, 0] = Scale(windows[s][t]);
            for (int h = 0; h < horizon; h++) target[b, h] = Scale(targets[s][h]);
        }
        return (history, target);
    }

    // The scaled prediction of the ForecastHorizon values after a window of WindowSize raw values.
    private double[] ScaledForecast(Vector<T> window)
    {
        if (_trainingSeries.Length == 0) throw new InvalidOperationException("The model must be trained before predicting.");
        int w = _options.WindowSize;
        if (window.Length != w)
            throw new ArgumentException($"DeepAnT predicts from a window of WindowSize ({w}) values; got {window.Length}.", nameof(window));
        var input = new Tensor<T>(new[] { 1, w, 1 });
        for (int t = 0; t < w; t++) input[0, t, 0] = Scale(window[t]);
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        var prediction = _network.Forward(input, training: false, _random);
        var result = new double[_options.ForecastHorizon];
        for (int h = 0; h < result.Length; h++) result[h] = _numOps.ToDouble(prediction[0, h]);
        return result;
    }

    /// <summary>
    /// Predicts the series. A one-column time index is answered with the predictor's forecasts: with as many rows as
    /// the training series, the one-step forecast of each observation from the WindowSize before it; otherwise the
    /// forecasts of the steps after the training series. Any other input is a matrix of windows, each predicted.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (TryForecastTimeIndex(input, _trainingSeries, _options.WindowSize, PredictSingle, out var forecasts))
            return forecasts;
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++) predictions[i] = PredictSingle(input.GetRow(i));
        return predictions;
    }

    /// <summary>The predicted value after a window of WindowSize raw values.</summary>
    public override T PredictSingle(Vector<T> input) => Unscale(ScaledForecast(input)[0]);

    /// <summary>
    /// Forecasts the <paramref name="steps"/> values after <paramref name="history"/>: each is the one-step prediction
    /// from the last WindowSize values, appended to the history for the next. A history shorter than the window is
    /// extended to the left by repeating its first value.
    /// </summary>
    public override Vector<T> Forecast(Vector<T> history, int steps)
    {
        if (history is null) throw new ArgumentNullException(nameof(history));
        if (history.Length == 0) throw new ArgumentException("DeepAnT needs at least one observed value to forecast from.", nameof(history));
        if (steps <= 0) throw new ArgumentException("Number of forecast steps must be positive.", nameof(steps));
        int w = _options.WindowSize;
        var values = history.ToList();
        var window = new Vector<T>(w);
        var forecasts = new Vector<T>(steps);
        for (int s = 0; s < steps; s++)
        {
            for (int t = 0; t < w; t++) window[t] = values[Math.Max(0, values.Count - w + t)];
            forecasts[s] = PredictSingle(window);
            values.Add(forecasts[s]);
        }
        return forecasts;
    }

    /// <summary>
    /// The anomaly score of every value after the first WindowSize: the Euclidean distance (§IV-B) between the value
    /// and its one-step prediction from the WindowSize values before it, in the normalized units the threshold uses.
    /// </summary>
    public Vector<T> ComputeAnomalyScores(Vector<T> data)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        int w = _options.WindowSize;
        if (data.Length <= w)
            throw new ArgumentException($"Data length must be greater than WindowSize ({w}).", nameof(data));
        var scores = new Vector<T>(data.Length - w);
        var window = new Vector<T>(w);
        for (int i = 0; i < scores.Length; i++)
        {
            for (int t = 0; t < w; t++) window[t] = data[i + t];
            double predicted = ScaledForecast(window)[0];
            double actual = _numOps.ToDouble(Scale(data[i + w]));
            scores[i] = _numOps.FromDouble(Math.Abs(actual - predicted));
        }
        return scores;
    }

    /// <summary>
    /// Flags every value after the first WindowSize whose anomaly score exceeds AnomalyThreshold.
    /// </summary>
    /// <returns>One flag per value from index WindowSize on; true marks an anomaly.</returns>
    public bool[] DetectAnomalies(Vector<T> data)
    {
        var scores = ComputeAnomalyScores(data);
        var anomalies = new bool[scores.Length];
        for (int i = 0; i < scores.Length; i++) anomalies[i] = _numOps.ToDouble(scores[i]) > _options.AnomalyThreshold;
        return anomalies;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DeepANT",
            Description = "DeepAnT: a convolutional time-series predictor whose prediction error scores anomalies",
            Complexity = ParameterCount,
            FeatureCount = _options.WindowSize,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "WindowSize", _options.WindowSize },
                { "ForecastHorizon", _options.ForecastHorizon },
                { "AnomalyThreshold", _options.AnomalyThreshold }
            }
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
        => new DeepANT<T>(new DeepANTOptions<T>(_options));

    /// <inheritdoc />
    /// <remarks>The network's parameters are saved by the base through the registered layers; this adds the
    /// training series and the scaler.</remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_trainingSeries.Length);
        for (int i = 0; i < _trainingSeries.Length; i++) writer.Write(_numOps.ToDouble(_trainingSeries[i]));
        writer.Write(_scaleMin);
        writer.Write(_scaleRange);
    }

    /// <inheritdoc />
    protected override void DeserializeCore(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        _trainingSeries = new Vector<T>(length);
        for (int i = 0; i < length; i++) _trainingSeries[i] = _numOps.FromDouble(reader.ReadDouble());
        _scaleMin = reader.ReadDouble();
        _scaleRange = reader.ReadDouble();
    }
}

/// <summary>
/// Options for the DeepAnT model. Defaults follow the paper where it states a value (32 filters; window 45 and
/// threshold 0.5, its A1 setting in Table 1) and otherwise the PyTorch implementation at github.com/datacubeR/DeepAnt;
/// see <see cref="DeepANT{T}"/>.
/// </summary>
public class DeepANTOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>The history window w: the values the predictor reads (the paper's Table 1 uses 35 or 45).</summary>
    public int WindowSize { get; set; } = 45;

    /// <summary>The prediction window p_w: the values predicted after each window (the paper predicts 1).</summary>
    public int ForecastHorizon { get; set; } = 1;

    /// <summary>The filters of each convolution (the paper's 32).</summary>
    public int Filters { get; set; } = 32;

    /// <summary>The width of each convolution. Not stated in the paper; 3 as in the datacubeR implementation.</summary>
    public int KernelSize { get; set; } = 3;

    /// <summary>The width and stride of each max-pooling. Not stated in the paper; 2 as in the datacubeR implementation.</summary>
    public int PoolSize { get; set; } = 2;

    /// <summary>The fully connected layer's width. Not stated in the paper; 40 as in the datacubeR implementation.</summary>
    public int HiddenUnits { get; set; } = 40;

    /// <summary>The dropout after the fully connected layer. Not in the paper; 0.25 as in the datacubeR implementation.</summary>
    public double DropoutRate { get; set; } = 0.25;

    /// <summary>The SGD learning rate. Stated by neither source; Keras' SGD default.</summary>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>The training epochs. Not stated in the paper; 30 as in the datacubeR implementation.</summary>
    public int Epochs { get; set; } = 30;

    /// <summary>The windows per SGD step. Not stated in the paper; 32 as in the datacubeR implementation.</summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>The anomaly-score threshold in normalized units (the paper's Table 1: 0.50 to 0.75; 0.5 for A1).</summary>
    public double AnomalyThreshold { get; set; } = 0.5;

    public DeepANTOptions() { }

    public DeepANTOptions(DeepANTOptions<T> other)
        : base(other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        WindowSize = other.WindowSize;
        ForecastHorizon = other.ForecastHorizon;
        Filters = other.Filters;
        KernelSize = other.KernelSize;
        PoolSize = other.PoolSize;
        HiddenUnits = other.HiddenUnits;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
        AnomalyThreshold = other.AnomalyThreshold;

        // Copy TimeSeriesRegressionOptions properties
        LagOrder = other.LagOrder;
        IncludeTrend = other.IncludeTrend;
        SeasonalPeriod = other.SeasonalPeriod;
        AutocorrelationCorrection = other.AutocorrelationCorrection;
        ModelType = other.ModelType;
        LossFunction = other.LossFunction;

        // Copy RegressionOptions properties
        DecompositionMethod = other.DecompositionMethod;
        UseIntercept = other.UseIntercept;
    }
}
