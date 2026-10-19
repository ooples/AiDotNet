using AiDotNet.Optimizers;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements the Informer model for efficient long-sequence time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>The Long-Sequence Forecasting Problem:</b>
/// Traditional Transformer models achieve state-of-the-art results in many sequence modeling tasks,
/// but they struggle with long time series because self-attention has O(L^2) time and memory complexity.
/// For a sequence of 1000 time steps, vanilla attention requires 1 million operations per layer.
/// This makes long-horizon forecasting computationally prohibitive.
/// </para>
/// <para>
/// <b>The Informer Solution:</b>
/// Informer (Zhou et al., AAAI 2021) introduces three key innovations:
/// 1. ProbSparse Self-Attention (O(L log L) complexity)
/// 2. Self-Attention Distilling for sequence compression
/// 3. Generative-Style Decoder for parallel multi-step forecasting
/// </para>
/// <para>
/// This class forecasts a univariate series (the reference's <c>features='S'</c>) with the network the official code
/// builds (<see cref="InformerNetwork{T}"/>, shared with the Finance Informer and checked against Informer2020), and
/// trains it as the reference's experiment does. The temporal embedding needs timestamps, which the time-series
/// interface does not carry, so it is not used.
/// </para>
/// <para><b>For Beginners:</b> Informer makes transformers practical for long time series
/// forecasting. Regular transformers get very slow with long sequences because every time
/// step looks at every other time step. Informer speeds this up by only looking at the most
/// important connections (ProbSparse attention), compressing the sequence as it goes through
/// layers, and predicting all future values at once instead of one at a time.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create Informer model for efficient long-sequence time series forecasting
/// var options = new InformerOptions&lt;double&gt;();
///
/// // Prepare long-horizon time series data
/// var history = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
///     115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140 });
/// var trainingMatrix = new Matrix&lt;double&gt;(history.Length - 1, 1);
///
/// // Train using ProbSparse self-attention for O(L log L) efficiency
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new InformerModel&lt;double&gt;(options))
///     .Build(trainingMatrix, history.SubVector(1, history.Length - 1));
///
/// // Generate multi-step forecasts in parallel via generative decoder
/// var forecast = result.Predict(trainingMatrix);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", "https://arxiv.org/abs/2012.07436", Year = 2021, Authors = "Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, DecayRate = 0.5, StepSize = 1,
                Schedule = LearningRateSchedulerType.Step,
                ScheduleStepMode = SchedulerStepMode.StepPerEpoch,
                Source = "Zhou et al. 2021, Sec. 4: the proposed methods are optimized with Adam, its "
                        + "learning rate starting from 1e-4 and decaying two times smaller every epoch.")]
public partial class InformerModel<T> : TimeSeriesModelBase<T>, ISupportsLossFunction<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Informer is a point forecaster trained with mean squared error (the reference's <c>loss='mse'</c>); any
    /// pointwise loss is meaningful.
    /// </remarks>
    public void SetLossFunction(ILossFunction<T> lossFunction) => ApplyLossFunction(lossFunction);

    private readonly InformerOptions<T> _options;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly List<LayerBase<T>> _layers = new();
    private readonly InformerNetwork<T> _network;
    private readonly Random _random;
    [Buffer]
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    // The reference's StandardScaler, fitted on the training series (mean and population standard deviation).
    private double _scaleMean;
    private double _scaleStd = 1.0;

    /// <summary>
    /// Initializes a new instance of the Informer model with the specified options.
    /// </summary>
    public InformerModel(InformerOptions<T>? options = null)
        : base(options ?? new InformerOptions<T>())
    {
        _options = (InformerOptions<T>)Options;
        var o = _options;
        if (o.LookbackWindow <= 0 || o.ForecastHorizon <= 0)
            throw new ArgumentException("LookbackWindow and ForecastHorizon must be positive.", nameof(options));
        if (o.LabelLength < 0 || o.LabelLength > o.LookbackWindow)
            throw new ArgumentException("LabelLength must be between 0 and LookbackWindow.", nameof(options));
        if (o.EmbeddingDim <= 0 || o.FeedForwardDim <= 0 || o.NumEncoderLayers <= 0 || o.NumDecoderLayers <= 0
            || o.NumAttentionHeads <= 0 || o.SamplingFactor <= 0)
            throw new ArgumentException("The Informer sizes must be positive.", nameof(options));
        if (o.Epochs <= 0 || o.BatchSize <= 0 || o.LearningRate <= 0)
            throw new ArgumentException("Epochs, BatchSize and LearningRate must be positive.", nameof(options));

        var seeds = InformerNetwork<T>.Seeds(o.Seed);
        _random = RandomHelper.CreateSeededRandom(seeds.Sampling);
        // Univariate forecasting (the reference's features='S'): one input channel, one output.
        _network = new InformerNetwork<T>(Engine, _layers, new InformerConfiguration(
            EncoderInputs: 1, DecoderInputs: 1, Outputs: 1, o.EmbeddingDim, o.NumAttentionHeads, o.NumEncoderLayers,
            o.NumDecoderLayers, o.FeedForwardDim, o.SamplingFactor, o.DropoutRate, o.Distil, o.Mix, o.Attention, o.UseGelu,
            TimeFeatures: 0, o.LabelLength, o.ForecastHorizon));
        _network.InitializeLikeReference(RandomHelper.CreateSeededRandom(seeds.Initialization));
    }

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        for (int i = 0; i < _layers.Count; i++)
            RegisterParameterComponent($"informer.layer{i}", _layers[i]);
    }

    private T Scale(T value) => _numOps.FromDouble((_numOps.ToDouble(value) - _scaleMean) / _scaleStd);

    private T Unscale(double value) => _numOps.FromDouble(value * _scaleStd + _scaleMean);

    /// <summary>
    /// Trains the Informer as the reference's experiment does: the series is standardized with statistics of the
    /// training data, every window of LookbackWindow inputs and LabelLength + ForecastHorizon decoder steps is a
    /// sample, and Adam minimizes the MSE of the forecast over shuffled batches (the last, incomplete batch dropped),
    /// its learning rate halved after every epoch.
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        var o = _options;
        int lookback = o.LookbackWindow, horizon = o.ForecastHorizon;
        int samples = y.Length - lookback - horizon + 1;
        if (samples <= 0)
            throw new ArgumentException(
                $"Not enough data to build a single training sample. Require at least {lookback + horizon} points, got {y.Length}.", nameof(y));
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++) _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = _numOps.FromDouble(y.Length);

        double mean = 0;
        for (int i = 0; i < y.Length; i++) mean += _numOps.ToDouble(y[i]);
        mean /= y.Length;
        double variance = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double d = _numOps.ToDouble(y[i]) - mean;
            variance += d * d;
        }
        _scaleMean = mean;
        _scaleStd = Math.Sqrt(variance / y.Length);
        if (_scaleStd < 1e-12) _scaleStd = 1.0;
        var scaled = new T[y.Length];
        for (int i = 0; i < y.Length; i++) scaled[i] = Scale(y[i]);

        var optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(null, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = false,
            LearningRateScheduler = new StepLRScheduler(o.LearningRate, stepSize: 1, gamma: 0.5),
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch,
        });
        var loss = TrainingLoss;
        var parameters = Training.TapeTrainingStep<T>.CollectParameters(_layers);
        int batchSize = Math.Min(o.BatchSize, samples);
        bool timeBounded = o.MaxTrainingTimeSeconds > 0;
        int epochs = timeBounded ? int.MaxValue : o.Epochs;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            if (timeBounded && TrainingCancellationToken.IsCancellationRequested) break;
            TrainingCancellationToken.ThrowIfCancellationRequested();
            var order = Enumerable.Range(0, samples).OrderBy(_ => _random.Next()).ToArray();
            double lossSum = 0;
            int batches = 0;
            for (int start = 0; start + batchSize <= samples; start += batchSize)
            {
                if (timeBounded && TrainingCancellationToken.IsCancellationRequested) break;
                TrainingCancellationToken.ThrowIfCancellationRequested();
                var (history, target) = Batch(scaled, order, start, batchSize);
                // One update on the shared eager step: the network is trained directly, outside the model's Layers chain.
                T lossValue = Training.TapeTrainingStepper<T>.EagerObjectiveStep(
                    parameters, () => loss.ComputeTapeLoss(_network.Forward(history, null, _network.DecoderInput(history), null, training: true, _random), target), optimizer);
                lossSum += _numOps.ToDouble(lossValue);
                batches++;
            }
            optimizer.OnEpochEnd();
            if (batches > 0 && !ReportEpoch(epoch, timeBounded ? 0 : o.Epochs, _numOps.FromDouble(lossSum / batches)))
                break;
        }
    }

    // The windows of the given samples: history [B, LookbackWindow, 1] and the forecast target [B, ForecastHorizon, 1].
    private (Tensor<T> history, Tensor<T> target) Batch(T[] series, int[] order, int start, int count)
    {
        int lookback = _options.LookbackWindow, horizon = _options.ForecastHorizon;
        var history = new Tensor<T>(new[] { count, lookback, 1 });
        var target = new Tensor<T>(new[] { count, horizon, 1 });
        for (int b = 0; b < count; b++)
        {
            int s = order[start + b];
            for (int t = 0; t < lookback; t++) history[b, t, 0] = series[s + t];
            for (int h = 0; h < horizon; h++) target[b, h, 0] = series[s + lookback + h];
        }
        return (history, target);
    }

    // The forecast (unscaled) of the ForecastHorizon steps after a history of raw values, the last LookbackWindow of
    // them; a shorter history is extended to the left by repeating its first value.
    private double[] ForecastFrom(IReadOnlyList<T> history)
    {
        if (history.Count == 0) throw new ArgumentException("Informer needs at least one observed value to forecast from.");
        int lookback = _options.LookbackWindow;
        var window = new Tensor<T>(new[] { 1, lookback, 1 });
        int offset = history.Count - lookback;
        for (int t = 0; t < lookback; t++) window[0, t, 0] = Scale(history[Math.Max(0, offset + t)]);
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        var forecast = _network.Forward(window, null, _network.DecoderInput(window), null, training: false,
            InformerNetwork<T>.InferenceRandom(_options.Seed));
        var result = new double[_options.ForecastHorizon];
        for (int h = 0; h < result.Length; h++) result[h] = _numOps.ToDouble(Unscale(_numOps.ToDouble(forecast[0, h, 0])));
        return result;
    }

    /// <summary>
    /// Predicts the series: with as many rows as the training series, the one-step forecast of each observation from
    /// the values before it; otherwise the forecasts of the next input.Rows steps after the training series (the
    /// input values are not used), continued past ForecastHorizon by appending the forecasts to the history.
    /// </summary>
    /// <remarks>The first observation has no history to forecast from; its in-sample prediction is the training mean.
    /// Later observations with fewer than LookbackWindow predecessors use them, extended to the left by repeating the
    /// first.</remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (_trainingSeries.Length == 0) throw new InvalidOperationException("The model must be trained before predicting.");
        int n = input.Rows;
        var predictions = new Vector<T>(n);
        if (n == _trainingSeries.Length)
        {
            var observed = _trainingSeries.ToArray();
            predictions[0] = GuardPrediction(Unscale(0.0));
            for (int i = 1; i < n; i++)
                predictions[i] = GuardPrediction(_numOps.FromDouble(ForecastFrom(new ArraySegment<T>(observed, 0, i))[0]));
            return predictions;
        }
        var history = _trainingSeries.ToList();
        int produced = 0;
        while (produced < n)
        {
            var forecast = ForecastFrom(history);
            for (int h = 0; h < forecast.Length && produced < n; h++)
            {
                var value = GuardPrediction(_numOps.FromDouble(forecast[h]));
                predictions[produced++] = value;
                history.Add(value);
            }
        }
        return predictions;
    }

    /// <summary>
    /// Predicts the value after the given history (its last LookbackWindow values).
    /// </summary>
    public override T PredictSingle(Vector<T> input) => _numOps.FromDouble(ForecastFrom(input.ToArray())[0]);

    /// <summary>
    /// Forecasts the ForecastHorizon values after the given history in one forward pass of the generative decoder.
    /// </summary>
    public Vector<T> ForecastHorizon(Vector<T> input)
    {
        var forecast = ForecastFrom(input.ToArray());
        var result = new Vector<T>(forecast.Length);
        for (int h = 0; h < forecast.Length; h++) result[h] = _numOps.FromDouble(forecast[h]);
        return result;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = _options;
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["EmbeddingDim"] = o.EmbeddingDim,
                ["FeedForwardDim"] = o.FeedForwardDim,
                ["NumEncoderLayers"] = o.NumEncoderLayers,
                ["NumDecoderLayers"] = o.NumDecoderLayers,
                ["NumAttentionHeads"] = o.NumAttentionHeads,
                ["LookbackWindow"] = o.LookbackWindow,
                ["LabelLength"] = o.LabelLength,
                ["ForecastHorizon"] = o.ForecastHorizon,
                ["Attention"] = o.Attention.ToString(),
                ["SamplingFactor"] = o.SamplingFactor,
                ["Distil"] = o.Distil,
            }
        };
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
        => new InformerModel<T>(new InformerOptions<T>(_options));

    /// <inheritdoc />
    /// <remarks>The network's parameters are saved by the base through the registered layers; this adds the
    /// training series and the scaler.</remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_trainingSeries.Length);
        for (int i = 0; i < _trainingSeries.Length; i++) writer.Write(_numOps.ToDouble(_trainingSeries[i]));
        writer.Write(_scaleMean);
        writer.Write(_scaleStd);
    }

    /// <inheritdoc />
    protected override void DeserializeCore(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        _trainingSeries = new Vector<T>(length);
        for (int i = 0; i < length; i++) _trainingSeries[i] = _numOps.FromDouble(reader.ReadDouble());
        _scaleMean = reader.ReadDouble();
        _scaleStd = reader.ReadDouble();
    }
}
