using System.Collections.Concurrent;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements the Autoformer model for long-term time series forecasting with decomposition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>The Long-Term Forecasting Challenge:</b>
/// Long-term time series forecasting requires models that can capture both fine-grained seasonal
/// patterns and long-term trends. Traditional approaches struggle because:
/// - RNNs have difficulty with long-range dependencies
/// - Transformers treat time series like text, ignoring continuous nature
/// - Neither explicitly models trend and seasonality separately
/// </para>
/// <para>
/// <b>The Autoformer Solution (Wu et al., NeurIPS 2021):</b>
/// Autoformer introduces three key innovations:
///
/// 1. <b>Series Decomposition Block:</b>
///    Progressive separation of trend and seasonal components at each layer.
///    Uses moving average to extract trend, remainder is seasonal.
///    Formula: Trend = MovingAvg(X), Seasonal = X - Trend
///
/// 2. <b>Auto-Correlation Mechanism:</b>
///    Replaces point-wise self-attention with period-based dependencies.
///    Finds correlations between sub-series and aggregates similar sub-sequences
///    based on their correlation strength.
///
/// 3. <b>Progressive Decomposition Architecture:</b>
///    Each encoder/decoder layer further refines the decomposition.
///    Seasonal and trend branches are processed separately and accumulated.
/// </para>
/// <para>
/// <b>For Beginners:</b> Autoformer is like having two experts work together:
/// - One expert focuses on the long-term direction (trend)
/// - One expert focuses on repeating patterns (seasonality)
///
/// Instead of looking at individual data points, it looks at how patterns repeat over time.
/// If today's pattern looks like last week's pattern, that's useful information!
///
/// Example use cases:
/// - Electricity demand forecasting (daily/weekly patterns)
/// - Retail sales prediction (seasonal buying patterns)
/// - Traffic flow prediction (rush hour patterns)
/// </para>
/// <para>
/// This class forecasts a univariate series (the reference's <c>features='S'</c>) with the network the official code
/// builds (<see cref="AutoformerNetwork{T}"/>, shared with the Finance Autoformer and checked against thuml/Autoformer),
/// and trains it as the reference's experiment does: the series standardized with training statistics, every window a
/// sample, Adam on the MSE over shuffled batches (the last incomplete batch dropped), the learning rate halved after every
/// epoch. The temporal embedding needs timestamps, which the time-series interface does not carry, so it is not used.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var inputMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var trainingLabels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var trainingMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Create an Autoformer with auto-correlation for long-range forecasting
/// var options = new AutoformerOptions&lt;double&gt; { LookbackWindow = 96, ForecastHorizon = 24 };
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new AutoformerModel&lt;double&gt;(options))
///     .Build(trainingMatrix, trainingLabels);
/// Vector&lt;double&gt; forecast = result.Predict(inputMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting", "https://arxiv.org/abs/2106.13008", Year = 2021, Authors = "Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long")]
public partial class AutoformerModel<T> : TimeSeriesModelBase<T>, ISupportsLossFunction<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Autoformer is a point forecaster trained with mean squared error (the reference's <c>loss='mse'</c>); any
    /// pointwise loss is meaningful.
    /// </remarks>
    public void SetLossFunction(ILossFunction<T> lossFunction) => ApplyLossFunction(lossFunction);

    private readonly AutoformerOptions<T> _options;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly List<LayerBase<T>> _layers = new();
    private readonly AutoformerNetwork<T> _network;
    private readonly Random _random;
    [Buffer]
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    // The reference's StandardScaler, fitted on the training series (mean and population standard deviation).
    private double _scaleMean;
    private double _scaleStd = 1.0;

    /// <summary>
    /// Initializes a new instance of the Autoformer model with the specified options.
    /// </summary>
    public AutoformerModel(AutoformerOptions<T>? options = null)
        : base(options ?? new AutoformerOptions<T>())
    {
        _options = (AutoformerOptions<T>)Options;
        var o = _options;
        if (o.LookbackWindow <= 0 || o.ForecastHorizon <= 0)
            throw new ArgumentException("LookbackWindow and ForecastHorizon must be positive.", nameof(options));
        if (o.LabelLength < 0 || o.LabelLength > o.LookbackWindow)
            throw new ArgumentException("LabelLength must be between 0 and LookbackWindow.", nameof(options));
        if (o.EmbeddingDim <= 0 || o.FeedForwardDim <= 0 || o.NumEncoderLayers <= 0 || o.NumDecoderLayers <= 0
            || o.NumAttentionHeads <= 0 || o.AutoCorrelationFactor <= 0)
            throw new ArgumentException("The Autoformer sizes must be positive.", nameof(options));
        if (o.MovingAverageKernel <= 0 || o.MovingAverageKernel % 2 == 0)
            throw new ArgumentException("MovingAverageKernel must be a positive odd number.", nameof(options));
        if (o.Epochs <= 0 || o.BatchSize <= 0 || o.LearningRate <= 0)
            throw new ArgumentException("Epochs, BatchSize and LearningRate must be positive.", nameof(options));

        int seed = o.Seed ?? 42;
        _random = RandomHelper.CreateSeededRandom(seed + 1);
        // Univariate forecasting (the reference's features='S'): one input channel, one output.
        _network = new AutoformerNetwork<T>(Engine, _layers, new AutoformerConfiguration(
            EncoderInputs: 1, DecoderInputs: 1, Outputs: 1, o.EmbeddingDim, o.NumAttentionHeads, o.NumEncoderLayers,
            o.NumDecoderLayers, o.FeedForwardDim, o.MovingAverageKernel, o.AutoCorrelationFactor, o.DropoutRate, o.UseGelu,
            TimeFeatures: 0, o.LabelLength, o.ForecastHorizon));
        _network.InitializeLikeReference(RandomHelper.CreateSeededRandom(seed));
    }

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        for (int i = 0; i < _layers.Count; i++)
            RegisterParameterComponent($"autoformer.layer{i}", _layers[i]);
    }

    private T Unscale(double value) => _numOps.FromDouble(value * _scaleStd + _scaleMean);

    // (x - mean) / std of every element.
    private Tensor<T> ScaleAll(Tensor<T> values)
        => Engine.TensorDivideScalar(Engine.TensorSubtractScalar(values, _numOps.FromDouble(_scaleMean)),
            _numOps.FromDouble(_scaleStd));

    // x * std + mean of every element.
    private Tensor<T> UnscaleAll(Tensor<T> values)
        => Engine.TensorAddScalar(Engine.TensorMultiplyScalar(values, _numOps.FromDouble(_scaleStd)),
            _numOps.FromDouble(_scaleMean));

    /// <summary>
    /// Trains the Autoformer as the reference's experiment does: the series is standardized with statistics of the
    /// training data, every window of LookbackWindow inputs and ForecastHorizon targets is a sample, and Adam minimizes the MSE of the forecast over shuffled batches (the last, incomplete batch dropped),
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
        _trainingSeries = y.Clone();
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = _numOps.FromDouble(y.Length);

        // Population mean and standard deviation of the training series, accumulated in double as master did.
        var series = Tensor<T>.FromVector(y);
        (_scaleMean, _scaleStd) = TimeSeriesEngineOps<T>.PopulationMeanStd(y);
        if (_scaleStd < 1e-12) _scaleStd = 1.0;
        // Every window of the scaled series: [samples, the history and then the target values of each sample].
        var windows = Engine.TensorUnfold(ScaleAll(series), 0, lookback + horizon, 1);

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
                var (history, target) = Batch(windows, order, start, batchSize);
                // Forward, loss, backward and the Adam update: one fused compiled plan when it applies (CPU or GPU),
                // the eager tape otherwise.
                T lossValue = TrainTapeBatch(Array.Empty<ITrainableLayer<T>>(), history, target,
                    step => _network.Forward(step, null, null, training: true, _random), loss.ComputeTapeLoss, optimizer,
                    extraParameters: parameters);
                lossSum += _numOps.ToDouble(lossValue);
                batches++;
            }
            optimizer.OnEpochEnd();
            if (batches > 0 && !ReportEpoch(epoch, timeBounded ? 0 : o.Epochs, _numOps.FromDouble(lossSum / batches)))
                break;
        }
    }

    // The windows of the given samples: history [B, LookbackWindow, 1] and the forecast target [B, ForecastHorizon, 1].
    private (Tensor<T> history, Tensor<T> target) Batch(Tensor<T> windows, int[] order, int start, int count)
    {
        int lookback = _options.LookbackWindow, horizon = _options.ForecastHorizon;
        var rows = new Tensor<int>(new[] { count }, new Vector<int>(new ArraySegment<int>(order, start, count)));
        var picked = Engine.TensorIndexSelect(windows, rows, 0);
        var history = Engine.TensorSlice(picked, new[] { 0, 0 }, new[] { count, lookback });
        var target = Engine.TensorSlice(picked, new[] { 0, lookback }, new[] { count, horizon });
        return (Engine.Reshape(history, new[] { count, lookback, 1 }), Engine.Reshape(target, new[] { count, horizon, 1 }));
    }

    // The unscaled forecast [1, ForecastHorizon] of the steps after a history of raw values, the last LookbackWindow
    // of them; a shorter history is extended to the left by repeating its first value.
    private Tensor<T> ForecastFrom(Vector<T> history) => ForecastFrom(history, new[] { 0 }, new[] { history.Length });

    // The unscaled forecasts [count, ForecastHorizon] of several histories in one forward pass (each window is
    // forecast independently).
    // History b is values[firsts[b] .. ends[b]); its window is the last LookbackWindow of them, extended to the left
    // by repeating values[firsts[b]].
    private Tensor<T> ForecastFrom(Vector<T> values, int[] firsts, int[] ends)
    {
        int lookback = _options.LookbackWindow, count = ends.Length;
        var positions = new int[count * lookback];
        for (int b = 0; b < count; b++)
        {
            if (ends[b] <= firsts[b]) throw new ArgumentException("Autoformer needs at least one observed value to forecast from.");
            for (int t = 0; t < lookback; t++) positions[b * lookback + t] = Math.Max(firsts[b], ends[b] - lookback + t);
        }
        var gathered = Engine.TensorIndexSelect(Tensor<T>.FromVector(values),
            new Tensor<int>(new[] { positions.Length }, new Vector<int>(positions)), 0);
        var window = Engine.Reshape(ScaleAll(gathered), new[] { count, lookback, 1 });
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        // Inference takes each series' own lags and draws nothing at random (dropout is off).
        var forecast = _network.Forward(window, null, null, training: false, _random);
        return UnscaleAll(Engine.Reshape(forecast, new[] { count, _options.ForecastHorizon }));
    }

    // The first forecast step of every row of [count, ForecastHorizon] forecasts, as [count, 1].
    private Tensor<T> FirstSteps(Tensor<T> forecasts)
        => Engine.TensorSlice(forecasts, new[] { 0, 0 }, new[] { forecasts.Shape[0], 1 });

    /// <summary>
    /// Predicts the series: with as many rows as the training series, the one-step forecast of each observation from
    /// the values before it; with one column (time steps), the forecasts of the next input.Rows steps after the
    /// training series, continued past ForecastHorizon by appending the forecasts to the history; otherwise each row is
    /// a history window and its prediction is <see cref="PredictSingle"/> of that row, all rows in one batched forward.
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
        if (n == 0) return predictions;
        if (n == _trainingSeries.Length)
        {
            var first = new Vector<T>(new[] { GuardPrediction(Unscale(0.0)) });
            if (n == 1) return first;
            var inSample = ForecastFrom(_trainingSeries, new int[n - 1], Enumerable.Range(1, n - 1).ToArray());
            return Vector<T>.Concatenate(first, GuardPredictions(FirstSteps(inSample)));
        }
        if (input.Columns == 1)
        {
            // Time steps after the series: the model's own forecast, continued from its forecasts.
            var history = _trainingSeries;
            int produced = 0;
            while (produced < n)
            {
                var forecast = GuardPredictions(ForecastFrom(history));
                var taken = forecast.GetSubVector(0, Math.Min(forecast.Length, n - produced));
                history = Vector<T>.Concatenate(history, taken);
                produced += taken.Length;
            }
            return history.GetSubVector(_trainingSeries.Length, n);
        }
        int columns = input.Columns;
        var firsts = Enumerable.Range(0, n).Select(i => i * columns).ToArray();
        var ends = Enumerable.Range(1, n).Select(i => i * columns).ToArray();
        return FirstSteps(ForecastFrom(input.ToRowVector(), firsts, ends)).ToVector();
    }

    /// <summary>
    /// Predicts the value after the given history (its last LookbackWindow values).
    /// </summary>
    public override T PredictSingle(Vector<T> input) => FirstSteps(ForecastFrom(input)).ToVector()[0];

    /// <summary>
    /// Forecasts the ForecastHorizon values after the given history in one forward pass.
    /// </summary>
    public Vector<T> PredictMultiple(Vector<T> input)
    {
        return ForecastFrom(input).ToVector();
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
                ["MovingAverageKernel"] = o.MovingAverageKernel,
                ["AutoCorrelationFactor"] = o.AutoCorrelationFactor,
            }
        };
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
        => new AutoformerModel<T>(new AutoformerOptions<T>(_options));

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
