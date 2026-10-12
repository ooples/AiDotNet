using AiDotNet.LearningRateSchedulers;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Base;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.TimeSeries;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Finance.Forecasting.Neural;

/// <summary>
/// DeepAR probabilistic autoregressive forecasting model using LSTM networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// DeepAR is a probabilistic forecasting model that produces forecast distributions rather than
/// point predictions. It uses autoregressive recurrent neural networks to learn temporal patterns
/// and outputs distribution parameters (e.g., mean and standard deviation for Gaussian).
/// </para>
/// <para>
/// <b>For Beginners:</b> DeepAR is special because it doesn't just predict a single value - it
/// predicts a probability distribution. This means you get:
/// - A most likely value (the median of simulated futures)
/// - A measure of confidence (the spread of those futures)
/// - The ability to generate prediction intervals (e.g., 95% confidence bounds)
///
/// Key features:
/// - <b>Autoregressive:</b> Each prediction depends on previous predictions
/// - <b>Probabilistic:</b> Outputs full distributions, not just point forecasts
/// - <b>Multi-series:</b> Can learn patterns across many related time series
/// - <b>Covariates:</b> Can include additional features like holidays or promotions
/// </para>
/// <para>
/// <b>Reference:</b> Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive
/// Recurrent Networks", International Journal of Forecasting 2020.
/// https://arxiv.org/abs/1704.04110
/// </para>
/// <para>
/// The network is the paper's (<see cref="AiDotNet.TimeSeries.DeepARNetwork{T}"/>, shared with
/// <see cref="AiDotNet.TimeSeries.DeepARModel{T}"/> and checked against a PyTorch transcription). Each input column is a
/// series; with several, a learned item embedding tells them apart (§3.4). Training maximizes the likelihood of the
/// whole window, input then target (§3.2); forecasts are medians or quantiles of ancestral sample paths.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 1, outputSize: 1);   // one series; [batch, LookbackWindow, 1] in, [batch, ForecastHorizon, 1] out
/// var model = new DeepAR&lt;double&gt;(architecture);
/// var onnxModel = new DeepAR&lt;double&gt;(architecture, "deepar.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", "https://arxiv.org/abs/1704.04110", Year = 2020, Authors = "David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski")]
[PaperOptimizer(OptimizerKind.Adam, Provenance = RecipeProvenance.PerDataset,
                Source = "Salinas et al. 2020, Sec. 4: Adam with early stopping. The paper tunes the learning rate manually per dataset, so there is no single value to declare and the provenance says so rather than the omission being silent.")]
public partial class DeepAR<T> : ForecastingModelBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly DeepAROptions<T> _options;
    private readonly DeepARLikelihood _likelihood;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly Random _random;
    private DeepARNetwork<T>? _network;

    public override ModelOptions GetOptions() => _options;

    public override int PatchSize => 1;

    public override int Stride => 1;

    /// <summary>Each input column is its own series: DeepAR is one global model over many series.</summary>
    public override bool IsChannelIndependent => true;

    private static DeepAROptions<T> OptionsOrDefault(DeepAROptions<T>? options) => options ?? new DeepAROptions<T>();

    /// <summary>
    /// Creates a DeepAR with the paper's defaults for a single series.
    /// </summary>
    public DeepAR()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 1,
            outputSize: 1))
    {
    }

    /// <summary>
    /// Creates a DeepAR that runs a pre-trained ONNX model.
    /// </summary>
    public DeepAR(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        DeepAROptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, onnxModelPath,
               OptionsOrDefault(options).LookbackWindow,
               OptionsOrDefault(options).ForecastHorizon,
               architecture.InputSize)
    {
        _options = OptionsOrDefault(options);
        Options = _options;
        _likelihood = Validate(_options);
        _random = RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate });
        SetBaseTrainOptimizer(_optimizer);
    }

    /// <summary>
    /// Creates a trainable DeepAR.
    /// </summary>
    /// <param name="architecture">The architecture; its input size is the number of series, each forecast by the one
    /// global model and, when there are several, told apart by a learned item embedding (§3.4).</param>
    /// <param name="options">The DeepAR options.</param>
    /// <param name="optimizer">The optimizer; Adam at <see cref="DeepAROptions{T}.LearningRate"/> when null.</param>
    /// <param name="lossFunction">Unused for training, which maximizes the likelihood; accepted for symmetry.</param>
    public DeepAR(
        NeuralNetworkArchitecture<T> architecture,
        DeepAROptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture,
               OptionsOrDefault(options).LookbackWindow,
               OptionsOrDefault(options).ForecastHorizon,
               architecture.InputSize,
               lossFunction)
    {
        _options = OptionsOrDefault(options);
        Options = _options;
        _likelihood = Validate(_options);
        _random = RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate });
        SetBaseTrainOptimizer(_optimizer);
        InitializeLayers();
    }

    private static DeepARLikelihood Validate(DeepAROptions<T> options)
    {
        if (options.HiddenSize <= 0) throw new ArgumentException("HiddenSize must be positive.");
        if (options.NumLayers <= 0) throw new ArgumentException("NumLayers must be positive.");
        if (options.NumSamples <= 0) throw new ArgumentException("NumSamples must be positive.");
        if (options.CovariateSize != 0)
            throw new ArgumentException("The tensor interface carries no future covariates; use DeepARModel for covariates.");
        return options.ResolveLikelihood();
    }

    /// <inheritdoc />
    /// <remarks>Builds the shared <see cref="DeepARNetwork{T}"/> (the paper's network); all of its layers are in
    /// <c>Layers</c>.</remarks>
    protected override void InitializeLayers()
    {
        if (!UseNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            throw new ArgumentException("DeepAR builds its own recurrent network; custom layers in the architecture are not supported.");
        var layers = new List<LayerBase<T>>();
        int items = NumFeatures > 1 ? NumFeatures : 0;
        _network = new DeepARNetwork<T>(Engine, layers, new DeepARConfiguration(
            0, _options.HiddenSize, _options.NumLayers, _options.DropoutRate, _likelihood, _options.ScaleHandling,
            items, items > 0 ? _options.EmbeddingDimension : 0));
        _network.InitializeLikeReference(RandomHelper.CreateSeededRandom(_options.Seed ?? 42));
        Layers.AddRange(layers);
    }

    public override bool SupportsTraining => UseNativeMode;

    /// <inheritdoc />
    /// <remarks>The likelihood is evaluated step by step with host-built inputs, so training runs the true forward
    /// every step rather than a traced replay.</remarks>
    protected override bool SupportsFusedCompiledTraining => false;

    /// <inheritdoc />
    /// <remarks>The forward is a recurrence, not a fold of <c>Layers</c>; the activations are those the forecast
    /// computes.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
        => UseNativeMode ? ObservedLayerActivations(input) : new Dictionary<string, Tensor<T>>();

    protected override Tensor<T> PredictCore(Tensor<T> input) => Forecast(input);

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "DeepAR",
            Description = $"DeepAR probabilistic forecasting ({_likelihood} likelihood)",
            Complexity = ParameterCount,
        };
        metadata.AdditionalInfo["SequenceLength"] = SequenceLength;
        metadata.AdditionalInfo["PredictionHorizon"] = PredictionHorizon;
        metadata.AdditionalInfo["NumFeatures"] = NumFeatures;
        metadata.AdditionalInfo["HiddenSize"] = _options.HiddenSize;
        metadata.AdditionalInfo["NumLayers"] = _options.NumLayers;
        metadata.AdditionalInfo["LikelihoodType"] = _likelihood.ToString();
        metadata.AdditionalInfo["NumSamples"] = _options.NumSamples;
        metadata.AdditionalInfo["UseNativeMode"] = UseNativeMode;
        return metadata;
    }

    // ------------------------------------------------------------------ data layout

    private DeepARNetwork<T> Network => _network ?? throw new InvalidOperationException("The native network is not initialized.");

    private Tensor<T> Batched(Tensor<T> input)
    {
        var x = input.Rank == 2 ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }) : input;
        if (x.Rank != 3 || x.Shape[2] != NumFeatures)
            throw new ArgumentException(
                $"DeepAR expects [batch, length, {NumFeatures}] (or [length, {NumFeatures}]), got [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
        return x;
    }

    // [B, L, S] → series rows [B·S, L + 1] (z₀ = 0: the input starts the series) and their item ids.
    private (double[,] Rows, int[]? Items) Series(Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1], series = x.Shape[2];
        var rows = new double[batch * series, length + 1];
        var items = series > 1 ? new int[batch * series] : null;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < series; s++)
            {
                int r = b * series + s;
                if (items is not null) items[r] = s;
                for (int t = 0; t < length; t++) rows[r, t + 1] = NumOps.ToDouble(x[b, t, s]);
            }
        return (rows, items);
    }

    private Tensor<T> Unrows(Func<int, int, double> value, int batch, int horizon, int series)
    {
        var y = new Tensor<T>(new[] { batch, horizon, series });
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < horizon; h++)
                for (int s = 0; s < series; s++) y[b, h, s] = NumOps.FromDouble(value(b * series + s, h));
        return y;
    }

    private Random InferenceRandom() => RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);

    // ------------------------------------------------------------------ forecasting

    /// <summary>
    /// Forecasts the PredictionHorizon steps after each series of the input <c>[B, L, S]</c>: the medians of
    /// NumSamples ancestral sample paths <c>[B, H, S]</c>, or with <paramref name="quantiles"/> the quantiles laid out
    /// as <c>[B, H, S·Q]</c> (series-major).
    /// </summary>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (historicalData is null) throw new ArgumentNullException(nameof(historicalData));
        return UseNativeMode ? ForecastNative(historicalData, quantiles) : ForecastOnnx(historicalData);
    }

    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        SetTrainingMode(false);
        var x = Batched(input);
        var (rows, items) = Series(x);
        var paths = Network.Sample(rows, null, null, PredictionHorizon, _options.NumSamples, InferenceRandom(), items);
        int batch = x.Shape[0], series = x.Shape[2], horizon = PredictionHorizon;
        double Q(int row, int h, double q)
        {
            var step = paths[row].Select(p => p[h]).OrderBy(v => v).ToArray();
            return step[Math.Max(0, Math.Min((int)Math.Floor(q * step.Length), step.Length - 1))];
        }
        if (quantiles is null || quantiles.Length == 0)
            return Unrows((row, h) => Q(row, h, 0.5), batch, horizon, series);
        var result = new Tensor<T>(new[] { batch, horizon, series * quantiles.Length });
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < horizon; h++)
                for (int s = 0; s < series; s++)
                    for (int k = 0; k < quantiles.Length; k++)
                        result[b, h, s * quantiles.Length + k] = NumOps.FromDouble(Q(b * series + s, h, quantiles[k]));
        return result;
    }

    /// <inheritdoc />
    /// <remarks>The differentiable mean path (<see cref="DeepARNetwork{T}.MeanPath"/>), <c>[B, H, S]</c>.</remarks>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input)
    {
        var x = Batched(input);
        var (rows, items) = Series(x);
        var path = Network.MeanPath(rows, null, null, PredictionHorizon, training: true, _random, items);   // [B·S, H]
        int batch = x.Shape[0], series = x.Shape[2];
        return Engine.TensorPermute(Engine.Reshape(path, new[] { batch, series, PredictionHorizon }), new[] { 0, 2, 1 });
    }

    // The window loss of an input [B, L, S] followed by its target [B, H, S]: the likelihood of every step (§3.2), the
    // input as the conditioning range.
    private Tensor<T> WindowLoss(Tensor<T> input, Tensor<T> target, bool training)
    {
        var x = Batched(input);
        var y = target.Rank == 2 ? Engine.Reshape(target, new[] { 1, target.Shape[0], target.Shape[1] }) : target;
        int batch = x.Shape[0], length = x.Shape[1], series = x.Shape[2], horizon = y.Shape[1];
        if (y.Rank != 3 || y.Shape[0] != batch || y.Shape[2] != series)
            throw new ArgumentException($"The target must be [{batch}, horizon, {series}].", nameof(target));
        var (history, items) = Series(x);
        var window = new double[batch * series, length + horizon + 1];
        for (int r = 0; r < batch * series; r++)
        {
            for (int t = 0; t <= length; t++) window[r, t] = history[r, t];
            for (int h = 0; h < horizon; h++) window[r, length + 1 + h] = NumOps.ToDouble(y[r / series, h, r % series]);
        }
        return Network.Loss(window, null, length, training, _random, items);
    }

    /// <inheritdoc />
    /// <remarks>One Adam step on the negative log-likelihood of the window (input, then target).</remarks>
    protected override void RunTrainingStep(Tensor<T> input, Tensor<T> expectedOutput)
        => TrainWithCustomLoss(input, _ => WindowLoss(input, expectedOutput, training: true), _optimizer);

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The negative log-likelihood training minimizes.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return WindowLoss(input, target, training: false)[0];
    }

    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps), "Steps must be at least 1.");
        var x = Batched(input);
        var (rows, items) = Series(x);
        var paths = Network.Sample(rows, null, null, steps, _options.NumSamples, InferenceRandom(), items);
        return Unrows((row, h) =>
        {
            var step = paths[row].Select(p => p[h]).OrderBy(v => v).ToArray();
            return step[step.Length / 2];
        }, x.Shape[0], steps, x.Shape[2]);
    }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (actuals is null) throw new ArgumentNullException(nameof(actuals));
        double mae = 0, mse = 0;
        int count = Math.Min(predictions.Length, actuals.Length);
        for (int i = 0; i < count; i++)
        {
            double d = NumOps.ToDouble(predictions.Data.Span[i]) - NumOps.ToDouble(actuals.Data.Span[i]);
            mae += Math.Abs(d);
            mse += d * d;
        }
        return new Dictionary<string, T>
        {
            ["MAE"] = NumOps.FromDouble(mae / Math.Max(count, 1)),
            ["MSE"] = NumOps.FromDouble(mse / Math.Max(count, 1)),
            ["RMSE"] = NumOps.FromDouble(Math.Sqrt(mse / Math.Max(count, 1))),
        };
    }

    /// <inheritdoc />
    /// <remarks>Each series divided by the paper's scale ν of its own values (§3.3), as the network sees it.</remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        if (!UseNativeMode) return input;
        var x = Batched(input);
        var (rows, _) = Series(x);
        int batch = x.Shape[0], length = x.Shape[1], series = x.Shape[2];
        var values = new double[rows.GetLength(0), length];
        for (int r = 0; r < values.GetLength(0); r++)
            for (int t = 0; t < length; t++) values[r, t] = rows[r, t + 1];
        var scale = Network.Scale(values);
        var y = new Tensor<T>(new[] { batch, length, series });
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < length; t++)
                for (int s = 0; s < series; s++) y[b, t, s] = NumOps.FromDouble(values[b * series + s, t] / scale[b * series + s]);
        return y;
    }

    public override Dictionary<string, T> GetFinancialMetrics()
    {
        return new Dictionary<string, T>
        {
            ["SequenceLength"] = NumOps.FromDouble(SequenceLength),
            ["PredictionHorizon"] = NumOps.FromDouble(PredictionHorizon),
            ["NumFeatures"] = NumOps.FromDouble(NumFeatures),
            ["NumSamples"] = NumOps.FromDouble(_options.NumSamples),
            ["ParameterCount"] = NumOps.FromDouble(GetParameterCount()),
        };
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");
        var inputName = OnnxSession.InputMetadata.Keys.First();
        var onnxInput = new OnnxTensors.DenseTensor<float>(
            input.ToArray().Select(v => Convert.ToSingle(v)).ToArray(), input._shape.ToArray());
        using var results = OnnxSession.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) });
        var output = results.First().AsTensor<float>();
        return new Tensor<T>(output.ToArray().Select(v => NumOps.FromDouble(v)).ToArray(), output.Dimensions.ToArray());
    }

    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> prediction, int stepsUsed)
    {
        int batch = input.Shape[0], length = input.Shape[1], series = input.Shape[2];
        var shifted = new Tensor<T>(new[] { batch, length, series });
        int keep = length - stepsUsed;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < series; s++)
            {
                for (int t = 0; t < keep; t++) shifted[b, t, s] = input[b, t + stepsUsed, s];
                for (int t = 0; t < stepsUsed; t++) shifted[b, keep + t, s] = prediction[b, t, s];
            }
        return shifted;
    }

    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        if (predictions.Count == 0) return new Tensor<T>(new[] { 1, totalSteps, NumFeatures });
        int batch = predictions[0].Shape[0];
        var result = new Tensor<T>(new[] { batch, totalSteps, NumFeatures });
        int step = 0;
        foreach (var p in predictions)
        {
            int add = Math.Min(p.Shape[1], totalSteps - step);
            for (int b = 0; b < batch; b++)
                for (int t = 0; t < add; t++)
                    for (int f = 0; f < NumFeatures; f++) result[b, step + t, f] = p[b, t, f];
            step += add;
        }
        return result;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) OnnxSession?.Dispose();
        base.Dispose(disposing);
    }
}
