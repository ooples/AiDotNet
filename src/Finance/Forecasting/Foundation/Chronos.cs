using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Base;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.TimeSeries;

namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// Chronos-T5 (Ansari et al. 2024) with the finance models' tensor API: probabilistic forecasting by a T5 language
/// model over mean-scaled, uniformly binned time series tokens.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This model and <see cref="ChronosFoundationModel{T}"/> share one network, which matches the official
/// chronos-forecasting code and Hugging Face's T5 to 1e-12: the MeanScaleUniformBins tokenizer, the T5 v1.0
/// encoder-decoder, the token cross-entropy, and forecasts as the median of sampled paths (temperature, top-k, top-p).
/// The released checkpoints load with <see cref="LoadPretrainedWeights"/> into a model built from
/// <see cref="ChronosOptions{T}.FromPretrainedConfig"/>.
/// </para>
/// <para>
/// <see cref="Forecast"/> reads a context <c>[L]</c> (or a batch <c>[B, L]</c>) and returns the PredictionLength-step
/// median forecast; with quantiles it returns them instead, quantile-major. Training takes a context and the
/// PredictionLength values after it (or a batch of each) and takes one step of the paper's recipe (AdamW with weight
/// decay 0.01, the learning rate annealed linearly to zero over TrainingSteps) on the cross-entropy of their tokens.
/// </para>
/// <para><b>For Beginners:</b> Chronos turns numbers into "words" (bins) and uses a language model to predict the next
/// words. Sampling many possible continuations and taking their median gives the forecast; their spread gives its
/// uncertainty.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = new ChronosFinanceOptions&lt;double&gt;(ChronosOptions&lt;double&gt;.FromPretrainedConfig("chronos-t5-small/config.json"));
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression, inputSize: options.ContextLength, outputSize: options.PredictionLength);
/// var model = new Chronos&lt;double&gt;(architecture, options);
/// model.LoadPretrainedWeights("chronos-t5-small/model.safetensors");
/// Tensor&lt;double&gt; forecast = model.Forecast(context);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Chronos: Learning the Language of Time Series", "https://arxiv.org/abs/2403.07815", Year = 2024, Authors = "Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, Yuyang Wang")]
public partial class Chronos<T> : TimeSeriesFoundationModelBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly ChronosFinanceOptions<T>? _options;
    private ChronosT5<T>? _core;
    private Random? _trainingRandom;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <summary>
    /// Creates a Chronos-T5, randomly initialized as the official training does.
    /// </summary>
    /// <param name="architecture">The architecture; its input and output sizes are the context and prediction
    /// lengths. Custom layers are not supported.</param>
    /// <param name="options">The sizes, decoding and training settings; the defaults are Chronos-T5 (Small) and the
    /// paper's recipe.</param>
    /// <param name="optimizer">An optimizer to train with instead of the paper's AdamW.</param>
    public Chronos(
        NeuralNetworkArchitecture<T> architecture,
        ChronosFinanceOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new ChronosFinanceOptions<T>();
        Options = _options;
        InitializeLayers();
        _optimizer = optimizer ?? Core.CreateOptimizer(Training());
        SetBaseTrainOptimizer(_optimizer);
    }

    private ChronosT5<T> Core => _core ?? throw new InvalidOperationException("The Chronos network is not built.");

    private ChronosTraining Training() => new(_options!.LearningRate, _options.WeightDecay, _options.TrainingSteps,
        _options.BatchSize, _options.MinPast, _options.MissingValueProbability, _options.MaxGradNorm);

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (_options is null || _core is not null) return;
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("Chronos builds Chronos-T5 from its options; custom layers are not supported.");
        var layers = new List<LayerBase<T>>();
        _core = new ChronosT5<T>(Engine, layers, _options.Configuration());
        _core.InitializeLikeReference(RandomHelper.CreateSeededRandom(_options.Seed ?? 42));
        _trainingRandom = RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1000);
        Layers.AddRange(layers);
    }

    /// <summary>Loads a released Chronos-T5 checkpoint's <c>model.safetensors</c>; build the model with
    /// <see cref="ChronosOptions{T}.FromPretrainedConfig"/> of the same checkpoint.</summary>
    public void LoadPretrainedWeights(string path) => Core.LoadPretrainedWeights(path);

    #region Properties

    /// <inheritdoc/>
    public override int SequenceLength => _options!.ContextLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _options!.PredictionLength;

    /// <inheritdoc/>
    public override int NumFeatures => 1;

    /// <inheritdoc/>
    public override int PatchSize => 1;

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public override bool UseNativeMode => true;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>The vocabulary: the bins plus PAD and EOS.</summary>
    public int NumTokens => _options!.NumTokens;

    /// <summary>The released size whose d_model is nearest this model's (Tiny 256, Mini 384, Small 512, Base 768,
    /// Large 1024).</summary>
    public override FoundationModelSize ModelSize
    {
        get
        {
            var sizes = new[] { (256, FoundationModelSize.Tiny), (384, FoundationModelSize.Mini), (512, FoundationModelSize.Small),
                (768, FoundationModelSize.Base), (1024, FoundationModelSize.Large) };
            return sizes.OrderBy(s => Math.Abs(s.Item1 - _options!.ModelDim)).First().Item2;
        }
    }

    /// <summary>The context the model reads; a longer history is truncated to its last ContextLength values.</summary>
    public override int MaxContextLength => _options!.ContextLength;

    /// <summary>The steps one sampling pass predicts; <see cref="AutoregressiveForecast"/> continues past it.</summary>
    public override int MaxPredictionHorizon => _options!.PredictionLength;

    #endregion

    #region Forecasting

    // Each forecast draws from a sampler seeded by Options.Seed, so the same call returns the same forecast.
    private Random Sampler() => RandomHelper.CreateSeededRandom((_options!.Seed ?? 42) + 7);

    private double[][] Rows(Tensor<T> values, string name)
    {
        var data = values.ToArray();
        int[] shape = values.Shape.ToArray();
        int rows, length;
        if (shape.Length == 1) (rows, length) = (1, shape[0]);
        else if (shape.Length == 2) (rows, length) = (shape[0], shape[1]);
        else if (shape.Length == 3 && shape[2] == 1) (rows, length) = (shape[0], shape[1]);
        else throw new ArgumentException($"Chronos reads series [L], [B, L] or [B, L, 1]; got [{string.Join(", ", shape)}].", name);
        if (length == 0) throw new ArgumentException("A series has no values.", name);
        var result = new double[rows][];
        for (int r = 0; r < rows; r++)
        {
            result[r] = new double[length];
            for (int t = 0; t < length; t++) result[r][t] = NumOps.ToDouble(data[r * length + t]);
        }
        return result;
    }

    private Tensor<T> Shaped(Tensor<T> input, double[][] rows, int width)
    {
        bool single = input.Rank == 1;
        var output = new Tensor<T>(single ? new[] { width } : new[] { rows.Length, width });
        for (int r = 0; r < rows.Length; r++)
            for (int h = 0; h < width; h++) output[r * width + h] = NumOps.FromDouble(rows[r][h]);
        return output;
    }

    private double[][] Medians(double[][] contexts, int steps)
    {
        var result = new double[contexts.Length][];
        for (int r = 0; r < contexts.Length; r++)
        {
            var paths = Core.SamplePaths(contexts[r], steps, Sampler());
            result[r] = new double[steps];
            for (int h = 0; h < steps; h++) result[r][h] = ChronosT5<T>.Median(paths.Select(p => p[h]));
        }
        return result;
    }

    /// <summary>
    /// The PredictionLength-step forecast of a context <c>[L]</c> (or of each row of <c>[B, L]</c>): the median of the
    /// sampled paths, or, given <paramref name="quantiles"/>, those quantiles of them (torch.quantile's linear
    /// interpolation), quantile-major.
    /// </summary>
    protected override Tensor<T> ForecastCore(Tensor<T> historicalData, double[]? quantiles)
    {
        if (historicalData is null) throw new ArgumentNullException(nameof(historicalData));
        var contexts = Rows(historicalData, nameof(historicalData));
        int steps = _options!.PredictionLength;
        if (quantiles is null || quantiles.Length == 0) return Shaped(historicalData, Medians(contexts, steps), steps);
        if (quantiles.Any(q => !(q >= 0 && q <= 1))) throw new ArgumentOutOfRangeException(nameof(quantiles), "Quantiles must be in [0, 1].");
        var result = new double[contexts.Length][];
        for (int r = 0; r < contexts.Length; r++)
        {
            var paths = Core.SamplePaths(contexts[r], steps, Sampler());
            result[r] = new double[quantiles.Length * steps];
            for (int q = 0; q < quantiles.Length; q++)
                for (int h = 0; h < steps; h++)
                    result[r][q * steps + h] = ChronosT5<T>.Quantile(paths.Select(p => p[h]), quantiles[q]);
        }
        return Shaped(historicalData, result, quantiles.Length * steps);
    }

    /// <summary>The <paramref name="steps"/>-step median forecast, continuing past PredictionLength as the official
    /// pipeline does (the context extended by the median so far).</summary>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps), "The number of steps must be positive.");
        return Shaped(input, Medians(Rows(input, nameof(input)), steps), steps);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input) => Forecast(input, quantiles: null);

    /// <summary>
    /// The encoder states of the context's tokens (the official pipeline's <c>embed</c>): <c>[L + 1, d_model]</c> for
    /// a context <c>[L]</c> (the last ContextLength values and EOS).
    /// </summary>
    public override Tensor<T> Embed(Tensor<T> series)
    {
        if (series is null) throw new ArgumentNullException(nameof(series));
        var rows = Rows(series, nameof(series));
        if (rows.Length != 1) throw new ArgumentException("Embed reads one series.", nameof(series));
        var (ids, mask, _) = Core.Tokenizer.Context(rows[0]);
        var tokens = new Tensor<T>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) tokens[i] = NumOps.FromDouble(ids[i]);
        return Core.Network.Encode(tokens, IsTrainingMode, _trainingRandom!, mask);
    }

    /// <summary>The differentiable function of the input this model defines: the encoder states (see
    /// <see cref="Embed"/>). Training does not use it; it takes the token cross-entropy of a context and its future.</summary>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input) => Embed(input);

    #endregion

    #region Training

    /// <summary>
    /// One step of the paper's recipe on contexts and the PredictionLength values after each (one series each, or a
    /// batch of rows): the cross-entropy of the future's tokens, averaged over the batch's tokens, with dropout on.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var windows = Windows(input, target);
        var parameters = AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(Layers.OfType<LayerBase<T>>().ToList());
        LastLoss = AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(parameters,
            () => Core.BatchLoss(windows, training: true, _trainingRandom!), _optimizer!,
            onGradients: gradients => PublishParameterGradients(gradients));
        return true;
    }

    private List<(double[] Context, double[] Future)> Windows(Tensor<T> input, Tensor<T> target)
    {
        var contexts = Rows(input, nameof(input));
        var futures = Rows(target, nameof(target));
        if (contexts.Length != futures.Length)
            throw new ArgumentException($"{contexts.Length} contexts but {futures.Length} targets.", nameof(target));
        if (futures[0].Length != _options!.PredictionLength)
            throw new ArgumentException(
                $"Chronos trains on the PredictionLength ({_options.PredictionLength}) values after each context; got {futures[0].Length}.",
                nameof(target));
        return contexts.Zip(futures, (c, f) => (c, f)).ToList();
    }

    /// <summary>Chronos trains on a supervised objective: the token cross-entropy of a context's future.</summary>
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    /// <summary>The future values are the target as given.</summary>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <summary>The objective training descends, without dropout: the token cross-entropy of the target given the
    /// input, averaged over the batch's tokens. (Its forecasts are sampled values, outside that loss's domain.)</summary>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return Core.BatchLoss(Windows(input, target), training: false, _trainingRandom!)[0];
    }

    /// <summary>The activations of the layer calls a forecast makes (the network is not a chain of its layers).</summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    #endregion

    #region Metrics

    /// <inheritdoc/>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        T mse = NumOps.Zero, mae = NumOps.Zero;
        int count = Math.Min(predictions.Length, actuals.Length);
        for (int i = 0; i < count; i++)
        {
            var diff = NumOps.Subtract(predictions[i], actuals[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
        }
        if (count > 0)
        {
            mse = NumOps.Divide(mse, NumOps.FromDouble(count));
            mae = NumOps.Divide(mae, NumOps.FromDouble(count));
        }
        return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) };
    }

    /// <summary>Chronos normalizes inside its tokenizer (mean scaling), so the input passes through unchanged.</summary>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input) => input;

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_options!.ContextLength),
            ["PredictionLength"] = NumOps.FromDouble(_options.PredictionLength),
            ["NumTokens"] = NumOps.FromDouble(_options.NumTokens),
            ["ModelDim"] = NumOps.FromDouble(_options.ModelDim),
            ["NumLayers"] = NumOps.FromDouble(_options.NumLayers),
            ["NumSamples"] = NumOps.FromDouble(_options.NumSamples),
            ["LastLoss"] = LastLoss is not null ? LastLoss : NumOps.Zero,
        };
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "Chronos-T5",
            Description = "Chronos-T5: a T5 language model over mean-scaled, uniformly binned time series tokens",
            Complexity = ParameterCount,
            FeatureCount = 1,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ContextLength", _options!.ContextLength },
                { "PredictionLength", _options.PredictionLength },
                { "NumTokens", _options.NumTokens },
                { "ModelDim", _options.ModelDim },
                { "NumLayers", _options.NumLayers },
                { "ModelSize", ModelSize.ToString() },
            }
        };
    }

    #endregion
}
