using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements Chronos-T5 (Ansari et al. 2024), a language model over tokenized time series, for probabilistic
/// forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Chronos scales a series by the mean absolute value of its context and quantizes each scaled value into one of
/// NumTokens − 2 uniform bins on [−15, 15] (§3.1); a T5 encoder-decoder reads the context's tokens and is trained to
/// predict the future's with the cross-entropy (§3.2). A forecast samples token paths autoregressively and maps them
/// back to values; the point forecast is the median of the samples (§5). The tokenizer, the sampling and the training
/// follow the official chronos-forecasting code, and the model matches it and Hugging Face's T5 to 1e-12.
/// </para>
/// <para>
/// The defaults are Chronos-T5 (Small) with the paper's training recipe (§5.2): AdamW with weight decay 0.01, the
/// learning rate 0.001 annealed linearly to zero over 200K steps of 256 windows each, and the transformers defaults for
/// everything else. Training on a single series is far shorter in practice; set TrainingSteps and BatchSize, or load
/// a released checkpoint with <see cref="LoadPretrainedWeights"/> after building the model with
/// <see cref="ChronosOptions{T}.FromPretrainedConfig"/>.
/// </para>
/// <para><b>For Beginners:</b> Chronos turns numbers into "words" (bins) and uses a language model to predict the next
/// words. Sampling many possible continuations and taking their median gives the forecast; their spread gives its
/// uncertainty.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = ChronosOptions&lt;double&gt;.FromPretrainedConfig("chronos-t5-small/config.json");
/// var model = new ChronosFoundationModel&lt;double&gt;(options);
/// model.LoadPretrainedWeights("chronos-t5-small/model.safetensors");
/// Vector&lt;double&gt; forecast = model.Forecast(history, 24);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Chronos: Learning the Language of Time Series", "https://arxiv.org/abs/2403.07815", Year = 2024, Authors = "Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, Yuyang Wang")]
public partial class ChronosFoundationModel<T> : TimeSeriesModelBase<T>
{
    private readonly ChronosOptions<T> _options;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly List<NeuralNetworks.Layers.LayerBase<T>> _layers = new();
    private readonly ChronosT5<T> _core;
    private readonly Random _random;
    [Buffer]
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new Chronos-T5 model, randomly initialized as the official training does.
    /// </summary>
    public ChronosFoundationModel(ChronosOptions<T>? options = null)
        : base(options ?? new ChronosOptions<T>())
    {
        _options = (ChronosOptions<T>)Options;
        _core = new ChronosT5<T>(Engine, _layers, _options.Configuration());
        _core.InitializeLikeReference(RandomHelper.CreateSeededRandom(SeedOr(42)));
        _random = RandomHelper.CreateSeededRandom(SeedOr(1042));
    }

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        for (int i = 0; i < _layers.Count; i++)
            RegisterParameterComponent($"chronos.layer{i}", _layers[i]);
    }

    /// <summary>Loads a released Chronos-T5 checkpoint's <c>model.safetensors</c>; build the model with
    /// <see cref="ChronosOptions{T}.FromPretrainedConfig"/> of the same checkpoint.</summary>
    public void LoadPretrainedWeights(string path)
    {
        _core.LoadPretrainedWeights(path);
        if (_trainingSeries.Length == 0) IsTrained = true;
    }

    /// <summary>
    /// Trains on the series <paramref name="y"/> (the input carries only its time index) with the official recipe:
    /// random windows of the series, missing-value augmentation, and AdamW on the token cross-entropy (see
    /// <see cref="ChronosOptions{T}"/>). MaxTrainingTimeSeconds, when set, ends training early.
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        var o = _options;
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++) _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = _numOps.FromDouble(y.Length);
        var series = new double[y.Length];
        for (int i = 0; i < y.Length; i++) series[i] = _numOps.ToDouble(y[i]);
        bool timeBounded = o.MaxTrainingTimeSeconds > 0;
        _core.Train(series, new ChronosTraining(o.LearningRate, o.WeightDecay, o.TrainingSteps, o.BatchSize, o.MinPast,
                o.MissingValueProbability, o.MaxGradNorm),
            _random, TrainingCancellationToken, timeBounded,
            (step, loss) => ReportEpoch(step, o.TrainingSteps, _numOps.FromDouble(loss)));
    }

    private double[] History(Vector<T> history)
    {
        if (history is null) throw new ArgumentNullException(nameof(history));
        if (history.Length == 0) throw new ArgumentException("Chronos needs at least one observed value to forecast from.", nameof(history));
        var values = new double[history.Length];
        for (int i = 0; i < values.Length; i++) values[i] = _numOps.ToDouble(history[i]);
        return values;
    }

    // Each forecast draws from a sampler seeded by Options.Seed, so the same call returns the same forecast.
    private Random Sampler() => RandomHelper.CreateSeededRandom(SeedOr(7));

    /// <summary>
    /// The NumSamples sample paths of the <paramref name="steps"/> values after <paramref name="history"/> (the
    /// official pipeline's <c>predict</c>), one row per sample.
    /// </summary>
    public Matrix<T> SampleForecasts(Vector<T> history, int steps)
    {
        var paths = _core.SamplePaths(History(history), steps, Sampler());
        var result = new Matrix<T>(paths.Length, steps);
        for (int s = 0; s < paths.Length; s++)
            for (int h = 0; h < steps; h++) result[s, h] = _numOps.FromDouble(paths[s][h]);
        return result;
    }

    /// <summary>The point forecast of the <paramref name="steps"/> values after <paramref name="history"/>: the median
    /// of the sample paths at each step.</summary>
    public override Vector<T> Forecast(Vector<T> history, int steps)
    {
        var paths = _core.SamplePaths(History(history), steps, Sampler());
        var forecast = new Vector<T>(steps);
        for (int h = 0; h < steps; h++) forecast[h] = _numOps.FromDouble(ChronosT5<T>.Median(paths.Select(p => p[h])));
        return forecast;
    }

    /// <summary>
    /// The given quantiles of the sample paths at each of the <paramref name="steps"/> forecast steps (torch.quantile's
    /// linear interpolation, as the official <c>predict_quantiles</c>).
    /// </summary>
    public Dictionary<double, Vector<T>> ForecastWithQuantiles(Vector<T> history, double[] quantiles, int steps)
    {
        if (quantiles is null || quantiles.Length == 0) throw new ArgumentException("Give at least one quantile.", nameof(quantiles));
        if (quantiles.Any(q => !(q >= 0 && q <= 1))) throw new ArgumentException("Quantiles must be in [0, 1].", nameof(quantiles));
        var paths = _core.SamplePaths(History(history), steps, Sampler());
        var result = new Dictionary<double, Vector<T>>();
        foreach (double q in quantiles)
        {
            var values = new Vector<T>(steps);
            for (int h = 0; h < steps; h++) values[h] = _numOps.FromDouble(ChronosT5<T>.Quantile(paths.Select(p => p[h]), q));
            result[q] = values;
        }
        return result;
    }

    /// <summary>
    /// Predicts the series. A one-column time index is answered with forecasts: with as many rows as the training
    /// series, the one-step forecast of each observation from the observations before it (up to ContextLength);
    /// otherwise the forecasts of the steps after the training series. Any other input is a matrix of contexts, each
    /// row forecast one step.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (TryForecastTimeIndex(input, _trainingSeries, _options.ContextLength, PredictSingle, out var forecasts, Forecast,
                extendShortHistory: false))
            return forecasts;
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++) predictions[i] = PredictSingle(input.GetRow(i));
        return predictions;
    }

    /// <summary>The one-step point forecast after the context <paramref name="input"/>.</summary>
    public override T PredictSingle(Vector<T> input) => Forecast(input, 1)[0];

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
                { "ContextLength", _options.ContextLength },
                { "PredictionLength", _options.PredictionLength },
                { "NumTokens", _options.NumTokens },
                { "ModelDim", _options.ModelDim },
                { "NumLayers", _options.NumLayers },
                { "NumSamples", _options.NumSamples }
            }
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
        => new ChronosFoundationModel<T>(new ChronosOptions<T>(_options));

    /// <inheritdoc />
    /// <remarks>The network's parameters are saved by the base through the registered layers; this adds the
    /// training series.</remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_trainingSeries.Length);
        for (int i = 0; i < _trainingSeries.Length; i++) writer.Write(_numOps.ToDouble(_trainingSeries[i]));
    }

    /// <inheritdoc />
    protected override void DeserializeCore(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        _trainingSeries = new Vector<T>(length);
        for (int i = 0; i < length; i++) _trainingSeries[i] = _numOps.FromDouble(reader.ReadDouble());
    }
}

/// <summary>
/// Options for <see cref="ChronosFoundationModel{T}"/>. The defaults are Chronos-T5 (Small) and the paper's training
/// recipe (Ansari et al. 2024, §5.2); the released sizes are Tiny (d_model 256, d_ff 1024, 4 heads, 4 layers), Mini
/// (384, 1536, 8, 4), Small (512, 2048, 8, 6), Base (768, 3072, 12, 12) and Large (1024, 4096, 16, 24), all with
/// d_kv 64.
/// </summary>
public class ChronosOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>The context the model reads: the last ContextLength values (the paper's 512).</summary>
    public int ContextLength { get; set; } = 512;

    /// <summary>The steps one sampling pass predicts, and the label length in training (the paper's 64).</summary>
    public int PredictionLength { get; set; } = 64;

    /// <summary>The vocabulary: the bins plus PAD and EOS (the paper's 4096).</summary>
    public int NumTokens { get; set; } = 4096;

    /// <summary>The lowest bin center, in units of the mean scale (the paper's −15).</summary>
    public double LowLimit { get; set; } = -15.0;

    /// <summary>The highest bin center, in units of the mean scale (the paper's 15).</summary>
    public double HighLimit { get; set; } = 15.0;

    /// <summary>T5's d_model.</summary>
    public int ModelDim { get; set; } = 512;

    /// <summary>T5's d_ff.</summary>
    public int FeedForwardDim { get; set; } = 2048;

    /// <summary>T5's d_kv, the width of each attention head.</summary>
    public int KeyValueDim { get; set; } = 64;

    /// <summary>T5's attention heads.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>The encoder's and the decoder's blocks.</summary>
    public int NumLayers { get; set; } = 6;

    /// <summary>T5's dropout rate (the released configurations' 0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>T5's relative attention buckets.</summary>
    public int RelativeAttentionBuckets { get; set; } = 32;

    /// <summary>T5's relative attention maximum distance.</summary>
    public int RelativeAttentionMaxDistance { get; set; } = 128;

    /// <summary>The sample paths a forecast draws (the paper's 20).</summary>
    public int NumSamples { get; set; } = 20;

    /// <summary>The sampling temperature (the official 1).</summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>The top-k cut of sampling (the official 50).</summary>
    public int TopK { get; set; } = 50;

    /// <summary>The nucleus cut of sampling (the official 1, none).</summary>
    public double TopP { get; set; } = 1.0;

    /// <summary>AdamW's learning rate, decayed linearly to zero over TrainingSteps (the paper's 0.001).</summary>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>AdamW's weight decay (the paper's 0.01).</summary>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>The optimizer steps (the paper's 200K).</summary>
    public int TrainingSteps { get; set; } = 200_000;

    /// <summary>The windows per step (the paper's effective batch of 256).</summary>
    public int BatchSize { get; set; } = 256;

    /// <summary>The values a training window's split point is at least past the start (the official 60).</summary>
    public int MinPast { get; set; } = 60;

    /// <summary>The largest per-window probability of dropping a value to missing in training (the official 0.2).</summary>
    public double MissingValueProbability { get; set; } = 0.2;

    /// <summary>The global gradient norm clip (the transformers default 1; 0 disables it).</summary>
    public double MaxGradNorm { get; set; } = 1.0;

    public ChronosOptions() { }

    public ChronosOptions(ChronosOptions<T> other)
        : base(other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength;
        PredictionLength = other.PredictionLength;
        NumTokens = other.NumTokens;
        LowLimit = other.LowLimit;
        HighLimit = other.HighLimit;
        ModelDim = other.ModelDim;
        FeedForwardDim = other.FeedForwardDim;
        KeyValueDim = other.KeyValueDim;
        NumHeads = other.NumHeads;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        RelativeAttentionBuckets = other.RelativeAttentionBuckets;
        RelativeAttentionMaxDistance = other.RelativeAttentionMaxDistance;
        NumSamples = other.NumSamples;
        Temperature = other.Temperature;
        TopK = other.TopK;
        TopP = other.TopP;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        TrainingSteps = other.TrainingSteps;
        BatchSize = other.BatchSize;
        MinPast = other.MinPast;
        MissingValueProbability = other.MissingValueProbability;
        MaxGradNorm = other.MaxGradNorm;
        LagOrder = other.LagOrder;
        IncludeTrend = other.IncludeTrend;
        SeasonalPeriod = other.SeasonalPeriod;
        AutocorrelationCorrection = other.AutocorrelationCorrection;
        ModelType = other.ModelType;
        LossFunction = other.LossFunction;
        DecompositionMethod = other.DecompositionMethod;
        UseIntercept = other.UseIntercept;
    }

    /// <summary>The options of a released checkpoint, read from its <c>config.json</c>; the training settings keep
    /// their defaults.</summary>
    public static ChronosOptions<T> FromPretrainedConfig(string configJsonPath)
    {
        var c = ChronosT5<T>.ReadConfiguration(File.ReadAllText(configJsonPath));
        return new ChronosOptions<T>
        {
            ContextLength = c.ContextLength, PredictionLength = c.PredictionLength, NumTokens = c.NumTokens,
            LowLimit = c.LowLimit, HighLimit = c.HighLimit, ModelDim = c.ModelDim, FeedForwardDim = c.FeedForwardDim,
            KeyValueDim = c.KeyValueDim, NumHeads = c.Heads, NumLayers = c.Layers, DropoutRate = c.Dropout,
            RelativeAttentionBuckets = c.RelativeBuckets, RelativeAttentionMaxDistance = c.RelativeMaxDistance,
            NumSamples = c.NumSamples, Temperature = c.Temperature, TopK = c.TopK, TopP = c.TopP,
        };
    }

    internal ChronosConfiguration Configuration()
    {
        if (TrainingSteps < 1 || BatchSize < 1 || MinPast < 0)
            throw new ArgumentException("TrainingSteps and BatchSize must be positive and MinPast non-negative.");
        if (!(LearningRate > 0) || WeightDecay < 0 || MaxGradNorm < 0)
            throw new ArgumentException("LearningRate must be positive, and WeightDecay and MaxGradNorm non-negative.");
        if (MissingValueProbability < 0 || MissingValueProbability >= 1)
            throw new ArgumentException("MissingValueProbability must be in [0, 1).");
        return new ChronosConfiguration(ContextLength, PredictionLength, NumTokens, LowLimit, HighLimit, ModelDim,
            FeedForwardDim, KeyValueDim, NumHeads, NumLayers, DropoutRate, RelativeAttentionBuckets,
            RelativeAttentionMaxDistance, NumSamples, Temperature, TopK, TopP);
    }
}
