using System.IO;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TimeSeries;

/// <summary>
/// The sizes and decoding settings of a Chronos-T5 model (Ansari et al. 2024; the official <c>ChronosConfig</c> and the
/// T5 configuration of the released checkpoints).
/// </summary>
internal sealed record ChronosConfiguration(
    int ContextLength, int PredictionLength, int NumTokens, double LowLimit, double HighLimit, int ModelDim,
    int FeedForwardDim, int KeyValueDim, int Heads, int Layers, double Dropout, int RelativeBuckets,
    int RelativeMaxDistance, int NumSamples, double Temperature, int TopK, double TopP)
{
    /// <summary>The T5 encoder-decoder these sizes describe: T5 v1.0 (ReLU feed-forward, tied embeddings), with the
    /// vocabulary resized to the token count and the pad token as the decoder start.</summary>
    public T5Configuration T5 => new(NumTokens, ModelDim, FeedForwardDim, KeyValueDim, Heads, Layers, Layers,
        RelativeBuckets, RelativeMaxDistance, Dropout, LayerNormEpsilon: 1e-6, DecoderStartTokenId: ChronosTokenizer.PadTokenId,
        EndTokenId: ChronosTokenizer.EosTokenId, GatedGelu: false, TieWordEmbeddings: true, PadTokenId: ChronosTokenizer.PadTokenId);

    /// <summary>Checks every size and setting.</summary>
    public void Validate()
    {
        if (ContextLength < 1 || PredictionLength < 1)
            throw new ArgumentException("ContextLength and PredictionLength must be positive.");
        if (NumTokens <= ChronosTokenizer.SpecialTokens + 1)
            throw new ArgumentException($"NumTokens must exceed {ChronosTokenizer.SpecialTokens + 1} (the special tokens and one bin).");
        if (!(HighLimit > LowLimit)) throw new ArgumentException("HighLimit must exceed LowLimit.");
        if (ModelDim < 1 || FeedForwardDim < 1 || KeyValueDim < 1 || Heads < 1 || Layers < 1)
            throw new ArgumentException("The T5 sizes must be positive.");
        if (RelativeBuckets < 2 || RelativeMaxDistance < 1) throw new ArgumentException("The relative position sizes must be positive.");
        if (Dropout < 0 || Dropout >= 1) throw new ArgumentException("Dropout must be in [0, 1).");
        if (NumSamples < 1) throw new ArgumentException("NumSamples must be positive.");
        if (!(Temperature > 0)) throw new ArgumentException("Temperature must be positive.");
        if (!(TopP > 0 && TopP <= 1)) throw new ArgumentException("TopP must be in (0, 1].");
    }
}

/// <summary>
/// Chronos's <c>MeanScaleUniformBins</c> tokenizer (Ansari et al. 2024, §3.1; the official <c>chronos.chronos</c>):
/// a series is divided by the mean absolute value of its observed context, and each scaled value becomes the token of
/// its bin among NumTokens − 3 uniformly spaced centers on [LowLimit, HighLimit] (edges midway between centers), after
/// the PAD (0) and EOS (1) tokens. Missing values become PAD and are masked; an EOS token ends each sequence. As in the
/// reference, values are tokenized in single precision.
/// </summary>
internal sealed class ChronosTokenizer
{
    public const int SpecialTokens = 2;
    public const int PadTokenId = 0;
    public const int EosTokenId = 1;

    private readonly int _numTokens;
    private readonly int _contextLength;

    /// <summary>The bin centers (torch.linspace in single precision).</summary>
    public float[] Centers { get; }

    /// <summary>The bin edges: −1e20, the midpoints of successive centers, and 1e20.</summary>
    public float[] Boundaries { get; }

    public ChronosTokenizer(int numTokens, double lowLimit, double highLimit, int contextLength)
    {
        _numTokens = numTokens;
        _contextLength = contextLength;
        int count = numTokens - SpecialTokens - 1;
        // torch.linspace's CPU kernel: step = (end − start) / (steps − 1); the first half counts up from start, the
        // second down from end, all in the tensor's precision.
        float start = (float)lowLimit, end = (float)highLimit;
        float step = count > 1 ? (end - start) / (count - 1) : 0f;
        int halfway = count / 2;
        Centers = new float[count];
        for (int i = 0; i < count; i++) Centers[i] = i < halfway ? start + step * i : end - step * (count - i - 1);
        Boundaries = new float[count + 1];
        Boundaries[0] = -1e20f;
        for (int i = 1; i < count; i++) Boundaries[i] = (Centers[i] + Centers[i - 1]) / 2f;
        Boundaries[count] = 1e20f;
    }

    /// <summary>
    /// The context's tokens followed by EOS, their mask (false where a value is missing) and the mean scale, from the
    /// last ContextLength values (<c>context_input_transform</c>).
    /// </summary>
    /// <remarks>The scale is the mean absolute observed value (1 when that is not positive), accumulated in double
    /// precision and rounded to single, where the reference sums in single precision; the two agree to rounding.</remarks>
    public (int[] Ids, bool[] Mask, float Scale) Context(IReadOnlyList<double> context)
    {
        int length = Math.Min(context.Count, _contextLength), offset = context.Count - length;
        double sum = 0;
        int observed = 0;
        for (int i = 0; i < length; i++)
        {
            float v = (float)context[offset + i];
            if (float.IsNaN(v)) continue;
            sum += Math.Abs(v);
            observed++;
        }
        float scale = observed > 0 ? (float)sum / observed : float.NaN;
        if (!(scale > 0)) scale = 1f;
        var values = new double[length];
        for (int i = 0; i < length; i++) values[i] = context[offset + i];
        var (ids, mask) = Tokens(values, scale);
        return (ids, mask, scale);
    }

    /// <summary>The future values' tokens followed by EOS, and their mask, on the context's scale
    /// (<c>label_input_transform</c>).</summary>
    public (int[] Ids, bool[] Mask) Label(IReadOnlyList<double> label, float scale) => Tokens(label, scale);

    private (int[] Ids, bool[] Mask) Tokens(IReadOnlyList<double> values, float scale)
    {
        var ids = new int[values.Count + 1];
        var mask = new bool[values.Count + 1];
        for (int i = 0; i < values.Count; i++)
        {
            float v = (float)values[i];
            if (float.IsNaN(v))
            {
                ids[i] = PadTokenId;
                continue;
            }
            mask[i] = true;
            float scaled = v / scale;
            // torch.bucketize(right=True): the number of edges at or below the value.
            int lo = 0, hi = Boundaries.Length;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (Boundaries[mid] <= scaled) lo = mid + 1;
                else hi = mid;
            }
            ids[i] = Math.Max(0, Math.Min(lo + SpecialTokens, _numTokens - 1));
        }
        ids[values.Count] = EosTokenId;
        mask[values.Count] = true;
        return (ids, mask);
    }

    /// <summary>The value of a sampled token on the given scale (<c>output_transform</c>).</summary>
    public double Value(int token, float scale)
    {
        int index = Math.Max(0, Math.Min(token - SpecialTokens - 1, Centers.Length - 1));
        return Centers[index] * scale;
    }
}

/// <summary>The official training recipe (Ansari et al. 2024, §5.2, and the official <c>train.py</c>).</summary>
internal sealed record ChronosTraining(
    double LearningRate, double WeightDecay, int Steps, int BatchSize, int MinPast, double DropProbability, double MaxGradNorm);

/// <summary>
/// Chronos-T5: the tokenizer, a T5 encoder-decoder over its tokens, the cross-entropy objective, sampled forecasts
/// and training on a series. Both Chronos models (<see cref="ChronosFoundationModel{T}"/> and
/// <c>Finance.Forecasting.Foundation.Chronos</c>) are built on it.
/// </summary>
internal sealed class ChronosT5<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers;

    public ChronosConfiguration Configuration { get; }
    public ChronosTokenizer Tokenizer { get; }
    public T5Seq2Seq<T> Network { get; }

    public ChronosT5(IEngine engine, List<LayerBase<T>> layers, ChronosConfiguration configuration)
    {
        configuration.Validate();
        _engine = engine;
        _layers = layers;
        Configuration = configuration;
        Tokenizer = new ChronosTokenizer(configuration.NumTokens, configuration.LowLimit, configuration.HighLimit,
            configuration.ContextLength);
        Network = new T5Seq2Seq<T>(engine, layers, configuration.T5);
    }

    /// <summary>The official training's random initialization: Hugging Face's T5 initialization with
    /// <c>initializer_factor</c> 0.05 (train.py: "the default initializer_factor (1.0) in transformers is too large").</summary>
    public void InitializeLikeReference(Random random) => Network.InitializeLikeHuggingFace(random, factor: 0.05);

    /// <summary>Loads a Hugging Face <c>T5ForConditionalGeneration</c> state dict (a released Chronos-T5 checkpoint).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadHuggingFaceWeights(read);

    /// <summary>Loads a released checkpoint's <c>model.safetensors</c>, checking every tensor's shape.</summary>
    public void LoadPretrainedWeights(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("The checkpoint path is empty.", nameof(path));
        using var stream = File.OpenRead(path);
        var file = SafetensorsReader.Read(stream);
        LoadTorchWeights((name, shape) =>
        {
            var tensor = file.Get(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            var actual = tensor.Shape.Select(d => (int)d).ToArray();
            if (!actual.SequenceEqual(shape))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}] in the checkpoint but [{string.Join(", ", shape)}] in " +
                    "this model; build the model with the checkpoint's configuration.");
            return file.ReadAsDouble(name);
        });
    }

    /// <summary>
    /// Reads a released checkpoint's <c>config.json</c>: the T5 sizes and its <c>chronos_config</c>.
    /// </summary>
    public static ChronosConfiguration ReadConfiguration(string configJson)
    {
        var root = JObject.Parse(configJson);
        var chronos = root["chronos_config"] as JObject ?? throw new InvalidDataException("Not a Chronos config: no chronos_config.");
        if ((string?)chronos["model_type"] != "seq2seq" || (string?)chronos["tokenizer_class"] != "MeanScaleUniformBins")
            throw new InvalidDataException("Only seq2seq Chronos-T5 models with the MeanScaleUniformBins tokenizer are supported.");
        if ((int?)chronos["n_special_tokens"] != ChronosTokenizer.SpecialTokens || (int?)chronos["pad_token_id"] != ChronosTokenizer.PadTokenId
            || (int?)chronos["eos_token_id"] != ChronosTokenizer.EosTokenId || (bool?)chronos["use_eos_token"] != true)
            throw new InvalidDataException("Only the released special tokens (PAD 0, EOS 1, appended EOS) are supported.");
        if ((string?)root["feed_forward_proj"] is { } projection && projection != "relu")
            throw new InvalidDataException($"Only the ReLU feed-forward of T5 v1.0 is supported, not '{projection}'.");
        int layers = (int)root["num_layers"]!;
        if (root["num_decoder_layers"] is { Type: not JTokenType.Null } decoderLayers && (int)decoderLayers != layers)
            throw new InvalidDataException("Encoder and decoder depths differ.");
        var kwargs = chronos["tokenizer_kwargs"] as JObject;
        return new ChronosConfiguration(
            (int)chronos["context_length"]!, (int)chronos["prediction_length"]!, (int)chronos["n_tokens"]!,
            (double?)kwargs?["low_limit"] ?? -15.0, (double?)kwargs?["high_limit"] ?? 15.0,
            (int)root["d_model"]!, (int)root["d_ff"]!, (int)root["d_kv"]!, (int)root["num_heads"]!, layers,
            (double?)root["dropout_rate"] ?? 0.1, (int?)root["relative_attention_num_buckets"] ?? 32,
            (int?)root["relative_attention_max_distance"] ?? 128, (int)chronos["num_samples"]!, (double)chronos["temperature"]!,
            (int)chronos["top_k"]!, (double)chronos["top_p"]!);
    }

    private static Tensor<T> Ids(int[] ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    /// <summary>
    /// The cross-entropy of the future values' tokens (and EOS) given the context (§3.2), as the official training
    /// computes it with <c>T5ForConditionalGeneration(labels=...)</c>: missing future values are ignored (−100) and
    /// missing context values masked. <paramref name="counted"/> is the number of tokens the mean is over.
    /// </summary>
    public Tensor<T> Loss(IReadOnlyList<double> context, IReadOnlyList<double> future, bool training, Random random, out int counted)
    {
        var (ids, mask, scale) = Tokenizer.Context(context);
        var (labelIds, labelMask) = Tokenizer.Label(future, scale);
        counted = 0;
        for (int i = 0; i < labelIds.Length; i++)
        {
            if (labelMask[i]) counted++;
            else labelIds[i] = -100;
        }
        return Network.Loss(Ids(ids), labelIds, training, random, inputMask: mask);
    }

    /// <summary>
    /// The loss of a batch of windows as <c>T5ForConditionalGeneration</c> computes it on the batch: the mean
    /// cross-entropy over every counted label token of every window (so a window weighs by its counted tokens).
    /// </summary>
    public Tensor<T> BatchLoss(IReadOnlyList<(double[] Context, double[] Future)> windows, bool training, Random random)
    {
        if (windows.Count == 0) throw new ArgumentException("The batch is empty.", nameof(windows));
        var losses = new List<(Tensor<T> Loss, int Count)>(windows.Count);
        int total = 0;
        foreach (var (context, future) in windows)
        {
            var term = Loss(context, future, training, random, out int counted);
            losses.Add((term, counted));
            total += counted;
        }
        return losses.Select(l => _engine.TensorMultiplyScalar(l.Loss, NumOps.FromDouble((double)l.Count / total)))
            .Aggregate((a, b) => _engine.TensorAdd(a, b));
    }

    /// <summary>
    /// <see cref="Configuration"/>.NumSamples sample paths of the <paramref name="steps"/> values after
    /// <paramref name="context"/>, as the official pipeline's <c>predict</c> draws them: the tokens of up to
    /// PredictionLength steps are sampled (temperature, top-k, top-p; EOS suppressed until the length is reached, as
    /// <c>min_new_tokens</c> does) and mapped back to values; a longer horizon continues from the context extended by
    /// the median of the samples so far.
    /// </summary>
    public double[][] SamplePaths(IReadOnlyList<double> context, int steps, Random random)
    {
        if (steps < 1) throw new ArgumentException("The number of forecast steps must be positive.", nameof(steps));
        var c = Configuration;
        var history = context.ToList();
        var paths = new double[c.NumSamples][];
        for (int s = 0; s < paths.Length; s++) paths[s] = new double[steps];
        int produced = 0;
        using var _ = new NoGradScope<T>();
        while (produced < steps)
        {
            var (ids, mask, scale) = Tokenizer.Context(history);
            var memory = Network.Encode(Ids(ids), training: false, random, mask);
            int length = Math.Min(steps - produced, c.PredictionLength);
            for (int s = 0; s < paths.Length; s++)
            {
                var tokens = Network.Sample(memory, mask, Array.Empty<int>(), c.Temperature, c.TopK, c.TopP, length, random,
                    (token, _) => token != ChronosTokenizer.EosTokenId);
                for (int h = 0; h < length; h++) paths[s][produced + h] = Tokenizer.Value(tokens[h], scale);
            }
            for (int h = 0; h < length; h++) history.Add(Median(paths.Select(p => p[produced + h])));
            produced += length;
        }
        return paths;
    }

    /// <summary>torch.median: the lower of the two middle values for an even count.</summary>
    public static double Median(IEnumerable<double> values)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        return sorted[(sorted.Length - 1) / 2];
    }

    /// <summary>torch.quantile's default (linear interpolation between the closest ranks).</summary>
    public static double Quantile(IEnumerable<double> values, double q)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        double position = q * (sorted.Length - 1);
        int below = (int)Math.Floor(position), above = Math.Min(below + 1, sorted.Length - 1);
        return sorted[below] + (position - below) * (sorted[above] - sorted[below]);
    }

    /// <summary>
    /// The official optimizer: AdamW (betas 0.9 and 0.999, ε 1e-8) with the given weight decay on every parameter except
    /// the relative attention bias tables (the transformers Trainer skips parameters named "bias"), the learning rate
    /// decayed linearly to zero over Steps, and global-norm clipping at MaxGradNorm.
    /// </summary>
    public AdamWOptimizer<T, Tensor<T>, Tensor<T>> CreateOptimizer(ChronosTraining settings)
    {
        var parameters = Training.TapeTrainingStep<T>.CollectParameters(_layers);
        var noDecay = new HashSet<Tensor<T>>(
            Training.TapeTrainingStep<T>.CollectParameters(new LayerBase<T>[] { Network.EncoderBias }
                .Concat(Network.DecoderBias is null ? Array.Empty<LayerBase<T>>() : new LayerBase<T>[] { Network.DecoderBias })),
            TensorReferenceComparer<Tensor<T>>.Instance);
        var mask = new Vector<T>(parameters.Sum(p => p.Length));
        int offset = 0;
        foreach (var p in parameters)
        {
            var value = noDecay.Contains(p) ? NumOps.Zero : NumOps.One;
            for (int i = 0; i < p.Length; i++) mask[offset + i] = value;
            offset += p.Length;
        }
        return new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(null, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = settings.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            WeightDecay = settings.WeightDecay,
            WeightDecayMask = mask,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = settings.MaxGradNorm > 0,
            GradientClippingMethod = GradientClippingMethod.ByNorm,
            MaxGradientNorm = settings.MaxGradNorm > 0 ? settings.MaxGradNorm : 1.0,
            LearningRateScheduler = new PolynomialLRScheduler(settings.LearningRate, settings.Steps, power: 1.0, endLearningRate: 0.0),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
    }

    /// <summary>
    /// Trains on <paramref name="series"/> with the official recipe: each step slices BatchSize random windows (a
    /// split point at least MinPast values in and PredictionLength from the end; the context is up to ContextLength
    /// values before it and the label the PredictionLength after), drops each value to missing with a probability
    /// drawn from U(0, DropProbability) per window, and minimizes the cross-entropy of the label tokens, averaged over
    /// the batch's tokens, with AdamW. The learning rate decays linearly to zero over Steps, gradients are clipped to a
    /// global norm of MaxGradNorm, and weight decay skips the relative attention bias tables, as the transformers
    /// Trainer skips parameters named "bias". Dropout is on.
    /// </summary>
    public void Train(IReadOnlyList<double> series, ChronosTraining settings, Random random, CancellationToken cancellation,
        bool stopAtCancellation, Func<int, double, bool>? report = null)
    {
        var c = Configuration;
        int lastSplit = series.Count - c.PredictionLength;
        if (lastSplit < settings.MinPast)
            throw new ArgumentException($"Training needs a series of at least MinPast + PredictionLength ({settings.MinPast + c.PredictionLength}) " +
                $"values; got {series.Count}.", nameof(series));
        if (settings.Steps < 1 || settings.BatchSize < 1) throw new ArgumentException("Steps and BatchSize must be positive.");

        var parameters = Training.TapeTrainingStep<T>.CollectParameters(_layers);
        var optimizer = CreateOptimizer(settings);

        for (int step = 0; step < settings.Steps; step++)
        {
            if (stopAtCancellation && cancellation.IsCancellationRequested) break;
            cancellation.ThrowIfCancellationRequested();
            var windows = new List<(double[] Context, double[] Future)>(settings.BatchSize);
            while (windows.Count < settings.BatchSize) windows.Add(Window(series, settings, random));

            // One update on the shared eager step (which also advances the per-step schedule). The objective is built
            // once per step; the windows are drawn before it so a re-evaluation would replay them.
            T lossValue = Training.TapeTrainingStepper<T>.EagerObjectiveStep(parameters,
                () => BatchLoss(windows, training: true, random), optimizer);
            if (report is not null && !report(step, NumOps.ToDouble(lossValue))) break;
        }
    }

    // One training window (the official InstanceSplitter on a randomly sampled split point, after the missing-value
    // augmentation); a window whose context has no observed value is redrawn, as the official filter drops it.
    private (double[] Context, double[] Future) Window(IReadOnlyList<double> series, ChronosTraining settings, Random random)
    {
        var c = Configuration;
        while (true)
        {
            int split = settings.MinPast + random.Next(series.Count - c.PredictionLength - settings.MinPast + 1);
            int start = Math.Max(0, split - c.ContextLength);
            double drop = random.NextDouble() * settings.DropProbability;
            var context = new double[split - start];
            var future = new double[c.PredictionLength];
            bool observed = false;
            for (int i = 0; i < context.Length; i++)
            {
                context[i] = drop > 0 && random.NextDouble() < drop ? double.NaN : series[start + i];
                observed |= !double.IsNaN(context[i]);
            }
            for (int h = 0; h < future.Length; h++)
                future[h] = drop > 0 && random.NextDouble() < drop ? double.NaN : series[split + h];
            if (observed) return (context, future);
        }
    }
}
