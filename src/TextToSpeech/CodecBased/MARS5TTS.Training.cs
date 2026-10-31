using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class MARS5TTS<T>
{
    private readonly Dictionary<Mars5Stage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private Random? _trainingRandom;

    /// <inheritdoc />
    /// <remarks>MARS5 trains on recordings at <see cref="TtsModelOptions.SampleRate"/> (<see cref="TtsTrainingSample{T}.Audio"/>)
    /// with their transcripts (<see cref="TtsTrainingSample{T}.Text"/>; without one, <see cref="TtsTrainingSample{T}.Tokens"/>
    /// are the text tokens) and optionally another recording of the speaker (<see cref="TtsTrainingSample{T}.SpeakerReference"/>;
    /// a crop of the recording otherwise).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? Optimizer() : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    private readonly Dictionary<Tensor<T>, int[,]> _codes = new(TensorReferenceComparer<Tensor<T>>.Instance);

    private int[,] CodesOf(Tensor<T> audio)
    {
        if (_codes.TryGetValue(audio, out var codes)) return codes;
        if (_codes.Count > 64) _codes.Clear();
        return _codes[audio] = Encode(audio);
    }

    private List<int> TextOf(TtsTrainingSample<T> sample)
    {
        if (sample.Text is { } text) return TextTokens(text.Trim());
        int vocabulary = TextTokenizer.VocabSize;
        var tokens = new List<int>();
        for (int i = 0; i < sample.Tokens.Length; i++)
        {
            int id = (int)Math.Round(Ops.ToDouble(sample.Tokens.GetFlat(i)));
            if (id < 0 || id >= vocabulary) throw new ArgumentOutOfRangeException(nameof(sample), $"Without Text, the sample's Tokens are text tokens in [0, {vocabulary}).");
            tokens.Add(id);
        }
        if (tokens.Count == 0) throw new ArgumentException("The sample has no text.", nameof(sample));
        return tokens;
    }

    // A random crop of between min and max seconds of codes (all of them when shorter).
    private int[,] Crop(int[,] codes, double minSeconds, double maxSeconds, Random random)
    {
        int frames = codes.GetLength(0), codebooks = codes.GetLength(1);
        double rate = (double)_options.SampleRate / _options.HopSize;
        int length = Math.Min(frames, Math.Max(1, (int)Math.Round((minSeconds + random.NextDouble() * (maxSeconds - minSeconds)) * rate)));
        int start = random.Next(frames - length + 1);
        var crop = new int[length, codebooks];
        for (int f = 0; f < length; f++)
            for (int q = 0; q < codebooks; q++) crop[f, q] = codes[start + f, q];
        return crop;
    }

    /// <inheritdoc />
    /// <remarks>
    /// One step of <see cref="MARS5TTSOptions.CurrentStage"/>. The AR minimizes next-token cross-entropy over
    /// <c>[start, text, end, speech tokens, end of speech]</c> after the speaker embedding of a reference, the text
    /// positions' mean weighted by <see cref="MARS5TTSOptions.TextLossWeight"/> and the speech positions' at 1. The NAR
    /// minimizes the multinomial diffusion's variational loss at a random step over all eight codebooks, the noisy codes'
    /// first codebook replaced by the truth.
    /// </remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs MARS5's networks; this model was built from caller-supplied layers.");
        var random = _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed);
        var objective = Objective(sample, random);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => objective(), Optimizer());
    }

    /// <inheritdoc />
    /// <remarks>The objective at a fixed draw (the training seed's first), so repeated evaluations compare.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs MARS5's networks; this model was built from caller-supplied layers.");
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Objective(sample, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed))()[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private Func<Tensor<T>> Objective(TtsTrainingSample<T> sample, Random random)
    {
        Validation.Guard.NotNull(sample);
        var o = _options;
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on recordings; set the sample's Audio.", nameof(sample));
        var text = TextOf(sample);
        int[,] codes;
        int[,]? separate;
        using (new NoGradScope<T>())
        {
            codes = CodesOf(audio);
            separate = sample.SpeakerReference is { } r ? CodesOf(r) : null;
        }
        if (codes.GetLength(0) == 0) throw new ArgumentException("The recording is shorter than one codec frame.", nameof(sample));

        if (o.CurrentStage == Mars5Stage.Autoregressive)
        {
            var reference = separate ?? Crop(codes, o.ArReferenceMinSeconds, o.ArReferenceMaxSeconds, random);
            int textVocab = TextTokenizer.VocabSize;
            var q0 = Enumerable.Range(0, codes.GetLength(0)).Select(f => codes[f, 0]).ToList();
            var speech = SpeechTokenizer.Encode(q0).Select(t => t + textVocab).ToList();
            var sequence = text.Concat(speech).Concat(new[] { EndOfSpeech }).ToList();
            if (sequence.Count > o.SlidingWindow) throw new ArgumentException("The utterance is longer than MARS5's attention window.", nameof(sample));
            var inputs = sequence.Take(sequence.Count - 1).ToList();
            var targets = sequence.Skip(1).ToArray();
            int textTargets = text.Count - 1;
            return () =>
            {
                var logits = Autoregressive.Logits(inputs, Autoregressive.Speaker(reference));
                var speechLoss = CrossEntropy(Engine.TensorNarrow(logits, 0, textTargets, targets.Length - textTargets), targets.Skip(textTargets).ToArray());
                if (textTargets == 0) return speechLoss;
                var textLoss = CrossEntropy(Engine.TensorNarrow(logits, 0, 0, textTargets), targets.Take(textTargets).ToArray());
                return Engine.TensorAdd(speechLoss, Engine.TensorMultiplyScalar(textLoss, Ops.FromDouble(o.TextLossWeight)));
            };
        }

        var prompt = separate ?? Crop(codes, o.NarPromptMinSeconds, o.NarPromptMaxSeconds, random);
        var diffusion = new Mars5Diffusion(o.CodebookSize + 1, o.DiffusionSteps);
        int frames = codes.GetLength(0), codebooks = o.NumCodebooks;
        int t = random.Next(o.DiffusionSteps);
        var flat = new int[frames * codebooks];
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < codebooks; c++) flat[f * codebooks + c] = codes[f, c];
        var logX0 = diffusion.OneHotRows(flat);
        var marginal = diffusion.QPredRows(logX0, t);
        var uniform = new double[marginal.Length];
        for (int i = 0; i < uniform.Length; i++) uniform[i] = random.NextDouble();
        var noisy = diffusion.SampleCategorical(marginal, uniform);
        var xt = new int[frames, codebooks];
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < codebooks; c++) xt[f, c] = c == 0 ? codes[f, 0] : noisy[f * codebooks + c];
        var xtFlat = new int[frames * codebooks];
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < codebooks; c++) xtFlat[f * codebooks + c] = xt[f, c];
        var logXt = diffusion.OneHotRows(xtFlat);
        int seed = random.Next();
        return () =>
        {
            var logits = NonAutoregressive.Forward(text, prompt, xt, t, dropCondition: false, training: IsTrainingMode,
                AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
            var logProbabilities = Engine.TensorLogSoftmax(Engine.Reshape(logits, new[] { frames * codebooks, o.CodebookSize + 1 }), axis: 1);
            return diffusion.Loss(Engine, logX0, logXt, logProbabilities, t, o.IncludeKlPrior);
        };
    }

    // The mean cross-entropy of logits [n, V] against targets [n].
    private Tensor<T> CrossEntropy(Tensor<T> logits, IReadOnlyList<int> targets)
    {
        int n = logits.Shape[0], v = logits.Shape[1];
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(logits, axis: 1), new[] { n * v, 1 });
        var picks = new Tensor<int>(new[] { n }, new Vector<int>(Enumerable.Range(0, n).Select(i => i * v + targets[i]).ToArray()));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorIndexSelect(logProbabilities, picks, 0), null), Ops.FromDouble(-1.0 / n));
    }

    private IReadOnlyList<LayerBase<T>> StageLayers() => _options.CurrentStage == Mars5Stage.Autoregressive ? _arLayers : _narLayers;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        var stage = _options.CurrentStage;
        if (_optimizers.TryGetValue(stage, out var cached)) return cached;
        var o = _options;
        bool ar = stage == Mars5Stage.Autoregressive;
        double peak = ar ? o.ArLearningRate : o.NarLearningRate, end = ar ? o.ArEndLearningRate : o.NarEndLearningRate;
        int endSteps = ar ? o.ArEndSteps : o.NarEndSteps, warmup = o.WarmupSteps;
        // Linear warmup from zero to the peak, then linear decay to the end rate at endSteps, held after.
        Func<int, double> schedule = s =>
        {
            if (warmup > 0 && s < warmup) return (s + 1) / (double)warmup;
            if (s >= endSteps) return end / peak;
            double progress = (s - warmup) / (double)Math.Max(1, endSteps - warmup);
            return (peak + (end - peak) * progress) / peak;
        };
        var parameters = Training.TapeTrainingStep<T>.CollectParameters(StageLayers().Cast<ILayer<T>>().ToList(), -1).ToList();
        var optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = peak, Beta1 = ar ? o.ArBeta1 : o.NarBeta1, Beta2 = ar ? o.ArBeta2 : o.NarBeta2, Epsilon = o.Epsilon,
                WeightDecay = ar ? o.ArWeightDecay : o.NarWeightDecay, MaskParameters = parameters,
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(peak, schedule),
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            }), ar ? "ar" : "nar");
        _optimizers[stage] = optimizer;
        return optimizer;
    }

    /// <inheritdoc />
    /// <remarks>Training updates the current stage's network; the codec and Vocos are frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(StageLayers().Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    /// <inheritdoc />
    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        _optimizers.Clear();
    }
}
