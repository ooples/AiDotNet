using AiDotNet.Audio.Resampling;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

public partial class MaskGCT<T>
{
    private readonly Dictionary<MaskGctStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private Random? _trainingRandom;

    /// <inheritdoc />
    /// <remarks>MaskGCT trains on recordings: the 24 kHz recording (<see cref="TtsTrainingSample{T}.Audio"/>) and its
    /// transcript's phone ids (<see cref="TtsTrainingSample{T}.Tokens"/>, from <see cref="Phonemize"/>); the duration stage
    /// also reads the phones' durations in semantic frames (<see cref="TtsTrainingSample{T}.Durations"/>).</remarks>
    protected override TtsSupervision RequiredSupervision =>
        _options.CurrentStage == MaskGctStage.Duration ? TtsSupervision.Durations : TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    /// <remarks>The current stage's AdamW with the paper's warmup and inverse-square-root decay.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? Optimizer() : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <inheritdoc />
    /// <remarks>One step of <see cref="MaskGCTOptions.CurrentStage"/>'s objective: the masked-token cross-entropy of the
    /// text-to-semantic or a semantic-to-acoustic model (the release's <c>forward_diffusion</c> and <c>loss_t</c>: a random
    /// step, a random prompt prefix kept unless dropped with <see cref="MaskGCTOptions.PromptDropProbability"/>, tokens
    /// masked with probability sin(πt/2)), or the duration predictor's flow-matching loss.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
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
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
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

    // Recordings' tokens (the frozen encoders run once per recording).
    private readonly Dictionary<Tensor<T>, (int[] Semantic, int[,] Acoustic)> _tokenCache = new(TensorReferenceComparer<Tensor<T>>.Instance);

    private (int[] Semantic, int[,] Acoustic) Tokens(Tensor<T> audio, bool acoustic)
    {
        if (_tokenCache.TryGetValue(audio, out var cached) && (!acoustic || cached.Acoustic.Length > 0)) return cached;
        var samples = Doubles(audio);
        var semantic = cached.Semantic ?? SemanticTokens(SoxrResampler.LibrosaResample(samples, _options.SampleRate, _options.SemanticSampleRate));
        var codes = acoustic ? AcousticCodec.Encode(samples) : new int[0, 0];
        if (_tokenCache.Count > 64) _tokenCache.Clear();
        return _tokenCache[audio] = (semantic, codes);
    }

    // The prompt length the release draws: none with the drop probability, else uniform in [min(n // 4, 5), high).
    private int PromptLength(int n, int high, Random random)
    {
        if (random.NextDouble() <= _options.PromptDropProbability) return 0;
        int low = Math.Min(n / 4, 5);
        if (high <= low)
            throw new ArgumentException($"A training utterance of {n} frames is too short to draw a prompt from (torch.randint({low}, {high})).");
        return random.Next(low, high);
    }

    // The positions masked with probability p past the prompt; at least one (the first after the prompt).
    private static bool[] Mask(int n, int prompt, double p, Random random)
    {
        var mask = new bool[n];
        bool any = false;
        for (int i = prompt; i < n; i++) any |= mask[i] = random.NextDouble() < p;
        if (!any && prompt < n) mask[prompt] = true;
        return mask;
    }

    private static double Step(Random random) => Math.Min(1.0, Math.Max(1e-5, random.NextDouble()));

    // The cross-entropy of logits [n, V] against the targets at the masked positions, averaged over them.
    private Tensor<T> MaskedCrossEntropy(Tensor<T> logits, IReadOnlyList<int> targets, IReadOnlyList<bool> masked, int offset)
    {
        int n = logits.Shape[0], v = logits.Shape[1];
        var picks = new List<int>();
        for (int i = 0; i < n; i++)
            if (masked[offset + i]) picks.Add(i * v + targets[offset + i]);
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(logits, axis: 1), new[] { n * v, 1 });
        var picked = Engine.TensorIndexSelect(logProbabilities, new Tensor<int>(new[] { picks.Count }, new Vector<int>(picks.ToArray())), 0);
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(picked, null), Ops.FromDouble(-1.0 / picks.Count));
    }

    private Func<Tensor<T>> Objective(TtsTrainingSample<T> sample, Random random)
    {
        Validation.Guard.NotNull(sample);
        var phones = PhoneIds(sample.Tokens, nameof(sample));
        if (phones.Length == 0) throw new ArgumentException("The sample has no phones.", nameof(sample));
        switch (_options.CurrentStage)
        {
            case MaskGctStage.Duration:
            {
                var durations = sample.Durations ?? throw new ArgumentException("The duration stage trains on the phones' durations; set Durations.", nameof(sample));
                if (durations.Length != phones.Length) throw new ArgumentException("Give one duration per phone.", nameof(sample));
                int n = phones.Length;
                int prompt = PromptLength(n, (int)(n * 0.4), random);
                double t = random.NextDouble();
                var noise = Enumerable.Range(0, n).Select(_ => Normal(random)).ToArray();
                return () => DurationPredictor.Loss(phones, durations, prompt, t, noise);
            }
            case MaskGctStage.TextToSemantic:
            {
                var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample));
                var semantic = Tokens(audio, acoustic: false).Semantic;
                int n = semantic.Length;
                double t = Step(random);
                int prompt = PromptLength(n, (int)(n * 0.4), random);
                var mask = Mask(n, prompt, Math.Max(MaskGctSampling.MaskProbability(t), 0.2), random);
                var promptTokens = semantic.Take(prompt).ToArray();
                var target = semantic.Skip(prompt).ToArray();
                var targetMask = mask.Skip(prompt).ToArray();
                return () => MaskedCrossEntropy(TextToSemantic.Logits(phones, promptTokens, target, targetMask, t), semantic, mask, prompt);
            }
            default:
            {
                var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample));
                var (semantic, codes) = Tokens(audio, acoustic: true);
                bool first = _options.CurrentStage == MaskGctStage.FirstAcousticLayer;
                var model = first ? FirstLayerModel : FullModel;
                int n = Math.Min(semantic.Length, codes.GetLength(0)), quantizers = model.Quantizers;
                var frames = new int[n, quantizers];
                for (int i = 0; i < n; i++)
                    for (int q = 0; q < quantizers; q++) frames[i, q] = codes[i, q];
                var cond = semantic.Take(n).ToArray();
                double t = Step(random);
                int layer = Layer(quantizers, model.PredictFirstLayer, random);
                int prompt = PromptLength(n, n / 2, random);
                var mask = Mask(n, prompt, MaskGctSampling.MaskProbability(t), random);
                var targets = Enumerable.Range(0, n).Select(i => frames[i, layer]).ToArray();
                return () => MaskedCrossEntropy(model.TrainingLogits(cond, frames, layer, prompt, mask, t), targets, mask, 0);
            }
        }
    }

    // mask_layer with the configs' "linear" schedule: weights Q − i over the layers (the full model never draws the first).
    private static int Layer(int quantizers, bool predictFirstLayer, Random random)
    {
        var weights = Enumerable.Range(0, quantizers).Select(i => predictFirstLayer ? quantizers - i : i == 0 ? 0 : quantizers - (i - 1)).ToArray();
        double draw = random.NextDouble() * weights.Sum(), cumulative = 0;
        for (int i = 0; i < quantizers; i++)
        {
            cumulative += weights[i];
            if (weights[i] > 0 && draw < cumulative) return i;
        }
        return Array.FindLastIndex(weights, w => w > 0);
    }

    private IReadOnlyList<LayerBase<T>> StageLayers() => _options.CurrentStage switch
    {
        MaskGctStage.TextToSemantic => _t2sLayers,
        MaskGctStage.FirstAcousticLayer => _s2aFirstLayers,
        MaskGctStage.AcousticLayers => _s2aFullLayers,
        _ => _durationLayers,
    };

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        var stage = _options.CurrentStage;
        if (_optimizers.TryGetValue(stage, out var cached)) return cached;
        var o = _options;
        var parameters = Training.TapeTrainingStep<T>.CollectParameters(StageLayers().Cast<ILayer<T>>().ToList(), -1).ToList();
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate, Beta1 = o.Beta1, Beta2 = o.Beta2, Epsilon = o.Epsilon, WeightDecay = o.WeightDecay,
            // The inverse-square-root schedule peaking at the learning rate after the warmup (a Noam schedule of unit width).
            LearningRateScheduler = new NoamSchedule(1, o.WarmupSteps, o.LearningRate * Math.Sqrt(o.WarmupSteps)),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            MaskParameters = parameters,
            WeightDecayMask = new Vector<T>(Enumerable.Repeat(Ops.One, parameters.Sum(p => p.Length)).ToArray()),
        };
        var optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
        _optimizers[stage] = optimizer;
        return optimizer;
    }

    /// <inheritdoc />
    /// <remarks>Training updates <see cref="MaskGCTOptions.CurrentStage"/>'s network; the encoders, the codecs and the other
    /// networks are frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(StageLayers().Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
