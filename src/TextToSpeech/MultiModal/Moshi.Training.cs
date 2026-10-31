using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>A training example for Moshi's language model: Moshi's (the main speaker's) 24 kHz speech, the user's if there
/// is one, and the main speaker's words with their times.</summary>
public sealed class MoshiConversation<T>
{
    /// <summary>The main speaker's speech.</summary>
    public required Tensor<T> Moshi { get; init; }

    /// <summary>The other side's speech, or null for single-stream audio (pre-training uses the main stream only).</summary>
    public Tensor<T>? User { get; init; }

    /// <summary>The main speaker's words and their start and end times; null trains the audio with the text stream masked.</summary>
    public IReadOnlyList<MoshiWord>? Words { get; init; }
}

public partial class Moshi<T>
{
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private readonly Dictionary<Tensor<T>, Tensor<T>> _mimiAverage = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private Random? _trainingRandom;
    private string _trainedGroup = "language model";

    private Random TrainingRandom => _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue + 1);

    /// <summary>Noise clips (24 kHz) for the instruct stage's user-stream augmentation (§4.4 uses the Deep Noise Suppression
    /// challenge's); without them that augmentation adds no noise.</summary>
    public IReadOnlyList<Tensor<T>> AugmentationNoise { get; set; } = Array.Empty<Tensor<T>>();

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? LanguageModelOptimizer(textOnly: false) : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    // ---------------------------------------------------------------- the language model

    private double TextAudioDelay() => _options.Variant switch
    {
        MoshiVariant.Tts => _options.TextAudioDelaySeconds,
        MoshiVariant.Asr => -_options.TextAudioDelaySeconds,
        _ => 0.0,
    };

    private static double[] Samples(Tensor<T> audio)
    {
        var samples = new double[audio.Length];
        for (int i = 0; i < samples.Length; i++) samples[i] = Ops.ToDouble(audio.GetFlat(i));
        return samples;
    }

    /// <summary>
    /// The joint sequence a conversation trains on at the current <see cref="MoshiOptions.Stage"/>: pre-training reads the
    /// main stream alone, masks the text 30% of the time and shifts it by U[−0.6, 0.6] s; the instruct stage augments the
    /// user's stream; the TTS and ASR variants put the text ahead of or behind the audio by their delay.
    /// </summary>
    internal int[,] ConversationSequence(MoshiConversation<T> conversation, Random? random)
    {
        var core = Core;
        var o = _options;
        bool pre = o.Stage == MoshiTrainingStage.PreTraining;
        var moshi = Samples(conversation.Moshi);
        double[]? user = pre || conversation.User is null ? null : Samples(conversation.User);
        if (user is not null && random is not null && o.Stage == MoshiTrainingStage.InstructFineTuning)
            user = core.AugmentUser(user, moshi, AugmentationNoise.Select(Samples).ToList(), random);
        int[,] moshiCodes, userCodes;
        using (new NoGradScope<T>())
        {
            moshiCodes = core.AudioCodes(moshi);
            userCodes = user is null ? null! : core.AudioCodes(user);
        }
        int frames = moshiCodes.GetLength(1);
        double delay = TextAudioDelay();
        if (pre && o.Variant == MoshiVariant.Dialogue && random is not null) delay = (2 * random.NextDouble() - 1) * o.TextDelaySeconds;
        bool masked = conversation.Words is null || pre && random is not null && random.NextDouble() < o.TextMaskProbability;
        var text = masked ? Enumerable.Repeat(-1, frames).ToArray() : core.TextStream(conversation.Words!, frames, delay);
        return core.Sequence(text, moshiCodes, user is null ? null : userCodes);
    }

    /// <summary>One language-model step on a conversation at the current stage (Eq. 7's loss). Returns the loss.</summary>
    public T TrainConversation(MoshiConversation<T> conversation)
    {
        ThrowIfDisposed();
        if (conversation is null) throw new ArgumentNullException(nameof(conversation));
        var codes = ConversationSequence(conversation, TrainingRandom);
        return TrainCodes(codes, textOnly: false);
    }

    /// <summary>One step on a text-only batch (§4.4: half of pre-training's batches, 10% of post-training's), the text in
    /// Helium's tokens, with its own optimizer state.</summary>
    public T TrainText(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        return TrainCodes(Core.TextOnlySequence(text), textOnly: true);
    }

    private T TrainCodes(int[,] codes, bool textOnly)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        var core = Core;
        var marker = IndexTtsOps.Ids<T>(new[] { 0 });
        _trainedGroup = "language model";
        return TrainWithCustomObjective(marker, marker, (_, _) => core.Loss(codes), LanguageModelOptimizer(textOnly));
    }

    // A sample's sequence: its tokens at their durations (frames at 12.5 Hz) when it has them, else the text masked.
    private int[,] SampleSequence(TtsTrainingSample<T> sample)
    {
        var core = Core;
        if (sample.Audio is null) throw new ArgumentException("Moshi trains on the recording; set the sample's Audio.", nameof(sample));
        int[,] moshi;
        using (new NoGradScope<T>()) moshi = core.AudioCodes(sample.Audio);
        int frames = moshi.GetLength(1);
        var text = Enumerable.Repeat(-1, frames).ToArray();
        if (sample.Durations is not null && sample.Durations.Length == sample.Tokens.Length)
        {
            var words = new List<MoshiTokenSpan>();
            int start = 0;
            for (int i = 0; i < sample.Tokens.Length; i++)
            {
                int id = (int)Math.Round(Ops.ToDouble(sample.Tokens.GetFlat(i)));
                words.Add(new MoshiTokenSpan(new[] { id }, start / core.FrameRate, (start + sample.Durations[i]) / core.FrameRate));
                start += sample.Durations[i];
            }
            text = core.TextStream(words, frames, TextAudioDelay());
        }
        return core.Sequence(text, moshi, null);
    }

    /// <inheritdoc />
    /// <remarks>One language-model step on the recording as single-stream audio; the sample's tokens, placed by its
    /// <see cref="TtsTrainingSample{T}.Durations"/> (frames at 12.5 Hz), are the text stream, or the text is masked when the
    /// sample carries no durations (the paper aligns with Whisper's word times; <see cref="TrainConversation"/> takes
    /// them).</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        return TrainCodes(SampleSequence(sample), textOnly: false);
    }

    /// <inheritdoc />
    /// <remarks>Eq. 7's loss of the sample's sequence.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            var codes = SampleSequence(sample);
            using var _ = new NoGradScope<T>();
            return Core.Loss(codes).GetFlat(0);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> LanguageModelOptimizer(bool textOnly)
    {
        string key = $"{(textOnly ? "text" : "audio")}:{_options.Stage}";
        if (_optimizers.TryGetValue(key, out var existing)) return existing;
        var optimizer = Core.CreateLanguageModelOptimizer(this, _lmLayers, textOnly);
        return _optimizers[key] = textOnly ? optimizer : PaperOptimizerFactory.VerifyHandBuilt(this, optimizer);
    }

    // ---------------------------------------------------------------- Mimi

    /// <summary>
    /// One step of Mimi's training (§3.3) on 24 kHz speech (the paper's random 12 s windows): the codebooks' moving
    /// averages and the reconstruction from a gradient-free pass, the discriminator's hinge step on it, then the codec's
    /// step — the balanced adversarial (4) and feature (4) losses, the commitment penalty and the WavLM distillation — each
    /// sub-quantizer quantizing half the time and the acoustic codebooks dropped out, then the weights' moving average.
    /// Needs <see cref="MoshiOptions.IncludeMimiTraining"/>. Returns the codec's loss.
    /// </summary>
    public T TrainMimi(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        var core = Core;
        var training = core.MimiTraining ?? throw new InvalidOperationException("Training Mimi needs MoshiOptions.IncludeMimiTraining.");
        var o = _options;
        var random = TrainingRandom;
        var wave = Engine.Reshape(audio, new[] { 1, 1, audio.Length });
        int acousticMax = Math.Min(o.Codebooks, core.Mimi.Configuration.Codebooks) - core.Mimi.Configuration.SemanticCodebooks;
        int acoustic = o.MimiQuantizerDropout ? 1 + random.Next(acousticMax) : acousticMax;
        bool quantizeSemantic = random.NextDouble() < o.MimiQuantizationRate, quantizeAcoustic = random.NextDouble() < o.MimiQuantizationRate;
        var real = Engine.Reshape(audio, new[] { audio.Length });
        Tensor<T>? targets = training.DistillationTargets(real);

        Tensor<T> generated;
        using (new NoGradScope<T>())
            generated = core.Mimi.TrainingForward(wave, acoustic, quantizeSemantic, quantizeAcoustic, random).Audio;
        var fake = Engine.Reshape(generated, new[] { generated.Length });
        _trainedGroup = "mimi discriminator";
        TrainWithCustomObjective(real, real, (_, _) => training.DiscriminatorLoss(real, fake), MimiOptimizer("mimi discriminator"));

        _trainedGroup = "mimi";
        double reported = 0;
        TrainWithCustomObjective(real, real, (_, _) =>
        {
            var (output, semantic, penalty) = core.Mimi.TrainingForward(wave, acoustic, quantizeSemantic, quantizeAcoustic, random, updateCodebooks: false);
            var (objective, value) = training.GeneratorObjective(real, Engine.Reshape(output, new[] { output.Length }), semantic, penalty, targets);
            reported = value;
            return objective;
        }, MimiOptimizer("mimi"));
        UpdateMimiAverage();
        return Ops.FromDouble(reported);
    }

    private IEnumerable<LayerBase<T>> MimiGeneratorLayers => _mimiLayers.Concat(_distillationLayers);

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> MimiOptimizer(string group)
    {
        if (_optimizers.TryGetValue(group, out var existing)) return existing;
        var o = _options;
        bool generator = group == "mimi";
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.MimiLearningRate, Beta1 = 0.5, Beta2 = 0.9, Epsilon = 1e-8, WeightDecay = generator ? o.MimiWeightDecay : 0.0,
            UseAdaptiveLearningRate = false, UseAdaptiveBetas = false,
        });
        if (generator)
        {
            // Weight decay on the Transformers' parameters only.
            var transformers = new HashSet<LayerBase<T>>(Core.Mimi.TransformerLayers);
            var parameters = new List<Tensor<T>>();
            var decays = new List<T>();
            foreach (var layer in MimiGeneratorLayers)
                foreach (var p in Training.TapeTrainingStep<T>.CollectParameters(new ILayer<T>[] { layer }, -1))
                {
                    parameters.Add(p);
                    for (int i = 0; i < p.Length; i++) decays.Add(transformers.Contains(layer) ? Ops.One : Ops.Zero);
                }
            var options = (AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)optimizer.GetOptions();
            options.WeightDecayMask = new Vector<T>(decays.ToArray());
            options.MaskParameters = parameters;
        }
        return _optimizers[group] = generator ? PaperOptimizerFactory.VerifyHandBuilt(this, optimizer, "mimi") : optimizer;
    }

    // ema ← d·ema + (1 − d)·w over Mimi's parameters after each codec step.
    private void UpdateMimiAverage()
    {
        double d = _options.MimiWeightAverageDecay;
        foreach (var p in Training.TapeTrainingStep<T>.CollectParameters(_mimiLayers.Cast<ILayer<T>>().ToList(), -1))
        {
            if (!_mimiAverage.TryGetValue(p, out var average))
            {
                _mimiAverage[p] = new Tensor<T>(p.Shape.ToArray(), p.ToVector());
                continue;
            }
            for (int i = 0; i < p.Length; i++)
                average.SetFlat(i, Ops.FromDouble(d * Ops.ToDouble(average.GetFlat(i)) + (1 - d) * Ops.ToDouble(p.GetFlat(i))));
        }
    }

    /// <summary>Replaces Mimi's weights by their moving average over the training steps (§3.3's EMA at
    /// <see cref="MoshiOptions.MimiWeightAverageDecay"/>), as the trained codec is used.</summary>
    public void ApplyMimiWeightAverage()
    {
        ThrowIfDisposed();
        foreach (var pair in _mimiAverage)
        {
            for (int i = 0; i < pair.Key.Length; i++) pair.Key.SetFlat(i, pair.Value.GetFlat(i));
            Engine.InvalidatePersistentTensor(pair.Key);
        }
    }

    /// <inheritdoc />
    /// <remarks>A language-model step updates the RQ-Transformer; a Mimi step the codec and the distillation projection
    /// or, on its turn, the discriminator; WavLM stays frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        IEnumerable<LayerBase<T>> layers = _trainedGroup switch
        {
            "mimi" => MimiGeneratorLayers,
            "mimi discriminator" => _discriminatorLayers,
            _ => _lmLayers,
        };
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
