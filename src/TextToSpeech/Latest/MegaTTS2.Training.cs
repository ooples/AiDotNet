using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.Latest;

public partial class MegaTTS2<T>
{
    private enum Turn
    {
        Acoustic,
        Discriminator,
        Prosody,
        Duration,
    }

    private Turn _turn = Turn.Acoustic;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _acousticOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _prosodyOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _durationOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private Random? _trainingRandom;

    /// <summary>A training utterance: its phonemes, their durations and 1-based frame map, its mel spectrogram
    /// <c>[frames, mels]</c>, and the MRTE's reference mel spectrogram.</summary>
    internal sealed record Utterance(int[] Phones, int[] Durations, int[] Mel2Ph, Tensor<T> Mel, Tensor<T> Reference);

    /// <summary>What the autoencoder makes of an utterance: the mel spectrogram and the VQ loss.</summary>
    internal sealed record Reconstruction(Tensor<T> Mel, Tensor<T> VqLoss);

    /// <summary>The language models' training sequence: the P-LLM's tokens, features, positions, speakers and targets, and
    /// the duration model's.</summary>
    internal sealed record LanguageSequence(int[] Tokens, Tensor<T> Features, int[] Positions, int[] Speakers, int[] Targets,
        int[] DurationTokens, Tensor<T> DurationFeatures, int[] DurationPositions, int[] DurationSpeakers, int[] Durations, bool[] DurationKept);

    private Random TrainingRandom => _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed);

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        throw new NotSupportedException($"{GetType().Name} trains on phoneme durations from a forced aligner: use Train(TtsTrainingSample) with Durations set.");
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>In the acoustic stage, one generator step and one discriminator step, as SyntaSpeech alternates them: the L2
    /// mel loss, the VQ loss with CVQ-VAE's dynamic initialization, and the LSGAN loss of the multi-length discriminator
    /// (Jiang et al. 2024, §3.2); then the discriminator on the recording's mel spectrogram against the generated one.</para>
    /// <para>In the language-model stage, this sentence alone; use <see cref="TrainLanguageModels"/> for the paper's
    /// multi-sentence sequences.</para>
    /// </remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        if (_options.Stage == MegaTts2Stage.LanguageModels) return TrainLanguageModels(new[] { new[] { sample } });
        var o = _options;
        var random = TrainingRandom;
        var utterance = Prepare(sample, random);
        int frames = utterance.Mel.Shape[0];
        bool adversarial = o.AdversarialWeight > 0 && o.DiscriminatorWindows.All(w => w <= frames);
        var generatorStarts = adversarial ? Starts(frames, random) : null;
        Tensor<T>? generated = null;
        var loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) =>
        {
            var (total, reconstruction) = AcousticLoss(utterance, generatorStarts, update: true);
            generated = reconstruction.Mel;
            return total;
        }, Optimizer(Turn.Acoustic));
        if (adversarial && generated is not null)
        {
            // The discriminator reads the generator step's output, detached, with fresh windows for each mel.
            var fake = new Tensor<T>(generated._shape, new Vector<T>(generated.ToArray()));
            var realStarts = Starts(frames, random);
            var fakeStarts = Starts(frames, random);
            _turn = Turn.Discriminator;
            try
            {
                TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) =>
                {
                    var real = Network.Discriminator.Forward(Batch(utterance.Mel), realStarts)!;
                    var forged = Network.Discriminator.Forward(Batch(fake), fakeStarts)!;
                    return Engine.TensorAdd(MeanSquare(real, 1), MeanSquare(forged, 0));
                }, Optimizer(Turn.Discriminator));
            }
            finally
            {
                _turn = Turn.Acoustic;
                // The step's loss is the generator's; the discriminator's turn does not replace it.
                LastLoss = loss;
            }
        }
        return loss;
    }

    /// <inheritdoc />
    /// <remarks>The stage's objective without the adversarial term (which depends on the discriminator's random windows)
    /// and without moving the codebook: the mel and VQ losses, or the P-LLM's cross-entropy plus the duration model's
    /// squared error.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        if (!HasPaperLayers) throw NotNative();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            if (_options.Stage == MegaTts2Stage.LanguageModels)
            {
                var s = LanguageSequenceOf(new[] { new[] { sample } });
                return NumOps.Add(ProsodyLoss(s)[0], DurationLoss(s)[0]);
            }
            return AcousticLoss(Prepare(sample, null), null, update: false).Loss[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// One step of the second stage on sentences grouped by speaker (§3.3): each sentence's prosody codes and content from
    /// the frozen autoencoder, every speaker's sentences one sequence with its own positions, attention kept within each
    /// speaker; the P-LLM's teacher-forced cross-entropy over the codes and each sentence's end, and the duration model's
    /// squared error of the phoneme durations.
    /// </summary>
    /// <returns>The sum of the two models' losses.</returns>
    public T TrainLanguageModels(IReadOnlyList<IReadOnlyList<TtsTrainingSample<T>>> speakers)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw NotNative();
        if (speakers is null || speakers.Count == 0 || speakers.Any(s => s is null || s.Count == 0))
            throw new ArgumentException("Every speaker has sentences.", nameof(speakers));
        var s = LanguageSequenceOf(speakers);
        if (s.Tokens.Length > _options.ProsodyTokenBudget)
            throw new ArgumentException($"The sentences make {s.Tokens.Length} prosody tokens; a step reads at most {_options.ProsodyTokenBudget}.", nameof(speakers));
        var tokens = speakers[0][0].Tokens;
        try
        {
            _turn = Turn.Prosody;
            var prosody = TrainWithCustomObjective(tokens, tokens, (_, _) => ProsodyLoss(s), Optimizer(Turn.Prosody));
            _turn = Turn.Duration;
            var duration = TrainWithCustomObjective(tokens, tokens, (_, _) => DurationLoss(s), Optimizer(Turn.Duration));
            var total = NumOps.Add(prosody, duration);
            LastLoss = total;
            return total;
        }
        finally
        {
            _turn = Turn.Acoustic;
        }
    }

    // ---------------------------------------------------------------- the acoustic stage

    /// <summary>A sample's phonemes, durations, mel spectrogram (its own, or the recording's) and the MRTE's reference,
    /// at most <see cref="MegaTTS2Options.TimbreReferenceFrames"/> frames of it: drawn at random in training, the first
    /// ones otherwise.</summary>
    internal Utterance Prepare(TtsTrainingSample<T> sample, Random? random)
    {
        Validation.Guard.NotNull(sample);
        var phones = Ids(sample.Tokens, nameof(sample));
        var mel = sample.Mel ?? (sample.Audio is not null ? MelOf(Doubles(sample.Audio))
            : throw new ArgumentException($"{GetType().Name} trains on the mel spectrogram; set Mel or Audio.", nameof(sample)));
        if (mel.Shape.Length != 2 || mel.Shape[1] != _options.MelChannels)
            throw new ArgumentException($"The mel spectrogram is [frames, {_options.MelChannels}].", nameof(sample));
        var durations = sample.Durations
            ?? throw new ArgumentException($"{GetType().Name} trains on phoneme durations; set Durations.", nameof(sample));
        if (durations.Length != phones.Length) throw new ArgumentException("There is one duration per phoneme.", nameof(sample));
        if (durations.Any(d => d < 0) || durations.Sum() != mel.Shape[0])
            throw new ArgumentException($"The durations sum to the mel spectrogram's {mel.Shape[0]} frames.", nameof(sample));
        var reference = sample.SpeakerReference is null ? mel
            : sample.SpeakerReference.Shape.Length == 1 ? MelOf(Doubles(sample.SpeakerReference)) : sample.SpeakerReference;
        if (reference.Shape.Length != 2 || reference.Shape[1] != _options.MelChannels)
            throw new ArgumentException($"The speaker reference is a recording or a [frames, {_options.MelChannels}] mel spectrogram.", nameof(sample));
        int limit = _options.TimbreReferenceFrames;
        if (limit > 0 && reference.Shape[0] > limit)
        {
            int start = random is null ? 0 : random.Next(reference.Shape[0] - limit + 1);
            reference = Engine.TensorSlice(reference, new[] { start, 0 }, new[] { limit, _options.MelChannels });
        }
        return new Utterance(phones, durations, MegaTts3Aligner<T>.LengthRegulate(durations), mel, reference);
    }

    /// <summary>The autoencoder on an utterance: the VQ encoder on its mel spectrogram, the content encoder on its
    /// phonemes, the MRTE on the reference; the decoded mel spectrogram and the VQ loss. With
    /// <paramref name="update"/>, the quantizer's dynamic initialization moves the codebook.</summary>
    internal Reconstruction Reconstruct(Utterance u, bool update)
    {
        var content = Content(u.Phones);
        var (quantized, _, vq) = Network.Quantizer.Forward(Network.ProsodyHidden(u.Mel), update, TrainingRandom);
        var prosody = Network.ProsodyAtFrames(quantized, u.Mel.Shape[0]);
        var timbre = Network.Timbre(content, Network.TimbreStates(u.Reference));
        return new Reconstruction(Network.DecodeMel(content, timbre, prosody, u.Mel2Ph), vq);
    }

    /// <summary>The generator's loss L_rec + L_VQ (+ the weighted L_Adv at window starts).</summary>
    internal (Tensor<T> Loss, Reconstruction Reconstruction) AcousticLoss(Utterance u, IReadOnlyList<int>? starts, bool update)
    {
        var r = Reconstruct(u, update);
        var loss = Engine.TensorAdd(MeanSquare(Engine.TensorSubtract(r.Mel, u.Mel), 0), Engine.Reshape(r.VqLoss, new[] { 1 }));
        if (starts is not null)
        {
            var score = Network.Discriminator.Forward(Batch(r.Mel), starts);
            if (score is not null)
                loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(MeanSquare(score, 1), NumOps.FromDouble(_options.AdversarialWeight)));
        }
        return (loss, r);
    }

    /// <summary>The mean of (x − c)² <c>[1]</c>.</summary>
    private Tensor<T> MeanSquare(Tensor<T> x, double c)
    {
        var d = c == 0 ? x : Engine.TensorAddScalar(x, NumOps.FromDouble(-c));
        var sum = Engine.ReduceSum(Engine.TensorMultiply(d, d), Enumerable.Range(0, d.Shape.Length).ToArray(), false);
        return Engine.TensorMultiplyScalar(Engine.Reshape(sum, new[] { 1 }), NumOps.FromDouble(1.0 / d.Length));
    }

    private Tensor<T> Batch(Tensor<T> mel) => Engine.Reshape(mel, new[] { 1, mel.Shape[0], mel.Shape[1] });

    // SyntaSpeech's clip: a uniform start for each window over the frames.
    private int[] Starts(int frames, Random random) =>
        _options.DiscriminatorWindows.Select(w => frames >= w ? random.Next(frames - w + 1) : 0).ToArray();

    // ---------------------------------------------------------------- the language models

    /// <summary>The second stage's sequence of speakers' sentences, read by the frozen autoencoder.</summary>
    internal LanguageSequence LanguageSequenceOf(IReadOnlyList<IReadOnlyList<TtsTrainingSample<T>>> speakers)
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var tokens = new List<int>();
            var features = new List<Tensor<T>>();
            var positions = new List<int>();
            var groups = new List<int>();
            var targets = new List<int>();
            var durationTokens = new List<int>();
            var durationFeatures = new List<Tensor<T>>();
            var durationPositions = new List<int>();
            var durationGroups = new List<int>();
            var durations = new List<int>();
            var kept = new List<bool>();
            for (int speaker = 0; speaker < speakers.Count; speaker++)
            {
                int position = 0, durationPosition = 0;
                foreach (var sentence in speakers[speaker])
                {
                    var u = Prepare(sentence, null);
                    var content = Content(u.Phones);
                    var codes = Network.Quantizer.Encode(Network.ProsodyHidden(u.Mel));
                    var (t, f) = ProsodyInputs(codes, Network.PooledContent(content, u.Mel2Ph));
                    tokens.AddRange(t);
                    features.Add(f);
                    targets.AddRange(codes.Select(c => c + 1).Append(0));
                    for (int i = 0; i < t.Length; i++)
                    {
                        positions.Add(position++);
                        groups.Add(speaker);
                    }
                    durationTokens.AddRange(DurationTokens(u.Durations));
                    durationFeatures.Add(Rows(content));
                    durations.AddRange(u.Durations);
                    kept.AddRange(u.Phones.Select(p => p != 0));
                    for (int i = 0; i < u.Phones.Length; i++)
                    {
                        durationPositions.Add(durationPosition++);
                        durationGroups.Add(speaker);
                    }
                }
            }
            Tensor<T> Join(List<Tensor<T>> parts) => parts.Count == 1 ? parts[0] : Engine.TensorConcatenate(parts.ToArray(), 0);
            return new LanguageSequence(tokens.ToArray(), Join(features), positions.ToArray(), groups.ToArray(), targets.ToArray(),
                durationTokens.ToArray(), Join(durationFeatures), durationPositions.ToArray(), durationGroups.ToArray(), durations.ToArray(), kept.ToArray());
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>The P-LLM's mean cross-entropy <c>[1]</c> of its targets, teacher-forced.</summary>
    internal Tensor<T> ProsodyLoss(LanguageSequence s)
    {
        int n = s.Targets.Length, v = _options.Codes + 1;
        var logits = Network.ProsodyModel.Forward(s.Tokens, s.Features, s.Positions, s.Speakers);
        var logp = Engine.TensorLogSoftmax(logits, axis: -1);
        var pick = new Tensor<T>(new[] { n, v });
        for (int i = 0; i < n; i++) pick[i, s.Targets[i]] = NumOps.FromDouble(-1.0 / n);
        return Engine.Reshape(Engine.ReduceSum(Engine.TensorMultiply(logp, pick), new[] { 0, 1 }, false), new[] { 1 });
    }

    /// <summary>The duration model's mean squared error <c>[1]</c> of the durations in frames, over the phonemes.</summary>
    internal Tensor<T> DurationLoss(LanguageSequence s)
    {
        int n = s.Durations.Length;
        double count = Math.Max(1, s.DurationKept.Count(k => k));
        var predicted = Network.DurationModel.Forward(s.DurationTokens, s.DurationFeatures, s.DurationPositions, s.DurationSpeakers);
        var target = new Tensor<T>(new[] { n }, new Vector<T>(s.Durations.Select(d => NumOps.FromDouble(d)).ToArray()));
        var weight = new Tensor<T>(new[] { n }, new Vector<T>(s.DurationKept.Select(k => NumOps.FromDouble(k ? 1.0 / count : 0)).ToArray()));
        var d = Engine.TensorSubtract(predicted, target);
        return Engine.Reshape(Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorMultiply(d, d), weight), new[] { 0 }, false), new[] { 1 });
    }

    // ---------------------------------------------------------------- optimizers

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer(Turn turn)
    {
        var o = _options;
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Build(string component, LearningRateSchedulerBase scheduler, double beta2, double epsilon)
        {
            var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = scheduler.CurrentLearningRate,
                Beta1 = o.Beta1,
                Beta2 = beta2,
                Epsilon = epsilon,
                UseAdaptiveBetas = false,
                EnableGradientClipping = o.MaxGradientNorm > 0,
                MaxGradientNorm = o.MaxGradientNorm > 0 ? o.MaxGradientNorm : 1.0,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            };
            return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options), component);
        }
        return turn switch
        {
            Turn.Acoustic => _acousticOptimizer ??= Build("acoustic", new NoamSchedule(o.HiddenDim, o.WarmupSteps, o.ScheduleFactor), o.Beta2, o.Epsilon),
            Turn.Prosody => _prosodyOptimizer ??= Build("prosody", new NoamSchedule(o.LmHidden, o.WarmupSteps, o.ScheduleFactor), o.Beta2, o.Epsilon),
            Turn.Duration => _durationOptimizer ??= Build("duration", new NoamSchedule(o.DurationHidden, o.WarmupSteps, o.ScheduleFactor), o.Beta2, o.Epsilon),
            // SyntaSpeech's discriminator AdamW: betas (0.9, 0.98).
            _ => _discriminatorOptimizer ??= Build("discriminator",
                new StepLRScheduler(o.DiscriminatorLearningRate, o.DiscriminatorDecaySteps, o.DiscriminatorDecay), 0.98, o.DiscriminatorEpsilon),
        };
    }

    /// <inheritdoc />
    /// <remarks>The generator's step trains the encoders and the decoder; the discriminator's step the discriminator; the
    /// second stage's steps the P-LLM and the duration model. The vocoder is trained apart (the paper uses a pre-trained
    /// HiFi-GAN).</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var layers = _turn switch
        {
            Turn.Discriminator => _discriminatorLayers,
            Turn.Prosody => _lmLayers,
            Turn.Duration => _durationLayers,
            _ => _acousticLayers,
        };
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
