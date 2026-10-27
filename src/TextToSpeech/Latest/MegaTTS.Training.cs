using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.Latest;

public partial class MegaTTS<T>
{
    private enum Turn
    {
        Acoustic,
        Discriminator,
        Prosody,
    }

    private Turn _turn = Turn.Acoustic;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _acousticOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _prosodyOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private Random? _trainingRandom;

    /// <summary>A training utterance: its phonemes, their 1-based frame map, its mel spectrogram <c>[frames, mels]</c>
    /// and the timbre encoder's reference mel spectrogram.</summary>
    internal sealed record Utterance(int[] Phones, int[] Durations, int[] Mel2Ph, Tensor<T> Mel, Tensor<T> Reference);

    /// <summary>What the acoustic model makes of an utterance: the mel spectrogram, the predicted durations <c>[L]</c>,
    /// and the VQ loss.</summary>
    internal sealed record Reconstruction(Tensor<T> Mel, Tensor<T> Durations, Tensor<T> VqLoss);

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        throw new NotSupportedException($"{GetType().Name} trains on phoneme durations from a forced aligner: use Train(TtsTrainingSample) with Durations set.");
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>In the acoustic stage, one generator step and one discriminator step, as SyntaSpeech alternates them: the L2
    /// mel loss, the VQ loss and the LSGAN loss of the multi-window discriminator (Jiang et al. 2023, Eq. 1–2) with
    /// NATSpeech's phoneme, word and sentence duration losses; then the discriminator on the recording's mel spectrogram
    /// against the generated one.</para>
    /// <para>In the prosody language model stage, the P-LLM's teacher-forced cross-entropy on this sentence alone; use
    /// <see cref="TrainProsodyModel"/> for the paper's contextual sentences.</para>
    /// </remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        if (_options.Stage == MegaTtsStage.ProsodyLanguageModel) return TrainProsodyModel(new[] { sample });
        var o = _options;
        var random = _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.TrainingSeed);
        var utterance = Prepare(sample);
        int frames = utterance.Mel.Shape[0];
        bool adversarial = o.AdversarialWeight > 0 && o.DiscriminatorWindows.All(w => w <= frames);
        var generatorStarts = adversarial ? Starts(frames, random) : null;
        Tensor<T>? generated = null;
        var loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) =>
        {
            var (total, reconstruction) = AcousticLoss(utterance, generatorStarts);
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
    /// <remarks>The stage's objective without the adversarial term (which depends on the discriminator's random
    /// windows): the acoustic stage's mel, VQ and duration losses, or the P-LLM's cross-entropy.</remarks>
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
            if (_options.Stage == MegaTtsStage.ProsodyLanguageModel)
            {
                var (previous, content, timbre, targets) = ProsodySequence(new[] { sample });
                return ProsodyLoss(previous, content, timbre, targets)[0];
            }
            return AcousticLoss(Prepare(sample), null).Loss[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// Trains the P-LLM on sentences of one speaker read as one sequence (the paper's Table 7 trains on 7 contextual
    /// sentences): each sentence's prosody codes from the frozen prosody encoder and its content from the frozen content
    /// encoder, joined; the first sentence's timbre; the teacher-forced cross-entropy of every code (Eq. 4).
    /// </summary>
    /// <returns>The step's loss.</returns>
    public T TrainProsodyModel(IReadOnlyList<TtsTrainingSample<T>> sentences)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw NotNative();
        if (sentences is null || sentences.Count == 0) throw new ArgumentException("There are no sentences.", nameof(sentences));
        if (sentences.Count > _options.ContextSentences)
            throw new ArgumentException($"The P-LLM reads at most {_options.ContextSentences} sentences at once.", nameof(sentences));
        var (previous, content, timbre, targets) = ProsodySequence(sentences);
        _turn = Turn.Prosody;
        try
        {
            var tokens = sentences[0].Tokens;
            return TrainWithCustomObjective(tokens, tokens, (_, _) => ProsodyLoss(previous, content, timbre, targets), Optimizer(Turn.Prosody));
        }
        finally
        {
            _turn = Turn.Acoustic;
        }
    }

    // ---------------------------------------------------------------- the acoustic stage

    /// <summary>A sample's phonemes, durations, mel spectrogram (its own, or the recording's) and reference.</summary>
    internal Utterance Prepare(TtsTrainingSample<T> sample)
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
        return new Utterance(phones, durations, MegaTts3Aligner<T>.LengthRegulate(durations), mel, reference);
    }

    /// <summary>The acoustic model on an utterance: the prosody encoder on its mel spectrogram with the VQ bottleneck,
    /// the content encoder on its phonemes, the timbre encoder on the reference; the decoded mel spectrogram and the
    /// predicted durations.</summary>
    internal Reconstruction Reconstruct(Utterance u)
    {
        int l = u.Phones.Length;
        var content = Content(u.Phones);
        var hidden = Network.ProsodyHidden(u.Mel, u.Mel2Ph, l);
        var mask = new Tensor<T>(new[] { 1, l, 1 });
        for (int i = 0; i < l; i++)
            if (u.Durations[i] > 0) mask[0, i, 0] = NumOps.One;
        var (quantized, _, vq) = Network.Quantizer.Forward(hidden, mask);
        var prosody = Network.ProsodyStates(quantized);
        var mel = Network.DecodeMel(content, prosody, Network.TimbreVector(u.Reference), u.Mel2Ph);
        return new Reconstruction(mel, Network.PredictDurations(content, prosody, u.Phones), vq);
    }

    /// <summary>The generator's loss: the mean squared mel error, the VQ loss, the duration losses, and with window
    /// starts the adversarial loss.</summary>
    internal (Tensor<T> Loss, Reconstruction Reconstruction) AcousticLoss(Utterance u, IReadOnlyList<int>? starts)
    {
        var o = _options;
        var r = Reconstruct(u);
        var loss = Engine.TensorAdd(MeanSquare(Engine.TensorSubtract(r.Mel, u.Mel), 0), Engine.Reshape(r.VqLoss, new[] { 1 }));
        loss = Engine.TensorAdd(loss, DurationLoss(r.Durations, u.Phones, u.Durations));
        if (starts is not null)
        {
            var score = Network.Discriminator.Forward(Batch(r.Mel), starts);
            if (score is not null)
                loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(MeanSquare(score, 1), NumOps.FromDouble(o.AdversarialWeight)));
        }
        return (loss, r);
    }

    /// <summary>
    /// NATSpeech's <c>add_dur_loss</c>: the squared error of log(d + 1) over the phonemes (padding excluded) weighted by
    /// <see cref="MegaTTSOptions.DurationWeight"/>; the same over the words (runs of non-silent phonemes) with a duration,
    /// weighted by <see cref="MegaTTSOptions.WordDurationWeight"/>; and of the sentence's total, weighted by
    /// <see cref="MegaTTSOptions.SentenceDurationWeight"/>.
    /// </summary>
    internal Tensor<T> DurationLoss(Tensor<T> predicted, int[] phones, int[] durations)
    {
        var o = _options;
        int l = phones.Length;
        Tensor<T> LogPlusOne(Tensor<T> x) => Engine.TensorLog(Engine.TensorAddScalar(x, NumOps.One));
        Tensor<T> Constant(IEnumerable<double> values, int[] shape) => new(shape, new Vector<T>(values.Select(v => NumOps.FromDouble(Math.Log(v + 1))).ToArray()));
        Tensor<T> WeightedMean(Tensor<T> error, double[] weights, double weight)
        {
            double count = weights.Sum();
            var w = new Tensor<T>(error._shape, new Vector<T>(weights.Select(v => NumOps.FromDouble(v)).ToArray()));
            var sum = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorMultiply(error, error), w), new[] { 0 }, false);
            return Engine.TensorMultiplyScalar(Engine.Reshape(sum, new[] { 1 }), NumOps.FromDouble(weight / count));
        }
        var nonPadding = phones.Select(p => p != 0 ? 1.0 : 0.0).ToArray();
        var target = durations.Select((d, i) => d * nonPadding[i]).ToArray();
        var loss = nonPadding.Sum() > 0
            ? WeightedMean(Engine.TensorSubtract(LogPlusOne(predicted), Constant(target, new[] { l })), nonPadding, o.DurationWeight)
            : new Tensor<T>(new[] { 1 });
        // Word ids: the running count of silent phonemes, 0 on the silent phonemes themselves.
        var silent = phones.Select(p => NatSpeechFrontEnd.IsSilence(_vocabulary[p])).ToArray();
        var wordId = new int[l];
        for (int i = 0, run = 0; i < l; i++)
        {
            if (silent[i]) run++;
            wordId[i] = silent[i] ? 0 : run;
        }
        int words = wordId.Length == 0 ? 0 : wordId.Max();
        if (o.WordDurationWeight > 0 && words > 0)
        {
            var gather = new Tensor<T>(new[] { words, l });
            var wordTarget = new double[words];
            for (int i = 0; i < l; i++)
                if (wordId[i] > 0)
                {
                    gather[wordId[i] - 1, i] = NumOps.One;
                    wordTarget[wordId[i] - 1] += target[i];
                }
            var keep = wordTarget.Select(d => d > 0 ? 1.0 : 0.0).ToArray();
            if (keep.Sum() > 0)
            {
                var wordPredicted = Engine.Reshape(Engine.TensorMatMul(gather, Engine.Reshape(predicted, new[] { l, 1 })), new[] { words });
                loss = Engine.TensorAdd(loss, WeightedMean(Engine.TensorSubtract(LogPlusOne(wordPredicted), Constant(wordTarget, new[] { words })), keep,
                    o.WordDurationWeight));
            }
        }
        if (o.SentenceDurationWeight > 0)
        {
            var total = Engine.Reshape(Engine.ReduceSum(predicted, new[] { 0 }, false), new[] { 1 });
            var error = Engine.TensorSubtract(LogPlusOne(total), Constant(new[] { target.Sum() }, new[] { 1 }));
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(Engine.TensorMultiply(error, error), NumOps.FromDouble(o.SentenceDurationWeight)));
        }
        return loss;
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

    // ---------------------------------------------------------------- the prosody language model

    /// <summary>Sentences read for the P-LLM by the frozen acoustic model: the start token and every code but the last,
    /// each phoneme's content <c>[1, L, h]</c>, the first sentence's timbre, and the codes.</summary>
    internal (int[] Previous, Tensor<T> Content, Tensor<T> Timbre, int[] Targets) ProsodySequence(IReadOnlyList<TtsTrainingSample<T>> sentences)
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var codes = new List<int>();
            var contents = new List<Tensor<T>>();
            Tensor<T>? timbre = null;
            foreach (var sentence in sentences)
            {
                var u = Prepare(sentence);
                contents.Add(Content(u.Phones));
                codes.AddRange(Row(Network.Quantizer.Encode(Network.ProsodyHidden(u.Mel, u.Mel2Ph, u.Phones.Length))));
                timbre ??= Network.TimbreVector(u.Reference);
            }
            var content = contents.Count == 1 ? contents[0] : Engine.TensorConcatenate(contents.ToArray(), 1);
            var previous = new[] { Network.StartCode }.Concat(codes.Take(codes.Count - 1)).ToArray();
            return (previous, content, timbre!, codes.ToArray());
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>The P-LLM's mean cross-entropy <c>[1]</c> of the codes, teacher-forced.</summary>
    internal Tensor<T> ProsodyLoss(int[] previous, Tensor<T> content, Tensor<T> timbre, int[] targets)
    {
        int n = targets.Length, v = _options.Codes + 2;
        var logp = Engine.TensorLogSoftmax(Network.ProsodyLogits(previous, content, timbre), axis: -1);
        var pick = new Tensor<T>(new[] { n, v });
        for (int i = 0; i < n; i++) pick[i, targets[i]] = NumOps.FromDouble(-1.0 / n);
        return Engine.Reshape(Engine.ReduceSum(Engine.TensorMultiply(logp, pick), new[] { 0, 1 }, false), new[] { 1 });
    }

    // ---------------------------------------------------------------- optimizers

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer(Turn turn)
    {
        var o = _options;
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Build(string component, LearningRateSchedulerBase scheduler, double epsilon)
        {
            var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = scheduler.CurrentLearningRate,
                Beta1 = o.Beta1,
                Beta2 = o.Beta2,
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
            Turn.Acoustic => _acousticOptimizer ??= Build("acoustic", new NoamSchedule(o.HiddenDim, o.WarmupSteps, o.ScheduleFactor), o.Epsilon),
            Turn.Prosody => _prosodyOptimizer ??= Build("prosody", new NoamSchedule(o.LmHidden, o.WarmupSteps, o.ScheduleFactor), o.Epsilon),
            _ => _discriminatorOptimizer ??= Build("discriminator",
                new StepLRScheduler(o.DiscriminatorLearningRate, o.DiscriminatorDecaySteps, o.DiscriminatorDecay), o.DiscriminatorEpsilon),
        };
    }

    /// <inheritdoc />
    /// <remarks>The generator's step trains the encoders, the duration predictor and the decoder; the discriminator's step
    /// the discriminator; the P-LLM's step the P-LLM. The vocoder is trained apart (the paper uses a pre-trained HiFi-GAN).</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var layers = _turn switch
        {
            Turn.Discriminator => _discriminatorLayers,
            Turn.Prosody => _lmLayers,
            _ => _acousticLayers,
        };
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
