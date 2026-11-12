using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class FireRedTTS<T>
{
    private readonly Dictionary<FireRedTtsStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private Random? _trainingRandom;
    private int _tokenizerSteps;

    /// <inheritdoc />
    /// <remarks>FireRedTTS trains on a recording's semantic tokens (<see cref="TtsTrainingSample{T}.CodecTokens"/>
    /// <c>[frames, 1]</c>, or the recording in <see cref="TtsTrainingSample{T}.Audio"/>, which the tokenizer encodes) and
    /// a recording of the speaker (<see cref="TtsTrainingSample{T}.SpeakerReference"/>, else the recording) for the
    /// speaker embedding and the flow's timbre. The flow and tokenizer stages also need the recording.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens | TtsSupervision.ReferenceRecording;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => 1;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => _options.CodebookSize * _options.CodebookSize;

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

    /// <inheritdoc />
    /// <remarks>One step of the current <see cref="FireRedTTSOptions.Stage"/>'s objective: the GPT's XTTS losses, the
    /// flow-matching loss (eq. 6), or the tokenizer's eq. 1 with dual decoding (the codebooks then take an EMA step).</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        Validation.Guard.NotNull(sample);
        var random = _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed);
        if (_options.Stage == FireRedTtsStage.MelCodec) return TrainMelCodec(sample, random);
        var objective = Objective(sample, training: true, random);
        var result = TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => objective(), Optimizer());
        if (_options.Stage == FireRedTtsStage.SpeechTokenizer) _tokenizerSteps++;
        return result;
    }

    /// <inheritdoc />
    /// <remarks>The current stage's objective without dropout or the EMA update, so repeated evaluations compare.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Objective(sample, training: false, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed))()[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private Func<Tensor<T>> Objective(TtsTrainingSample<T> sample, bool training, Random random) => _options.Stage switch
    {
        FireRedTtsStage.TextToSpeech => GptObjective(sample, training, random),
        FireRedTtsStage.FlowDecoder => FlowObjective(sample, training, random),
        FireRedTtsStage.SpeechTokenizer => TokenizerObjective(sample, training, random),
        FireRedTtsStage.MelCodec => CodecObjective(PrepareCodec(sample), training, random, null),
        _ => AcousticObjective(sample, training, random),
    };

    // ---------------------------------------------------------------- the streamable decoder

    private (Tensor<T> Mel, Tensor<T> Speaker) PrepareCodec(TtsTrainingSample<T> sample)
    {
        var audio = sample.Audio ?? throw new ArgumentException("The Mel codec trains on the recording's mel; set Audio.", nameof(sample));
        using var _ = new NoGradScope<T>();
        var mel = FlowMel(audio, _options.SampleRate);
        int frames = mel.Shape[0] - mel.Shape[0] % MelCodec.Downsampling;
        if (frames < MelCodec.Downsampling) throw new ArgumentException("The recording is shorter than one codec frame.", nameof(sample));
        return (Engine.TensorNarrow(mel, 0, 0, frames), SpeakerOf(SpeakerRecording(sample)));
    }

    // The codec's generator loss: L1 mel reconstruction, the commitment loss and, with window starts, the LSGAN loss of
    // the discriminator's score of the reconstruction; the reconstruction is handed out for the discriminator's turn.
    private Func<Tensor<T>> CodecObjective((Tensor<T> Mel, Tensor<T> Speaker) target, bool training, Random random, int[]? starts,
        Action<Tensor<T>>? reconstructed = null)
    {
        var o = _options;
        return () =>
        {
            var codec = MelCodec;
            var (_, quantized, _, commitment) = codec.Quantizer.Forward(codec.Encode(target.Mel), update: training, random);
            var mel = codec.Decode(quantized, target.Speaker);
            reconstructed?.Invoke(mel);
            int frames = target.Mel.Shape[0], channels = target.Mel.Shape[1];
            var l1 = Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorAbs(Engine.TensorSubtract(mel, target.Mel)), null),
                Ops.FromDouble(1.0 / ((double)frames * channels)));
            var loss = Engine.TensorAdd(Engine.TensorMultiplyScalar(l1, Ops.FromDouble(o.CodecReconstructionWeight)), commitment);
            if (starts is not null)
            {
                var score = MelDiscriminator.Forward(Engine.Reshape(mel, new[] { 1, frames, channels }), starts)!;
                loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(MeanSquareTo(score, 1), Ops.FromDouble(o.CodecAdversarialWeight)));
            }
            return loss;
        };
    }

    // (x − target)² averaged: LSGAN's losses.
    private Tensor<T> MeanSquareTo(Tensor<T> x, double target)
    {
        var d = Engine.TensorAddScalar(x, Ops.FromDouble(-target));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(d, d), null), Ops.FromDouble(1.0 / x.Length));
    }

    private int[] WindowStarts(int frames, Random random) =>
        _options.DiscriminatorWindows.Select(w => frames >= w ? random.Next(frames - w + 1) : 0).ToArray();

    /// <summary>One GAN step of the Mel codec: the codec against its losses (the LSGAN term when the mel fills the
    /// discriminator's windows), then the discriminator on the real mel and the detached reconstruction.</summary>
    private T TrainMelCodec(TtsTrainingSample<T> sample, Random random)
    {
        var target = PrepareCodec(sample);
        int frames = target.Mel.Shape[0];
        bool adversarial = _options.CodecAdversarialWeight > 0 && _options.DiscriminatorWindows.All(w => w <= frames);
        Tensor<T>? generated = null;
        var loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens,
            (_, _) => CodecObjective(target, training: true, random, adversarial ? WindowStarts(frames, random) : null, m => generated = m)(),
            Optimizer());
        if (adversarial && generated is not null)
        {
            var fake = new Tensor<T>(generated._shape, new Vector<T>(generated.ToArray()));
            var realStarts = WindowStarts(frames, random);
            var fakeStarts = WindowStarts(frames, random);
            int channels = target.Mel.Shape[1];
            _discriminatorTurn = true;
            try
            {
                TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) =>
                {
                    var real = MelDiscriminator.Forward(Engine.Reshape(target.Mel, new[] { 1, frames, channels }), realStarts)!;
                    var forged = MelDiscriminator.Forward(Engine.Reshape(fake, new[] { 1, frames, channels }), fakeStarts)!;
                    return Engine.TensorAdd(MeanSquareTo(real, 1), MeanSquareTo(forged, 0));
                }, DiscriminatorOptimizer());
            }
            finally
            {
                _discriminatorTurn = false;
                LastLoss = loss;
            }
        }
        return loss;
    }

    private bool _discriminatorTurn;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> DiscriminatorOptimizer() =>
        _discriminatorOptimizer ??= PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, "mel discriminator")
            ?? throw new InvalidOperationException("FireRedTTS declares the discriminator's optimizer.");

    private Func<Tensor<T>> AcousticObjective(TtsTrainingSample<T> sample, bool training, Random random)
    {
        var tokens = SemanticTokens(sample);
        var audio = sample.Audio ?? throw new ArgumentException("The acoustic LM trains on the recording's Mel codec streams; set Audio.", nameof(sample));
        int[,] streams;
        Tensor<T> speaker;
        using (new NoGradScope<T>())
        {
            streams = MelCodec.Tokenize(FlowMel(audio, _options.SampleRate));
            speaker = SpeakerOf(SpeakerRecording(sample));
        }
        int n = Math.Min(tokens.Length, streams.GetLength(0) / 2);
        if (n < 1) throw new ArgumentException("The recording is shorter than one token (40 ms).", nameof(sample));
        var kept = tokens.Take(n).ToArray();
        var frames = new int[2 * n, streams.GetLength(1)];
        for (int t = 0; t < 2 * n; t++)
            for (int q = 0; q < frames.GetLength(1); q++) frames[t, q] = streams[t, q];
        return () => AcousticLm.Loss(frames, kept, speaker, training, random);
    }

    // ---------------------------------------------------------------- inputs

    private readonly Dictionary<Tensor<T>, Tensor<T>> _speakerCache = new(TensorReferenceComparer<Tensor<T>>.Instance);

    // The speaker embedding of a recording at the model's rate (the frozen acoustic encoder), cached per recording.
    private Tensor<T> SpeakerOf(Tensor<T> recording)
    {
        if (!_speakerCache.TryGetValue(recording, out var speaker))
        {
            if (_speakerCache.Count > 64) _speakerCache.Clear();
            speaker = _speakerCache[recording] = SpeakerEmbedding(To16k(recording, _options.SampleRate));
        }
        return speaker;
    }

    private int[] SemanticTokens(TtsTrainingSample<T> sample)
    {
        if (sample.CodecTokens is { } given)
        {
            if (given.Rank == 2 && given.Shape[1] != 1) throw new ArgumentException("Semantic tokens are [frames, 1].", nameof(sample));
            var ids = Enumerable.Range(0, given.Shape[0]).Select(i => (int)Math.Round(Ops.ToDouble(given.GetFlat(i)))).ToArray();
            if (ids.Any(t => t < 0 || t >= CodecTokenVocabulary)) throw new ArgumentOutOfRangeException(nameof(sample), $"Semantic tokens lie in [0, {CodecTokenVocabulary}).");
            return ids;
        }
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on semantic tokens or the recording; set CodecTokens or Audio.", nameof(sample));
        return EncodeSemanticTokens(audio, _options.SampleRate);
    }

    private Tensor<T> SpeakerRecording(TtsTrainingSample<T> sample) =>
        sample.SpeakerReference ?? sample.Audio ?? throw new ArgumentException("The speaker embedding needs a recording of the speaker; set SpeakerReference or Audio.", nameof(sample));

    // ---------------------------------------------------------------- the language model

    private Func<Tensor<T>> GptObjective(TtsTrainingSample<T> sample, bool training, Random random)
    {
        var o = _options;
        int[] text;
        int[]? labels = null;
        if (o.InstructionTuning && sample.Text is { } markup)
            (text, labels) = FireRedTtsInstruction.Encode(TextTokenizer, FireRedTtsInstruction.Strip(markup), FireRedTtsInstruction.Parse(markup));
        else
            text = TextIds(sample.Tokens, nameof(sample));
        if (text.Length == 0) throw new ArgumentException("The sample has no text tokens.", nameof(sample));
        if (text.Length + 2 > o.MaxTextTokens + 2) throw new ArgumentException($"At most {o.MaxTextTokens} text tokens fit.", nameof(sample));
        var tokens = SemanticTokens(sample);
        if (tokens.Length + 5 > o.MaxAudioTokens + 3) throw new ArgumentException($"At most {o.MaxAudioTokens - 2} semantic tokens fit.", nameof(sample));
        var speaker = SpeakerOf(SpeakerRecording(sample));
        int? emotion = o.InstructionTuning ? sample.ClassLabel : null;
        if (emotion is < 0 or > 3) throw new ArgumentOutOfRangeException(nameof(sample), "The emotion (ClassLabel) is 0 … 3.");
        return () =>
        {
            var (textLoss, audioLoss) = GptLosses(speaker, text, labels, tokens, emotion, training, random);
            return Engine.TensorAdd(Engine.TensorMultiplyScalar(textLoss, Ops.FromDouble(o.TextLossWeight)),
                Engine.TensorMultiplyScalar(audioLoss, Ops.FromDouble(o.AudioLossWeight)));
        };
    }

    /// <summary>XTTS's text and audio cross-entropies (<c>GPT.forward</c>) of one utterance.</summary>
    internal (Tensor<T> Text, Tensor<T> Audio) GptLosses(Tensor<T> speaker, IReadOnlyList<int> text, IReadOnlyList<int>? labels,
        IReadOnlyList<int> tokens, int? emotion, bool training, Random? random)
    {
        var gpt = Gpt;
        var (inputs, targets, textTargets) = gpt.XttsSequences(text, tokens);
        var (textLogits, audioLogits) = gpt.Forward(gpt.Prefix(speaker, emotion), text, labels, inputs, training, random, maskFinalKeys: true);
        return (CrossEntropy(textLogits, textTargets), CrossEntropy(audioLogits, targets));
    }

    // The mean cross-entropy of the first targets.Count rows of logits [m, V] against the targets.
    private Tensor<T> CrossEntropy(Tensor<T> logits, IReadOnlyList<int> targets)
    {
        int n = targets.Count, v = logits.Shape[1];
        logits = Engine.TensorNarrow(logits, 0, 0, n);
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(logits, axis: 1), new[] { n * v, 1 });
        var picks = new Tensor<int>(new[] { n }, new Vector<int>(Enumerable.Range(0, n).Select(i => i * v + targets[i]).ToArray()));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorIndexSelect(logProbabilities, picks, 0), null), Ops.FromDouble(-1.0 / n));
    }

    // ---------------------------------------------------------------- the flow

    private Func<Tensor<T>> FlowObjective(TtsTrainingSample<T> sample, bool training, Random random)
    {
        var audio = sample.Audio ?? throw new ArgumentException("The flow trains on the recording's mel; set Audio.", nameof(sample));
        Tensor<T> mel, reference;
        int[] tokens;
        using (new NoGradScope<T>())
        {
            tokens = SemanticTokens(sample);
            mel = FlowMel(audio, _options.SampleRate);
            reference = sample.SpeakerReference is { } other ? FlowMel(other, _options.SampleRate) : mel;
        }
        int n = Math.Min(tokens.Length, mel.Shape[0] / 4);
        if (n < 1) throw new ArgumentException("The recording is shorter than one token (40 ms).", nameof(sample));
        var kept = tokens.Take(n).ToArray();
        var target = Engine.TensorNarrow(mel, 0, 0, 4 * n);
        return () => Flow.TrainingLoss(kept, target, reference, training, random);
    }

    // ---------------------------------------------------------------- the speech tokenizer

    /// <summary>Dual decoding's weight γ at the current step (DPQ-VAE: 1, falling linearly to the final weight between the
    /// decay's start and end).</summary>
    internal double DualDecodingWeight(int step)
    {
        var o = _options;
        if (step <= o.DualDecodingDecayStart) return 1.0;
        if (step >= o.DualDecodingDecayEnd) return o.DualDecodingFinalWeight;
        double f = (step - o.DualDecodingDecayStart) / (double)(o.DualDecodingDecayEnd - o.DualDecodingDecayStart);
        return 1.0 + f * (o.DualDecodingFinalWeight - 1.0);
    }

    private Func<Tensor<T>> TokenizerObjective(TtsTrainingSample<T> sample, bool training, Random random)
    {
        var o = _options;
        var audio = sample.Audio ?? throw new ArgumentException("The tokenizer trains on the recording; set Audio.", nameof(sample));
        var tokenizer = SpeechTokenizer;
        var wave = tokenizer.PadToTokens(To16k(audio, o.SampleRate));
        Tensor<T> ssl, mel;
        using (new NoGradScope<T>())
        {
            ssl = tokenizer.SslFeatures(wave);                        // HuBERT is frozen
            mel = SpeakerMel(wave);
        }
        double gamma = DualDecodingWeight(_tokenizerSteps);
        return () =>
        {
            foreach (var layer in _speakerLayers) layer.SetTrainingMode(training);
            var encoded = tokenizer.Encode(ssl);
            var (_, quantized, preQuantized, commitment) = tokenizer.Quantizer.Forward(encoded, update: training, random);
            // The acoustic encoder reads "Clip&Shuffle" of the mel in training (§3.1), the whole mel otherwise.
            var speaker = SpeakerEncoder.Forward(training ? tokenizer.ClipAndShuffle(mel, 100, random) : mel);
            var (sslQ, melQ) = tokenizer.Reconstruct(quantized, speaker);
            var (sslP, melP) = tokenizer.Reconstruct(preQuantized, speaker);
            var ssLoss = Engine.TensorAdd(MeanSquare(sslQ, ssl), Engine.TensorMultiplyScalar(MeanSquare(sslP, ssl), Ops.FromDouble(gamma)));
            var melLoss = Engine.TensorAdd(MeanSquare(melQ, mel), Engine.TensorMultiplyScalar(MeanSquare(melP, mel), Ops.FromDouble(gamma)));
            return Engine.TensorAdd(Engine.TensorMultiplyScalar(commitment, Ops.FromDouble(o.VqLossWeight)),
                Engine.TensorAdd(Engine.TensorMultiplyScalar(ssLoss, Ops.FromDouble(o.SslLossWeight)),
                    Engine.TensorMultiplyScalar(melLoss, Ops.FromDouble(o.MelLossWeight))));
        };
    }

    // The mean squared error over the frames both hold (the decoders' output can run one token past the target).
    private Tensor<T> MeanSquare(Tensor<T> output, Tensor<T> target)
    {
        int frames = Math.Min(output.Shape[0], target.Shape[0]), width = target.Shape[1];
        var error = Engine.TensorSubtract(Engine.TensorNarrow(output, 0, 0, frames), Engine.TensorNarrow(target, 0, 0, frames));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(error, error), null), Ops.FromDouble(1.0 / ((double)frames * width)));
    }

    // ---------------------------------------------------------------- optimizers

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        var stage = _options.Stage;
        if (_optimizers.TryGetValue(stage, out var optimizer)) return optimizer;
        optimizer = stage switch
        {
            FireRedTtsStage.TextToSpeech => GptOptimizer(),
            FireRedTtsStage.FlowDecoder => PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, "flow", _options.FlowWarmupSteps)
                ?? throw new InvalidOperationException("FireRedTTS declares the flow's optimizer."),
            FireRedTtsStage.MelCodec => PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, "mel codec")
                ?? throw new InvalidOperationException("FireRedTTS declares the codec's optimizer."),
            FireRedTtsStage.AcousticLanguageModel => PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, "acoustic language model",
                    _options.AcousticLmWarmupSteps)
                ?? throw new InvalidOperationException("FireRedTTS declares the acoustic LM's optimizer."),
            _ => PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, "speech tokenizer")
                ?? throw new InvalidOperationException("FireRedTTS declares the tokenizer's optimizer."),
        };
        return _optimizers[stage] = optimizer;
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GptOptimizer()
    {
        var o = _options;
        // XTTS's optimizer_wd_only_on_weights: biases, normalization parameters and embedding tables are not decayed.
        var parameters = new List<Tensor<T>>();
        var decay = new List<T>();
        foreach (var layer in _gptLayers)
        {
            bool exempt = layer is LayerNormalizationLayer<T> || layer is TiedEmbeddingLayer<T>;
            foreach (var p in Training.TapeTrainingStep<T>.CollectParameters(new List<ILayer<T>> { layer }, -1))
            {
                parameters.Add(p);
                var d = exempt || p.Shape.Length == 1 ? Ops.Zero : Ops.One;
                for (int i = 0; i < p.Length; i++) decay.Add(d);
            }
        }
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate, Beta1 = o.Beta1, Beta2 = o.Beta2, Epsilon = o.Epsilon, WeightDecay = o.WeightDecay,
            MaskParameters = parameters,
            WeightDecayMask = new Vector<T>(decay.ToArray()),
            LearningRateScheduler = new MultiStepLRScheduler(o.LearningRate, o.LearningRateMilestones, o.LearningRateDecay),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        };
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options), "language model");
    }

    /// <inheritdoc />
    /// <remarks>Each stage updates its own networks: the GPT; the flow; the tokenizer's encoder, quantizer projections and
    /// decoders with the acoustic encoder; the Mel codec (or, on its turn, the discriminator); or the acoustic LM. HuBERT and
    /// the vocoder stay as loaded.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var layers = _options.Stage switch
        {
            FireRedTtsStage.TextToSpeech => _gptLayers,
            FireRedTtsStage.FlowDecoder => _flowLayers,
            FireRedTtsStage.SpeechTokenizer => _tokenizerLayers.Concat(_speakerLayers).ToList(),
            FireRedTtsStage.MelCodec => _discriminatorTurn ? _discriminatorLayers : _codecLayers,
            _ => _acousticLayers,
        };
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
