using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.Latest;

public partial class IndexTTS2<T>
{
    private readonly Dictionary<IndexTts2Stage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _emotionOptimizer;
    private List<LayerBase<T>>? _trainingAdapter;          // set while the text-to-emotion adapter trains
    private Random? _trainingRandom;

    /// <inheritdoc />
    /// <remarks>IndexTTS2 trains on recordings: the recording at <see cref="TtsModelOptions.SampleRate"/>
    /// (<see cref="TtsTrainingSample{T}.Audio"/>), its text's token ids (<see cref="TtsTrainingSample{T}.Tokens"/>, from
    /// <see cref="TokenizeText"/>), another recording of the same speaker as the prompt
    /// (<see cref="TtsTrainingSample{T}.SpeakerReference"/>; the recording itself when absent) and, for the emotion stage,
    /// the speaker's index (<see cref="TtsTrainingSample{T}.SpeakerId"/>).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    /// <remarks>The current stage's AdamW (the paper: learning rate 2e-4).</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? Optimizer() : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <inheritdoc />
    /// <remarks>
    /// One step of <see cref="IndexTTS2Options.CurrentStage"/>'s objective. The text-to-semantic stages minimize Eq. 1:
    /// the cross-entropy of the semantic tokens and the end token after <c>[c (+ e), p, BT text ET, BA]</c>, with p the
    /// target length's duration token or zeros (stage 1 drops it with <see cref="IndexTTS2Options.DurationDropProbability"/>);
    /// the emotion stage adds α times the speaker classifier's cross-entropy on e behind a gradient reversal layer. The
    /// semantic-to-mel stage minimizes Eq. 2, the flow's L1 over the target frames after a random prompt split, its
    /// semantic features fused with the GPT latents in half the samples.
    /// </remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
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
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
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

    // A recording's frozen features (w2v-BERT, codes, CAM++), computed once per recording.
    private sealed class Recording
    {
        public Tensor<T> Features = null!;
        public int[] Codes = null!;
        public Tensor<T>? Style;
        public Tensor<T>? Mel;
    }

    private readonly Dictionary<Tensor<T>, Recording> _recordings = new(TensorReferenceComparer<Tensor<T>>.Instance);

    private Recording RecordingOf(Tensor<T> audio, bool mel)
    {
        if (!_recordings.TryGetValue(audio, out var r))
        {
            using var _ = new NoGradScope<T>();
            var audio16k = ResampleTorchaudio(Doubles(audio), _options.SampleRate, _options.SemanticSampleRate);
            var features = SemanticFeatures(Doubles(audio16k));
            if (_recordings.Count > 64) _recordings.Clear();
            r = _recordings[audio] = new Recording { Features = features, Codes = SemanticCodec.EncodeNormalized(features), Style = Style(audio16k) };
        }
        if (mel && r.Mel is null)
            using (new NoGradScope<T>()) r.Mel = MelSpectrogram(audio);
        return r;
    }

    private int[] TextOf(TtsTrainingSample<T> sample)
    {
        var text = TokenIds(sample.Tokens, nameof(sample));
        if (text.Length == 0) throw new ArgumentException("The sample has no text tokens.", nameof(sample));
        return text;
    }

    private Func<Tensor<T>> Objective(TtsTrainingSample<T> sample, Random random)
    {
        Validation.Guard.NotNull(sample);
        var o = _options;
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample));
        var text = TextOf(sample);
        var target = RecordingOf(audio, mel: o.CurrentStage == IndexTts2Stage.SemanticToMel);
        var prompt = sample.SpeakerReference is { } reference ? RecordingOf(reference, mel: false) : target;
        var gpt = TextToSemantic;
        if (target.Codes.Length + 2 > gpt.MaxDurationTokens) throw new ArgumentException("The recording has more semantic tokens than the model has positions.", nameof(sample));

        if (o.CurrentStage == IndexTts2Stage.SemanticToMel)
        {
            var mel = target.Mel!;
            int frames = mel.Shape[0];
            if (frames < 2) throw new ArgumentException("The recording is too short for a prompt split.", nameof(sample));
            // seed-vc's split: a prompt of ⌊u (F − 1)⌋ frames, none one time in ten.
            int promptFrames = random.NextDouble() < 0.1 ? 0 : (int)Math.Floor(random.NextDouble() * (frames - 1));
            bool fuse = random.NextDouble() < o.LatentFusionProbability;
            int? duration = DurationToken(target.Codes.Length, random, drop: false);
            return () =>
            {
                var semantic = SemanticCodec.Embed(target.Codes);
                if (fuse) semantic = SemanticToMel.Fuse(semantic, Detached(gpt, prompt, text, target.Codes, duration));
                var condition = SemanticToMel.Regulator.Forward(semantic, frames);
                return SemanticToMel.Loss(mel, promptFrames, target.Style!, condition, random);
            };
        }

        int? token = DurationToken(target.Codes.Length, random, drop: o.CurrentStage == IndexTts2Stage.TextToSemantic);
        int? speakerId = sample.SpeakerId;
        if (o.CurrentStage == IndexTts2Stage.Emotion && (speakerId is not int id || id < 0 || id >= Math.Max(1, o.SpeakerCount)))
            throw new ArgumentException($"The emotion stage's speaker classifier needs SpeakerId in [0, {Math.Max(1, o.SpeakerCount)}).", nameof(sample));
        return () =>
        {
            Tensor<T> speaker, emotion;
            switch (o.CurrentStage)
            {
                case IndexTts2Stage.TextToSemantic:
                    speaker = gpt.SpeakerConditioning(prompt.Features, training: true, random);
                    emotion = new Tensor<T>(new[] { 1, o.HiddenDim });
                    break;
                case IndexTts2Stage.Emotion:
                    using (new NoGradScope<T>()) speaker = gpt.SpeakerConditioning(prompt.Features);
                    emotion = gpt.EmotionVector(prompt.Features, training: true, random);
                    break;
                default:
                    using (new NoGradScope<T>())
                    {
                        speaker = gpt.SpeakerConditioning(prompt.Features);
                        emotion = gpt.EmotionVector(prompt.Features);
                    }
                    break;
            }
            var latents = gpt.Latents(gpt.Conditions(speaker, emotion, token), text, target.Codes, training: true, random);
            var targets = target.Codes.Concat(new[] { gpt.Configuration.StopMel }).ToArray();
            var loss = CrossEntropy(gpt.Logits(latents), targets);
            if (o.CurrentStage != IndexTts2Stage.Emotion) return loss;
            var adversary = CrossEntropy(SpeakerClassifier.Forward(ReverseGradient(emotion, o.SpeakerAdversaryWeight)), new[] { speakerId!.Value });
            return Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(adversary, Ops.FromDouble(o.SpeakerAdversaryWeight)));
        };
    }

    // The GPT latents of the target, off the tape (the S2M stage trains only the flow and the latent MLP).
    private Tensor<T> Detached(IndexTtsUnifiedVoice<T> gpt, Recording prompt, int[] text, int[] codes, int? duration)
    {
        using var _ = new NoGradScope<T>();
        var conditions = gpt.Conditions(gpt.SpeakerConditioning(prompt.Features), gpt.EmotionVector(prompt.Features), duration);
        var latents = Engine.TensorNarrow(gpt.Latents(conditions, text, codes), 0, 0, codes.Length);
        return new Tensor<T>(latents.Shape.ToArray(), latents.ToVector());
    }

    /// <summary>The duration token's length: the target's token count, or null (zeros) when dropped; the release's speed
    /// tokens take none.</summary>
    private int? DurationToken(int tokens, Random random, bool drop)
    {
        if (_options.Duration != IndexTts2DurationConditioning.Paper) return null;
        if (drop && random.NextDouble() < _options.DurationDropProbability) return null;
        return tokens;
    }

    /// <summary>A gradient reversal layer: the value of x, with the gradient scaled by −λ on the way back
    /// (<c>(1 + λ)·sg(x) − λ·x</c>).</summary>
    private Tensor<T> ReverseGradient(Tensor<T> x, double lambda)
    {
        var constant = new Tensor<T>(x.Shape.ToArray(), x.ToVector());
        return Engine.TensorSubtract(Engine.TensorMultiplyScalar(constant, Ops.FromDouble(1 + lambda)), Engine.TensorMultiplyScalar(x, Ops.FromDouble(lambda)));
    }

    // The mean cross-entropy of logits [n, V] against targets [n].
    private Tensor<T> CrossEntropy(Tensor<T> logits, IReadOnlyList<int> targets)
    {
        int n = logits.Shape[0], v = logits.Shape[1];
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(logits, axis: 1), new[] { n * v, 1 });
        var picks = new Tensor<int>(new[] { n }, new Vector<int>(Enumerable.Range(0, n).Select(i => i * v + targets[i]).ToArray()));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorIndexSelect(logProbabilities, picks, 0), null), Ops.FromDouble(-1.0 / n));
    }

    /// <summary>The layers the current stage trains.</summary>
    private IReadOnlyList<LayerBase<T>> StageLayers()
    {
        var gpt = TextToSemantic;
        switch (_options.CurrentStage)
        {
            case IndexTts2Stage.SemanticToMel:
                return _s2melLayers;
            case IndexTts2Stage.TextToSemantic:
                return _gptLayers;
            case IndexTts2Stage.Emotion:
                return _gptLayers.Except(gpt.SpeakerConditionerLayers).Concat(_classifierLayers).ToList();
            default:
                return _gptLayers.Except(gpt.SpeakerConditionerLayers).Except(gpt.EmotionConditionerLayers).ToList();
        }
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        var stage = _options.CurrentStage;
        if (_optimizers.TryGetValue(stage, out var cached)) return cached;
        var optimizer = AdamW(StageLayers());
        _optimizers[stage] = optimizer;
        return optimizer;
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> AdamW(IReadOnlyList<LayerBase<T>> layers)
    {
        var o = _options;
        var parameters = Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1).ToList();
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate, Beta1 = o.Beta1, Beta2 = o.Beta2, Epsilon = o.Epsilon, WeightDecay = o.WeightDecay,
            MaskParameters = parameters,
            WeightDecayMask = new Vector<T>(Enumerable.Repeat(Ops.One, parameters.Sum(p => p.Length)).ToArray()),
        };
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
    }

    /// <inheritdoc />
    /// <remarks>Training updates <see cref="IndexTTS2Options.CurrentStage"/>'s layers; w2v-BERT, the semantic codec,
    /// CAM++ and BigVGAN are frozen, as are the conditioners the stage freezes.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var layers = _trainingAdapter ?? StageLayers();
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    /// <summary>
    /// One distillation step of the text-to-emotion model (§ Text-to-Emotion, Eq. 4): the cross-entropy of the teacher's
    /// answer for <paramref name="text"/> — <paramref name="distribution"/>, one probability per emotion label — trained
    /// into the LoRA adapter only. Returns the loss before the step.
    /// </summary>
    public T TrainTextToEmotion(string text, IReadOnlyList<double> distribution)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (distribution is null) throw new ArgumentNullException(nameof(distribution));
        if (!HasPaperLayers) throw NotNative();
        var t2e = TextToEmotion();
        var lora = t2e.Model.Layers.SelectMany(l => new[] { l.QueryLora, l.ValueLora }).Where(a => a is not null)
            .SelectMany(a => new LayerBase<T>[] { a!.A, a.B }).ToList();
        _emotionOptimizer ??= AdamW(lora);
        var tokens = IndexTtsOps.Ids<T>(new[] { 0 });
        _trainingAdapter = lora;
        try
        {
            return TrainWithCustomObjective(tokens, tokens, (_, _) => t2e.Loss(text, distribution), _emotionOptimizer);
        }
        finally
        {
            _trainingAdapter = null;
        }
    }

    /// <summary>The text-to-emotion model's emotion scores for <paramref name="text"/>, one per
    /// <see cref="IndexTTS2Options.EmotionLabels"/>.</summary>
    public double[] DetectEmotion(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (!HasPaperLayers) throw NotNative();
        using var _ = new NoGradScope<T>();
        return TextToEmotion().Scores(text, _options.EmotionMaxTokens);
    }
}
