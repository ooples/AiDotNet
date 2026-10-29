using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class IndexTTS<T>
{
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private Random? _trainingRandom;

    /// <inheritdoc />
    /// <remarks>IndexTTS trains on recordings: the recording at 24 kHz (<see cref="TtsTrainingSample{T}.Audio"/>), its text's
    /// token ids (<see cref="TtsTrainingSample{T}.Tokens"/>, from <see cref="TokenizeText"/>) and another recording of the
    /// same speaker as the prompt (<see cref="TtsTrainingSample{T}.SpeakerReference"/>; the recording itself when
    /// absent).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    /// <remarks>AdamW at 5e-6 (XTTS's GPT recipe).</remarks>
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
    /// One step of XTTS's GPT objective on UnifiedVoice's <c>forward</c>: the recording's DVAE codes (those past
    /// ⌈samples / 1024⌉ + 1 set to the stop code, <c>set_mel_padding</c>), the GPT reading
    /// <c>[c, BT text ET, BM codes EM]</c> with c from the prompt's log mel, and
    /// <see cref="IndexTTSOptions.TextLossWeight"/> × the text cross-entropy (targets <c>text, ET, ET</c>) plus
    /// <see cref="IndexTTSOptions.MelLossWeight"/> × the mel cross-entropy (targets <c>codes, EM, EM</c>).
    /// </remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        var random = _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed);
        var (prompt, text, codes, samples) = TrainingInputs(sample);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => Objective(prompt, text, codes, samples, training: true, random),
            Optimizer());
    }

    /// <inheritdoc />
    /// <remarks>The objective without dropout, so repeated evaluations compare.</remarks>
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
            var (prompt, text, codes, samples) = TrainingInputs(sample);
            return Objective(prompt, text, codes, samples, training: false, null)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // A recording's frozen features (log mel, DVAE codes), computed once per recording.
    private sealed class Recording
    {
        public Tensor<T> Mel = null!;
        public int[] Codes = null!;
    }

    private readonly Dictionary<Tensor<T>, Recording> _recordings = new(TensorReferenceComparer<Tensor<T>>.Instance);

    private Recording RecordingOf(Tensor<T> audio)
    {
        if (!_recordings.TryGetValue(audio, out var r))
        {
            using var _ = new NoGradScope<T>();
            var mel = MelSpectrogram(audio);
            if (_recordings.Count > 64) _recordings.Clear();
            r = _recordings[audio] = new Recording { Mel = mel, Codes = Dvae.Codes(mel) };
        }
        return r;
    }

    private (Tensor<T> Prompt, int[] Text, int[] Codes, int Samples) TrainingInputs(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample));
        var text = TokenIds(sample.Tokens, nameof(sample));
        if (text.Length == 0) throw new ArgumentException("The sample has no text tokens.", nameof(sample));
        var target = RecordingOf(audio);
        var prompt = sample.SpeakerReference is { } reference ? RecordingOf(reference).Mel : target.Mel;
        if (target.Codes.Length + 3 > TextToMel.MelPositions.VocabularySize)
            throw new ArgumentException("The recording has more mel codes than the model has positions.", nameof(sample));
        return (prompt, text, target.Codes, audio.Length);
    }

    private Tensor<T> Objective(Tensor<T> prompt, int[] text, int[] codes, int samples, bool training, Random? random)
    {
        var (textLoss, melLoss) = Losses(prompt, text, codes, samples, training, random);
        var o = _options;
        return Engine.TensorAdd(Engine.TensorMultiplyScalar(textLoss, Ops.FromDouble(o.TextLossWeight)),
            Engine.TensorMultiplyScalar(melLoss, Ops.FromDouble(o.MelLossWeight)));
    }

    /// <summary>
    /// UnifiedVoice's <c>forward</c> losses: the text and the mel cross-entropy (each a mean over its positions) of the
    /// codes of a recording of <paramref name="samples"/> samples, given the prompt's log mel <c>[frames, mel]</c>.
    /// </summary>
    internal (Tensor<T> Text, Tensor<T> Mel) Losses(Tensor<T> prompt, IReadOnlyList<int> text, IReadOnlyList<int> codes, int samples,
        bool training = false, Random? random = null)
    {
        var gpt = TextToMel;
        var c = gpt.Configuration;
        var (textLogits, melLogits) = Logits(prompt, text, codes, samples, training, random);
        var padded = PaddedCodes(codes, samples);
        var textTargets = text.Concat(new[] { c.StopText, c.StopText }).ToArray();
        var melTargets = padded.Concat(new[] { c.StopMel, c.StopMel }).ToArray();
        return (CrossEntropy(textLogits, textTargets), CrossEntropy(melLogits, melTargets));
    }

    /// <summary>The text logits <c>[L + 2, text tokens]</c> and mel logits <c>[n + 2, MelCodes]</c> of the teacher-forced
    /// pass.</summary>
    internal (Tensor<T> Text, Tensor<T> Mel) Logits(Tensor<T> prompt, IReadOnlyList<int> text, IReadOnlyList<int> codes, int samples,
        bool training = false, Random? random = null)
    {
        var gpt = TextToMel;
        var conditions = gpt.SpeakerConditioning(prompt, training, random);
        var (textStates, melStates) = gpt.Forward(conditions, text, PaddedCodes(codes, samples), training, random);
        return (gpt.TextHead.Forward(textStates), gpt.MelHead.Forward(melStates));
    }

    /// <summary><c>set_mel_padding</c>: codes from index ⌈samples / SamplesPerCode⌉ + 1 on become the stop code.</summary>
    private int[] PaddedCodes(IReadOnlyList<int> codes, int samples)
    {
        int limit = (int)Math.Ceiling(samples / (double)_options.SamplesPerCode) + 1;
        int stop = TextToMel.Configuration.StopMel;
        return codes.Select((code, i) => i >= limit ? stop : code).ToArray();
    }

    // The mean cross-entropy of logits [n, V] against targets [n].
    private Tensor<T> CrossEntropy(Tensor<T> logits, IReadOnlyList<int> targets)
    {
        int n = logits.Shape[0], v = logits.Shape[1];
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(logits, axis: 1), new[] { n * v, 1 });
        var picks = new Tensor<int>(new[] { n }, new Vector<int>(Enumerable.Range(0, n).Select(i => i * v + targets[i]).ToArray()));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorIndexSelect(logProbabilities, picks, 0), null), Ops.FromDouble(-1.0 / n));
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        if (_optimizer is not null) return _optimizer;
        var o = _options;
        // XTTS's optimizer_wd_only_on_weights: biases, normalization parameters and embedding tables are not decayed;
        // everything else (weights, the perceiver's latents and gamma, the conformer's position biases) is.
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
        return _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
    }

    /// <inheritdoc />
    /// <remarks>Training updates the GPT (its conditioning encoder included); the DVAE, the speaker encoder and BigVGAN are
    /// frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_gptLayers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
