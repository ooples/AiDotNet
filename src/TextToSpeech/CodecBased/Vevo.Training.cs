using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class Vevo<T>
{
    private readonly Dictionary<VevoStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private Random? _trainingRandom;

    /// <inheritdoc />
    /// <remarks>Vevo trains on recordings at 24 kHz (<see cref="TtsTrainingSample{T}.Audio"/>) with their phone ids
    /// (<see cref="TtsTrainingSample{T}.Tokens"/>, from <see cref="Phonemize"/>); the stage
    /// (<see cref="VevoOptions.CurrentStage"/>) picks the network.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    /// <remarks>The current stage's AdamW.</remarks>
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
    /// One step of the current stage's objective, as Amphion's trainers compute it: the autoregressive transformers'
    /// cross-entropy on the recording's content-style tokens after its phones (Vevo-TTS) or, half the time, its content
    /// tokens (Vevo-Voice); the flow's mean absolute error on the recording's mel with a random prompt (dropped, with the
    /// condition, a fifth of the time).
    /// </remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        var random = _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed);
        var objective = Objective(sample, random, training: true);
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
            return Objective(sample, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed), training: false)()[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // A recording's frozen features, computed once per recording.
    private sealed class Recording
    {
        public Tensor<T> Mel = null!;
        public int[] ContentStyle = null!;
        public int[]? Content;
    }

    private readonly Dictionary<Tensor<T>, Recording> _recordings = new(TensorReferenceComparer<Tensor<T>>.Instance);

    private Recording RecordingOf(Tensor<T> audio, bool content)
    {
        if (!_recordings.TryGetValue(audio, out var r))
        {
            using var _ = new NoGradScope<T>();
            var audio16k = AtContentRate(audio);
            if (_recordings.Count > 64) _recordings.Clear();
            r = _recordings[audio] = new Recording { Mel = Mel(audio), ContentStyle = ContentStyleCodes(audio16k) };
            if (content) r.Content = ContentCodes(audio16k);
        }
        if (content && r.Content is null)
            using (new NoGradScope<T>()) r.Content = ContentCodes(AtContentRate(audio));
        return r;
    }

    private Func<Tensor<T>> Objective(TtsTrainingSample<T> sample, Random random, bool training)
    {
        Validation.Guard.NotNull(sample);
        var o = _options;
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample));
        switch (o.CurrentStage)
        {
            case VevoStage.TextToContentStyle:
            {
                var phones = PhoneIds(sample.Tokens, nameof(sample));
                var target = RecordingOf(audio, content: false);
                return () => TextToContentStyle.Loss(phones, target.ContentStyle, training);
            }
            case VevoStage.ContentToContentStyle:
            {
                // train_both_vc_and_tts: half the samples read the phones, half the content tokens shifted past them.
                bool tts = random.NextDouble() < 0.5;
                var target = RecordingOf(audio, content: !tts);
                var input = tts ? PhoneIds(sample.Tokens, nameof(sample)) : target.Content!.Select(c => c + o.PhoneVocabulary).ToArray();
                return () => ContentToContentStyle.Loss(input, target.ContentStyle, training);
            }
            default:
            {
                var target = RecordingOf(audio, content: false);
                int frames = Math.Min(target.Mel.Shape[0], target.ContentStyle.Length);
                if (frames < 2) throw new ArgumentException("The recording is too short for the flow's objective.", nameof(sample));
                var mel = Engine.TensorNarrow(target.Mel, 0, 0, frames);
                var codes = target.ContentStyle.Take(frames).ToArray();
                // forward_diffusion's draws: t, whether the prompt (and the condition) is dropped, the prompt's length, the noise.
                double t = random.NextDouble();
                bool drop = random.NextDouble() <= Flow.Configuration.PromptDropProbability;
                int low = Math.Min(frames / 4, 5), high = (int)(frames * 0.4);
                int prompt = high > low ? low + random.Next(high - low) : low;
                var noise = new Tensor<T>(new[] { frames, o.MelChannels });
                for (int i = 0; i < noise.Length; i++) noise[i] = Ops.FromDouble(Latest.IndexTtsOps.Gaussian(random));
                return () => Flow.Loss(mel, codes, t, drop, prompt, noise, training);
            }
        }
    }

    /// <summary>The layers the current stage trains.</summary>
    private IReadOnlyList<LayerBase<T>> StageLayers() => _options.CurrentStage switch
    {
        VevoStage.TextToContentStyle => _textArLayers,
        VevoStage.ContentToContentStyle => _voiceAr is not null ? _voiceArLayers
            : throw new InvalidOperationException("Voice conversion is off (VevoOptions.VoiceConversion)."),
        _ => _flowLayers,
    };

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        var stage = _options.CurrentStage;
        if (_optimizers.TryGetValue(stage, out var cached)) return cached;
        var o = _options;
        bool flow = stage == VevoStage.Acoustic;
        double rate = flow ? o.FlowLearningRate : o.LearningRate;
        int warmup = Math.Max(1, o.WarmupSteps);
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = rate, Beta1 = o.Beta1, Beta2 = o.Beta2, Epsilon = o.Epsilon, WeightDecay = o.WeightDecay,
            EnableGradientClipping = true, MaxGradientNorm = flow ? 0.2 : 1.0,
            // transformers' get_inverse_sqrt_schedule: linear warm-up, then √(warmup / step).
            LearningRateScheduler = new LambdaLRScheduler(rate, step => step < warmup ? step / (double)warmup : 1.0 / Math.Sqrt(step / (double)warmup)),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            MaskParameters = Training.TapeTrainingStep<T>.CollectParameters(StageLayers().Cast<ILayer<T>>().ToList(), -1).ToList(),
        };
        var optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
        _optimizers[stage] = optimizer;
        return optimizer;
    }

    /// <inheritdoc />
    /// <remarks>Training updates the current stage's network; HuBERT, both tokenizers and Vocos are frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(StageLayers().Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
