using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

public partial class MinMo<T>
{
    private readonly Dictionary<MinMoTrainingStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private Random? _trainingRandom;

    private Random TrainingRandom => _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue + 1);

    /// <inheritdoc />
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

    private static bool IsSpeechToText(MinMoTrainingStage stage) =>
        stage is MinMoTrainingStage.PreAlign or MinMoTrainingStage.FullAlign or MinMoTrainingStage.InstructionTuning;

    private static bool IsVoice(MinMoTrainingStage stage) =>
        stage is MinMoTrainingStage.TextToSpeechProjector or MinMoTrainingStage.TextToSpeech or MinMoTrainingStage.SpeechToSpeech;

    private void RequireStage(Func<MinMoTrainingStage, bool> fits, string task)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        if (!fits(_options.Stage)) throw new InvalidOperationException($"{task} does not train at stage {_options.Stage}; set MinMoOptions.Stage.");
    }

    private T Step(Func<Tensor<T>> objective)
    {
        var marker = IndexTtsOps.Ids<T>(new[] { 0 });
        return TrainWithCustomObjective(marker, marker, (_, _) => objective(), Optimizer());
    }

    /// <summary>
    /// One speech-to-text step (§3.4 stage 1): the ChatML prompt with the instruction and the recording, the answer's
    /// cross-entropy. Pre-alignment updates the input projector, full alignment the projector and the voice encoder,
    /// instruction tuning the LLM's LoRA. Returns the loss.
    /// </summary>
    public T TrainSpeechToText(Tensor<T> speech, string instruction, string answer)
    {
        RequireStage(IsSpeechToText, "Speech-to-text");
        if (speech is null) throw new ArgumentNullException(nameof(speech));
        if (instruction is null || answer is null) throw new ArgumentNullException(instruction is null ? nameof(instruction) : nameof(answer));
        var core = Core;
        return Step(() => core.SpeechToTextLoss(core.BuildPrompt(instruction, speech, _options.SampleRate), answer));
    }

    /// <summary>
    /// One text-to-speech step (§3.4 stage 2): the user asks "{instruction}: {text}", the assistant's answer
    /// (<paramref name="answer"/>, the text itself by default, or Table 3's tagged form) is spoken by the recording; the
    /// output projector, and at <see cref="MinMoTrainingStage.TextToSpeech"/> also the voice token LM, learn the speech
    /// tokens. Returns the loss.
    /// </summary>
    public T TrainTextToSpeech(string text, Tensor<T> recording, string? instruction = null, string? answer = null)
    {
        RequireStage(s => s is MinMoTrainingStage.TextToSpeechProjector or MinMoTrainingStage.TextToSpeech, "Text-to-speech");
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var core = Core;
        var random = TrainingRandom;
        return Step(() => core.VoiceLoss(core.BuildPrompt($"{instruction ?? _options.SynthesisInstruction}: {text}", null, _options.SampleRate),
            answer ?? text, recording, _options.SampleRate, random));
    }

    /// <summary>One speech-to-speech step (§3.4 stage 3): the user's recording as the prompt, the answer's text and its
    /// recording; the output projector and the voice token LM learn. Returns the loss.</summary>
    public T TrainSpeechToSpeech(Tensor<T> speech, string answer, Tensor<T> answerRecording, string? instruction = null)
    {
        RequireStage(s => s == MinMoTrainingStage.SpeechToSpeech, "Speech-to-speech");
        if (speech is null || answerRecording is null) throw new ArgumentNullException(speech is null ? nameof(speech) : nameof(answerRecording));
        if (answer is null) throw new ArgumentNullException(nameof(answer));
        var core = Core;
        var random = TrainingRandom;
        return Step(() => core.VoiceLoss(core.BuildPrompt(instruction ?? string.Empty, speech, _options.SampleRate), answer, answerRecording,
            _options.SampleRate, random));
    }

    /// <summary>
    /// One duplex step (§3.4 stage 4): the full duplex predictor learns an action per speech position of a conversation
    /// recording (−1 unlabelled), labelled by §3.3's rules — the assistant's turn starts where the user's ends
    /// (<see cref="MinMoDuplexAction.Respond"/>), the user takes the turn back (<see cref="MinMoDuplexAction.Stop"/>), and
    /// back-channels change nothing (<see cref="MinMoDuplexAction.Continue"/>). Returns the loss.
    /// </summary>
    public T TrainDuplex(Tensor<T> speech, IReadOnlyList<int> labels)
    {
        RequireStage(s => s == MinMoTrainingStage.Duplex, "Duplex");
        if (speech is null || labels is null) throw new ArgumentNullException(speech is null ? nameof(speech) : nameof(labels));
        var core = Core;
        return Step(() => core.DuplexLoss(core.BuildPrompt(string.Empty, speech, _options.SampleRate), labels));
    }

    /// <summary>The number of speech positions (duplex labels) a recording gives.</summary>
    public int SpeechPositions(Tensor<T> speech)
    {
        ThrowIfDisposed();
        using var _ = new NoGradScope<T>();
        return Core.SpeechEmbeddings(speech ?? throw new ArgumentNullException(nameof(speech)), _options.SampleRate).Shape[0];
    }

    // The sample's objective at the current stage: its tokens' text as the transcript (speech-to-text stages) or as the
    // text to speak (voice stages).
    private Func<Tensor<T>> SampleObjective(TtsTrainingSample<T> sample, Random random)
    {
        var core = Core;
        if (sample.Audio is null) throw new ArgumentException("MinMo trains on the recording; set the sample's Audio.", nameof(sample));
        string text = core.TextOf(IdsOf(sample.Tokens));
        var o = _options;
        if (IsSpeechToText(o.Stage)) return () => core.SpeechToTextLoss(core.BuildPrompt(o.TranscriptionInstruction, sample.Audio, o.SampleRate), text);
        if (IsVoice(o.Stage))
        {
            if (text.Trim().Length == 0) throw new ArgumentException("The sample's tokens have no text to speak.", nameof(sample));
            return () => core.VoiceLoss(core.BuildPrompt($"{o.SynthesisInstruction}: {text}", null, o.SampleRate), text, sample.Audio, o.SampleRate, random);
        }
        throw new ArgumentException("The duplex stage trains on per-position labels; use TrainDuplex.", nameof(sample));
    }

    /// <inheritdoc />
    /// <remarks>At a speech-to-text stage the recording is the speech and the tokens its transcript; at a voice stage the
    /// tokens are the text and the recording its speech.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        RequireStage(_ => true, "Training");
        var objective = SampleObjective(sample, TrainingRandom);
        return Step(objective);
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        RequireStage(_ => true, "Evaluation");
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            // The same draw each evaluation, so the bistream choice and sampling do not move the measurement.
            var objective = SampleObjective(sample, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue + 2));
            using var _ = new NoGradScope<T>();
            return objective().GetFlat(0);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        if (_optimizers.TryGetValue(_options.Stage, out var existing)) return existing;
        var o = _options;
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = o.WeightDecay,
            UseAdaptiveLearningRate = false, UseAdaptiveBetas = false,
            EnableGradientClipping = o.MaxGradientNorm > 0, MaxGradientNorm = o.MaxGradientNorm > 0 ? o.MaxGradientNorm : 1.0,
        });
        return _optimizers[o.Stage] = PaperOptimizerFactory.VerifyHandBuilt(this, optimizer);
    }

    // The modules §3.4 updates at each stage.
    private IEnumerable<LayerBase<T>> TrainedLayers() => _options.Stage switch
    {
        MinMoTrainingStage.PreAlign => _inputProjectorLayers,
        MinMoTrainingStage.FullAlign => _inputProjectorLayers.Concat(_encoderLayers),
        MinMoTrainingStage.InstructionTuning => Core.LoraLayers,
        MinMoTrainingStage.TextToSpeechProjector => _outputProjectorLayers,
        MinMoTrainingStage.TextToSpeech or MinMoTrainingStage.SpeechToSpeech => _outputProjectorLayers.Concat(Core.Voice.LanguageModelLayers),
        _ => _duplexLayers,
    };

    /// <inheritdoc />
    /// <remarks>The modules the current <see cref="MinMoOptions.Stage"/> updates (§3.4); everything else stays frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(TrainedLayers().Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
