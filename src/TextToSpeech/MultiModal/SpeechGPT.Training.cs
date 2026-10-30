using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

public partial class SpeechGPT<T>
{
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private Random? _trainingRandom;
    private string _trainedGroup = "language model";

    private Random TrainingRandom => _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue + 1);

    /// <inheritdoc />
    /// <remarks>SpeechGPT's TTS step trains on a 16 kHz recording (<see cref="TtsTrainingSample{T}.Audio"/>), whose
    /// merged units the response holds, and its text's token ids (<see cref="TtsTrainingSample{T}.Tokens"/>).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? LanguageModelOptimizer() : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <summary>An example's ids and the first position its loss reads.</summary>
    internal (int[] Ids, int FirstTarget) TtsExample(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var core = Core;
        var text = core.Tokenizer.Decode(TokenIds(sample.Tokens, nameof(sample)));
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample));
        var units = core.UnitsOf(audio);
        if (units.Length == 0) throw new ArgumentException("The recording gives no units.", nameof(sample));
        // Stage 1 reads the speech alone (eq. 3); the others the TTS instruction (§3.1), the loss on the response (eq. 4).
        if (_options.Stage == SpeechGptTrainingStage.ModalityAdaptation)
            return (core.Example("", SpeechGptCore<T>.UnitString(units)).Ids, 1);
        var (ids, prefix) = core.Example(_options.MetaInstruction + $"[Human]: {TtsQuestion(text)} <eoh>. [SpeechGPT]: ", SpeechGptCore<T>.UnitString(units) + "<eos>");
        return (ids, prefix);
    }

    private T TrainIds(Tensor<T> anchor, int[] ids, int firstTarget)
    {
        _trainedGroup = "language model";
        var lm = Core.LanguageModel;
        return TrainWithCustomObjective(anchor, anchor, (_, _) =>
        {
            lm.Training = true;
            try
            {
                return Core.Loss(ids, firstTarget);
            }
            finally
            {
                lm.Training = false;
            }
        }, LanguageModelOptimizer());
    }

    /// <inheritdoc />
    /// <remarks>One update of the configured stage on the text read aloud: stage 1 on the recording's units alone, the
    /// others on the TTS instruction with the loss on its units.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        var (ids, first) = TtsExample(sample);
        return TrainIds(sample.Tokens, ids, first);
    }

    /// <summary>Stage 1 (eq. 3): next-token prediction on the merged units of unlabelled 16 kHz speech.</summary>
    public T TrainContinuation(Tensor<T> recording)
    {
        ThrowIfDisposed();
        var units = UnitsOf(recording);
        if (units.Length == 0) throw new ArgumentException("The recording gives no units.", nameof(recording));
        var ids = Core.Example("", SpeechGptCore<T>.UnitString(units)).Ids;
        return TrainIds(IndexTtsOps.Ids<T>(ids), ids, 1);
    }

    /// <summary>
    /// An instruction step (eq. 4): <c>[Human]: {question} &lt;eoh&gt;. [SpeechGPT]: {response}</c> after the meta
    /// instruction, the loss on the response and the end. Questions and responses hold units as
    /// <c>&lt;sosp&gt;&lt;u&gt;…&lt;eosp&gt;</c>.
    /// </summary>
    public T TrainInstruction(string question, string response)
    {
        ThrowIfDisposed();
        if (question is null) throw new ArgumentNullException(nameof(question));
        if (response is null) throw new ArgumentNullException(nameof(response));
        var (ids, prefix) = Core.Example(_options.MetaInstruction + $"[Human]: {question} <eoh>. [SpeechGPT]: ", response);
        return TrainIds(IndexTtsOps.Ids<T>(ids), ids, prefix);
    }

    /// <summary>A cross-modal ASR step (§3.1): 16 kHz speech and its transcript.</summary>
    public T TrainTranscription(Tensor<T> recording, string transcript) => TrainInstruction(AsrQuestion(UnitsOf(recording)), transcript + "<eos>");

    /// <summary>
    /// A chain-of-modality step (App. C) on a quadruplet: the speech instruction (or null for a text one), its text, the
    /// text response, and the speech response (or null for a text one).
    /// </summary>
    public T TrainChainOfModality(Tensor<T>? speechInstruction, string textInstruction, string textResponse, Tensor<T>? speechResponse)
    {
        if (textInstruction is null) throw new ArgumentNullException(nameof(textInstruction));
        if (textResponse is null) throw new ArgumentNullException(nameof(textResponse));
        bool speechIn = speechInstruction is not null, speechOut = speechResponse is not null;
        string question = ChainOfModalityQuestion(speechIn ? SpeechGptCore<T>.UnitString(UnitsOf(speechInstruction!)) : textInstruction, speechIn, speechOut);
        string answer = (speechIn ? $"[tq] {textInstruction}; " : "") + $"[ta] {textResponse}"
            + (speechOut ? $"; [ua] {SpeechGptCore<T>.UnitString(UnitsOf(speechResponse!))}" : "") + "<eoa>";
        return TrainInstruction(question, answer);
    }

    /// <inheritdoc />
    /// <remarks>The configured stage's objective on the TTS example.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        var (ids, first) = TtsExample(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Core.Loss(ids, first)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private int StageSteps => _options.TotalSteps > 0 ? _options.TotalSteps : _options.Stage switch
    {
        SpeechGptTrainingStage.ModalityAdaptation => 900,
        SpeechGptTrainingStage.CrossModalInstruction => 4000,
        _ => 4200,
    };

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> LanguageModelOptimizer()
    {
        string key = "language model " + _options.Stage;
        if (_optimizers.TryGetValue(key, out var existing)) return existing;
        var o = _options;
        int total = Math.Max(1, StageSteps), warmup = (int)Math.Ceiling(o.WarmupFraction * total);
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = o.WeightDecay,
            EnableGradientClipping = o.MaxGradientNorm > 0, MaxGradientNorm = o.MaxGradientNorm > 0 ? o.MaxGradientNorm : 1.0,
            LearningRateScheduler = new LinearWarmupScheduler(o.LearningRate, warmup, total, 0.0, LinearWarmupScheduler.DecayMode.Cosine, 0.0),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        return _optimizers[key] = PaperOptimizerFactory.VerifyHandBuilt(this, optimizer);
    }

    // ---------------------------------------------------------------- the vocoder

    /// <summary>
    /// One GAN step of the multi-speaker unit vocoder (§4.1; speech-resynthesis' <c>train.py</c>) on 16 kHz speech of a
    /// speaker: its frame-level units and audio cut to a segment of <see cref="SpeechGPTOptions.VocoderSegmentSize"/>
    /// samples at a random unit, the discriminators' least-squares step on the vocoder's output, then the vocoder's:
    /// adversarial, feature matching (×2), mel L1 (×45) and duration loss. Needs
    /// <see cref="SpeechGPTOptions.IncludeVocoderDiscriminators"/>. Returns the vocoder's loss.
    /// </summary>
    public T TrainVocoder(Tensor<T> recording, int speaker = 0)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var core = Core;
        var vocoder = core.Vocoder ?? throw NotNative();
        var discriminators = core.Discriminators ?? throw new InvalidOperationException("Training the vocoder needs SpeechGPTOptions.IncludeVocoderDiscriminators.");
        var mel = core.LossMel!;
        int? spkr = _options.Vocoder == SpeechGptVocoder.MultiSpeaker ? speaker : null;
        var codes = core.UnitsOf(recording, merged: false);
        int hop = vocoder.Configuration.HopSize, frames = Math.Min(codes.Length, Math.Max(1, _options.VocoderSegmentSize / hop));
        var random = TrainingRandom;
        int start = random.Next(codes.Length - frames + 1);
        var segment = codes.Skip(start).Take(frames).ToArray();
        var real = new Tensor<T>(new[] { frames * hop });
        for (int i = 0; i < real.Length; i++) real[i] = start * hop + i < recording.Length ? recording.GetFlat(start * hop + i) : Ops.Zero;

        // The discriminators learn from the vocoder's output computed without gradients.
        _trainedGroup = "discriminator";
        TrainWithCustomObjective(real, real, (_, _) =>
        {
            Tensor<T> generated;
            using (new NoGradScope<T>()) generated = vocoder.TrainingForward(segment, spkr, random).Waveform;
            var onReal = discriminators.Forward(real);
            var onFake = discriminators.Forward(generated);
            Tensor<T>? total = null;
            for (int k = 0; k < onReal.Count; k++)
            {
                var r = Engine.TensorAddScalar(Engine.TensorNegate(onReal[k].Score), Ops.One);
                var term = Engine.TensorAdd(Mean(Engine.TensorMultiply(r, r)), Mean(Engine.TensorMultiply(onFake[k].Score, onFake[k].Score)));
                total = total is null ? term : Engine.TensorAdd(total, term);
            }
            return total!;
        }, VocoderOptimizer("discriminator"));

        _trainedGroup = "vocoder";
        return TrainWithCustomObjective(real, real, (_, _) =>
        {
            var (generated, durationLoss) = vocoder.TrainingForward(segment, spkr, random);
            var onFake = discriminators.Forward(generated);
            List<(Tensor<T> Score, List<Tensor<T>> Features)> onReal;
            using (new NoGradScope<T>()) onReal = discriminators.Forward(real);
            Tensor<T>? total = null;
            void Add(Tensor<T> term) => total = total is null ? term : Engine.TensorAdd(total, term);
            for (int k = 0; k < onFake.Count; k++)
            {
                var d = Engine.TensorAddScalar(Engine.TensorNegate(onFake[k].Score), Ops.One);
                Add(Mean(Engine.TensorMultiply(d, d)));
                // speech-resynthesis' feature_loss: the summed mean L1 of the feature maps, times 2.
                for (int l = 0; l < onFake[k].Features.Count; l++)
                    Add(Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(onReal[k].Features[l], onFake[k].Features[l]))), Ops.FromDouble(2.0)));
            }
            Tensor<T> realMel;
            using (new NoGradScope<T>()) realMel = mel.Forward(real);
            Add(Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(mel.Forward(generated), realMel))), Ops.FromDouble(45.0)));
            if (durationLoss is not null) Add(durationLoss);
            return total!;
        }, VocoderOptimizer("vocoder"));
    }

    private Tensor<T> Mean(Tensor<T> x) => Engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), keepDims: false);

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> VocoderOptimizer(string group)
    {
        if (_optimizers.TryGetValue(group, out var existing)) return existing;
        var o = _options;
        int perEpoch = Math.Max(1, o.VocoderUpdatesPerEpoch);
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.VocoderLearningRate, Beta1 = 0.8, Beta2 = 0.99, WeightDecay = 0.01,
            LearningRateScheduler = new LambdaLRScheduler(o.VocoderLearningRate, step => Math.Pow(o.VocoderLearningRateDecay, step / perEpoch)),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        return _optimizers[group] = group == "vocoder" ? PaperOptimizerFactory.VerifyHandBuilt(this, optimizer, "vocoder") : optimizer;
    }

    /// <inheritdoc />
    /// <remarks>A language-model step updates LLaMA (stage 3: only its LoRA adapters); a vocoder step the vocoder or, on
    /// their turn, the discriminators; mHuBERT stays frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        IEnumerable<LayerBase<T>> layers = _trainedGroup switch
        {
            "vocoder" => _vocoderLayers,
            "discriminator" => _discriminatorLayers,
            _ when _options.Stage == SpeechGptTrainingStage.ChainOfModality => Core.LanguageModel.Layers
                .SelectMany(l => new[] { l.QueryLora, l.ValueLora }).Where(l => l is not null).SelectMany(l => new LayerBase<T>[] { l!.A, l.B }),
            _ => _lmLayers,
        };
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
