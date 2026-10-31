using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

public partial class LlamaOmni<T>
{
    private readonly Dictionary<LlamaOmniTrainingStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private Random? _trainingRandom;

    /// <inheritdoc />
    /// <remarks>LLaMA-Omni's TTS step trains on a 16 kHz recording (<see cref="TtsTrainingSample{T}.Audio"/>), whose
    /// deduplicated mHuBERT units are the speech decoder's target, and its text's token ids
    /// (<see cref="TtsTrainingSample{T}.Tokens"/>, from <see cref="TokenizeText"/>).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? StageOptimizer() : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <summary>A training example: the prompt (with its speech embeddings, when it has speech), the response's ids ending
    /// with <c>&lt;|eot_id|&gt;</c> (the release's targets, §2.3), and the units of the response's speech.</summary>
    internal sealed record Example((int[] Before, int[] After, bool HasSpeech) Prompt, Tensor<T>? Speech, int[] Response, int[]? Units);

    /// <summary>
    /// The objective of the configured stage (§2.5): the response's next-token cross-entropy (eq. 3), the CTC loss of its
    /// units from the response's hidden states (eq. 5; the language model frozen), or the release's joint sum. The decoder
    /// reads, per <see cref="LlamaOmniOptions.Alignment"/>, the state that predicts each response token (the paper's
    /// c_i) or the one at its position (the release's training code).
    /// </summary>
    internal Tensor<T> Objective(Example example)
    {
        var o = _options;
        int n = example.Response.Length, d = o.HiddenDim;
        var inputs = Inputs(example.Prompt, example.Speech, example.Response);
        int length = inputs.Shape[1];
        bool text = o.Stage != LlamaOmniTrainingStage.SpeechDecoder, speech = o.Stage != LlamaOmniTrainingStage.TextResponse;
        if (speech && (example.Units is null || example.Units.Length == 0)) throw new ArgumentException("The speech decoder trains on the response's units.");
        Tensor<T> hidden;
        Tensor<T>? logits = null;
        if (text)
        {
            (hidden, logits) = LanguageModel.Forward(inputs);
        }
        else
        {
            using (new NoGradScope<T>()) hidden = LanguageModel.Forward(inputs).Hidden;
        }
        Tensor<T>? loss = null;
        if (text)
        {
            // Response token j sits at position length − n + j and is predicted from the position before it.
            int vocabulary = o.VocabSize;
            var rows = Engine.TensorSlice(Engine.Reshape(logits!, new[] { length, vocabulary }), new[] { length - n - 1, 0 }, new[] { n, vocabulary });
            var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(rows, axis: 1), new[] { n * vocabulary, 1 });
            var picks = new Tensor<int>(new[] { n }, new Vector<int>(example.Response.Select((id, j) => j * vocabulary + id).ToArray()));
            loss = Engine.TensorMultiplyScalar(Engine.TensorNegate(Engine.ReduceSum(Engine.TensorIndexSelect(logProbabilities, picks, 0), null, false)),
                Ops.FromDouble(1.0 / n));
        }
        if (speech)
        {
            int start = o.Alignment == LlamaOmniAlignment.Paper ? length - n - 1 : length - n;
            var states = Engine.TensorSlice(Engine.Reshape(hidden, new[] { length, d }), new[] { start, 0 }, new[] { n, d });
            var ctc = SpeechDecoder.Loss(new[] { (states, example.Units!) });
            loss = loss is null ? ctc : Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(ctc, Ops.FromDouble(o.CtcLossWeight)));
        }
        return loss!;
    }

    private int[] UnitsOf(Tensor<T> recording)
    {
        var audio = new double[recording.Length];
        for (int i = 0; i < audio.Length; i++) audio[i] = Ops.ToDouble(recording.GetFlat(i));
        // §2.4: consecutive identical indices are merged.
        return UnitExtractor.Units(audio, deduplicate: true);
    }

    private Example TtsExample(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var text = TokenIds(sample.Tokens, nameof(sample));
        if (text.Length == 0) throw new ArgumentException("The sample has no text tokens.", nameof(sample));
        int[]? units = null;
        if (_options.Stage != LlamaOmniTrainingStage.TextResponse)
        {
            var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains its speech decoder on the recording; set Audio.", nameof(sample));
            units = UnitsOf(audio);
            if (units.Length == 0) throw new ArgumentException("The recording gives no units.", nameof(sample));
        }
        // As Synthesize reads it: the text as the user's turn, then as the response.
        return new Example(PromptIds(DecodeText(text)), null, text.Concat(new[] { EotId }).ToArray(), units);
    }

    /// <inheritdoc />
    /// <remarks>One update of the configured stage on the text read aloud (by default the speech decoder's CTC on the
    /// recording's units, §2.5).</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        var example = TtsExample(sample);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => Objective(example), StageOptimizer());
    }

    /// <summary>
    /// One update on an InstructS2S triplet (§3): a 16 kHz speech instruction, the text response and the response's
    /// 16 kHz speech (whose units the decoder stages read; not needed for the text stage). Returns the loss.
    /// </summary>
    public T TrainTurn(Tensor<T> instruction, string response, Tensor<T>? responseSpeech = null)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        if (instruction is null) throw new ArgumentNullException(nameof(instruction));
        if (string.IsNullOrWhiteSpace(response)) throw new ArgumentException("The response is empty.", nameof(response));
        int[]? units = null;
        if (_options.Stage != LlamaOmniTrainingStage.TextResponse)
        {
            units = UnitsOf(responseSpeech ?? throw new ArgumentException("The speech decoder trains on the response's speech.", nameof(responseSpeech)));
            if (units.Length == 0) throw new ArgumentException("The response's speech gives no units.", nameof(responseSpeech));
        }
        var prompt = PromptIds(SpeechPlaceholder + "\n" + _options.SpeechInstruction);
        Tensor<T> speech;
        using (new NoGradScope<T>()) speech = _options.Stage == LlamaOmniTrainingStage.SpeechDecoder ? SpeechEmbeddings(instruction) : default!;
        var ids = TokenizeText(response.Trim()).Concat(new[] { EotId }).ToArray();
        var anchor = IndexTtsOps.Ids<T>(ids);
        return TrainWithCustomObjective(anchor, anchor, (_, _) => Objective(new Example(prompt,
            _options.Stage == LlamaOmniTrainingStage.SpeechDecoder ? speech : SpeechEmbeddings(instruction), ids, units)), StageOptimizer());
    }

    /// <inheritdoc />
    /// <remarks>The configured stage's objective.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        var example = TtsExample(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Objective(example)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> StageOptimizer()
    {
        var o = _options;
        if (_optimizers.TryGetValue(o.Stage, out var existing)) return existing;
        double peak = o.Stage == LlamaOmniTrainingStage.TextResponse ? o.LearningRate : o.SpeechDecoderLearningRate;
        int total = Math.Max(1, o.TotalSteps), warmup = (int)Math.Ceiling(o.WarmupFraction * total);
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = peak, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.0,
            EnableGradientClipping = o.MaxGradientNorm > 0, MaxGradientNorm = o.MaxGradientNorm > 0 ? o.MaxGradientNorm : 1.0,
            // §4.1: cosine, the first 3 % of steps warming up (transformers' get_cosine_schedule_with_warmup).
            LearningRateScheduler = new LinearWarmupScheduler(peak, warmup, total, 0.0, LinearWarmupScheduler.DecayMode.Cosine, 0.0),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        };
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
        _optimizers[o.Stage] = o.Stage == LlamaOmniTrainingStage.SpeechDecoder ? PaperOptimizerFactory.VerifyHandBuilt(this, optimizer) : optimizer;
        return _optimizers[o.Stage];
    }

    /// <inheritdoc />
    /// <remarks>Stage 1 updates the adaptor and the language model, stage 2 the speech decoder, the joint option all
    /// three; the speech encoder, the vocoder and the unit extractor stay frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var layers = _options.Stage switch
        {
            LlamaOmniTrainingStage.TextResponse => _adaptorLayers.Concat(_lmLayers),
            LlamaOmniTrainingStage.SpeechDecoder => _decoderLayers,
            _ => _adaptorLayers.Concat(_lmLayers).Concat(_decoderLayers),
        };
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
