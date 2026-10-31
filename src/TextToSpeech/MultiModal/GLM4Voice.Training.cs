using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>One conversational turn for GLM-4-Voice's supervised fine-tuning (§4.2): the user's speech or text, and the
/// reply's text and speech tokens.</summary>
public sealed class GlmVoiceDialogueTurn
{
    /// <summary>The user's speech tokens (Q_s); when null, <see cref="UserText"/> is the input.</summary>
    public int[]? UserSpeechTokens { get; init; }

    /// <summary>The user's text (Q_t), used when there are no speech tokens.</summary>
    public string? UserText { get; init; }

    /// <summary>The reply's text (A_t).</summary>
    public string ResponseText { get; init; } = string.Empty;

    /// <summary>The reply's speech tokens (A_s).</summary>
    public int[] ResponseSpeechTokens { get; init; } = Array.Empty<int>();
}

public partial class GLM4Voice<T>
{
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _lmOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _tokenizerOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _decoderOptimizer;
    private Random? _trainingRandom;
    private AiDotNet.TextToSpeech.Vocoders.DifferentiableMel<T>? _mel;
    private List<LayerBase<T>>? _trainedLayers;

    private Random TrainingRandom => _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue + 1);

    /// <inheritdoc />
    /// <remarks>GLM-4-Voice's TTS step trains on a recording at the model's rate
    /// (<see cref="TtsTrainingSample{T}.Audio"/>), whose speech tokens are the target, and its text's token ids
    /// (<see cref="TtsTrainingSample{T}.Tokens"/>, from <see cref="TokenizeText"/>).</remarks>
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

    // ---------------------------------------------------------------- sequences

    /// <summary>
    /// The TTS training sequence of a text and its speech (<see cref="GLM4VoiceOptions.TtsFormat"/>), with its targets:
    /// the speech tokens and what ends them. Forced-text thoughts are the chat template with the text as the user's turn
    /// and the reply interleaving it with the speech (the speech half of §4.2.2's split); the interleaved base format is
    /// <c>{text}&lt;|begin_of_audio|&gt;{speech}&lt;|end_of_audio|&gt;</c>, a supervised TTS pair of pre-training (Zeng
    /// et al. 2024a §2.3.1: "only trained to predict tokens in the target parts").
    /// </summary>
    internal (int[] Ids, bool[] Targets) TtsSequence(int[] text, int[] codes)
    {
        var speech = codes.Select(c => AudioOffset + c).ToArray();
        if (_options.TtsFormat == GlmVoiceTtsFormat.InterleavedBase)
        {
            var ids = Tokenizer.Prefix.Concat(text).Concat(new[] { Special("<|begin_of_audio|>") }).ToList();
            int start = ids.Count;
            ids.AddRange(speech);
            ids.Add(Special("<|end_of_audio|>"));
            return (ids.ToArray(), Enumerable.Range(0, ids.Count).Select(i => i >= start).ToArray());
        }
        var turn = new GlmVoiceDialogueTurn { UserText = string.Empty, ResponseSpeechTokens = codes };
        var (sequence, _, speechTargets) = DialogueSequence(turn, text, TokenizeText("\n").Concat(text).ToArray());
        return (sequence, speechTargets);
    }

    /// <summary>
    /// A turn's sequence and the two target masks of §4.2.2's split: the text reply's tokens, and the speech reply's
    /// tokens with what ends them. Streaming thoughts interleave the reply 13 : 26 and end it with
    /// <c>&lt;|user|&gt;</c>; text first writes <c>transcript\n{text}&lt;|assistant|&gt;\n&lt;|begin_of_audio|&gt;{speech}&lt;|end_of_audio|&gt;</c>.
    /// </summary>
    internal (int[] Ids, bool[] TextTargets, bool[] SpeechTargets) DialogueSequence(GlmVoiceDialogueTurn turn, int[]? responseText = null,
        int[]? userIds = null)
    {
        if (turn is null) throw new ArgumentNullException(nameof(turn));
        if (turn.UserSpeechTokens is null && turn.UserText is null) throw new ArgumentException("The turn needs the user's speech tokens or text.", nameof(turn));
        if (turn.ResponseSpeechTokens.Any(c => c < 0 || c >= _options.SpeechCodebookSize))
            throw new ArgumentOutOfRangeException(nameof(turn), $"Speech tokens lie in [0, {_options.SpeechCodebookSize}).");
        var prompt = userIds is not null ? PromptAround(userIds, speechInput: false) : ChatPrompt(turn.UserText, turn.UserSpeechTokens);
        var text = responseText ?? TokenizeText(turn.ResponseText);
        var speech = turn.ResponseSpeechTokens.Select(c => AudioOffset + c).ToArray();
        var ids = new List<int>(prompt);
        var textTargets = new List<bool>(prompt.Select(_ => false));
        var speechTargets = new List<bool>(prompt.Select(_ => false));
        void Add(IEnumerable<int> part, bool isText, bool isSpeech)
        {
            foreach (int id in part)
            {
                ids.Add(id);
                textTargets.Add(isText);
                speechTargets.Add(isSpeech);
            }
        }
        if (_options.Generation == GlmVoiceGeneration.StreamingThoughts)
        {
            foreach (int id in Interleave(text, speech)) Add(new[] { id }, !IsSpeech(id), IsSpeech(id));
            Add(new[] { Special("<|user|>") }, false, true);
        }
        else
        {
            Add(text, true, false);
            Add(new[] { Special("<|assistant|>") }, true, false);
            Add(TokenizeText("\n").Concat(new[] { Special("<|begin_of_audio|>") }), false, false);
            Add(speech.Concat(new[] { Special("<|end_of_audio|>") }), false, true);
        }
        if (ids.Count > _options.MaxTextLength) throw new ArgumentException($"The turn's {ids.Count} tokens exceed {_options.MaxTextLength}.", nameof(turn));
        return (ids.ToArray(), textTargets.ToArray(), speechTargets.ToArray());
    }

    // ---------------------------------------------------------------- the language model's objective

    /// <summary>The next-token cross-entropy of <paramref name="ids"/> averaged over the positions
    /// <paramref name="targets"/> marks (each predicted from the ones before it).</summary>
    internal Tensor<T> LanguageModelLoss(int[] ids, bool[] targets, bool training, Random? random)
    {
        if (ids.Length != targets.Length) throw new ArgumentException("Each id needs a target flag.", nameof(targets));
        if (ids.Any(id => id < 0 || id >= _options.VocabSize)) throw new ArgumentOutOfRangeException(nameof(ids), $"Token ids lie in [0, {_options.VocabSize}).");
        var positions = Enumerable.Range(1, ids.Length - 1).Where(i => targets[i]).ToArray();
        if (positions.Length == 0) throw new ArgumentException("The sequence has no target tokens.", nameof(targets));
        var lm = LanguageModel;
        var hidden = lm.Forward(lm.Embed(ids.Take(ids.Length - 1).ToArray()), null, training, random);
        var rows = new Tensor<int>(new[] { positions.Length }, new Vector<int>(positions.Select(i => i - 1).ToArray()));
        var logits = lm.Logits(Engine.TensorIndexSelect(hidden, rows, 0));                                      // [n, V]
        int vocabulary = _options.VocabSize;
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(logits, axis: 1), new[] { positions.Length * vocabulary, 1 });
        var picks = new Tensor<int>(new[] { positions.Length }, new Vector<int>(positions.Select((p, n) => n * vocabulary + ids[p]).ToArray()));
        var nll = Engine.TensorNegate(Engine.ReduceSum(Engine.TensorIndexSelect(logProbabilities, picks, 0), null, false));
        return Engine.TensorMultiplyScalar(nll, Ops.FromDouble(1.0 / positions.Length));
    }

    private int[] TextOf(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var text = TokenIds(sample.Tokens, nameof(sample));
        if (text.Length == 0) throw new ArgumentException("The sample has no text tokens.", nameof(sample));
        if (text.Any(IsSpeech)) throw new ArgumentException("The sample's text tokens include speech tokens.", nameof(sample));
        return text;
    }

    private (int[] Ids, bool[] Targets) TtsExample(TtsTrainingSample<T> sample)
    {
        var text = TextOf(sample);
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample));
        int[] codes;
        using (new NoGradScope<T>()) codes = SpeechTokensOf(To16k(audio));
        if (codes.Length == 0) throw new ArgumentException("The recording gives no speech tokens.", nameof(sample));
        return TtsSequence(text, codes);
    }

    /// <inheritdoc />
    /// <remarks>One language-model update of the TTS sequence (<see cref="GLM4VoiceOptions.TtsFormat"/>), its loss on the
    /// speech tokens only. The speech tokenizer and decoder train through <see cref="TrainSpeechTokenizer"/> and
    /// <see cref="TrainSpeechDecoder"/>.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        var (ids, targets) = TtsExample(sample);
        return TrainSequence(sample.Tokens, ids, targets);
    }

    /// <summary>
    /// One language-model update of any sequence: <paramref name="targets"/> marks the tokens the loss predicts (all of
    /// them for pre-training's text, speech and interleaved data; the target part of a supervised pair, §4.1). Returns
    /// the loss.
    /// </summary>
    public T TrainLanguageModel(int[] ids, bool[]? targets = null)
    {
        ThrowIfDisposed();
        if (ids is null) throw new ArgumentNullException(nameof(ids));
        if (!HasPaperLayers) throw NotNative();
        if (ids.Length < 2) throw new ArgumentException("A sequence needs at least two tokens.", nameof(ids));
        return TrainSequence(IndexTtsOps.Ids<T>(ids), ids, targets ?? ids.Select(_ => true).ToArray());
    }

    /// <summary>
    /// The fine-tuning of one turn in epoch <paramref name="epoch"/> (§4.2.2): the turn split in two, the text reply's
    /// loss with the speech masked while <paramref name="epoch"/> &lt; <see cref="GLM4VoiceOptions.TextEpochs"/>, and the
    /// speech reply's loss with the text masked while it is below <see cref="GLM4VoiceOptions.SpeechEpochs"/>. Returns
    /// the mean of the losses it trained, or NaN when the epoch trains neither.
    /// </summary>
    public double TrainDialogue(GlmVoiceDialogueTurn turn, int epoch = 0)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        if (epoch < 0) throw new ArgumentOutOfRangeException(nameof(epoch));
        var (ids, textTargets, speechTargets) = DialogueSequence(turn);
        var anchor = IndexTtsOps.Ids<T>(ids);
        var losses = new List<double>();
        if (epoch < _options.TextEpochs && textTargets.Any(t => t)) losses.Add(Ops.ToDouble(TrainSequence(anchor, ids, textTargets)));
        if (epoch < _options.SpeechEpochs && speechTargets.Any(t => t)) losses.Add(Ops.ToDouble(TrainSequence(anchor, ids, speechTargets)));
        return losses.Count == 0 ? double.NaN : losses.Average();
    }

    private T TrainSequence(Tensor<T> anchor, int[] ids, bool[] targets)
    {
        var random = TrainingRandom;
        _trainedLayers = _lmLayers;
        return TrainWithCustomObjective(anchor, anchor, (_, _) => LanguageModelLoss(ids, targets, true, random), LanguageModelOptimizer());
    }

    /// <inheritdoc />
    /// <remarks>The TTS objective with dropout off.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        var (ids, targets) = TtsExample(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return LanguageModelLoss(ids, targets, false, null)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> LanguageModelOptimizer()
    {
        if (_lmOptimizer is not null) return _lmOptimizer;
        var o = _options;
        double peak = o.Stage == GlmVoiceTrainingStage.PreTraining ? o.PreTrainingLearningRate : o.LearningRate;
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = peak, Beta1 = 0.9, Beta2 = 0.95, WeightDecay = o.WeightDecay,
            EnableGradientClipping = o.MaxGradientNorm > 0, MaxGradientNorm = o.MaxGradientNorm > 0 ? o.MaxGradientNorm : 1.0,
            // §4.1.1 and §4.2.2: a linear decay to a tenth of the peak.
            LearningRateScheduler = new PolynomialLRScheduler(peak, Math.Max(1, o.DecaySteps), 1.0, peak / 10),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        };
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
        _lmOptimizer = o.Stage == GlmVoiceTrainingStage.FineTuning ? PaperOptimizerFactory.VerifyHandBuilt(this, optimizer) : optimizer;
        return _lmOptimizer;
    }

    // ---------------------------------------------------------------- the speech tokenizer

    /// <summary>
    /// One update of the speech tokenizer (§3.1; Zeng et al. 2024a App. B.1) on 16 kHz speech and its transcript's
    /// Whisper token ids (<paramref name="labels"/>, ending with Whisper's end of text): the ASR cross-entropy through
    /// the quantized bottleneck plus the scaled commitment loss, then the codebook's EMA update and, every
    /// <see cref="GLM4VoiceOptions.RestartInterval"/> updates, the restart of dead codes at random frames. Needs
    /// <see cref="GLM4VoiceOptions.IncludeSpeechTokenizerTraining"/>. Returns the loss.
    /// </summary>
    public T TrainSpeechTokenizer(Tensor<T> recording, IReadOnlyList<int> labels)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (labels is null || labels.Count == 0) throw new ArgumentException("The transcript needs Whisper token ids.", nameof(labels));
        if (!_options.IncludeSpeechTokenizerTraining)
            throw new InvalidOperationException("Training the speech tokenizer needs GLM4VoiceOptions.IncludeSpeechTokenizerTraining.");
        var tokenizer = SpeechTokenizer;
        if (labels.Any(id => id < 0 || id >= _options.AsrVocabulary)) throw new ArgumentOutOfRangeException(nameof(labels), $"Whisper ids lie in [0, {_options.AsrVocabulary}).");
        var samples = Samples(recording);
        if (samples.Length > 30 * SpeechSampleRate) throw new ArgumentException("The tokenizer trains on chunks of at most 30 seconds.", nameof(recording));
        var (features, valid) = tokenizer.Features(samples);
        var decoderInput = new[] { _options.AsrDecoderStartId }.Concat(labels.Take(labels.Count - 1)).ToArray();
        var random = TrainingRandom;
        GlmSpeechTokenizer<T>.EncoderOutput? last = null;
        _trainedLayers = _speechTokenizerLayers;
        var loss = TrainWithCustomObjective(recording, recording, (_, _) =>
        {
            var output = tokenizer.Encode(features, valid, training: true, random);
            last = output;
            return Engine.TensorAdd(tokenizer.AsrLoss(decoderInput, labels, output.Hidden, output.Valid, true, random), tokenizer.CommitmentLoss(output));
        }, TokenizerOptimizer());
        // The release updates the EMA codebook in the training forward; the codebook takes no gradient.
        if (last is not null)
        {
            using var _ = new NoGradScope<T>();
            tokenizer.UpdateCodebook(last, (rows, count) =>
            {
                if (count > rows)
                    throw new InvalidOperationException($"{count} dead codes restart at distinct frames, but the batch has {rows} (random.sample in the release).");
                return Enumerable.Range(0, rows).OrderBy(_ => random.Next()).Take(count).ToArray();
            });
        }
        return loss;
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> TokenizerOptimizer() => _tokenizerOptimizer ??=
        PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, "speech tokenizer")
        ?? throw new InvalidOperationException($"{GetType().Name} declares the speech tokenizer's optimizer.");

    // ---------------------------------------------------------------- the speech decoder

    /// <summary>The decoder's mel <c>[1, 80, frames]</c> of 22.05 kHz audio (CosyVoice's <c>mel_spectrogram</c>: FFT
    /// 1024, hop 256, 0 – 8 kHz, not centred).</summary>
    public Tensor<T> MelSpectrogram(Tensor<T> audio)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        var o = _options;
        _mel ??= new AiDotNet.TextToSpeech.Vocoders.DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.FftSize, o.MelChannels, 0, 8000);
        var mel = _mel.Forward(Engine.Reshape(audio, new[] { audio.Length }));                                // [frames, mels]
        return Engine.Reshape(Engine.TensorTranspose(mel), new[] { 1, mel.Shape[1], mel.Shape[0] });
    }

    /// <summary>
    /// One update of the speech decoder's flow (§3.2; Zeng et al. 2024a App. B.2) on speech tokens and their 22.05 kHz
    /// audio: with <see cref="GLM4VoiceOptions.StreamingTruncation"/>, the first n·0.8 s (n drawn uniformly, the whole
    /// utterance included); the flow-matching loss with a prompt prefix of up to 80 % half the time and the conditions
    /// dropped at <see cref="GLM4VoiceOptions.TrainingCfgRate"/>. The speaker embedding is zero, as the release removes it.
    /// Returns the loss.
    /// </summary>
    public T TrainSpeechDecoder(IReadOnlyList<int> speechTokens, Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (speechTokens is null || speechTokens.Count == 0) throw new ArgumentException("There are no speech tokens.", nameof(speechTokens));
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        if (speechTokens.Any(c => c < 0 || c >= _options.SpeechCodebookSize))
            throw new ArgumentOutOfRangeException(nameof(speechTokens), $"Speech tokens lie in [0, {_options.SpeechCodebookSize}).");
        var decoder = SpeechDecoder;
        Tensor<T> mel;
        using (new NoGradScope<T>()) mel = MelSpectrogram(audio);
        var random = TrainingRandom;
        int[] tokens = speechTokens.ToArray();
        if (_options.StreamingTruncation) (tokens, mel) = decoder.Truncate(tokens, mel, random);
        var zeros = new Tensor<T>(new[] { decoder.Flow.Configuration.SpeakerDim });
        _trainedLayers = _flowLayers;
        return TrainWithCustomObjective(audio, audio, (_, _) => decoder.Flow.TrainingLoss(tokens, mel, zeros, true, random), DecoderOptimizer());
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> DecoderOptimizer() => _decoderOptimizer ??=
        PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, "speech decoder", 2500)
        ?? throw new InvalidOperationException($"{GetType().Name} declares the speech decoder's optimizer.");

    /// <inheritdoc />
    /// <remarks>A step updates the network it trains: the language model, the speech tokenizer or the flow. HiFT stays
    /// as loaded.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters((_trainedLayers ?? _lmLayers).Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
