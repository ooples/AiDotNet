using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>A word of an aligned transcript: its text and its start and end in seconds.</summary>
public readonly record struct SpiritLmWord(string Text, double Start, double End);

public partial class SpiritLM<T>
{
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private Random? _trainingRandom;
    private string _trainedGroup = "language model";

    private Random TrainingRandom => _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue + 1);

    /// <inheritdoc />
    /// <remarks>SpiritLM's TTS step trains on a 16 kHz recording (<see cref="TtsTrainingSample{T}.Audio"/>) and its text's
    /// token ids (<see cref="TtsTrainingSample{T}.Tokens"/>).</remarks>
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

    /// <summary>A text-only training sequence (§3.1): <c>[Text]{text}</c>.</summary>
    public string TextSequence(string text) => SpiritLmTokens.TextPrefix + (text ?? throw new ArgumentNullException(nameof(text)));

    /// <summary>A speech-only training sequence (§3.1): <c>[Speech]{speech tokens}</c>.</summary>
    public string SpeechSequence(Tensor<T> recording) => SpiritLmTokens.SpeechPrefix + SpeechString(recording);

    /// <summary>
    /// The time-stamped speech tokens of a recording, as <c>units_to_string</c> orders them: each stream's tokens at
    /// <c>i / rate</c> seconds after its deduplication, merged by time.
    /// </summary>
    internal List<(string Token, double Time)> TimedSpeechTokens(Tensor<T> recording)
    {
        var (hubert, pitch, style) = SpeechUnits(recording);
        var rates = new SpiritLmTokens.Rates();
        var tokens = new List<(string Token, double Time, int Order)>();
        void Add(string prefix, int[] values, double rate, bool deduplicate)
        {
            for (int i = 0; i < values.Length; i++)
                if (!deduplicate || i == 0 || values[i] != values[i - 1])
                    tokens.Add(($"[{prefix}{values[i]}]", i / rate, tokens.Count));
        }
        if (style is not null) Add("St", style, rates.Style, rates.DeduplicateStyle);
        if (pitch is not null) Add("Pi", pitch, rates.Pitch, rates.DeduplicatePitch);
        Add("Hu", hubert, rates.Hubert, rates.DeduplicateHubert);
        return tokens.OrderBy(t => t.Time).ThenBy(t => t.Order).Select(t => (t.Token, t.Time)).ToList();
    }

    /// <summary>
    /// An interleaved training sequence (§3.1, App. A): the transcript cut at word boundaries into spans that alternate
    /// between modalities, starting with either, text spans of <see cref="SpiritLMOptions.MinTextSpanWords"/>–
    /// <see cref="SpiritLMOptions.MaxTextSpanWords"/> words and speech spans of
    /// <see cref="SpiritLMOptions.MinSpeechSpanWords"/>–<see cref="SpiritLMOptions.MaxSpeechSpanWords"/>; a speech span is
    /// the recording's speech tokens timed from its first word's start to its last word's end, each run opened by its marker:
    /// <c>[Text]the cat [Speech][Hu3][Hu7]…[Text]the mat</c>.
    /// </summary>
    public string InterleavedSequence(Tensor<T> recording, IReadOnlyList<SpiritLmWord> words, Random random)
    {
        if (words is null || words.Count == 0) throw new ArgumentException("The transcript has no words.", nameof(words));
        if (random is null) throw new ArgumentNullException(nameof(random));
        var timed = TimedSpeechTokens(recording);
        var o = _options;
        var sb = new System.Text.StringBuilder();
        bool speech = random.Next(2) == 1;
        for (int at = 0; at < words.Count; speech = !speech)
        {
            int span = speech ? random.Next(o.MinSpeechSpanWords, o.MaxSpeechSpanWords + 1) : random.Next(o.MinTextSpanWords, o.MaxTextSpanWords + 1);
            int end = Math.Min(words.Count, at + span);
            if (speech)
            {
                double from = words[at].Start, to = words[end - 1].End;
                sb.Append(SpiritLmTokens.SpeechPrefix);
                foreach (var (token, time) in timed)
                    if (time >= from && time < to) sb.Append(token);
            }
            else
            {
                sb.Append(SpiritLmTokens.TextPrefix);
                sb.Append(string.Join(" ", words.Skip(at).Take(end - at).Select(w => w.Text)));
                if (end < words.Count) sb.Append(' ');
            }
            at = end;
        }
        return sb.ToString();
    }

    /// <summary>The TTS step's sequence: the text, then its speech, <c>[Text]{text}[Speech]{speech tokens}</c> — the
    /// interleaving of an aligned pair with one switch.</summary>
    internal string TtsSequence(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        string text = Decode(TokenIds(sample.Tokens, nameof(sample)));
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample));
        string speech = SpeechString(audio);
        if (speech.Length == 0) throw new ArgumentException("The recording gives no speech tokens.", nameof(sample));
        return TextSequence(text) + SpiritLmTokens.SpeechPrefix + speech;
    }

    // ---------------------------------------------------------------- the language model

    /// <summary>The next-token cross-entropy of <paramref name="ids"/>, averaged over every position after the first.</summary>
    internal Tensor<T> Loss(int[] ids)
    {
        if (ids.Length < 2) throw new ArgumentException("A sequence needs two tokens to predict one.");
        if (ids.Length > _options.MaxTextLength) throw new ArgumentException($"The sequence's {ids.Length} tokens exceed {_options.MaxTextLength}.");
        int count = ids.Length - 1, vocabulary = _options.VocabSize;
        var (_, logits) = LanguageModel.Forward(Embed(ids.Take(count).ToArray()));
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(Engine.Reshape(logits!, new[] { count, vocabulary }), axis: 1), new[] { count * vocabulary, 1 });
        var picks = new Tensor<int>(new[] { count }, new Vector<int>(Enumerable.Range(0, count).Select(j => j * vocabulary + ids[j + 1]).ToArray()));
        return Engine.TensorMultiplyScalar(Engine.TensorNegate(Engine.ReduceSum(Engine.TensorIndexSelect(logProbabilities, picks, 0), null, false)),
            Ops.FromDouble(1.0 / count));
    }

    private T TrainIds(Tensor<T> anchor, int[] ids)
    {
        _trainedGroup = "language model";
        var lm = LanguageModel;
        return TrainWithCustomObjective(anchor, anchor, (_, _) =>
        {
            lm.Training = true;
            try
            {
                return Loss(ids);
            }
            finally
            {
                lm.Training = false;
            }
        }, LanguageModelOptimizer());
    }

    /// <summary>One continued pre-training step on a sequence (<see cref="TextSequence"/>, <see cref="SpeechSequence"/>,
    /// <see cref="InterleavedSequence"/>), tokenized with <c>&lt;s&gt;</c>.</summary>
    public T TrainSequence(string sequence)
    {
        ThrowIfDisposed();
        var ids = Encode(sequence);
        return TrainIds(IndexTtsOps.Ids<T>(ids), ids);
    }

    /// <summary>One step on text alone (§3.1's text-only data).</summary>
    public T TrainText(string text) => TrainSequence(TextSequence(text));

    /// <summary>One step on speech alone (§3.1's speech-only data).</summary>
    public T TrainSpeech(Tensor<T> recording) => TrainSequence(SpeechSequence(recording));

    /// <summary>One step on an aligned recording interleaved at word boundaries (§3.1's speech+text data).</summary>
    public T TrainInterleaved(Tensor<T> recording, IReadOnlyList<SpiritLmWord> words) =>
        TrainSequence(InterleavedSequence(recording, words, TrainingRandom));

    /// <inheritdoc />
    /// <remarks>One step on the text followed by its speech (<c>[Text]…[Speech]…</c>).</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        var ids = Encode(TtsSequence(sample));
        return TrainIds(sample.Tokens, ids);
    }

    /// <inheritdoc />
    /// <remarks>The cross-entropy of the TTS step's sequence.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        var ids = Encode(TtsSequence(sample));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Loss(ids)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> LanguageModelOptimizer()
    {
        const string key = "language model";
        if (_optimizers.TryGetValue(key, out var existing)) return existing;
        var o = _options;
        // App. A: LLaMA 2's final rate held constant; LLaMA 2's AdamW.
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate, Beta1 = 0.9, Beta2 = 0.95, Epsilon = 1e-5, WeightDecay = o.WeightDecay,
            EnableGradientClipping = o.MaxGradientNorm > 0, MaxGradientNorm = o.MaxGradientNorm > 0 ? o.MaxGradientNorm : 1.0,
        });
        return _optimizers[key] = PaperOptimizerFactory.VerifyHandBuilt(this, optimizer);
    }

    // ---------------------------------------------------------------- the vocoder

    /// <summary>
    /// One GAN step of the unit vocoder (speech-resynthesis' and Expresso's <c>train.py</c>) on 16 kHz speech of a speaker:
    /// its frame-level units (with the per-frame pitch and style tokens for Expressive, or the utterance
    /// <paramref name="style"/> for Base) and audio cut to <see cref="SpiritLMOptions.VocoderSegmentSize"/> samples at a
    /// random unit; the discriminators' least-squares step, then the vocoder's adversarial, feature-matching (×2), mel L1
    /// (×45) and duration losses. Needs <see cref="SpiritLMOptions.IncludeVocoderDiscriminators"/>. Returns the vocoder's
    /// loss.
    /// </summary>
    public T TrainVocoder(Tensor<T> recording, int speaker = 0, int? style = null)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var (codes, pitch, styles) = SpeechUnits(recording);
        return TrainVocoder(recording, codes, pitch, styles, speaker, style);
    }

    /// <summary>
    /// The vocoder step on a recording whose speech tokens are already extracted (<see cref="SpeechUnits"/>), as the
    /// release's training reads them from a manifest: frame-level HuBERT units and, for Expressive, the pitch and style
    /// tokens at their own rates.
    /// </summary>
    public T TrainVocoder(Tensor<T> recording, int[] codes, int[]? pitch, int[]? styles, int speaker = 0, int? style = null)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (codes is null || codes.Length == 0) throw new ArgumentException("There are no units.", nameof(codes));
        if (IsExpressive && (pitch is null || styles is null)) throw new ArgumentException("SpiritLM Expressive's vocoder trains on pitch and style tokens.");
        var vocoder = Vocoder;
        var discriminators = _discriminators ?? throw new InvalidOperationException("Training the vocoder needs SpiritLMOptions.IncludeVocoderDiscriminators.");
        var mel = _lossMel!;
        int[]? framePitch = pitch is null ? null : CodeHiFiGan<T>.Upsample(pitch, codes.Length, vocoder.Configuration.PitchRatio);
        int[]? frameStyle = styles is null ? null : CodeHiFiGan<T>.Upsample(styles, codes.Length, vocoder.Configuration.StyleRatio);
        int? utteranceStyle = IsExpressive || _options.VocoderStyles == 0 ? null : style ?? _options.VocoderStyle;
        int hop = vocoder.Configuration.HopSize, frames = Math.Min(codes.Length, Math.Max(1, _options.VocoderSegmentSize / hop));
        var random = TrainingRandom;
        int start = random.Next(codes.Length - frames + 1);
        int[] Cut(int[] values) => values.Skip(start).Take(frames).ToArray();
        var segment = Cut(codes);
        var segmentPitch = framePitch is null ? null : Cut(framePitch);
        var segmentStyle = frameStyle is null ? null : Cut(frameStyle);
        var real = new Tensor<T>(new[] { frames * hop });
        for (int i = 0; i < real.Length; i++) real[i] = start * hop + i < recording.Length ? recording.GetFlat(start * hop + i) : Ops.Zero;
        (Tensor<T> Waveform, Tensor<T>? DurationLoss) Generate() => vocoder.TrainingForward(segment, speaker, random, utteranceStyle, segmentPitch, segmentStyle);

        _trainedGroup = "discriminator";
        TrainWithCustomObjective(real, real, (_, _) =>
        {
            Tensor<T> generated;
            using (new NoGradScope<T>()) generated = Generate().Waveform;
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
        }, SpeechOptimizer("discriminator"));

        _trainedGroup = "vocoder";
        return TrainWithCustomObjective(real, real, (_, _) =>
        {
            var (generated, durationLoss) = Generate();
            var onFake = discriminators.Forward(generated);
            List<(Tensor<T> Score, List<Tensor<T>> Features)> onReal;
            using (new NoGradScope<T>()) onReal = discriminators.Forward(real);
            Tensor<T>? total = null;
            void Add(Tensor<T> term) => total = total is null ? term : Engine.TensorAdd(total, term);
            for (int k = 0; k < onFake.Count; k++)
            {
                var d = Engine.TensorAddScalar(Engine.TensorNegate(onFake[k].Score), Ops.One);
                Add(Mean(Engine.TensorMultiply(d, d)));
                for (int l = 0; l < onFake[k].Features.Count; l++)
                    Add(Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(onReal[k].Features[l], onFake[k].Features[l]))), Ops.FromDouble(2.0)));
            }
            Tensor<T> realMel;
            using (new NoGradScope<T>()) realMel = mel.Forward(real);
            Add(Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(mel.Forward(generated), realMel))), Ops.FromDouble(45.0)));
            if (durationLoss is not null) Add(durationLoss);
            return total!;
        }, SpeechOptimizer("vocoder"));
    }

    // ---------------------------------------------------------------- the pitch VQ-VAE

    /// <summary>
    /// One step of SpiritLM Expressive's F0 VQ-VAE (§3.2) on 16 kHz speech: its F0 by pYAAPT (the paper's extractor for this
    /// model; <paramref name="extractor"/> to change it), interpolated and normalized by the voiced mean, with the voicing,
    /// cut to a multiple of 16 frames; the weighted L1 reconstruction and the commitment loss, the codebook moved by its
    /// moving averages. Returns the loss.
    /// </summary>
    public T TrainPitchQuantizer(Tensor<T> recording, SpiritLmPitchExtractor extractor = SpiritLmPitchExtractor.Yaapt)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var quantizer = PitchQuantizer;
        var (f0, voiced) = F0Of(Samples(recording), extractor);
        int multiple = 1 << quantizer.Configuration.Downsamplings, frames = f0.Length / multiple * multiple;
        if (frames == 0) throw new ArgumentException("The recording is too short for a pitch token.", nameof(recording));
        var normalized = NormalizeF0(f0, voiced).Take(frames).ToArray();
        var x = quantizer.Input(normalized, voiced.Take(frames).ToArray());
        var random = TrainingRandom;
        _trainedGroup = "pitch";
        return TrainWithCustomObjective(x, x, (_, _) => quantizer.TrainingLoss(x, random), SpeechOptimizer("pitch"));
    }

    private Tensor<T> Mean(Tensor<T> x) => Engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), keepDims: false);

    // The vocoder's, its discriminators' and the pitch VQ-VAE's AdamW (0.8, 0.99) at 2e-4, decayed by 0.999 per epoch.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> SpeechOptimizer(string group)
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
    /// <remarks>A language-model step updates LLaMA; a vocoder step the vocoder or, on their turn, the discriminators; a
    /// pitch step the VQ-VAE's convolutions; the HuBERT, FCPE and style tokenizers stay frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        IEnumerable<LayerBase<T>> layers = SpeechPropTrainedLayers(_trainedGroup) ?? _trainedGroup switch
        {
            "vocoder" => _vocoderLayers,
            "discriminator" => _discriminatorLayers,
            "pitch" => _pitchLayers,
            _ => _lmLayers,
        };
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
