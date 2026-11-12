using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Audio.Foundations;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>
/// SenseVoice-Small (An et al. 2024, FunAudioLLM §2.2): a non-autoregressive SAN-M encoder that recognizes speech and
/// classifies its language, emotion and audio events from four task queries prepended to the features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> SenseVoice-Small turns 16 kHz audio into text with one encoder pass, no decoder. Four
/// extra "question" vectors are put in front of the audio frames; the encoder's answers at those four positions are the
/// spoken language, the emotion, the audio event and whether the text is normalized, and the remaining positions spell
/// the transcript with CTC.</para>
/// <para>
/// The front end is the release's: 80-band Kaldi filterbanks of the waveform, seven frames stacked every six (low frame
/// rate) and mean-variance normalized, giving 560 features per 60 ms. Four query embeddings of the same width — the
/// language (auto 0, zh 3, en 4, yue 7, ja 11, ko 12, nospeech 13), the emotion (1), the event (2) and the text
/// normalization (with 14, without 15) — are prepended in that order. The encoder is
/// <see cref="SanmEncoder{T}"/>: 50 SAN-M blocks (self-attention plus an FSMN memory, width 512, 4 heads, feed-forward
/// 2048, kernel 11), a norm, 20 more blocks and a norm. A linear layer (<c>ctc_lo</c>) maps every position onto the
/// 25,055-token vocabulary: positions 4 onward are the CTC transcript and the first four the rich tokens. Training
/// minimizes the CTC loss of the transcript plus, when the targets carry the four rich tokens first (the release's
/// layout), their cross-entropy through the same layer.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 16000, outputSize: 25055);
/// var model = new SenseVoice&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs", "https://arxiv.org/abs/2407.04051", Year = 2024, Authors = "An et al.")]
public partial class SenseVoice<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    // The release's query ids: lid_dict, the emotion and event queries, and textnorm_dict.
    private static readonly Dictionary<string, int> LanguageQueries = new(StringComparer.OrdinalIgnoreCase)
    {
        ["auto"] = 0, ["zh"] = 3, ["en"] = 4, ["yue"] = 7, ["ja"] = 11, ["ko"] = 12, ["nospeech"] = 13,
    };
    private const int EmotionQuery = 1, EventQuery = 2, WithItnQuery = 14, WithoutItnQuery = 15, QueryCount = 16, RichTokens = 4;

    /// <inheritdoc />
    /// <remarks>The output vocabulary, blank (0) included.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly SenseVoiceOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private SanmEncoder<T>? _encoder;
    private EmbeddingLayer<T>? _queries;
    private DenseLayer<T>? _ctcOutput;
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => false;
    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    public SenseVoice(NeuralNetworkArchitecture<T> architecture, string modelPath, SenseVoiceOptions? options = null) : base(architecture)
    {
        _options = options ?? new SenseVoiceOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { "zh", "en", "yue", "ja", "ko" }; InitializeLayersCore();
    }

    public SenseVoice(NeuralNetworkArchitecture<T> architecture, SenseVoiceOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new SenseVoiceOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate, WeightDecay = _options.WeightDecay });
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { "zh", "en", "yue", "ja", "ko" }; InitializeLayersCore();
    }

    /// <summary>The encoder's CMVN shift and scale (FunASR's <c>am.mvn</c>, read by
    /// <see cref="SanmEncoder{T}.ReadNormalization"/>); identity until set.</summary>
    public (double[] Shift, double[] Scale) Normalization
    {
        get => (_encoder ?? throw new InvalidOperationException("No native network.")).Normalization;
        set => (_encoder ?? throw new InvalidOperationException("No native network.")).Normalization = value;
    }

    /// <summary>Transcribes audio: the CTC transcript of the frames after the four task queries, decoded greedily.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for SenseVoice.");
        var waveform = PreprocessAudio(audio);
        Tensor<T> logits;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            logits = _encoder is null ? Logits(waveform) : TranscriptLogits(waveform, LanguageQuery(language ?? _options.Language));
        var (tokens, confidence) = GreedyCtc(logits);
        return new TranscriptionResult<T>
        {
            Text = Text(tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <summary>The configured language: the vocabulary's language tokens belong to the tokenizer, so the rich output
    /// (<see cref="RichTokenIds"/>) is returned as token ids rather than named here.</summary>
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("SenseVoice does not support streaming.");

    /// <summary>The most likely vocabulary token at each of the four query positions (language, emotion, event, text
    /// normalization), the language query auto.</summary>
    public int[] RichTokenIds(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (_encoder is null) throw new NotSupportedException("The rich tokens are defined on the native network.");
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        var logits = AllLogits(PreprocessAudio(audio), 0);
        var ids = new int[RichTokens];
        for (int p = 0; p < RichTokens; p++)
        {
            int best = 0;
            for (int v = 1; v < _options.VocabSize; v++) if (NumOps.GreaterThan(logits[p, v], logits[p, best])) best = v;
            ids[p] = best;
        }
        return ids;
    }

    private static int LanguageQuery(string language) => LanguageQueries.TryGetValue(language, out int id) ? id : 0;

    // Each token is its vocabulary entry when a vocabulary is configured, otherwise its Unicode code point.
    private string Text(List<int> tokens)
    {
        var sb = new System.Text.StringBuilder();
        foreach (int t in tokens)
        {
            if (t < _options.Vocabulary.Length) { string s = _options.Vocabulary[t]; sb.Append(s.StartsWith("<", StringComparison.Ordinal) ? string.Empty : s.Replace("▁", " ")); }
            else if (t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t));
        }
        return sb.ToString().Trim();
    }

    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _encoder = null; _queries = null; _ctcOutput = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The paper network (see the class remarks). It used to be the generic Paraformer template.
        var o = _options;
        var configuration = new SanmEncoderConfiguration(o.NumMels, o.LfrM, o.LfrN, o.EncoderDim, o.NumAttentionHeads, o.FeedForwardDim,
            o.NumEncoderLayers, o.NumTpEncoderLayers, o.FsmnKernelSize);
        var layers = new List<LayerBase<T>>();
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(o.Seed ?? 42);
        _encoder = new SanmEncoder<T>(Engine, layers, configuration, random);
        // After the encoder's layers: the model reads waveforms, not the query ids the embedding reads.
        _queries = new EmbeddingLayer<T>(QueryCount, configuration.InputSize);
        layers.Add(_queries);
        _ctcOutput = new DenseLayer<T>(o.VocabSize, (IActivationFunction<T>)new IdentityActivation<T>());
        layers.Add(_ctcOutput);
        Layers.AddRange(layers);
    }

    // The recognizer reads the raw waveform, truncated to the longest window a transcription covers.
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_encoder is null) return base.PreprocessAudio(rawAudio);
        int length = Math.Min(rawAudio.Length, SampleRate * _options.MaxAudioLengthSeconds);
        var waveform = new Tensor<T>(new[] { length });
        rawAudio.Data.Span.Slice(0, length).CopyTo(waveform.Data.Span);
        return waveform;
    }

    private Tensor<T> Samples(Tensor<T> waveform, int offset, int count)
    {
        var samples = new Tensor<T>(new[] { count });
        waveform.Data.Span.Slice(offset, count).CopyTo(samples.Data.Span);
        return samples;
    }

    // ctc_lo over every position [4 + frames, vocabulary]: the queries [language, emotion, event, text normalization]
    // prepended to the stacked features, then the encoder.
    private Tensor<T> AllLogits(Tensor<T> waveform, int languageQuery)
    {
        var encoder = _encoder ?? throw new InvalidOperationException("No native network.");
        var queries = _queries ?? throw new InvalidOperationException("No native network.");
        var output = _ctcOutput ?? throw new InvalidOperationException("No native network.");
        var ids = new Tensor<T>(new[] { 1, RichTokens });
        ids[0, 0] = NumOps.FromDouble(languageQuery);
        ids[0, 1] = NumOps.FromDouble(EmotionQuery);
        ids[0, 2] = NumOps.FromDouble(EventQuery);
        ids[0, 3] = NumOps.FromDouble(_options.UseInverseTextNormalization ? WithItnQuery : WithoutItnQuery);
        var features = encoder.Features(waveform);
        var prefix = Engine.Reshape(queries.Forward(ids), new[] { RichTokens, features.Shape[1] });
        return output.Forward(encoder.Forward(Engine.TensorConcatenate(new[] { prefix, features }, 0)));
    }

    private Tensor<T> TranscriptLogits(Tensor<T> waveform, int languageQuery)
    {
        var all = AllLogits(waveform, languageQuery);
        return Engine.TensorSlice(all, new[] { RichTokens, 0 }, new[] { all.Shape[0] - RichTokens, all.Shape[1] });
    }

    // The transcript logits [frames, vocabulary] of a waveform [S], or [batch, frames, vocabulary] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_encoder is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++) outputs[r] = TranscriptLogits(Samples(input, r * samples, samples), LanguageQuery(_options.Language));
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _encoder is null ? base.ForwardForTraining(input) : Logits(input);

    private (List<int> Tokens, double Confidence) GreedyCtc(Tensor<T> logits)
    {
        var frames = Engine.Reshape(logits, new[] { logits.Length / logits.Shape[^1], logits.Shape[^1] });
        var tokens = new List<int>();
        double total = 0;
        int previous = -1;
        for (int t = 0; t < frames.Shape[0] && tokens.Count < _options.MaxTextLength; t++)
        {
            int best = 0;
            double max = double.NegativeInfinity;
            for (int v = 0; v < frames.Shape[1]; v++) { double value = NumOps.ToDouble(frames[t, v]); if (value > max) { max = value; best = v; } }
            double sum = 0;
            for (int v = 0; v < frames.Shape[1]; v++) sum += Math.Exp(NumOps.ToDouble(frames[t, v]) - max);
            if (best != previous && best > 0) { tokens.Add(best); total += 1.0 / sum; }
            previous = best;
        }
        return (tokens, tokens.Count > 0 ? total / tokens.Count : 0.0);
    }

    // The labels a training target names: label ids [length], or one distribution per frame [frames, vocabulary] read
    // as a CTC alignment (the most likely symbol per frame, repeats merged, blanks removed).
    private int[] Labels(Tensor<T> target)
    {
        if (target.Rank >= 2 && target.Shape[^1] == _options.VocabSize) return GreedyCtc(target).Tokens.ToArray();
        var labels = new int[target.Length];
        var span = target.Data.Span;
        for (int i = 0; i < labels.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(span[i]));
            if (value <= 0 || value >= _options.VocabSize) throw new ArgumentException($"Label {value} is the blank or outside the vocabulary.", nameof(target));
            labels[i] = (int)value;
        }
        return labels;
    }

    private int[][] RowLabels(Tensor<T> input, Tensor<T> target)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0], perRow = target.Length / rows;
        var labels = new int[rows][];
        for (int r = 0; r < rows; r++)
        {
            var row = rows == 1 ? target : Engine.Reshape(Engine.TensorSlice(Engine.Reshape(target, new[] { rows, perRow }), new[] { r, 0 }, new[] { 1, perRow }),
                target.Rank >= 3 ? target.Shape.ToArray().Skip(1).ToArray() : new[] { perRow });
            labels[r] = Labels(row);
        }
        return labels;
    }

    /// <summary>
    /// The release's loss of one utterance: the CTC loss of the transcript over the positions after the queries, plus,
    /// when <see cref="SenseVoiceOptions.TargetsIncludeRichTokens"/> is set, the cross-entropy of the first four labels
    /// (language, emotion, event, text normalization) at the four query positions, summed over them as the release's
    /// label-smoothing loss does without length normalization (smoothing 0).
    /// </summary>
    private Tensor<T> Loss(Tensor<T> waveform, int[] labels)
    {
        int rich = _options.TargetsIncludeRichTokens ? RichTokens : 0;
        if (labels.Length <= rich) throw new ArgumentException($"The target needs {rich} rich tokens and a transcript.", nameof(labels));
        var all = AllLogits(waveform, LanguageQuery(_options.Language));
        int frames = all.Shape[0] - RichTokens, v = _options.VocabSize;
        var transcript = labels.Skip(rich).ToArray();
        var speech = Engine.TensorSlice(all, new[] { RichTokens, 0 }, new[] { frames, v });
        var loss = Engine.TensorCTCLoss(Engine.Reshape(Engine.TensorLogSoftmax(speech, axis: -1), new[] { frames, 1, v }),
            new Tensor<int>(transcript, new[] { transcript.Length }), new[] { frames }, new[] { transcript.Length }, 0);
        if (rich == 0) return loss;
        var logProbabilities = Engine.TensorLogSoftmax(Engine.TensorSlice(all, new[] { 0, 0 }, new[] { RichTokens, v }), axis: -1);
        var oneHot = new Tensor<T>(new[] { RichTokens, v });
        for (int p = 0; p < RichTokens; p++) oneHot[p, labels[p]] = NumOps.One;
        var crossEntropy = Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(logProbabilities, oneHot), new[] { 0, 1 }, keepDims: false), NumOps.FromDouble(-1.0));
        return Engine.TensorAdd(Engine.Reshape(loss, new[] { 1 }), Engine.Reshape(crossEntropy, new[] { 1 }));
    }

    /// <inheritdoc />
    /// <remarks>The release's loss (<see cref="Loss"/>), averaged over the batch.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        if (_encoder is null) return false;
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var labels = RowLabels(input, target);
        TrainWithCustomLoss(input, _ =>
        {
            Tensor<T>? total = null;
            for (int r = 0; r < rows; r++)
            {
                var loss = Loss(Samples(input, r * samples, samples), labels[r]);
                total = total is null ? loss : Engine.TensorAdd(total, loss);
            }
            return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
        }, _optimizer);
        return true;
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes, averaged over the batch.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        if (_encoder is null) throw new NotSupportedException("The objective is defined on the native network.");
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var labels = RowLabels(input, target);
        double total = 0;
        for (int r = 0; r < rows; r++) total += NumOps.ToDouble(Engine.Reshape(Loss(Samples(input, r * samples, samples), labels[r]), new[] { 1 })[0]);
        return NumOps.FromDouble(total / rows);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "SenseVoice-Native" : "SenseVoice-ONNX", Description = "SenseVoice-Small: a SAN-M encoder with task queries and a CTC head (An et al. 2024)", FeatureCount = 1, Complexity = _options.NumEncoderLayers + _options.NumTpEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SenseVoice<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
