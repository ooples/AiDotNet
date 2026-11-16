using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>
/// Code-switching ASR: a hybrid CTC/attention model trained jointly with an EXPLICIT
/// language-identification task, for speech that mixes languages inside one utterance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Zeng, Khassanov, Pham, Xu, Chng and Li, "On the End-to-End Solution to Mandarin-English
/// Code-switching Speech Recognition" (Interspeech 2019, arXiv:1811.00241). Code-switching "refers
/// to a linguistic phenomenon where a speaker uses different languages in an utterance or between
/// alternating utterances". The paper's headline result is that making the model say WHICH language
/// it is hearing, as a separate supervised task, improves the transcription itself:
/// </para>
/// <code>
///   L_MTL(Y|X) = lambda1 * L_att(Y|X) + (1 - lambda1) * L_ctc(Y|X) + lambda2 * L_lid(Z|X)
///
///   CTC weight (1 - lambda1) = 0.2        lambda2 = 0.2
/// </code>
/// <para>
/// The network is section 4.3's ESPnet model: log-mel features, a CNN front end (ESPnet's VGG2L, 4x in time), a
/// six-layer BLSTM encoder of 320 units, a one-layer 320-unit LSTM decoder with location-aware attention (content plus
/// location, Chorowski et al. 2015), and a CTC layer on the encoder. The LID task is a second decoder that emits the
/// language Z of each output token from the same attention (Figure 1, LID_shared) or its own (LID_indep), so its
/// labels come from the transcript: Mandarin characters and English BPE units occupy disjoint token ranges ("Mandarin
/// uses characters while English uses BPE units"). Attention sizes not stated by the paper are ESPnet's defaults.
/// Transcription decodes greedily with the attention decoder; the language reported is the LID decoder's.
/// </para>
/// <para>
/// <b>For Beginners:</b> Bilingual speakers often switch language mid-sentence. This model listens with a
/// bidirectional recurrent encoder, writes the transcript one token at a time while looking back at the audio, and in
/// parallel says which language each token is in. Being forced to notice the switch makes the transcription better.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 16000, outputSize: 6002);
/// var model = new CodeSwitchingASR&lt;double&gt;(architecture);
/// model.Train(waveform, transcriptTokenIds);
/// var result = model.Transcribe(waveform);
/// var language = model.DetectLanguage(waveform);   // from the learned LID decoder
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("On the End-to-End Solution to Mandarin-English Code-switching Speech Recognition",
    "https://arxiv.org/abs/1811.00241",
    Year = 2019,
    Authors = "Zhiping Zeng, Yerbolat Khassanov, Van Tung Pham, Haihua Xu, Eng Siong Chng, Haizhou Li")]
[AiDotNet.Attributes.PreprocessesInput("CodeSwitchingAsrNetwork converts the waveform to log-mel features and feeds them to the first convolution as a [B, 1, frames, bins] image, so Layers[0] never sees the raw samples.")]
[AiDotNet.Attributes.StackInputLayout(AiDotNet.Enums.TensorAxis.Batch, AiDotNet.Enums.TensorAxis.Channels, AiDotNet.Enums.TensorAxis.Height, AiDotNet.Enums.TensorAxis.Width)]
public partial class CodeSwitchingASR<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The CTC output: the inventory and the boundary token.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize + 1;

    private readonly CodeSwitchingASROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private CodeSwitchingAsrNetwork<T>? _network;
    private Random _random;
    internal CodeSwitchingAsrNetwork<T>? Network => _network;

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedLanguages { get; }

    /// <inheritdoc />
    public bool SupportsStreaming => false;

    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    /// <summary>
    /// Gets the token id at which the English BPE inventory begins. Ids below it are Mandarin
    /// characters; the CTC blank is id 0.
    /// </summary>
    public int EnglishTokenOffset => 1 + _options.MandarinTokenCount;

    /// <summary>
    /// Returns the language a token id belongs to, which is well-defined precisely because the two
    /// inventories are concatenated rather than unified.
    /// </summary>
    public string LanguageOfToken(int tokenId) =>
        tokenId <= 0 ? _options.Language : tokenId < EnglishTokenOffset ? "zh" : "en";

    public CodeSwitchingASR(NeuralNetworkArchitecture<T> architecture, string modelPath, CodeSwitchingASROptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CodeSwitchingASROptions();
        Options = _options;
        _useNativeMode = false;
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        base.SampleRate = _options.SampleRate;
        base.NumMels = 0;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = CodeSwitchingAsrNetwork<T>.Languages;
        InitializeLayersCore();
    }

    public CodeSwitchingASR(NeuralNetworkArchitecture<T> architecture, CodeSwitchingASROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CodeSwitchingASROptions();
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        base.SampleRate = _options.SampleRate;
        base.NumMels = 0;
        // The paper's task is Mandarin-English: the concatenated inventory and the LID labels are defined over that pair.
        SupportedLanguages = CodeSwitchingAsrNetwork<T>.Languages;
        InitializeLayersCore();
    }

    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _network = null;
        Layers.Clear();
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The paper network (section 4.3). It used to be a Conformer template with dense "decoder" layers trained by
        // mean squared error against the targets.
        var layers = new List<LayerBase<T>>();
        _network = new CodeSwitchingAsrNetwork<T>(Engine, layers, _options.ToConfiguration());
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        Layers.AddRange(layers);
    }

    // The recognizer reads the raw waveform, truncated to the longest window a transcription covers.
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_network is null) return rawAudio;
        int length = Math.Min(rawAudio.Length, SampleRate * _options.MaxAudioLengthSeconds);
        var waveform = new Tensor<T>(new[] { length });
        rawAudio.Data.Span.Slice(0, length).CopyTo(waveform.Data.Span);
        return waveform;
    }

    private CodeSwitchingAsrNetwork<T> RequireNetwork() =>
        _network ?? throw new NotSupportedException("This operation runs on the native paper network.");

    /// <summary>Transcribes audio with greedy attention decoding; the language is the LID decoder's.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for CodeSwitchingASR.");
        if (IsOnnxMode || _network is null)
        {
            var logits = PredictCore(audio);
            var frames = Engine.Reshape(logits, new[] { logits.Length / logits.Shape[^1], logits.Shape[^1] });
            var ctcTokens = Audio.Foundations.ConformerAsrNetwork<T>.GreedyCtc(frames);
            return Result(ctcTokens, language ?? _options.Language, audio);
        }
        var (tokens, languages) = Decode(audio);
        return Result(tokens, language ?? Dominant(languages), audio);
    }

    private TranscriptionResult<T> Result(List<int> tokens, string language, Tensor<T> audio) => new()
    {
        Text = Text(tokens), Language = language, Confidence = NumOps.FromDouble(tokens.Count > 0 ? 1.0 : 0.0),
        DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
    };

    // Each token is its vocabulary entry; tokens past the vocabulary list decode as Unicode code points.
    private string Text(List<int> tokens)
    {
        var sb = new System.Text.StringBuilder();
        foreach (int t in tokens)
        {
            if (t < _options.Vocabulary.Length) sb.Append(_options.Vocabulary[t]);
            else if (t > 0 && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t));
        }
        return sb.ToString().Trim();
    }

    private (List<int> Tokens, List<double[]> Languages) Decode(Tensor<T> audio)
    {
        var network = RequireNetwork();
        using var _ = new NoGradScope<T>();
        var encoded = network.Encode(network.Features(PreprocessAudio(audio)));
        return network.Greedy(encoded, _options.MaxTextLength);
    }

    // The language with the larger mean LID posterior over the decoded steps.
    private string Dominant(List<double[]> languages)
    {
        var mean = MeanLanguage(languages);
        return mean[0] >= mean[1] ? SupportedLanguages[0] : SupportedLanguages[1];
    }

    // The LID decoder's language posteriors averaged over the decoding steps, renormalized over the languages.
    private static double[] MeanLanguage(List<double[]> languages)
    {
        int count = CodeSwitchingAsrNetwork<T>.Languages.Length;
        var mean = new double[count];
        foreach (var step in languages)
        {
            double total = 0;
            for (int k = 0; k < count; k++) total += step[k];
            for (int k = 0; k < count; k++) mean[k] += total > 0 ? step[k] / total : 1.0 / count;
        }
        for (int k = 0; k < count; k++) mean[k] = languages.Count > 0 ? mean[k] / languages.Count : 1.0 / count;
        return mean;
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null,
        bool? includeTimestamps = null, CancellationToken cancellationToken = default)
        => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <summary>Reports the dominant language from the LEARNED LID decoder.</summary>
    public string DetectLanguage(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var probabilities = DetectLanguageProbabilities(audio);
        var entries = probabilities.ToArray();
        if (entries.Length == 0)
            return _options.Language;

        // The first language holding the largest posterior wins; NaN and -inf never win, and when nothing
        // else is left the configured language stands.
        var posteriors = new Tensor<T>(entries.Select(kv => kv.Value).ToArray(), new[] { 1, entries.Length });
        int bestIndex = CtcGreedyDecoding.FirstMaxIndexPerRow(posteriors).GetFlat(0);
        var winner = entries[bestIndex];
        return NumOps.ToDouble(winner.Value) > double.NegativeInfinity ? winner.Key : _options.Language;
    }

    /// <summary>The LID decoder's language posteriors averaged over the greedy decoding's steps (the step that ends
    /// the transcript included) and normalized over the two languages.</summary>
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var result = new Dictionary<string, T>();
        if (IsOnnxMode || _network is null)
        {
            // No native LID decoder to consult: a flat distribution rather than a confident guess.
            foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(1.0 / SupportedLanguages.Count);
            return result;
        }
        var mean = MeanLanguage(Decode(audio).Languages);
        for (int k = 0; k < mean.Length; k++) result[SupportedLanguages[k]] = NumOps.FromDouble(mean[k]);
        return result;
    }

    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null)
        => throw new NotSupportedException("CodeSwitchingASR does not support streaming.");

    // CTC logits [frames, V] for a waveform [S], or [batch, frames, V] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input, bool training)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        var rows = Rows(input);
        var outputs = new Tensor<T>[rows.Length];
        for (int r = 0; r < rows.Length; r++)
            outputs[r] = _network.CtcLogits(_network.Encode(_network.Features(rows[r]), training, training ? _random : null));
        return input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    private Tensor<T>[] Rows(Tensor<T> input)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var flat = Engine.Reshape(input, new[] { rows, samples });
        var result = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
            result[r] = PreprocessAudio(Engine.Reshape(Engine.TensorSlice(flat, new[] { r, 0 }, new[] { 1, samples }), new[] { samples }));
        return result;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input, training: false); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input, training: true);

    // The transcript a training target names: token ids [length], or one distribution per frame [frames, V] read as a
    // CTC alignment (the most likely token per frame, repeats merged, blanks and the boundary removed).
    private int[] Labels(Tensor<T> target, int boundary)
    {
        int width = target.Rank >= 2 ? target.Shape[^1] : 0;
        if (width == boundary || width == boundary + 1)
        {
            var frames = Engine.Reshape(target, new[] { target.Length / width, width });
            return Audio.Foundations.ConformerAsrNetwork<T>.GreedyCtc(frames).Where(t => t < boundary).ToArray();
        }
        var labels = new int[target.Length];
        var span = target.Data.Span;
        for (int i = 0; i < labels.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(span[i]));
            if (value <= 0 || value >= boundary) throw new ArgumentException($"Token {value} is the blank, the boundary or outside the inventory.", nameof(target));
            labels[i] = (int)value;
        }
        return labels;
    }

    // One (waveform, transcript) pair per row; a target whose last axis is the vocabulary keeps it in each row.
    private (Tensor<T> Waveform, int[] Labels)[] Batch(Tensor<T> input, Tensor<T> target, int boundary)
    {
        var rows = Rows(input);
        int perRow = target.Length / rows.Length, width = target.Rank >= 2 ? target.Shape[^1] : 0;
        bool distributions = width == boundary || width == boundary + 1;
        var batch = new (Tensor<T>, int[])[rows.Length];
        for (int r = 0; r < rows.Length; r++)
        {
            var row = rows.Length == 1 ? target : Engine.Reshape(Engine.TensorSlice(Engine.Reshape(target, new[] { rows.Length, perRow }), new[] { r, 0 }, new[] { 1, perRow }),
                distributions ? new[] { perRow / width, width } : new[] { perRow });
            batch[r] = (rows[r], Labels(row, boundary));
        }
        return batch;
    }

    /// <inheritdoc />
    /// <remarks>Equation 2: L = λ1 L_att + (1 − λ1) L_ctc + λ2 L_lid on one tape over the shared encoder, each loss
    /// summed over the utterance's tokens and averaged over the batch.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        var batch = Batch(input, target, network.Boundary);
        TrainWithCustomLoss(input, _ =>
        {
            Tensor<T>? total = null;
            foreach (var (waveform, labels) in batch)
            {
                var loss = network.Loss(network.Features(waveform), labels, training: true, _random).Total;
                total = total is null ? loss : Engine.TensorAdd(total, loss);
            }
            return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / batch.Length));
        }, _optimizer);
        return true;
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes, without dropout, averaged over the batch.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var network = RequireNetwork();
        using var _ = new NoGradScope<T>();
        var batch = Batch(input, target, network.Boundary);
        double total = 0;
        foreach (var (waveform, labels) in batch) total += NumOps.ToDouble(network.Loss(network.Features(waveform), labels).Total[0]);
        return NumOps.FromDouble(total / batch.Length);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = _useNativeMode ? "CodeSwitchingASR-Native" : "CodeSwitchingASR-ONNX",
        Description = "Hybrid CTC/attention BLSTM code-switching ASR with an explicit LID task (Zeng et al. 2019)",
        FeatureCount = _options.NumMels,
        Complexity = _options.NumEncoderLayers,
        AdditionalInfo = BaseAudioMetadataInfo()
    };

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CodeSwitchingASR<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }
}
