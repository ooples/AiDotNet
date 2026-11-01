using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>
/// Qwen3-ASR-1.7B (Qwen Team, 2026): an LLM-based recognizer of the AuT audio encoder and the Qwen3-1.7B decoder, which names the
/// language and then writes the transcript.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Qwen3-ASR-1.7B turns speech into a few vectors per second with its audio encoder and lets the
/// Qwen3-1.7B language model answer a chat prompt that contains them: it writes "language English", a marker, and then the
/// transcript. You can force the language, and put hotwords or background text in the prompt as context.</para>
/// <para>
/// The report (§2) builds Qwen3-ASR-1.7B from Qwen3-1.7B, a projector and an AuT encoder (300M parameters, width 1,024) over 128-band Fbank features at a
/// 12.5 Hz token rate. The release's <c>modeling_qwen3_asr.py</c> fixes the encoder: 100-frame chunks of the Whisper
/// log-mel spectrogram, three 3×3 stride-2 GELU convolutions (480 channels), a bias-free linear map to the width with
/// sinusoidal positions restarting every chunk, pre-norm Transformer layers attending within windows of eight chunks, a
/// layer norm, and the GELU projector to the decoder's width. The decoder reads the projector's output in place of the
/// audio pads in the release's chat template (system context; user audio between <c>&lt;|audio_start|&gt;</c> and
/// <c>&lt;|audio_end|&gt;</c>; assistant), and answers <c>language {name}&lt;asr_text&gt;{transcript}</c>; a forced
/// language moves its prefix into the prompt. Its interleaved multimodal rotary embedding gives audio and text the same
/// position on all three axes, which is the ordinary rotary embedding. Training is the release's supervised fine-tuning
/// of every parameter on the cross-entropy of the answer and <c>&lt;|im_end|&gt;</c> after the prompt.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 160000, outputSize: 151936);
/// var model = new Qwen3ASR&lt;float&gt;(architecture);
/// var result = model.Transcribe(waveform);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Qwen3-ASR Technical Report", "https://arxiv.org/abs/2601.21337", Year = 2026, Authors = "Shi et al.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-5, WeightDecay = 0, Schedule = LearningRateSchedulerType.LinearWarmup, MaxGradientNorm = 1,
                Source = "The release's finetuning/qwen3_asr_sft.py: Hugging Face Trainer AdamW at 2e-5, linear schedule, warmup ratio 0.02.")]
public partial class Qwen3ASR<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The decoder's vocabulary.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly Qwen3ASROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private Qwen3AsrNetwork<T>? _network;
    internal Qwen3AsrNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public Qwen3ASR(NeuralNetworkArchitecture<T> architecture, string modelPath, Qwen3ASROptions? options = null) : base(architecture)
    {
        _options = options ?? new Qwen3ASROptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public Qwen3ASR(NeuralNetworkArchitecture<T> architecture, Qwen3ASROptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new Qwen3ASROptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, warmupStepsOverride: _options.WarmupSteps) ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <inheritdoc />
    /// <remarks>The model's optimizer, unless a base training optimizer was set explicitly.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    // The forced language; "auto" and empty let the model name it.
    private string? ForcedLanguage(string? language)
    {
        string? chosen = language ?? _options.Language;
        return string.IsNullOrEmpty(chosen) || string.Equals(chosen, "auto", StringComparison.OrdinalIgnoreCase) ? null : chosen;
    }

    /// <summary>Transcribes audio: the decoder's greedy answer, split at <c>&lt;asr_text&gt;</c> into the language it names
    /// and the transcript.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for Qwen3-ASR-1.7B.");
        var network = _network ?? throw new NotSupportedException("Transcription runs on the native network.");
        string? forced = ForcedLanguage(language);
        int[] tokens;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            tokens = network.Greedy(network.Prompt(network.Encode(network.Spectrogram(Waveform(audio))), _options.Context, forced), _options.MaxTextLength);
        var (named, text) = Split(network, tokens);
        return new TranscriptionResult<T>
        {
            Text = text, Language = forced ?? named ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Length > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    // The answer's language (the words after "language" before <asr_text>) and its transcript (after it).
    private (string? Language, string Text) Split(Qwen3AsrNetwork<T> network, int[] tokens)
    {
        if (!network.UsesQwenVocabulary) return (null, Text(tokens));
        int marker = Array.IndexOf(tokens, network.TextMarker);
        if (marker < 0) return (null, network.Tokenizer.Decode(tokens).Trim());
        string prefix = network.Tokenizer.Decode(tokens.Take(marker)).Trim();
        string? named = prefix.StartsWith("language", StringComparison.Ordinal) ? prefix.Substring("language".Length).Trim() : null;
        return (string.IsNullOrEmpty(named) ? null : named, network.Tokenizer.Decode(tokens.Skip(marker + 1)).Trim());
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <summary>The language the model names before the transcript.</summary>
    public string DetectLanguage(Tensor<T> audio) => Transcribe(audio, "auto").Language;

    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [DetectLanguage(audio)] = NumOps.One };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("Streaming is not implemented for Qwen3-ASR-1.7B.");

    // Each token is its vocabulary entry when a vocabulary is configured, otherwise its Unicode code point.
    private string Text(IEnumerable<int> tokens)
    {
        var vocabulary = _options.Vocabulary;
        var sb = new System.Text.StringBuilder();
        foreach (int t in tokens)
        {
            if (vocabulary is not null && t < vocabulary.Length) sb.Append(vocabulary[t]);
            else if (t <= 0x10FFFF && (t < 0xD800 || t > 0xDFFF)) sb.Append(char.ConvertFromUtf32(t));
        }
        return sb.ToString().Trim();
    }

    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The AuT encoder and the decoder (see the class remarks). It used to be a generic Conformer template.
        var o = _options;
        var layers = new List<LayerBase<T>>();
        _network = new Qwen3AsrNetwork<T>(Engine, new Qwen3AsrConfiguration
        {
            Mels = o.NumMels, EncoderWidth = o.EncoderDim, EncoderLayers = o.NumEncoderLayers, EncoderHeads = o.NumAttentionHeads,
            EncoderFeedForward = o.EncoderFeedForwardDim, DownsampleChannels = o.DownsampleChannels, ChunkFrames = o.ChunkFrames,
            WindowChunks = o.WindowChunks,
            Llm = new LlamaConfiguration
            {
                VocabSize = o.VocabSize, Hidden = o.LlmHiddenDim, Layers = o.NumLlmLayers, Heads = o.LlmHeads, KeyValueHeads = o.LlmKeyValueHeads,
                HeadDim = o.LlmHeadDim, Intermediate = o.LlmIntermediateDim, RmsEpsilon = o.LlmRmsEpsilon, RopeTheta = o.LlmRopeTheta,
                QueryKeyNorm = true, TieWordEmbeddings = true,
            },
        }, layers, Tensors.Helpers.RandomHelper.CreateSeededRandom(o.Seed ?? 42));
        Layers.AddRange(layers);
    }

    // The 16 kHz waveform, truncated to the longest recording transcribed.
    private Tensor<T> Waveform(Tensor<T> audio)
    {
        int length = Math.Min(audio.Length, SampleRate * _options.MaxAudioLengthSeconds);
        var waveform = new Tensor<T>(new[] { length });
        audio.Data.Span.Slice(0, length).CopyTo(waveform.Data.Span);
        return waveform;
    }

    private Tensor<T> Row(Tensor<T> input, int row, int rows)
    {
        if (rows == 1) return input;
        int samples = input.Length / rows;
        return Engine.Reshape(Engine.TensorSlice(Engine.Reshape(input, new[] { rows, samples }), new[] { row, 0 }, new[] { 1, samples }), new[] { samples });
    }

    private Tensor<T> PromptOf(Qwen3AsrNetwork<T> network, Tensor<T> audio) =>
        network.Prompt(network.Encode(network.Spectrogram(Waveform(audio))), _options.Context, ForcedLanguage(null));

    // The logits [1, vocabulary] of the answer's first token for a waveform [S], or [batch, 1, vocabulary] for [batch, S].
    private Tensor<T> FirstLogits(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++) outputs[r] = _network.FirstLogits(PromptOf(_network, Row(input, r, rows)));
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return FirstLogits(input); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : FirstLogits(input);

    // The answer a training target names: ids [length] (the transcript when the language is forced, otherwise the whole
    // answer with its language prefix), or one distribution per token [length, vocabulary], read as its most likely
    // tokens; the end token is never a target.
    private int[] Answer(Qwen3AsrNetwork<T> network, Tensor<T> target)
    {
        int vocabulary = _options.VocabSize, end = network.EndToken;
        var span = target.Data.Span;
        if (target.Rank >= 2 && target.Shape[^1] == vocabulary)
        {
            var tokens = new int[target.Length / vocabulary];
            for (int r = 0; r < tokens.Length; r++)
            {
                int best = -1;
                for (int v = 0; v < vocabulary; v++)
                    if (v != end && (best < 0 || NumOps.GreaterThan(span[r * vocabulary + v], span[r * vocabulary + best]))) best = v;
                tokens[r] = best;
            }
            return tokens;
        }
        var ids = new int[target.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(span[i]));
            if (value < 0 || value >= vocabulary || (int)value == end)
                throw new ArgumentException($"Token {value} is outside the vocabulary or the end token.", nameof(target));
            ids[i] = (int)value;
        }
        return ids;
    }

    private Tensor<T> BatchLoss(Qwen3AsrNetwork<T> network, Tensor<T> input, Tensor<T> target)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var loss = Engine.Reshape(network.Loss(PromptOf(network, Row(input, r, rows)), Answer(network, Row(target, r, rows))), new[] { 1 });
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
    }

    /// <inheritdoc />
    /// <remarks>The answer's cross-entropy after the prompt, averaged over the batch.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        TrainWithCustomLoss(input, _ => BatchLoss(network, input, target));
        return true;
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var network = _network ?? throw new NotSupportedException("The objective is defined on the native network.");
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        return BatchLoss(network, input, target)[0];
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "Qwen3ASR-Native" : "Qwen3ASR-ONNX", Description = "Qwen3-ASR-1.7B: AuT encoder and Qwen3-1.7B (Qwen Team 2026)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumLlmLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Qwen3ASR<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
