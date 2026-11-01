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
/// Fun-ASR-nano (Tongyi Lab, 2025): an LLM-based recognizer of a SenseVoice-Small audio encoder, a Transformer adaptor and
/// the Qwen3-0.6B decoder, with a CTC decoder whose first-pass hypothesis drives hotword retrieval.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Fun-ASR-nano listens to speech, turns it into vectors a language model can read, and lets
/// the language model (Qwen3-0.6B) write the transcript as the answer to a chat prompt that asks for one. A separate,
/// faster CTC decoder produces a rough first transcript that is used to look up the hotwords to put in the prompt.</para>
/// <para>
/// The report (§3, Figure 2) gives four parts: an audio encoder (0.2B parameters for nano), an adaptor of two Transformer
/// encoder layers, a CTC decoder on the audio encoder, and the LLM decoder. The release (Fun-ASR-Nano-2512's config and
/// <c>FunASRNano</c> model) fixes them: SenseVoice-Small's encoder (50 + 20 SAN-M blocks of width 512 over 80-band
/// filterbanks stacked seven every six frames); the adaptor, a linear-ReLU-linear projection to the LLM width and two
/// Transformer encoder layers; Qwen3-0.6B, reading the adaptor's output in place of the speech inside a ChatML prompt
/// ("You are a helpful assistant." as system, the instruction "语音转写：" before the speech); and the CTC decoder, five
/// Transformer encoder layers and a linear head over a 60,515-token vocabulary on the detached encoder output. The LLM's
/// loss is the transcript's cross-entropy after the prompt. Training follows the report's §4.2 stages
/// (<see cref="FunASRNanoOptions.TrainingStage"/>): only the modules the stage names are updated, LoRA adapters (rank 16)
/// stand in for the LLM, and the CTC stage trains the CTC decoder alone with targets in its own vocabulary. Those stages
/// presuppose a pretrained Qwen3, so the default is joint training of every parameter on the LLM's loss. Without Qwen's
/// 151,936-token vocabulary (a smaller <see cref="FunASRNanoOptions.VocabSize"/>) the prompt has no text: the LLM reads
/// the speech alone and its last token ends the answer.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 480000, outputSize: 151936);
/// var model = new FunASRNano&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Fun-ASR Technical Report", "https://arxiv.org/abs/2509.12508", Year = 2025, Authors = "Tongyi Fun Team")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-6, WeightDecay = 0, WarmupSteps = 2500, Schedule = LearningRateSchedulerType.Noam, MaxGradientNorm = 5,
                Source = "The Fun-ASR-Nano-2512 release config: AdamW at 5e-6, weight decay 0, warmuplr with 2500 warmup steps, grad_clip 5.")]
public partial class FunASRNano<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The LLM's vocabulary.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly FunASRNanoOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private FunAsrNanoNetwork<T>? _network;
    internal FunAsrNanoNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public FunASRNano(NeuralNetworkArchitecture<T> architecture, string modelPath, FunASRNanoOptions? options = null) : base(architecture)
    {
        _options = options ?? new FunASRNanoOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public FunASRNano(NeuralNetworkArchitecture<T> architecture, FunASRNanoOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new FunASRNanoOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, warmupStepsOverride: _options.WarmupSteps) ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <inheritdoc />
    /// <remarks>The model's optimizer, unless a base training optimizer was set explicitly.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    // The language the prompt names; "auto" and empty leave it unnamed, as the release's default does.
    private string? PromptLanguage(string? language)
    {
        string? chosen = language ?? _options.Language;
        return string.IsNullOrEmpty(chosen) || string.Equals(chosen, "auto", StringComparison.OrdinalIgnoreCase) ? null : chosen;
    }

    private string Instruction(string? language) =>
        FunAsrNanoNetwork<T>.Instruction(_options.Hotwords, PromptLanguage(language), _options.InverseTextNormalization);

    /// <summary>Transcribes audio: the LLM's greedy answer to the transcription prompt, with
    /// <see cref="FunASRNanoOptions.Hotwords"/> in its context section.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for Fun-ASR-nano.");
        var network = _network ?? throw new NotSupportedException("Transcription runs on the native network.");
        int[] tokens;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            tokens = network.Greedy(network.Prompt(network.Encode(Waveform(audio)), Instruction(language)), _options.MaxTextLength);
        return new TranscriptionResult<T>
        {
            Text = Text(network, tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Length > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    /// <summary>The CTC decoder's first-pass hypothesis (ids in the CTC vocabulary), which the report uses to retrieve the
    /// hotwords for the prompt.</summary>
    public int[] FirstPassTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var network = _network ?? throw new NotSupportedException("The CTC decoder runs on the native network.");
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        return network.CtcGreedy(network.Encode(Waveform(audio)));
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("Fun-ASR-nano does not support streaming.");

    // Qwen's tokenizer decodes when the vocabulary is Qwen's; otherwise each token is its vocabulary entry or code point.
    private string Text(FunAsrNanoNetwork<T> network, int[] tokens)
    {
        if (network.UsesQwenVocabulary) return network.Tokenizer.Decode(tokens).Trim();
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
        // The encoder, adaptor, LLM and CTC decoder (see the class remarks). It used to be a generic Conformer template.
        var layers = new List<LayerBase<T>>();
        _network = new FunAsrNanoNetwork<T>(Engine, _options, layers, Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? 42));
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

    // The logits [1, vocabulary] of the answer's first token for a waveform [S], or [batch, 1, vocabulary] for [batch, S].
    private Tensor<T> FirstLogits(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        var outputs = new Tensor<T>[rows];
        string instruction = Instruction(null);
        for (int r = 0; r < rows; r++) outputs[r] = _network.FirstLogits(_network.Prompt(_network.Encode(Waveform(Row(input, r, rows))), instruction));
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return FirstLogits(input); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : FirstLogits(input);

    // The token ids a training target names: ids [length] within the vocabulary the stage trains, or one distribution per
    // token [length, vocabulary], read as its most likely tokens; the reserved token (end or blank) is never a target.
    private int[] Tokens(Tensor<T> target, int vocabulary, int reserved)
    {
        var span = target.Data.Span;
        if (target.Rank >= 2 && target.Shape[^1] == vocabulary)
        {
            var tokens = new int[target.Length / vocabulary];
            for (int r = 0; r < tokens.Length; r++)
            {
                int best = -1;
                for (int v = 0; v < vocabulary; v++)
                    if (v != reserved && (best < 0 || NumOps.GreaterThan(span[r * vocabulary + v], span[r * vocabulary + best]))) best = v;
                tokens[r] = best;
            }
            return tokens;
        }
        var ids = new int[target.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(span[i]));
            if (value < 0 || value >= vocabulary || (int)value == reserved)
                throw new ArgumentException($"Token {value} is outside the vocabulary or reserved.", nameof(target));
            ids[i] = (int)value;
        }
        return ids;
    }

    // The stage's loss [1] of one utterance: the CTC loss for the CTC stage, the LLM's cross-entropy otherwise.
    // Joint training reads the target as LLM tokens; the CTC decoder trains in its own stage, on its own vocabulary.
    private Tensor<T> UtteranceLoss(FunAsrNanoNetwork<T> network, Tensor<T> audio, Tensor<T> target)
    {
        var encoded = network.Encode(Waveform(audio));
        if (_options.TrainingStage == FunASRNanoTrainingStage.CtcDecoder)
            return network.CtcLoss(encoded, Tokens(target, _options.CtcVocabSize, network.Blank));
        var loss = network.LlmLoss(network.Prompt(encoded, Instruction(null)), Tokens(target, _options.VocabSize, network.EndToken));
        return Engine.Reshape(loss, new[] { 1 });
    }

    private Tensor<T> BatchLoss(FunAsrNanoNetwork<T> network, Tensor<T> input, Tensor<T> target)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var loss = UtteranceLoss(network, Row(input, r, rows), Row(target, r, rows));
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
    }

    /// <inheritdoc />
    /// <remarks>The current stage's loss, averaged over the batch, with LoRA dropout active.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        network.Llm.Training = true;
        try { TrainWithCustomLoss(input, _ => BatchLoss(network, input, target)); }
        finally { network.Llm.Training = false; }
        return true;
    }

    // The modules the report's §4.2 stage updates.
    private IEnumerable<LayerBase<T>> TrainedLayers(FunAsrNanoNetwork<T> network) => _options.TrainingStage switch
    {
        FunASRNanoTrainingStage.AdaptorAlignment => network.AdaptorLayers,
        FunASRNanoTrainingStage.EncoderAndAdaptor => network.EncoderLayers.Concat(network.AdaptorLayers),
        FunASRNanoTrainingStage.LlmLora => network.LoraLayers,
        FunASRNanoTrainingStage.CtcDecoder => network.CtcLayers,
        _ => network.EncoderLayers.Concat(network.AdaptorLayers).Concat(network.LoraLayers),
    };

    /// <inheritdoc />
    /// <remarks>The modules the current <see cref="FunASRNanoOptions.TrainingStage"/> updates; everything else, the LLM's
    /// own weights included, stays frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        var network = _network;
        if (network is null || _options.TrainingStage == FunASRNanoTrainingStage.Joint) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(TrainedLayers(network).Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
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
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "FunASRNano-Native" : "FunASRNano-ONNX", Description = "Fun-ASR-nano: SenseVoice-Small encoder, adaptor and Qwen3-0.6B (Tongyi Lab 2025)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumTpEncoderLayers + _options.NumLlmLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FunASRNano<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
