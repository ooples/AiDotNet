using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Audio.Foundations;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>
/// Seed-ASR (Bai et al., 2024): ByteDance's audio-conditioned LLM recognizer, which feeds the LUISE Conformer encoder's
/// output through a frame-splicing converter into a large language model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Seed-ASR listens with a large Conformer encoder (LUISE), joins every four of its 40 ms
/// frames into one 160 ms speech token, and hands those tokens to a language model, which writes the transcript. Text
/// placed before the speech (a conversation, a document) can help it with names and rare words.</para>
/// <para>
/// §3.2–3.4 and Figure 2 of the paper: LUISE is a 32-block Conformer of nearly 2 billion parameters over mel filterbanks,
/// emitting 40 ms frames, pretrained BERT-style on masked frame labels; the converter splices four consecutive frames
/// along the features and applies a linear projection into the LLM; SFT trains the encoder and the converter with the
/// LLM fixed, with the cross-entropy over the transcript's tokens only; context SFT places context text before the
/// speech; inference is greedy. The paper does not publish LUISE's width, the LLM (a mixture-of-experts model of over
/// ten billion parameters), its tokenizer or the optimizer: see <see cref="SeedASROptions"/> for the defaults chosen.
/// A model built here starts from random weights and a fixed random LLM cannot fit a transcript, so the default updates
/// every parameter; the paper's scope is <see cref="SpeechLlmTrainingScope.EncoderAdapterAndLora"/>.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Your recording: 16 kHz mono samples in [-1, 1], shaped [1, samples]. Noise stands in for it here.
/// var waveform = Tensor&lt;float&gt;.CreateRandom(1, 16000);
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 160000, outputSize: 32000);
/// var model = new SeedASR&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[AiDotNet.Attributes.PreprocessesInput("ConformerAsrNetwork converts the waveform to log-mel features and feeds them to the first convolution as a [B, 1, frames, bins] image, so Layers[0] never sees the raw samples.")]
[AiDotNet.Attributes.StackInputLayout(AiDotNet.Enums.TensorAxis.Batch, AiDotNet.Enums.TensorAxis.Channels, AiDotNet.Enums.TensorAxis.Height, AiDotNet.Enums.TensorAxis.Width)]
[ResearchPaper("Seed-ASR: Understanding Diverse Speech and Contexts with LLM-based Speech Recognition", "https://arxiv.org/abs/2407.04675", Year = 2024, Authors = "Bai et al.")]
public partial class SeedASR<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The decoder's vocabulary.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly SeedASROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private SeedAsrNetwork<T>? _network;
    internal SeedAsrNetwork<T>? Network => _network;
    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => false;
    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    public SeedASR(NeuralNetworkArchitecture<T> architecture, string modelPath, SeedASROptions? options = null) : base(architecture)
    {
        _options = options ?? new SeedASROptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { "zh", "en" }; InitializeLayersCore();
    }

    public SeedASR(NeuralNetworkArchitecture<T> architecture, SeedASROptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new SeedASROptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate, WeightDecay = _options.WeightDecay,
            });
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { "zh", "en" }; InitializeLayersCore();
    }

    /// <inheritdoc />
    /// <remarks>The model's optimizer, unless a base training optimizer was set explicitly.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <summary>Transcribes audio: the LLM's greedy continuation of the context and the speech, at most
    /// <see cref="SeedASROptions.MaxTextLength"/> tokens.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for Seed-ASR.");
        var network = _network ?? throw new NotSupportedException("Transcription runs on the native network.");
        int[] tokens;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            tokens = network.Greedy(network.Prompt(network.Speech(TruncateAudioWaveform(audio, _options.MaxAudioLengthSeconds), training: false), Context), _options.MaxTextLength);
        return new TranscriptionResult<T>
        {
            Text = Text(tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Length > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    /// <inheritdoc/>
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    /// <inheritdoc/>
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    /// <inheritdoc/>
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
    /// <inheritdoc/>
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("Seed-ASR does not support streaming.");

    // The context tokens before the speech (none for plain recognition).
    private IReadOnlyList<int> Context => _options.Context ?? Array.Empty<int>();

    // Each token is its vocabulary entry or code point (the paper's tokenizer is unpublished).
    private string Text(int[] tokens)
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
        // The encoder, adapter and LLM (see the class remarks). It used to be a generic LLM-ASR template.
        var o = _options;
        var layers = new List<LayerBase<T>>();
        if (o.SpliceFrames < 1) throw new ArgumentException("The converter splices at least one frame.", nameof(Options));
        _network = new SeedAsrNetwork<T>(Engine, new ConformerAsrConfiguration
        {
            SampleRate = o.SampleRate, NumMels = o.NumMels, EncoderDim = o.EncoderDim, NumEncoderLayers = o.NumEncoderLayers,
            NumAttentionHeads = o.NumAttentionHeads, FeedForwardMultiplier = o.FeedForwardMultiplier, ConvKernel = o.ConvKernel,
            Subsampling = ConformerAsrSubsampling.Conv4x, VocabSize = 1,
        }, o.SpliceFrames, new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.LlmDim, Layers = o.NumLlmLayers, Heads = o.NumLlmAttentionHeads, KeyValueHeads = o.NumLlmKvHeads,
            Intermediate = o.LlmFeedForwardDim, RmsEpsilon = o.LlmRmsEpsilon, RopeTheta = o.LlmRopeTheta,
        }, layers, Tensors.Helpers.RandomHelper.CreateSeededRandom(o.Seed ?? 42));
        Layers.AddRange(layers);
    }

    private Tensor<T> FirstLogits(Tensor<T> input, bool training) =>
        ForwardNativeWaveformBatch(input, _network, (net, wave) =>
            net.FirstLogits(net.Prompt(net.Speech(TruncateAudioWaveform(wave, _options.MaxAudioLengthSeconds), training), Context)));

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return FirstLogits(input, training: false); }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : FirstLogits(input, training: true);

    private int[] Transcript(SeedAsrNetwork<T> network, Tensor<T> target) =>
        ParseVocabularyTargetTokens(target, _options.VocabSize, network.EndToken);

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target) =>
        TrainNativeWaveformBatch(
            input,
            target,
            _network,
            null,
            Transcript,
            (net, wave, transcript) => net.Loss(net.Prompt(net.Speech(TruncateAudioWaveform(wave, _options.MaxAudioLengthSeconds), training: true), Context), transcript),
            beforeTrain: net => net.Llm.Training = true,
            afterTrain: net => net.Llm.Training = false);

    /// <inheritdoc />
    /// <remarks>With <see cref="SpeechLlmTrainingScope.EncoderAdapterAndLora"/>, LUISE and the converter with the LLM fixed
    /// (§3.3); otherwise every parameter.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        var network = _network;
        if (network is null || _options.TrainingScope == SpeechLlmTrainingScope.AllParameters) return parameters;
        var trainedLayers = network.EncoderLayers.Concat(network.ConverterLayers);
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(trainedLayers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target) => EvaluateNativeWaveformBatchObjective(input, target, _network, Transcript, (net, wave, transcript) => net.Loss(net.Prompt(net.Speech(TruncateAudioWaveform(wave, _options.MaxAudioLengthSeconds), training: false), Context), transcript));

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "SeedASR-Native" : "SeedASR-ONNX", Description = "Seed-ASR: LUISE Conformer encoder, frame-splicing converter and LLM (Bai et al. 2024)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumLlmLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SeedASR<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
