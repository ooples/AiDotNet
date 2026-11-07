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
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
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
            tokens = network.Greedy(network.Prompt(network.Speech(Samples(audio), training: false), Context), _options.MaxTextLength);
        return new TranscriptionResult<T>
        {
            Text = Text(tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Length > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
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

    // The 16 kHz waveform's samples, truncated to the longest recording transcribed.
    private double[] Samples(Tensor<T> audio)
    {
        int length = Math.Min(audio.Length, SampleRate * _options.MaxAudioLengthSeconds);
        var samples = new double[length];
        var span = audio.Data.Span;
        for (int i = 0; i < length; i++) samples[i] = NumOps.ToDouble(span[i]);
        return samples;
    }

    private Tensor<T> Row(Tensor<T> input, int row, int rows)
    {
        if (rows == 1) return input;
        int samples = input.Length / rows;
        return Engine.Reshape(Engine.TensorSlice(Engine.Reshape(input, new[] { rows, samples }), new[] { row, 0 }, new[] { 1, samples }), new[] { samples });
    }

    // The logits [1, vocabulary] of the answer's first token for a waveform [S], or [batch, 1, vocabulary] for [batch, S].
    private Tensor<T> FirstLogits(Tensor<T> input, bool training)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
            outputs[r] = _network.FirstLogits(_network.Prompt(_network.Speech(Samples(Row(input, r, rows)), training), Context));
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return FirstLogits(input, training: false); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : FirstLogits(input, training: true);

    // The transcript a training target names: ids [length], or one distribution per token [length, vocabulary], read as
    // its most likely tokens; the end token is never a target.
    private int[] Transcript(SeedAsrNetwork<T> network, Tensor<T> target)
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

    private Tensor<T> BatchLoss(SeedAsrNetwork<T> network, Tensor<T> input, Tensor<T> target, bool training)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var prompt = network.Prompt(network.Speech(Samples(Row(input, r, rows)), training), Context);
            var loss = Engine.Reshape(network.Loss(prompt, Transcript(network, Row(target, r, rows))), new[] { 1 });
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
    }

    /// <inheritdoc />
    /// <remarks>The transcript's cross-entropy after the prompt, averaged over the batch.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        network.Llm.Training = true;
        try { TrainWithCustomLoss(input, _ => BatchLoss(network, input, target, training: true)); }
        finally { network.Llm.Training = false; }
        return true;
    }

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
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var network = _network ?? throw new NotSupportedException("The objective is defined on the native network.");
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        return BatchLoss(network, input, target, training: false)[0];
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "SeedASR-Native" : "SeedASR-ONNX", Description = "Seed-ASR: LUISE Conformer encoder, frame-splicing converter and LLM (Bai et al. 2024)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumLlmLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SeedASR<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
