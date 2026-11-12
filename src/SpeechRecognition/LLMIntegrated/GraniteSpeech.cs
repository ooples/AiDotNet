using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Audio;
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
/// Granite-speech (Saon et al., IBM 2025): a conformer CTC acoustic encoder, a window Q-Former speech modality adapter
/// and a Granite LLM with LoRA adapters, which writes the transcript of the audio it is given in its chat prompt.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Granite-speech listens with a conformer encoder, squeezes every 15 encoder frames into 3
/// vectors with a small query transformer, and places those vectors in a chat message to IBM's Granite language model,
/// which answers with the transcript.</para>
/// <para>
/// §II of the paper and the release (<c>modeling_granite_speech.py</c>, the processor and the model card) fix the
/// details. The encoder reads 80-band log-mels in pairs of stacked frames (160 values, 50 per second) through conformer
/// blocks with block self-attention of 200 frames and Shaw relative positions, and self-conditioned CTC at the middle
/// layer. The projector pads the encoder frames to windows of 15 and reads each with a two-layer Q-Former's 3 learned
/// queries (5× downsampling), then maps them into the LLM's width; they replace the <c>&lt;|audio|&gt;</c> tokens of
/// the chat prompt. The LLM is granite-3.3-instruct with its embedding, residual, attention and logits multipliers and
/// LoRA of rank 64 on the query and value projections. The paper trains the encoder with CTC (0.2 weight on the middle
/// layer), then the projector and LoRA adapters on the frozen encoder and LLM with the transcript's cross-entropy
/// (<see cref="GraniteSpeechTrainingStage"/>); a model built here starts from random weights, so the default trains
/// every parameter on the transcript.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 160000, outputSize: 49160);
/// var model = new GraniteSpeech&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Granite-speech: open-source speech-aware LLMs with strong English ASR capabilities", "https://arxiv.org/abs/2505.08699", Year = 2025, Authors = "IBM Research")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, ReferenceBatchSize = 256,
                MinLearningRate = 5e-6, Schedule = LearningRateSchedulerType.Cyclic,
                CyclicPolicy = CyclicLRScheduler.CyclicMode.Triangular,
                Phase = TrainingPhase.PreTraining,
                Source = "Saon et al. 2025, Sec. 4: the acoustic encoder trains for 20 epochs at a "
                        + "batch size of 256 utterances under a triangular learning rate schedule that "
                        + "ramps from 5e-5 to 5e-4 over the first 6 epochs and decays to 5e-6 "
                        + "thereafter.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, ReferenceBatchSize = 128,
                WarmupSteps = 1000, Phase = TrainingPhase.FineTuning,
                Source = "Saon et al. 2025, Sec. 4: the speech-aware LLM stage trains over three epochs "
                        + "and 660000 updates at a peak learning rate of 1e-4 with a 1000-step warm-up, "
                        + "at a batch size of 128 utterances.")]
public partial class GraniteSpeech<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The decoder's vocabulary.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly GraniteSpeechOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private GraniteSpeechNetwork<T>? _network;
    internal GraniteSpeechNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => false;
    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    public GraniteSpeech(NeuralNetworkArchitecture<T> architecture, string modelPath, GraniteSpeechOptions? options = null) : base(architecture)
    {
        _options = options ?? new GraniteSpeechOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public GraniteSpeech(NeuralNetworkArchitecture<T> architecture, GraniteSpeechOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new GraniteSpeechOptions(); Options = _options; _useNativeMode = true;
        // The encoder's CTC stage follows §4's acoustic-encoder recipe; the transcript stages follow the speech-aware
        // LLM recipe, with its warmup from the options.
        bool encoderStage = _options.TrainingStage == GraniteSpeechTrainingStage.EncoderCtc;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, warmupStepsOverride: encoderStage ? 0 : _options.WarmupSteps,
                phase: encoderStage ? TrainingPhase.PreTraining : TrainingPhase.FineTuning)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <inheritdoc />
    /// <remarks>The model's optimizer, unless a base training optimizer was set explicitly.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <summary>Transcribes audio: the LLM's greedy answer to the transcription prompt, at most
    /// <see cref="GraniteSpeechOptions.MaxTextLength"/> tokens.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for Granite-speech.");
        var network = _network ?? throw new NotSupportedException("Transcription runs on the native network.");
        int[] tokens;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var speech = network.Speech(Samples(audio), training: false);
            tokens = network.Greedy(network.Prompt(speech, _options.SystemPrompt, _options.Instruction), Math.Max(1, _options.MaxTextLength));
        }
        return new TranscriptionResult<T>
        {
            Text = Text(network, tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Length > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("Granite-speech does not support streaming.");

    // Granite's tokenizer decodes when the vocabulary is Granite's; otherwise each token is its vocabulary entry or code point.
    private string Text(GraniteSpeechNetwork<T> network, int[] tokens)
    {
        if (network.UsesGraniteVocabulary) return network.Tokenizer.Decode(tokens).Trim();
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
        // The encoder, projector and LLM (see the class remarks). It used to be a generic LLM-ASR template.
        var o = _options;
        var layers = new List<LayerBase<T>>();
        _network = new GraniteSpeechNetwork<T>(Engine, new GraniteSpeechConfiguration
        {
            SampleRate = o.SampleRate, Mels = o.NumMels, EncoderWidth = o.EncoderDim, EncoderLayers = o.NumEncoderLayers,
            EncoderHeads = o.NumAttentionHeads, EncoderHeadDim = o.EncoderHeadDim, FeedForwardMultiplier = o.FeedForwardMultiplier,
            ContextSize = o.AttentionContextSize, MaxRelativePosition = o.MaxRelativePosition, ConvKernel = o.ConvKernel,
            ConvExpansion = o.ConvExpansionFactor, CtcOutputs = o.CtcVocabSize, ProjectorWidth = o.ProjectorDim,
            ProjectorLayers = o.NumProjectorLayers, ProjectorHeads = o.NumProjectorHeads, ProjectorIntermediate = o.ProjectorFeedForwardDim,
            WindowSize = o.ProjectorWindowSize, DownsampleRate = o.ProjectorDownsampleRate,
            Llm = new LlamaConfiguration
            {
                VocabSize = o.VocabSize, Hidden = o.LlmDim, Layers = o.NumLlmLayers, Heads = o.NumLlmAttentionHeads, KeyValueHeads = o.NumLlmKvHeads,
                Intermediate = o.LlmFeedForwardDim, RmsEpsilon = o.LlmRmsEpsilon, RopeTheta = o.LlmRopeTheta, TieWordEmbeddings = true,
                EmbeddingMultiplier = o.EmbeddingMultiplier, ResidualMultiplier = o.ResidualMultiplier, AttentionMultiplier = o.AttentionMultiplier,
                LogitsScaling = o.LogitsScaling, LoraRank = o.LoraRank, LoraAlpha = o.LoraAlpha, LoraDropout = o.LoraDropout,
                LoraTargets = LlamaLoraTargets.Query | LlamaLoraTargets.Value,
            },
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
            outputs[r] = _network.FirstLogits(_network.Prompt(_network.Speech(Samples(Row(input, r, rows)), training), _options.SystemPrompt, _options.Instruction));
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return FirstLogits(input, training: false); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : FirstLogits(input, training: true);

    // The labels a training target names: ids [length], or one distribution per token [length, outputs], read as its
    // most likely tokens; the excluded id (the end token, or the CTC blank) is never a label.
    private int[] Labels(Tensor<T> target, int outputs, int excluded)
    {
        var span = target.Data.Span;
        if (target.Rank >= 2 && target.Shape[^1] == outputs)
        {
            var tokens = new int[target.Length / outputs];
            for (int r = 0; r < tokens.Length; r++)
            {
                int best = -1;
                for (int v = 0; v < outputs; v++)
                    if (v != excluded && (best < 0 || NumOps.GreaterThan(span[r * outputs + v], span[r * outputs + best]))) best = v;
                tokens[r] = best;
            }
            return tokens;
        }
        var ids = new int[target.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(span[i]));
            if (value < 0 || value >= outputs || (int)value == excluded)
                throw new ArgumentException($"Label {value} is outside [0, {outputs}) or is the reserved id {excluded}.", nameof(target));
            ids[i] = (int)value;
        }
        return ids;
    }

    private Tensor<T> BatchLoss(GraniteSpeechNetwork<T> network, Tensor<T> input, Tensor<T> target, bool training)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var samples = Samples(Row(input, r, rows));
            var labelsOf = Row(target, r, rows);
            Tensor<T> loss = _options.TrainingStage == GraniteSpeechTrainingStage.EncoderCtc
                ? network.CtcLoss(network.Features(samples), Labels(labelsOf, _options.CtcVocabSize, GraniteSpeechNetwork<T>.Blank), training)
                : network.Loss(network.Prompt(network.Speech(samples, training), _options.SystemPrompt, _options.Instruction),
                    Labels(labelsOf, _options.VocabSize, network.EndToken));
            loss = Engine.Reshape(loss, new[] { 1 });
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
    }

    /// <inheritdoc />
    /// <remarks>The stage's loss averaged over the batch: the transcript's cross-entropy after the prompt, or the
    /// encoder's self-conditioned CTC.</remarks>
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
    /// <remarks>The encoder for <see cref="GraniteSpeechTrainingStage.EncoderCtc"/>; the projector and the LoRA adapters
    /// for <see cref="GraniteSpeechTrainingStage.ProjectorAndLora"/>; otherwise every parameter.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        var network = _network;
        if (network is null || _options.TrainingStage == GraniteSpeechTrainingStage.Joint) return parameters;
        IEnumerable<LayerBase<T>> trainedLayers = _options.TrainingStage == GraniteSpeechTrainingStage.EncoderCtc
            ? network.EncoderLayers
            : network.ProjectorLayers.Concat(network.Llm.LoraLayers);
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
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "GraniteSpeech-Native" : "GraniteSpeech-ONNX", Description = "Granite-speech: conformer CTC encoder, window Q-Former and Granite LLM (IBM 2025)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumLlmLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GraniteSpeech<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
