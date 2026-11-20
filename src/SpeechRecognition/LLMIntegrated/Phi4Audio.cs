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
/// Phi-4-multimodal's speech recognition (Abdin et al., Microsoft 2025): a conformer audio encoder and a projector that
/// feed Phi-4-Mini, which carries the speech LoRA adapters ("Mixture of LoRAs") and writes the transcript.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Phi-4-multimodal hears through an audio encoder that turns each 80 ms of speech into one
/// vector, maps those vectors into the language model's embedding space with a small two-layer network, and asks the
/// Phi-4-Mini language model to "Transcribe the audio clip into text." Speech-specific LoRA adapters adjust the language
/// model while leaving its text abilities untouched.</para>
/// <para>
/// §2.2.1 and the release (<c>speech_conformer_encoder.py</c>, <c>modeling_phi4mm.py</c>, <c>processing_phi4mm.py</c>)
/// fix the details: SpeechLib log filterbanks (pre-emphasis 0.97, Hamming 400, 512-point FFT, 80 bands to 7,690 Hz); a
/// global mean-variance normalization; NeMo's depthwise-striding convolutions (8× subsampling); 24 conformer blocks of
/// width 1,024 with GLU feed-forward blocks, T5 relative attention biases and a causal depthwise convolution, attending
/// fully within chunks of 500 frames; a GELU MLP to 3,072; and Phi-4-Mini with partial rotary embeddings, LongRoPE and the
/// speech LoRA (rank 320, alpha 640) on its fused qkv, o, gate_up and down projections. The prompt is
/// <c>&lt;|user|&gt;&lt;|audio_1|&gt;Transcribe the audio clip into text.&lt;|end|&gt;&lt;|assistant|&gt;</c> and the
/// answer ends with <c>&lt;|end|&gt;&lt;|endoftext|&gt;</c>. §2.2.2 trains the encoder and projector on a frozen decoder,
/// then the projector and the speech LoRA (<see cref="Phi4AudioTrainingStage"/>); a model built here starts from random
/// weights, so the default trains every parameter.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 160000, outputSize: 200064);
/// var model = new Phi4Audio&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Phi-4-mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs", "https://arxiv.org/abs/2503.01743", Year = 2025, Authors = "Abdin et al.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 4e-5, Phase = TrainingPhase.PreTraining,
                Source = "Abdin et al. 2025, Sec. 2.2.2: speech pre-training updates the audio encoder and projector at a "
                        + "learning rate of 4e-5 for 50k steps with the language decoder frozen. The paper names no "
                        + "optimizer; AdamW is the release's sample_finetune_speech.py choice.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.95, Epsilon = 1e-7, WeightDecay = 0.01,
                MaxGradientNorm = 1, Phase = TrainingPhase.FineTuning,
                Source = "Abdin et al. 2025, Sec. 2.2.2: speech post-training updates the audio projector and LoRA_A at a "
                        + "learning rate of 1e-4 for 50k steps with the encoder frozen. Betas, epsilon, weight decay and "
                        + "clipping are the release's sample_finetune_speech.py (adamw_torch).")]
public partial class Phi4Audio<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    // The release's rope_scaling.long_factor (short_factor is all ones), for 48 rotary frequencies.
    private static readonly double[] LongRopeLongFactors =
    {
        1.0, 1.118320672, 1.250641126, 1.398617824, 1.564103225, 1.74916897, 1.956131817, 2.187582649, 2.446418898, 2.735880826,
        3.059592084, 3.421605075, 3.826451687, 4.279200023, 4.785517845, 5.351743533, 5.984965424, 6.693110555, 7.485043894,
        8.370679318, 9.36110372, 10.4687158, 11.70738129, 13.09260651, 14.64173252, 16.37415215, 18.31155283, 20.47818807,
        22.90118105, 25.61086418, 28.64115884, 32.03, 32.1, 32.13, 32.23, 32.6, 32.61, 32.64, 32.66, 32.7, 32.71, 32.93, 32.97,
        33.28, 33.49, 33.5, 44.16, 47.77,
    };

    /// <inheritdoc />
    /// <remarks>The decoder's vocabulary.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly Phi4AudioOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private Phi4AudioNetwork<T>? _network;
    internal Phi4AudioNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => false;
    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    public Phi4Audio(NeuralNetworkArchitecture<T> architecture, string modelPath, Phi4AudioOptions? options = null) : base(architecture)
    {
        _options = options ?? new Phi4AudioOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public Phi4Audio(NeuralNetworkArchitecture<T> architecture, Phi4AudioOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new Phi4AudioOptions(); Options = _options; _useNativeMode = true;
        // Encoder alignment follows the speech pre-training recipe; the transcript stages the post-training recipe.
        var phase = _options.TrainingStage == Phi4AudioTrainingStage.EncoderAlignment ? TrainingPhase.PreTraining : TrainingPhase.FineTuning;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, phase: phase)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <inheritdoc />
    /// <remarks>The model's optimizer, unless a base training optimizer was set explicitly.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <summary>Transcribes audio: the decoder's greedy answer to the ASR prompt, at most
    /// <see cref="Phi4AudioOptions.MaxTextLength"/> tokens.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for Phi-4-multimodal.");
        var network = _network ?? throw new NotSupportedException("Transcription runs on the native network.");
        int[] tokens;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            tokens = network.Greedy(network.Prompt(network.Speech(Samples(audio)), _options.Instruction), Math.Max(1, _options.MaxTextLength));
        return new TranscriptionResult<T>
        {
            Text = Text(network, tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Length > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("Phi-4-multimodal does not support streaming.");

    // Phi's tokenizer decodes when the vocabulary is Phi's; otherwise each token is its vocabulary entry or code point.
    private string Text(Phi4AudioNetwork<T> network, int[] tokens)
    {
        if (network.UsesPhiVocabulary) return network.Tokenizer.Decode(tokens).Trim();
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
        // The audio encoder, projector and Phi-4-Mini (see the class remarks). It used to be a generic LLM-ASR template.
        var o = _options;
        var llm = new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.LlmDim, Layers = o.NumLlmLayers, Heads = o.NumLlmAttentionHeads, KeyValueHeads = o.NumLlmKvHeads,
            Intermediate = o.LlmFeedForwardDim, RmsEpsilon = o.LlmRmsEpsilon, RopeTheta = o.LlmRopeTheta, TieWordEmbeddings = true,
            PartialRotaryFactor = o.PartialRotaryFactor, LoraRank = o.LoraRank, LoraAlpha = o.LoraAlpha, LoraDropout = o.LoraDropout,
            LoraTargets = LlamaLoraTargets.All, FusedLora = true, RopeOriginalMaxPositions = 4096,
        };
        if (o.UseLongRope)
        {
            if (llm.RotaryDim / 2 != LongRopeLongFactors.Length)
                throw new ArgumentException($"The release's LongRoPE factors cover {LongRopeLongFactors.Length} rotary frequencies; disable UseLongRope for other head widths.", nameof(Options));
            llm = llm with
            {
                LongRopeShortFactors = Enumerable.Repeat(1.0, LongRopeLongFactors.Length).ToArray(),
                LongRopeLongFactors = LongRopeLongFactors, LongRopeMaxPositions = 131072,
            };
        }
        var layers = new List<LayerBase<T>>();
        _network = new Phi4AudioNetwork<T>(Engine, new Phi4AudioConfiguration
        {
            SampleRate = o.SampleRate, Mels = o.NumMels, EncoderWidth = o.EncoderDim, EncoderLayers = o.NumEncoderLayers,
            EncoderHeads = o.NumAttentionHeads, EncoderFeedForward = o.EncoderFeedForwardDim, SubsamplingChannels = o.SubsamplingChannels,
            ConvKernel = o.ConvKernel, RelativeBiasMaxDistance = o.RelativeBiasMaxDistance, Llm = llm,
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
    private Tensor<T> FirstLogits(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
            outputs[r] = _network.FirstLogits(_network.Prompt(_network.Speech(Samples(Row(input, r, rows))), _options.Instruction));
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return FirstLogits(input); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : FirstLogits(input);

    // The transcript a training target names: ids [length], or one distribution per token [length, vocabulary], read as
    // its most likely tokens; end tokens are never transcript tokens.
    private int[] Transcript(Phi4AudioNetwork<T> network, Tensor<T> target)
    {
        int vocabulary = _options.VocabSize;
        var ends = new HashSet<int>(network.EndTokens);
        var span = target.Data.Span;
        if (target.Rank >= 2 && target.Shape[^1] == vocabulary)
        {
            var tokens = new int[target.Length / vocabulary];
            for (int r = 0; r < tokens.Length; r++)
            {
                int best = -1;
                for (int v = 0; v < vocabulary; v++)
                    if (!ends.Contains(v) && (best < 0 || NumOps.GreaterThan(span[r * vocabulary + v], span[r * vocabulary + best]))) best = v;
                tokens[r] = best;
            }
            return tokens;
        }
        var ids = new int[target.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(span[i]));
            if (value < 0 || value >= vocabulary || ends.Contains((int)value))
                throw new ArgumentException($"Token {value} is outside the vocabulary or an end token.", nameof(target));
            ids[i] = (int)value;
        }
        return ids;
    }

    private Tensor<T> BatchLoss(Phi4AudioNetwork<T> network, Tensor<T> input, Tensor<T> target)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var prompt = network.Prompt(network.Speech(Samples(Row(input, r, rows))), _options.Instruction);
            var loss = Engine.Reshape(network.Loss(prompt, Transcript(network, Row(target, r, rows))), new[] { 1 });
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
    }

    /// <inheritdoc />
    /// <remarks>The transcript's cross-entropy after the prompt, averaged over the batch, with LoRA dropout active.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        network.Llm.Training = true;
        try { TrainWithCustomLoss(input, _ => BatchLoss(network, input, target)); }
        finally { network.Llm.Training = false; }
        return true;
    }

    /// <inheritdoc />
    /// <remarks>The encoder and projector for <see cref="Phi4AudioTrainingStage.EncoderAlignment"/>; the projector and the
    /// speech LoRA for <see cref="Phi4AudioTrainingStage.ProjectorAndLora"/>; otherwise every parameter.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        var network = _network;
        if (network is null || _options.TrainingStage == Phi4AudioTrainingStage.Joint) return parameters;
        IEnumerable<LayerBase<T>> trainedLayers = _options.TrainingStage == Phi4AudioTrainingStage.EncoderAlignment
            ? network.EncoderLayers.Concat(network.ProjectorLayers)
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
        return BatchLoss(network, input, target)[0];
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "Phi4Audio-Native" : "Phi4Audio-ONNX", Description = "Phi-4-multimodal speech: conformer encoder, projector and Phi-4-Mini with speech LoRA (Microsoft 2025)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumLlmLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Phi4Audio<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
