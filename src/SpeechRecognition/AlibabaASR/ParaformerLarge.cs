using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Audio.Foundations;
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
/// Paraformer-large (Gao et al. 2023, FunASR): the released Mandarin Paraformer, a SAN-M encoder and decoder around a CIF
/// predictor.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Paraformer-large is the production-size Paraformer: it predicts how many tokens the audio
/// holds, gathers the audio belonging to each, and writes all of them in one decoder pass, which makes it many times
/// faster than a recognizer that writes one token at a time.</para>
/// <para>
/// This is the release <c>speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404</c> (about 220M parameters): 80-band
/// Kaldi filterbanks, seven frames stacked every six and mean-variance normalized; a 50-block SAN-M encoder
/// (<see cref="SanmEncoder{T}"/>: width 512, 4 heads, feed-forward 2048, FSMN kernel 11); the convolutional CIF predictor
/// (<c>CifPredictorV2</c>, context 1 each side, tail weight 0.45); and a 16-block SAN-M decoder (feed-forward, FSMN memory
/// and cross-attention) with a final feed-forward block, over 8,404 characters. Every target ends with the end token
/// (2), which the predictor counts. Training minimizes the length-normalized label-smoothed (0.1) decoder loss with the
/// glancing sampler (λ = 0.75) plus the MAE of the token count; there is no CTC branch. The release's SpecAugment on the
/// stacked features is not applied. The optimizer is Adam at 5e-4 with 30,000 warmup steps and gradient clipping at 5.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 16000, outputSize: 8404);
/// var model = new ParaformerLarge&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("FunASR: A Fundamental End-to-End Speech Recognition Toolkit", "https://arxiv.org/abs/2305.11013", Year = 2023, Authors = "Gao et al.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 5e-4, WarmupSteps = 30000, Schedule = LearningRateSchedulerType.Noam, MaxGradientNorm = 5,
                Source = "The release's config.yaml (speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404): Adam at 5e-4, warmuplr with 30000 warmup steps, grad_clip 5.")]
public partial class ParaformerLarge<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The output vocabulary, blank (0), start (1) and end (2) included.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly ParaformerLargeOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private SanmEncoder<T>? _encoder;
    private ParaformerNetwork<T>? _network;
    private Random _random = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);
    internal ParaformerNetwork<T>? Network => _network;
    internal SanmEncoder<T>? Encoder => _encoder;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public ParaformerLarge(NeuralNetworkArchitecture<T> architecture, string modelPath, ParaformerLargeOptions? options = null) : base(architecture)
    {
        _options = options ?? new ParaformerLargeOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public ParaformerLarge(NeuralNetworkArchitecture<T> architecture, ParaformerLargeOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new ParaformerLargeOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, warmupStepsOverride: _options.WarmupSteps) ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <inheritdoc />
    /// <remarks>The model's optimizer, unless a base training optimizer was set explicitly.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <summary>The encoder's CMVN shift and scale (FunASR's <c>am.mvn</c>, read by
    /// <see cref="SanmEncoder{T}.ReadNormalization"/>); identity until set.</summary>
    public (double[] Shift, double[] Scale) Normalization
    {
        get => (_encoder ?? throw new InvalidOperationException("No native network.")).Normalization;
        set => (_encoder ?? throw new InvalidOperationException("No native network.")).Normalization = value;
    }

    /// <summary>Transcribes audio: the CIF token count and acoustic embeddings, then one parallel decoder pass.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for Paraformer-large.");
        var waveform = PreprocessAudio(audio);
        List<int> tokens;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            if (_network is null || _encoder is null) tokens = ConformerAsrNetwork<T>.GreedyCtc(Logits(waveform));
            else tokens = _network.Greedy(_encoder.Encode(waveform), _options.MaxTextLength);
        }
        return new TranscriptionResult<T>
        {
            Text = Text(tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Count > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("Paraformer-large does not support streaming.");

    // Each token is its vocabulary entry when a vocabulary is configured, otherwise its Unicode code point.
    private string Text(List<int> tokens)
    {
        var vocabulary = _options.Vocabulary;
        var sb = new System.Text.StringBuilder();
        foreach (int t in tokens)
        {
            if (vocabulary is not null && t < vocabulary.Length) { string s = vocabulary[t]; if (!s.StartsWith("<", StringComparison.Ordinal)) sb.Append(s); }
            else if (t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t));
        }
        return sb.ToString().Trim();
    }

    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _encoder = null; _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The paper network (see the class remarks). It used to be a generic Paraformer template.
        var o = _options;
        var layers = new List<LayerBase<T>>();
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(o.Seed ?? 42);
        _encoder = new SanmEncoder<T>(Engine, layers, new SanmEncoderConfiguration(o.NumMels, o.LfrM, o.LfrN, o.EncoderDim, o.NumAttentionHeads,
            o.FeedForwardDim, o.NumEncoderLayers, 0, o.FsmnKernelSize, 1e-12), random);
        _network = new ParaformerNetwork<T>(Engine, layers, new ParaformerConfiguration
        {
            Width = o.EncoderDim, Heads = o.NumAttentionHeads, FeedForward = o.FeedForwardDim, DecoderLayers = o.NumDecoderLayers,
            Decoder = ParaformerDecoderKind.Sanm, Predictor = CifPredictorKind.Convolution, FsmnKernel = o.FsmnKernelSize, VocabSize = o.VocabSize,
            TailThreshold = o.TailThreshold, SamplingRatio = o.SamplingRatio, LabelSmoothing = o.LabelSmoothing, CtcWeight = 0,
            PredictorWeight = o.PredictorWeight, AppendEndToken = true, LengthNormalizedLoss = true,
        });
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((o.Seed ?? 42) + 1);
        Layers.AddRange(layers);
    }

    // The recognizer reads the raw waveform, truncated to the longest window a transcription covers.
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_network is null) return base.PreprocessAudio(rawAudio);
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

    // The decoder's logits [1, vocabulary] for the first acoustic token of a waveform [S] (at least one token is
    // integrated), or [batch, 1, vocabulary] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null || _encoder is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
        {
            var encoded = _encoder.Encode(Samples(input, r * samples, samples));
            var acoustic = _network.AcousticEmbeddings(encoded, minimumTokens: 1) ?? throw new InvalidOperationException("No acoustic token.");
            var logits = _network.Decode(encoded, acoustic);
            outputs[r] = Engine.TensorSlice(logits, new[] { 0, 0 }, new[] { 1, logits.Shape[1] });
        }
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input);

    // The labels a training target names: label ids [length], or one distribution per position [n, vocabulary] read as a
    // CTC alignment (the most likely symbol per row, repeats merged, blanks removed).
    private int[] Labels(Tensor<T> target)
    {
        if (target.Rank >= 2 && target.Shape[^1] == _options.VocabSize)
            return ConformerAsrNetwork<T>.GreedyCtc(Engine.Reshape(target, new[] { target.Length / _options.VocabSize, _options.VocabSize })).ToArray();
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

    /// <inheritdoc />
    /// <remarks>The release's loss (<see cref="ParaformerNetwork{T}.Loss"/>), averaged over the batch.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var encoder = _encoder;
        var network = _network;
        if (encoder is null || network is null) return false;
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var labels = RowLabels(input, target);
        TrainWithCustomLoss(input, _ =>
        {
            Tensor<T>? total = null;
            for (int r = 0; r < rows; r++)
            {
                var loss = network.Loss(encoder.Encode(Samples(input, r * samples, samples)), labels[r], _random, null);
                total = total is null ? loss : Engine.TensorAdd(total, loss);
            }
            return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
        });
        return true;
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes without the glancing sampler, averaged over the
    /// batch.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var encoder = _encoder ?? throw new NotSupportedException("The objective is defined on the native network.");
        var network = _network ?? throw new NotSupportedException("The objective is defined on the native network.");
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var labels = RowLabels(input, target);
        double total = 0;
        for (int r = 0; r < rows; r++)
            total += NumOps.ToDouble(network.Loss(encoder.Encode(Samples(input, r * samples, samples)), labels[r], _random, null, sample: false)[0]);
        return NumOps.FromDouble(total / rows);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "ParaformerLarge-Native" : "ParaformerLarge-ONNX", Description = "Paraformer-large: a SAN-M encoder, CIF predictor and parallel SAN-M decoder (Gao et al. 2023)", FeatureCount = 1, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ParaformerLarge<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
