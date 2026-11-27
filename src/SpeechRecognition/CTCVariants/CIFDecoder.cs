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

namespace AiDotNet.SpeechRecognition.CTCVariants;

/// <summary>
/// The CIF-based recognizer of Dong and Xu (2020): continuous integrate-and-fire between a self-attention encoder and a
/// non-autoregressive decoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Speech has far more audio frames than words. CIF learns a weight for every encoded frame,
/// adds the weights up as it moves through the audio, and each time the running total reaches one it "fires", handing
/// the decoder one summary of the frames that made up that label. The decoder then writes every label at once.</para>
/// <para>
/// The network (<see cref="CifAsrNetwork{T}"/>) is the paper's section 3.3 with the sizes of section 4: the
/// self-attention aligner's encoder (a strided convolution and a multiplicative unit over filterbanks with deltas, then
/// five self-attention layers at each of three time resolutions with pooling between them, encoding at 1/8 of the frame
/// rate), a convolutional weight predictor, CIF with threshold 1, and a two-layer non-autoregressive decoder. Training
/// scales the weights to the target length and minimizes the label-smoothed (0.2) cross-entropy on the targets ending with
/// &lt;EOS&gt;, plus 0.5 times the encoder's CTC loss and the quantity loss; inference fires once more when the residual
/// weight exceeds 0.5. Speed perturbation, SpecAugment, dropout, scheduled sampling, beam search and the language model
/// rescoring are not applied. The paper states no optimizer; AdamW at 2e-4 is the default.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Your recording: 16 kHz mono samples in [-1, 1], shaped [1, samples]. Noise stands in for it here.
/// var waveform = Tensor&lt;float&gt;.CreateRandom(1, 16000);
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 16000, outputSize: 5230);
/// var model = new CIFDecoder&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition", "https://arxiv.org/abs/1905.11235", Year = 2020, Authors = "Dong and Xu")]
[AiDotNet.Attributes.PreprocessesInput("CifAsrNetwork converts the waveform to log-mel features and feeds them to the first convolution as a [B, 1, frames, bins] image, so Layers[0] never sees the raw samples.")]
[AiDotNet.Attributes.StackInputLayout(AiDotNet.Enums.TensorAxis.Batch, AiDotNet.Enums.TensorAxis.Channels, AiDotNet.Enums.TensorAxis.Height, AiDotNet.Enums.TensorAxis.Width)]
public partial class CIFDecoder<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The output labels, blank (0), end of sentence (1) and pad (2) included.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly CIFDecoderOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private CifAsrNetwork<T>? _network;
    internal CifAsrNetwork<T>? Network => _network;
    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => false;
    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    public CIFDecoder(NeuralNetworkArchitecture<T> architecture, string modelPath, CIFDecoderOptions? options = null) : base(architecture)
    {
        _options = options ?? new CIFDecoderOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public CIFDecoder(NeuralNetworkArchitecture<T> architecture, CIFDecoderOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new CIFDecoderOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate });
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <inheritdoc />
    /// <remarks>The model's optimizer, unless a base training optimizer was set explicitly.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <summary>Transcribes audio: CIF over the encoder output, then one parallel decoder pass.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for CIF.");
        var waveform = PreprocessAudio(audio);
        List<int> tokens;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            if (_network is null) tokens = ConformerAsrNetwork<T>.GreedyCtc(Logits(waveform));
            else tokens = _network.Greedy(_network.Encode(_network.FeatureTensor(waveform)), _options.MaxTextLength);
        }
        return new TranscriptionResult<T>
        {
            Text = Text(tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Count > 0 ? 1.0 : 0.0),
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
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("This CIF model does not support streaming.");

    // Each label is its vocabulary entry when a vocabulary is configured, otherwise its Unicode code point.
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
        _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The paper network (see the class remarks). It used to be a generic Conformer CTC template.
        var o = _options;
        var layers = new List<LayerBase<T>>();
        _network = new CifAsrNetwork<T>(Engine, layers, new CifAsrConfiguration
        {
            Mels = o.NumMels, Filters = o.NumFilters, Width = o.EncoderDim, Heads = o.NumAttentionHeads, FeedForward = o.FeedForwardDim,
            SansPerResolution = o.NumLayersPerResolution, Resolutions = o.NumResolutions, PredictorWindow = o.PredictorWindow,
            DecoderLayers = o.NumDecoderLayers, VocabSize = o.VocabSize, TailThreshold = o.TailThreshold, LabelSmoothing = o.LabelSmoothing,
            CtcWeight = o.CtcWeight, QuantityWeight = o.QuantityWeight,
        });
        Layers.AddRange(layers);
    }

    private protected override int? NativeWaveformMaxAudioLengthSeconds => _network is null ? null : _options.MaxAudioLengthSeconds;

    // The decoder's logits [1, labels] for the first integrated embedding of a waveform [S] (the whole weight integrated
    // when it is under one), or [batch, 1, labels] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input) =>
        ForwardNativeWaveformBatch(input, _network, (net, wave) =>
        {
            var encoded = net.Encode(net.FeatureTensor(wave));
            return net.Decode(ParaformerNetwork<T>.Integrate(Engine, encoded, net.Alphas(encoded), 1));
        });

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input); }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input);

    private int[] Labels(Tensor<T> target) => ParseCtcOrTokenTargetLabels(target, _options.VocabSize, ConformerAsrNetwork<T>.GreedyCtc);

    /// <inheritdoc />
    /// <remarks>The paper's loss (<see cref="CifAsrNetwork{T}.Loss"/>), averaged over the batch.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target) =>
        TrainNativeWaveformBatch(input, target, _network, _optimizer, (_, row) => Labels(row),
            (net, wave, rowLabels) => net.Loss(net.Encode(net.FeatureTensor(wave)), rowLabels));

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target) =>
        EvaluateNativeWaveformBatchObjective(input, target, _network, (_, row) => Labels(row),
            (net, wave, rowLabels) => net.Loss(net.Encode(net.FeatureTensor(wave)), rowLabels));

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "CIFDecoder-Native" : "CIFDecoder-ONNX", Description = "CIF: a self-attention encoder, continuous integrate-and-fire and a non-autoregressive decoder (Dong and Xu 2020)", FeatureCount = 1, Complexity = _options.NumLayersPerResolution * _options.NumResolutions + _options.NumDecoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CIFDecoder<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
