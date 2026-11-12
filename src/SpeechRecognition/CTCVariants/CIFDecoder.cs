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
            else tokens = _network.Greedy(_network.Encode(_network.Features(waveform)), _options.MaxTextLength);
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

    // The decoder's logits [1, labels] for the first integrated embedding of a waveform [S] (the whole weight integrated
    // when it is under one), or [batch, 1, labels] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
        {
            var encoded = _network.Encode(_network.Features(Samples(input, r * samples, samples)));
            var logits = _network.Decode(ParaformerNetwork<T>.Integrate(Engine, encoded, _network.Alphas(encoded), 1));
            outputs[r] = logits;
        }
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input);

    // The labels a training target names: label ids [length], or one distribution per position [n, labels] read as a CTC
    // alignment (the most likely label per row, repeats merged, blanks removed).
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

    // Equation 1's loss [1], averaged over the batch.
    private Tensor<T> BatchLoss(CifAsrNetwork<T> network, Tensor<T> input, int[][] labels)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var loss = network.Loss(network.Encode(network.Features(Samples(input, r * samples, samples))), labels[r]);
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
    }

    /// <inheritdoc />
    /// <remarks>The paper's loss (<see cref="CifAsrNetwork{T}.Loss"/>), averaged over the batch.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        var labels = RowLabels(input, target);
        TrainWithCustomLoss(input, _ => BatchLoss(network, input, labels));
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
        return BatchLoss(network, input, RowLabels(input, target))[0];
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "CIFDecoder-Native" : "CIFDecoder-ONNX", Description = "CIF: a self-attention encoder, continuous integrate-and-fire and a non-autoregressive decoder (Dong and Xu 2020)", FeatureCount = 1, Complexity = _options.NumLayersPerResolution * _options.NumResolutions + _options.NumDecoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CIFDecoder<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
