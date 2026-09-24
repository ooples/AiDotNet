using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// wav2vec 2.0 self-supervised speech representation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// wav2vec 2.0 (Baevski et al., 2020, Meta) learns speech representations via contrastive
/// learning over quantized latent speech units. With just 10 minutes of labeled data, it
/// achieves WER 1.8% on LibriSpeech test-clean when fine-tuned for ASR. It pioneered the
/// self-supervised approach later extended by HuBERT and WavLM.
/// </para>
/// <para>
/// <b>For Beginners:</b> wav2vec 2.0 was the breakthrough that showed AI could learn to
/// understand speech with very little labeled data. It works by:
/// 1. Converting raw audio into features with a CNN
/// 2. Masking some features (hiding them)
/// 3. Learning to predict the masked parts from context
/// This is similar to how GPT predicts the next word, but for audio.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1, outputSize: 768);
/// var model = new Wav2Vec2&lt;float&gt;(arch, "wav2vec2_base.onnx");
/// var embeddings = model.ExtractEmbeddings(audioWaveform);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", "https://arxiv.org/abs/2006.11477", Year = 2020, Authors = "Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 5e-4, Variant = "base",
                Schedule = LearningRateSchedulerType.LinearWarmup, WarmupFraction = 0.08,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear, MinLearningRate = 0,
                Source = "Baevski et al. 2020, pre-training setup: Adam, warmup over the first 8% of updates to a peak of 5e-4 for BASE and 3e-4 for LARGE, then linear decay. This row is the BASE peak rate.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 3e-4, Variant = "large",
                Schedule = LearningRateSchedulerType.LinearWarmup, WarmupFraction = 0.08,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear, MinLearningRate = 0,
                Source = "Baevski et al. 2020, pre-training setup: Adam, warmup over the first 8% of updates to a peak of 5e-4 for BASE and 3e-4 for LARGE, then linear decay. This row is the LARGE peak rate.")]
public partial class Wav2Vec2<T> : AudioNeuralNetworkBase<T>, IAudioFoundationModel<T>, IPaperOptimizerVariant
{
    /// <inheritdoc />
    /// <remarks>
    /// Measured: <c>PredictCore</c> folds <c>Layers</c> and <c>PostprocessOutput</c> is the identity.
    /// <c>CreateDefaultFoundationModelLayers</c> ends with the last
    /// <c>TransformerEncoderBlock&lt;T&gt;(hiddenDim, ...)</c> and adds no head, so Predict returns
    /// contextual representations at <c>_options.HiddenDim</c>. <c>FeatureEncoderDim</c> is the CNN
    /// front-end width, projected up to <c>HiddenDim</c> before the encoder; there is no CTC
    /// vocabulary projection in this stack.
    /// </remarks>
    protected override int OutputFeatureWidth => _options.HiddenDim;

    #region Fields

    private readonly Wav2Vec2Options _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IAudioFoundationModel Properties

    /// <inheritdoc />
    public int EmbeddingDimension => _options.HiddenDim;

    /// <inheritdoc />
    public int NumLayers => _options.NumLayers;

    #endregion

    #region Constructors

    public Wav2Vec2(NeuralNetworkArchitecture<T> architecture, string modelPath, Wav2Vec2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new Wav2Vec2Options();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public Wav2Vec2(NeuralNetworkArchitecture<T> architecture, Wav2Vec2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new Wav2Vec2Options();
        _useNativeMode = true;
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    /// <summary>The variant this instance was built at, keying the paper recipe for it.</summary>
    /// <remarks>
    /// The paper states a different peak rate for BASE and LARGE, so one recipe for the class
    /// would be wrong for one of them. Read during construction, which is safe because the
    /// options are assigned before the optimizer is built.
    /// </remarks>
    public string PaperOptimizerVariant => _options.Variant;

    internal static async Task<Wav2Vec2<T>> CreateAsync(Wav2Vec2Options? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new Wav2Vec2Options();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("wav2vec2", $"wav2vec2_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: 1, outputSize: options.HiddenDim);
        return new Wav2Vec2<T>(arch, mp, options);
    }

    #endregion

    #region IAudioFoundationModel

    /// <inheritdoc />
    public Tensor<T> ExtractEmbeddings(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audio) : Predict(audio);
    }

    /// <inheritdoc />
    public Task<Tensor<T>> ExtractEmbeddingsAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
        => Task.Run(() => ExtractEmbeddings(audio), cancellationToken);

    /// <inheritdoc />
    public Tensor<T> ExtractLayerFeatures(Tensor<T> audio, int layerIndex = -1)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return ExtractEmbeddings(audio);

        int targetLayer = layerIndex < 0 ? _options.NumLayers + layerIndex : layerIndex;
        if (targetLayer < 0 || targetLayer >= _options.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex),
                $"Layer index {layerIndex} is out of range. Valid range: [{-_options.NumLayers}, {_options.NumLayers - 1}].");

        var c = audio;
        int currentLayer = 0;
        foreach (var l in Layers)
        {
            c = l.Forward(c);
            if (l is TransformerEncoderBlock<T>)
            {
                if (currentLayer == targetLayer) return c;
                currentLayer++;
            }
        }
        return c;
    }

    /// <inheritdoc />
    public Tensor<T> ExtractWeightedFeatures(Tensor<T> audio, T[]? layerWeights = null)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return ExtractEmbeddings(audio);
        var layerOutputs = new List<Tensor<T>>();
        var c = audio;
        foreach (var l in Layers)
        {
            c = l.Forward(c);
            if (l is TransformerEncoderBlock<T>)
                layerOutputs.Add(c);
        }
        if (layerOutputs.Count == 0) return c;
        var result = new Tensor<T>(layerOutputs[0].Shape.ToArray());
        int count = layerOutputs.Count;
        for (int li = 0; li < count; li++)
        {
            T w = layerWeights is not null && li < layerWeights.Length
                ? layerWeights[li]
                : NumOps.FromDouble(1.0 / count);
            var layerOut = layerOutputs[li];
            if (layerOut.Length == result.Length && layerOut.Rank == result.Rank)
            {
                var scaled = Engine.TensorMultiplyScalar(layerOut, w);
                result = Engine.TensorAdd(result, scaled);
            }
            else
            {
                for (int i = 0; i < result.Length && i < layerOut.Length; i++)
                    result[i] = NumOps.Add(result[i], NumOps.Multiply(layerOut[i], w));
            }
        }
        return result;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultFoundationModelLayers(
            featureEncoderDim: _options.FeatureEncoderDim, hiddenDim: _options.HiddenDim,
            numLayers: _options.NumLayers, numAttentionHeads: _options.NumAttentionHeads,
            feedForwardDim: _options.FeedForwardDim, dropoutRate: _options.DropoutRate));
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "wav2vec2-Native" : "wav2vec2-ONNX",
            Description = $"wav2vec 2.0 {_options.Variant} self-supervised speech model (Baevski et al., 2020)",
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["HiddenDim"] = _options.HiddenDim.ToString();
        return m;
    }





    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Wav2Vec2<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
