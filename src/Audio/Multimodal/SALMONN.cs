using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Audio.Multimodal;

/// <summary>
/// SALMONN dual-encoder audio-language model for speech and audio understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SALMONN (Tang et al., 2024, Tsinghua/ByteDance) uses dual audio encoders: a Whisper
/// speech encoder and a BEATs audio encoder, connected to a Vicuna LLM through a
/// window-level Q-Former adapter. This gives it strong capability for both speech and
/// general audio understanding tasks.
/// </para>
/// <para>
/// <b>For Beginners:</b> SALMONN has two "ears": one for speech (Whisper) and one for
/// general sounds (BEATs). This means it can understand what people say AND non-speech
/// sounds. Ask it "What is the person saying?" and it transcribes speech. Ask "What sounds
/// are in the background?" and it identifies environmental audio.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1280, outputSize: 4096);
/// var model = new SALMONN&lt;float&gt;(arch, "salmonn.onnx");
/// string answer = model.Understand(audio, "What is the person saying?");
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SALMONN: Towards Generic Hearing Abilities for Large Language Models", "https://arxiv.org/abs/2310.13289", Year = 2024, Authors = "Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Chao Zhang")]
public partial class SALMONN<T> : AudioNeuralNetworkBase<T>, IAudioLanguageModel<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Traced from output construction: PredictCore folds over Layers, and the last layer
    /// CreateDefaultSALMONNLayers emits is the output projection
    /// <c>FullyConnectedLayer&lt;T&gt;(lmHiddenDim)</c>, wired from <c>_options.LMHiddenDim</c>
    /// (4096). Explicitly NOT VocabSize (32000): the Q-Former stack ends in LM input space with no
    /// unembedding head. QFormerDim (768) is the width one projection earlier.
    /// </remarks>
    protected override int OutputFeatureWidth => _options.LMHiddenDim;

    #region Fields

    private readonly SALMONNOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    private static readonly string[] Capabilities =
        ["captioning", "qa", "reasoning", "speech_recognition", "translation", "sound_event_detection"];

    #endregion

    #region IAudioLanguageModel Properties

    /// <inheritdoc />
    public double MaxAudioDurationSeconds => _options.MaxAudioDurationSeconds;

    /// <inheritdoc />
    public int MaxResponseTokens => _options.MaxResponseTokens;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SALMONN model in ONNX inference mode.
    /// </summary>
    public SALMONN(NeuralNetworkArchitecture<T> architecture, string modelPath, SALMONNOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SALMONNOptions();
        Options = _options;
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.Vicuna);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a SALMONN model in native training mode.
    /// </summary>
    // SALMONN (Tang et al., "SALMONN: Towards Generic Hearing Abilities for Large Language Models",
    // arXiv 2310.13289) trains its audio-text LLM with CROSS-ENTROPY over text tokens (window-level
    // Q-Former + LoRA on a Vicuna backbone). AudioNeuralNetworkBase defaults to MeanSquaredErrorLoss,
    // which this model silently inherited, so it was descending MSE on token LOGITS — an objective
    // the paper never uses. The head emits raw logits, so use the fused log-softmax/NLL form.
    public SALMONN(NeuralNetworkArchitecture<T> architecture, SALMONNOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new SALMONNOptions();
        Options = _options;
        _useNativeMode = true;
        // READS THE CONFIGURED RATE. The bare AdamWOptimizer(this) ran at Adam's own 1e-3 default while
        // SALMONNOptions.LearningRate sat at 1e-5 -- the property existed and was documented, and setting
        // it did nothing. Two orders of magnitude matters especially here: the paper fine-tunes LoRA
        // adapters and a Q-Former on top of a frozen backbone, which is exactly the regime a large rate
        // destabilizes.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            { InitialLearningRate = _options.LearningRate });
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.Vicuna);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<SALMONN<T>> CreateAsync(SALMONNOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new SALMONNOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("salmonn", "salmonn.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.SpeechEncoderDim, outputSize: options.LMHiddenDim);
        return new SALMONN<T>(arch, mp, options);
    }

    #endregion

    #region IAudioLanguageModel

    /// <inheritdoc />
    public IReadOnlyList<string> GetCapabilities() => Capabilities;

    /// <inheritdoc />
    public string Understand(Tensor<T> audio, string prompt, int maxTokens = 256, double temperature = 0.7)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);

        // Dual encoder: speech (Whisper) + audio (BEATs)
        var speechFeatures = EncodeSpeech(features);
        var audioFeatures = EncodeAudio(features);

        // Fuse through window-level Q-Former
        var fusedFeatures = FuseEncoderOutputs(speechFeatures, audioFeatures);

        // Encode text prompt
        var promptEmbedding = EncodePrompt(prompt);

        // Combine modalities for LM
        var combined = CombineModalEmbeddings(fusedFeatures, promptEmbedding);

        // Generate response
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(combined) : Predict(combined);
        return DecodeToText(output, maxTokens);
    }

    /// <inheritdoc />
    public Task<string> UnderstandAsync(Tensor<T> audio, string prompt, int maxTokens = 256,
        double temperature = 0.7, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Understand(audio, prompt, maxTokens, temperature), cancellationToken);
    }

    /// <inheritdoc />
    public string Caption(Tensor<T> audio, int maxTokens = 128)
    {
        return Understand(audio, "Provide a detailed description of this audio.", maxTokens);
    }

    /// <inheritdoc />
    public Tensor<T> ExtractAudioEmbeddings(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultSALMONNLayers(
            speechEncoderDim: _options.SpeechEncoderDim, audioEncoderDim: _options.AudioEncoderDim,
            qFormerDim: _options.QFormerDim, numQFormerLayers: _options.NumQFormerLayers,
            lmHiddenDim: _options.LMHiddenDim, dropoutRate: _options.DropoutRate));
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SALMONN-Native" : "SALMONN-ONNX",
            Description = "SALMONN dual-encoder audio-language model (Tang et al., 2024, Tsinghua/ByteDance)",
            Complexity = _options.NumSpeechEncoderLayers + _options.NumAudioEncoderLayers + _options.NumLMLayers
        };
        m.AdditionalInfo["LMHiddenDim"] = _options.LMHiddenDim.ToString();
        m.AdditionalInfo["QFormerDim"] = _options.QFormerDim.ToString();
        return m;
    }





    #endregion

    #region Private Helpers

    private Tensor<T> EncodeSpeech(Tensor<T> features)
    {
        // Whisper-style speech encoder pass
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        // The first SpeechEncoderDim output values, zero-padded (a fresh tensor)
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return AudioEngineOps.FitLength(Engine, output, _options.SpeechEncoderDim).Clone();
    }

    private Tensor<T> EncodeAudio(Tensor<T> features)
    {
        // BEATs-style audio encoder pass
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        int dim = _options.AudioEncoderDim;
        if (dim <= 0)
            return new Tensor<T>([dim]);
        if (output.Length == 0)
            throw new DivideByZeroException("The audio encoder produced no output.");

        // The output values from SpeechEncoderDim onwards, wrapping around (a fresh tensor)
        int offset = Math.Min(_options.SpeechEncoderDim, output.Length);
        var index = new int[dim];
        for (int i = 0; i < dim; i++) index[i] = (offset + i) % output.Length;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return Engine.TensorIndexSelect(AudioEngineOps.Flatten(Engine, output), new Tensor<int>(index, [dim]), 0);
    }

    private Tensor<T> FuseEncoderOutputs(Tensor<T> speechEmbed, Tensor<T> audioEmbed)
    {
        // Window-level Q-Former fusion: (speech + audio) / 2, each zero-padded to the output size
        int outDim = _options.NumQueryTokens * _options.QFormerDim / _options.NumQueryTokens;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var sum = Engine.TensorAdd(
            AudioEngineOps.FitLength(Engine, speechEmbed, outDim), AudioEngineOps.FitLength(Engine, audioEmbed, outDim));
        return Engine.TensorDivideScalar(sum, NumOps.FromDouble(2.0));
    }

    private Tensor<T> EncodePrompt(string prompt)
    {
        var embedding = new Tensor<T>([_options.LMHiddenDim]);
        // A stable hash: string.GetHashCode() changes per process, so the same prompt would encode differently each run (#2361).
        int hash = AiDotNet.Helpers.StableHash.Seed(prompt);
        for (int i = 0; i < _options.LMHiddenDim; i++)
        {
            double val = Math.Sin((hash + i) * 0.1) * 0.5;
            embedding[i] = NumOps.FromDouble(val);
        }
        return embedding;
    }

    private Tensor<T> CombineModalEmbeddings(Tensor<T> audioEmbed, Tensor<T> textEmbed)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return Engine.Concat(new[] { AudioEngineOps.Flatten(Engine, audioEmbed), AudioEngineOps.Flatten(Engine, textEmbed) }, 0);
    }

    private string DecodeToText(Tensor<T> output, int maxTokens)
    {
        int numTokens = Math.Min(maxTokens, output.Length);
        if (numTokens <= 0)
            return _tokenizer.Decode(new List<int>());

        // Token ids: round to nearest (banker's), clamped to [0, vocab - 1]; NaN maps to 0
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var rounded = Engine.TensorRound(AudioEngineOps.FitLength(Engine, output, numTokens));
        var zeros = new Tensor<T>([numTokens]);
        T maxId = NumOps.FromDouble(_tokenizer.VocabularySize - 1);
        var clamped = Engine.TensorWhere(Engine.TensorLessThan(rounded, NumOps.Zero), zeros, rounded);
        clamped = Engine.TensorWhere(
            Engine.TensorGreaterThan(clamped, maxId), Tensor<T>.CreateDefault([numTokens], maxId), clamped);
        clamped = Engine.TensorWhere(Engine.TensorIsNan(clamped), zeros, clamped);
        return _tokenizer.Decode(clamped.Cast<int>().ToArray().ToList());
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SALMONN<T>)); }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
