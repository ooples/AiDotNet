using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Tacotron2 attention-based text-to-speech model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Tacotron2 is a classic neural TTS model that generates mel spectrograms from text.
/// It uses an encoder-attention-decoder architecture with:
/// <list type="bullet">
/// <item>Character/phoneme encoder with convolutional layers</item>
/// <item>Location-sensitive attention for alignment</item>
/// <item>Autoregressive LSTM decoder</item>
/// <item>Post-net for mel spectrogram refinement</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Tacotron2 is a two-stage TTS system:
///
/// Stage 1 (Tacotron2): Text -> Mel Spectrogram
/// Stage 2 (Vocoder): Mel Spectrogram -> Audio Waveform
///
/// Key characteristics:
/// - Autoregressive: Generates one mel frame at a time
/// - Attention-based: Learns to align text with audio
/// - High quality but slower than parallel models like VITS
///
/// Two ways to use this class:
/// 1. ONNX Mode: Load pretrained Tacotron2 models for inference
/// 2. Native Mode: Train your own TTS model from scratch
///
/// ONNX Mode Example:
/// <code>
/// var tacotron = new Tacotron2Model&lt;float&gt;(
///     architecture,
///     acousticModelPath: "tacotron2.onnx",
///     vocoderPath: "hifigan.onnx");
/// var audio = tacotron.Synthesize("Hello, world!");
/// </code>
///
/// Training Mode Example:
/// <code>
/// var tacotron = new Tacotron2Model&lt;float&gt;(architecture);
/// tacotron.Train(phonemeInput, expectedMelSpectrogram);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.TextToSpeech)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions", "https://arxiv.org/abs/1712.05884", Year = 2018, Authors = "Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu")]
public class Tacotron2Model<T> : AudioNeuralNetworkBase<T>, ITextToSpeech<T>
{
    private readonly Tacotron2ModelOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX models (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// Path to the acoustic model ONNX file.
    /// </summary>
    private readonly string? _acousticModelPath;

    /// <summary>
    /// Path to the vocoder ONNX file.
    /// </summary>
    private readonly string? _vocoderPath;

    /// <summary>
    /// ONNX acoustic model (Tacotron2).
    /// </summary>
    private readonly OnnxModel<T>? _acousticModel;

    /// <summary>
    /// ONNX vocoder model (HiFi-GAN or WaveGlow).
    /// </summary>
    private readonly OnnxModel<T>? _vocoder;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// Character/phoneme embedding layer.
    /// </summary>
    private ILayer<T>? _embedding;

    /// <summary>
    /// Encoder convolutional layers.
    /// </summary>
    private readonly List<ILayer<T>> _encoderConvLayers = [];

    /// <summary>
    /// Encoder LSTM layer.
    /// </summary>
    private ILayer<T>? _encoderLstm;

    /// <summary>
    /// Attention layers.
    /// </summary>
    private readonly List<ILayer<T>> _attentionLayers = [];

    /// <summary>
    /// Decoder LSTM layers.
    /// </summary>
    private readonly List<ILayer<T>> _decoderLstmLayers = [];

    /// <summary>
    /// Post-net layers for mel refinement.
    /// </summary>
    private readonly List<ILayer<T>> _postNetLayers = [];

    /// <summary>
    /// Stop token prediction layer.
    /// </summary>
    private ILayer<T>? _stopTokenLayer;

    /// <summary>
    /// Griffin-Lim vocoder fallback.
    /// </summary>
    private readonly GriffinLim<T>? _griffinLim;

    #endregion

    #region Shared Fields

    /// <summary>
    /// Text preprocessor for phoneme conversion.
    /// </summary>
    private readonly TtsPreprocessor _preprocessor;

    /// <summary>
    /// Optimizer for training.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Whether the model has been disposed.
    /// </summary>
    private bool _disposed;

    #endregion

    #region Model Architecture Parameters

    /// <summary>
    /// Character/phoneme vocabulary size.
    /// </summary>
    private int _vocabSize;

    /// <summary>
    /// Embedding dimension.
    /// </summary>
    private int _embeddingDim;

    /// <summary>
    /// Encoder hidden dimension.
    /// </summary>
    private int _encoderDim;

    /// <summary>
    /// Decoder hidden dimension.
    /// </summary>
    private int _decoderDim;

    /// <summary>
    /// Attention dimension.
    /// </summary>
    private int _attentionDim;

    /// <summary>
    /// Attention location filters.
    /// </summary>
    private int _attentionFilters;

    /// <summary>
    /// Pre-net dimension.
    /// </summary>
    private int _prenetDim;

    /// <summary>
    /// Post-net embedding dimension.
    /// </summary>
    private int _postnetEmbeddingDim;

    /// <summary>
    /// Number of encoder convolutional layers.
    /// </summary>
    private int _numEncoderConvLayers;

    /// <summary>
    /// Number of post-net convolutional layers.
    /// </summary>
    private int _numPostnetConvLayers;

    /// <summary>
    /// Number of mel frames to output per decoder step.
    /// </summary>
    private int _numMelsPerFrame;

    /// <summary>
    /// Maximum decoder steps.
    /// </summary>
    private int _maxDecoderSteps;

    /// <summary>
    /// Decoder stop threshold.
    /// </summary>
    private double _stopThreshold;

    /// <summary>
    /// FFT size for Griffin-Lim.
    /// </summary>
    private int _fftSize;

    /// <summary>
    /// Hop length for audio synthesis.
    /// </summary>
    private int _hopLength;

    /// <summary>
    /// Griffin-Lim iterations.
    /// </summary>
    private int _griffinLimIterations;

    /// <summary>
    /// Speaking rate multiplier.
    /// </summary>
    private double _speakingRate;

    #endregion

    #region ITextToSpeech Properties

    /// <summary>
    /// Gets the list of available built-in voices.
    /// </summary>
    public IReadOnlyList<VoiceInfo<T>> AvailableVoices { get; }

    /// <summary>
    /// Gets whether this model supports voice cloning from reference audio.
    /// </summary>
    public bool SupportsVoiceCloning => false;

    /// <summary>
    /// Gets whether this model supports emotional expression control.
    /// </summary>
    public bool SupportsEmotionControl => false;

    /// <summary>
    /// Gets whether this model supports streaming audio generation.
    /// </summary>
    public bool SupportsStreaming => false;

    #endregion

    #region Public Properties

    /// <summary>
    /// Gets whether the model is ready for synthesis.
    /// </summary>
    public bool IsReady => _useNativeMode ||
        (_acousticModel?.IsLoaded == true && (_vocoder?.IsLoaded == true || _griffinLim is not null));

    /// <summary>
    /// Gets the maximum decoder steps.
    /// </summary>
    public int MaxDecoderSteps => _maxDecoderSteps;

    /// <summary>
    /// Tacotron2's autoregressive decoder keeps several component views over
    /// the published layer graph. Rebuilding those aliases through the normal
    /// layer deserializer preserves trained predictions exactly; tensor-only
    /// COW rebinding leaves a small post-training decoder drift.
    /// </summary>
    protected override bool SupportsCopyOnWriteDeepCopy => false;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Tacotron2 model for ONNX inference with pretrained models.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="acousticModelPath">Path to the Tacotron2 ONNX model.</param>
    /// <param name="vocoderPath">Optional path to vocoder ONNX (HiFi-GAN/WaveGlow). Uses Griffin-Lim if null.</param>
    /// <param name="sampleRate">Output sample rate in Hz. Default is 22050.</param>
    /// <param name="numMels">Number of mel spectrogram channels. Default is 80.</param>
    /// <param name="speakingRate">Speaking rate multiplier. Default is 1.0.</param>
    /// <param name="maxDecoderSteps">Maximum decoder steps. Default is 1000.</param>
    /// <param name="stopThreshold">Stop token threshold. Default is 0.5.</param>
    /// <param name="fftSize">FFT size for Griffin-Lim. Default is 1024.</param>
    /// <param name="hopLength">Hop length. Default is 256.</param>
    /// <param name="griffinLimIterations">Griffin-Lim iterations. Default is 60.</param>
    /// <param name="onnxOptions">ONNX runtime options.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor with pretrained Tacotron2 models.
    ///
    /// You need at least an acoustic model (Tacotron2).
    /// The vocoder is optional - Griffin-Lim can be used as fallback.
    ///
    /// Example:
    /// <code>
    /// var tacotron = new Tacotron2Model&lt;float&gt;(
    ///     architecture,
    ///     acousticModelPath: "tacotron2.onnx",
    ///     vocoderPath: "hifigan.onnx");
    /// </code>
    /// </para>
    /// </remarks>
    public Tacotron2Model(
        NeuralNetworkArchitecture<T> architecture,
        string acousticModelPath,
        string? vocoderPath = null,
        int sampleRate = 22050,
        int numMels = 80,
        double speakingRate = 1.0,
        int maxDecoderSteps = 1000,
        double stopThreshold = 0.5,
        int fftSize = 1024,
        int hopLength = 256,
        int griffinLimIterations = 60,
        OnnxModelOptions? onnxOptions = null,
        Tacotron2ModelOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new Tacotron2ModelOptions();
        Options = _options;
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));
        if (acousticModelPath is null)
            throw new ArgumentNullException(nameof(acousticModelPath));

        _useNativeMode = false;
        _acousticModelPath = acousticModelPath;
        _vocoderPath = vocoderPath;

        // Store parameters
        SampleRate = sampleRate;
        NumMels = numMels;
        _speakingRate = speakingRate;
        _maxDecoderSteps = maxDecoderSteps;
        _stopThreshold = stopThreshold;
        _fftSize = fftSize;
        _hopLength = hopLength;
        _griffinLimIterations = griffinLimIterations;

        // Default architecture parameters (standard Tacotron2)
        _vocabSize = 148; // Standard phoneme vocabulary
        _embeddingDim = 512;
        _encoderDim = 512;
        _decoderDim = 1024;
        _attentionDim = 128;
        _attentionFilters = 32;
        _prenetDim = 256;
        _postnetEmbeddingDim = 512;
        _numEncoderConvLayers = 3;
        _numPostnetConvLayers = 5;
        _numMelsPerFrame = 2;

        // Initialize preprocessor
        _preprocessor = new TtsPreprocessor();

        // Load ONNX models
        var onnxOpts = onnxOptions ?? new OnnxModelOptions();
        _acousticModel = new OnnxModel<T>(acousticModelPath, onnxOpts);

        if (vocoderPath is not null && vocoderPath.Length > 0)
        {
            _vocoder = new OnnxModel<T>(vocoderPath, onnxOpts);
        }
        else
        {
            // Use Griffin-Lim as fallback vocoder
            _griffinLim = new GriffinLim<T>(
                nFft: fftSize,
                hopLength: hopLength,
                iterations: griffinLimIterations);
        }

        // Initialize available voices
        AvailableVoices = GetDefaultVoices();

        // Default loss function (MSE is standard for TTS mel-spectrogram prediction)
        _lossFunction = new MeanSquaredErrorLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Creates a Tacotron2 model for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sampleRate">Output sample rate in Hz. Default is 22050.</param>
    /// <param name="numMels">Number of mel spectrogram channels. Default is 80.</param>
    /// <param name="speakingRate">Speaking rate multiplier. Default is 1.0.</param>
    /// <param name="vocabSize">Character/phoneme vocabulary size. Default is 148.</param>
    /// <param name="embeddingDim">Embedding dimension. Default is 512.</param>
    /// <param name="encoderDim">Encoder hidden dimension. Default is 512.</param>
    /// <param name="decoderDim">Decoder hidden dimension. Default is 1024.</param>
    /// <param name="attentionDim">Attention dimension. Default is 128.</param>
    /// <param name="attentionFilters">Number of attention location filters. Default is 32.</param>
    /// <param name="prenetDim">Pre-net dimension. Default is 256.</param>
    /// <param name="postnetEmbeddingDim">Post-net embedding dimension. Default is 512.</param>
    /// <param name="numEncoderConvLayers">Number of encoder conv layers. Default is 3.</param>
    /// <param name="numPostnetConvLayers">Number of post-net conv layers. Default is 5.</param>
    /// <param name="numMelsPerFrame">Mel frames per decoder step. Default is 2.</param>
    /// <param name="maxDecoderSteps">Maximum decoder steps. Default is 1000.</param>
    /// <param name="stopThreshold">Stop token threshold. Default is 0.5.</param>
    /// <param name="fftSize">FFT size for Griffin-Lim. Default is 1024.</param>
    /// <param name="hopLength">Hop length. Default is 256.</param>
    /// <param name="griffinLimIterations">Griffin-Lim iterations. Default is 60.</param>
    /// <param name="optimizer">Optimizer for training. If null, uses Adam.</param>
    /// <param name="lossFunction">Loss function for training. If null, uses MSE.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train your own Tacotron2 model.
    ///
    /// Training Tacotron2 requires:
    /// 1. Paired text-audio data with aligned phoneme sequences
    /// 2. GPU training is recommended (many hours of training)
    /// 3. Teacher forcing is used during training
    ///
    /// Example:
    /// <code>
    /// var tacotron = new Tacotron2Model&lt;float&gt;(
    ///     architecture,
    ///     embeddingDim: 512,
    ///     encoderDim: 512,
    ///     decoderDim: 1024);
    ///
    /// // Training loop
    /// tacotron.Train(phonemeInput, expectedMelSpectrogram);
    /// </code>
    /// </para>
    /// </remarks>
    public Tacotron2Model(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 22050,
        int numMels = 80,
        double speakingRate = 1.0,
        int vocabSize = 148,
        int embeddingDim = 512,
        int encoderDim = 512,
        int decoderDim = 1024,
        int attentionDim = 128,
        int attentionFilters = 32,
        int prenetDim = 256,
        int postnetEmbeddingDim = 512,
        int numEncoderConvLayers = 3,
        int numPostnetConvLayers = 5,
        int numMelsPerFrame = 2,
        int maxDecoderSteps = 1000,
        double stopThreshold = 0.5,
        int fftSize = 1024,
        int hopLength = 256,
        int griffinLimIterations = 60,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        Tacotron2ModelOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new Tacotron2ModelOptions();
        Options = _options;
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));

        _useNativeMode = true;

        // Store parameters
        SampleRate = sampleRate;
        NumMels = numMels;
        _speakingRate = speakingRate;
        _vocabSize = vocabSize;
        _embeddingDim = embeddingDim;
        _encoderDim = encoderDim;
        _decoderDim = decoderDim;
        _attentionDim = attentionDim;
        _attentionFilters = attentionFilters;
        _prenetDim = prenetDim;
        _postnetEmbeddingDim = postnetEmbeddingDim;
        _numEncoderConvLayers = numEncoderConvLayers;
        _numPostnetConvLayers = numPostnetConvLayers;
        _numMelsPerFrame = numMelsPerFrame;
        _maxDecoderSteps = maxDecoderSteps;
        _stopThreshold = stopThreshold;
        _fftSize = fftSize;
        _hopLength = hopLength;
        _griffinLimIterations = griffinLimIterations;

        // Initialize preprocessor
        _preprocessor = new TtsPreprocessor();

        // Create Griffin-Lim vocoder
        _griffinLim = new GriffinLim<T>(
            nFft: fftSize,
            hopLength: hopLength,
            iterations: griffinLimIterations);

        // Initialize available voices
        AvailableVoices = GetDefaultVoices();

        // Initialize training components
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        // Paper training configuration (Shen et al. 2018, sec. 3): "Adam optimizer with beta1 = 0.9,
        // beta2 = 0.999, eps = 10^-6 and a learning rate of 10^-3 exponentially decaying to 10^-5".
        // The optimizer was previously built bare, so none of those values applied and the decoder ran
        // on framework defaults. Callers can still supply their own optimizer.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = 1e-3,
                MinLearningRate = 1e-5,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-6,
            });

        InitializeNativeLayers();
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes layers for ONNX inference mode.
    /// </summary>
    protected override void InitializeLayers()
    {
        // ONNX mode - no native layers needed
    }

    /// <summary>
    /// Initializes layers for native training mode.
    /// </summary>
    private void InitializeNativeLayers()
    {
        List<ILayer<T>> layers;
        bool builtDefaultLayers = false;
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            layers = Architecture.Layers.ToList();
            // First layer should be embedding if present
            if (layers.Count > 0 && layers[0] is EmbeddingLayer<T> emb)
            {
                _embedding = emb;
                layers.RemoveAt(0);
                Layers.Add(_embedding);
            }
        }
        else
        {
            builtDefaultLayers = true;
            _embedding = new EmbeddingLayer<T>(_vocabSize, _embeddingDim);
            Layers.Add(_embedding);

            layers = LayerHelper<T>.CreateTacotron2Layers(
                vocabSize: _vocabSize, embeddingDim: _embeddingDim, encoderDim: _encoderDim,
                decoderDim: _decoderDim, attentionDim: _attentionDim,
                attentionFilters: _attentionFilters, prenetDim: _prenetDim,
                numMels: NumMels, numMelsPerFrame: _numMelsPerFrame,
                numEncoderConvLayers: _numEncoderConvLayers,
                numPostnetConvLayers: _numPostnetConvLayers,
                postnetEmbeddingDim: _postnetEmbeddingDim).ToList();
        }

        _encoderConvLayers.Clear();
        _attentionLayers.Clear();
        _decoderLstmLayers.Clear();
        _postNetLayers.Clear();
        Layers.AddRange(layers);

        BindNativeLayersFromPublishedList();
        if (builtDefaultLayers)
            ResolveDefaultLayerShapes();
    }

    /// <summary>
    /// Rebinds the private paper-component views to the framework-visible
    /// <see cref="Layers"/> instances.
    /// </summary>
    /// <remarks>
    /// Base deserialization replaces every element in <see cref="Layers"/>.
    /// Keeping the constructor-created objects in these private lists made
    /// Predict bypass the restored/trained weights even though GetParameters
    /// correctly reported them. The native forward and all framework services
    /// must reference the same layer instances.
    /// </remarks>
    private void BindNativeLayersFromPublishedList()
    {
        _embedding = null;
        _encoderLstm = null;
        _stopTokenLayer = null;
        _encoderConvLayers.Clear();
        _attentionLayers.Clear();
        _decoderLstmLayers.Clear();
        _postNetLayers.Clear();

        // Distribute to internal sub-lists for forward pass
        int idx = 0;
        if (idx < Layers.Count && Layers[idx] is EmbeddingLayer<T> embedding)
            _embedding = embedding;
        if (_embedding is not null)
            idx++;

        for (int i = 0; i < _numEncoderConvLayers && idx < Layers.Count; i++)
            _encoderConvLayers.Add(Layers[idx++]);
        if (idx < Layers.Count)
            _encoderLstm = Layers[idx++];
        for (int i = 0; i < 4 && idx < Layers.Count; i++)
            _attentionLayers.Add(Layers[idx++]);
        for (int i = 0; i < 5 && idx < Layers.Count; i++) // prenet(2) + recurrent projections(2) + mel(1)
            _decoderLstmLayers.Add(Layers[idx++]);
        if (idx < Layers.Count)
            _stopTokenLayer = Layers[idx++];
        while (idx < Layers.Count)
            _postNetLayers.Add(Layers[idx++]);
    }

    /// <summary>
    /// Resolves every default layer's input width up front instead of letting it be inferred from
    /// whatever tensor happens to arrive first.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>DenseLayer</c> is always lazy: it carries a -1 input placeholder and resolves the real width
    /// on its first forward. That is fine while a model stays in memory, but a deserialized model has
    /// not run a forward pass yet, so its layers are still unresolved, report a parameter count of 0,
    /// and <c>SetParameters</c> skips them without complaint — the trained weights are dropped on the
    /// floor and a clone predicts differently from the model it was copied from. This is the #1221
    /// lazy-layer class of failure.
    /// </para>
    /// <para>
    /// Every width here is already fixed by the constructor arguments, so none of it needs to be
    /// discovered at runtime. The pairs that are not simply "previous layer's output" come from the
    /// two concatenations in the decoder loop: the decoder LSTM consumes pre-net output joined to the
    /// attention context, and both the mel projection and the stop-token head consume the decoder
    /// state joined to that same context.
    /// </para>
    /// <para>
    /// Only the layers this class builds are resolved. A caller supplying its own
    /// <c>Architecture.Layers</c> may use entirely different widths, so those are left to resolve
    /// themselves as before.
    /// </para>
    /// </remarks>
    private void ResolveDefaultLayerShapes()
    {
        int encoderOut = _encoderDim * 2;
        int decoderIn = _prenetDim + encoderOut;
        int contextIn = _decoderDim + encoderOut;

        foreach (var conv in _encoderConvLayers)
        {
            ResolveLayerInput(conv, _embeddingDim);
        }

        ResolveLayerInput(_encoderLstm, _embeddingDim);

        if (_attentionLayers.Count >= 4)
        {
            ResolveLayerInput(_attentionLayers[0], _decoderDim);       // query projection
            ResolveLayerInput(_attentionLayers[1], encoderOut);        // key projection
            ResolveLayerInput(_attentionLayers[2], _attentionFilters); // location projection
            ResolveLayerInput(_attentionLayers[3], _attentionDim);     // energy projection
        }

        if (_decoderLstmLayers.Count >= 5)
        {
            ResolveLayerInput(_decoderLstmLayers[0], NumMels * _numMelsPerFrame);
            ResolveLayerInput(_decoderLstmLayers[1], _prenetDim);
            ResolveLayerInput(_decoderLstmLayers[2], decoderIn);
            ResolveLayerInput(_decoderLstmLayers[3], _decoderDim);
            ResolveLayerInput(_decoderLstmLayers[4], contextIn); // mel projection
        }

        ResolveLayerInput(_stopTokenLayer, contextIn);

        for (int i = 0; i < _postNetLayers.Count; i++)
        {
            // The post-net refines the mel spectrogram, so it starts at NumMels and stays at the
            // post-net width until the last layer projects back down.
            ResolveLayerInput(_postNetLayers[i], i == 0 ? NumMels : _postnetEmbeddingDim);
        }
    }

    /// <summary>
    /// Pins one layer's input width. ResolveShapesOnly is declared on <see cref="LayerBase{T}"/>
    /// rather than on ILayer, and it deliberately resolves shapes WITHOUT allocating weights, so the
    /// first real forward still initializes them from the same RNG draw as before.
    /// </summary>
    private static void ResolveLayerInput(ILayer<T>? layer, int inputWidth)
    {
        if (layer is LayerBase<T> resolvable && inputWidth > 0)
        {
            resolvable.ResolveShapesOnly(new[] { inputWidth });
        }
    }

    private static IReadOnlyList<VoiceInfo<T>> GetDefaultVoices()
    {
        return new[]
        {
            new VoiceInfo<T>
            {
                Id = "default",
                Name = "Default Voice",
                Language = "en",
                Gender = VoiceGender.Neutral,
                Style = "neutral"
            }
        };
    }

    #endregion

    #region ITextToSpeech Implementation

    /// <summary>
    /// Synthesizes speech from text.
    /// </summary>
    public Tensor<T> Synthesize(
        string text,
        string? voiceId = null,
        double speakingRate = 1.0,
        double pitch = 0.0)
    {
        ThrowIfDisposed();

        // Preprocess text to phonemes
        var phonemes = _preprocessor.TextToPhonemes(text);

        // Create phoneme tensor
        var phonemeTensor = CreatePhonemeTensor(phonemes);

        // Apply speaking rate
        double effectiveRate = Math.Abs(speakingRate - 1.0) > 0.01 ? speakingRate : _speakingRate;

        // Generate mel spectrogram
        Tensor<T> melSpectrogram;
        if (_useNativeMode)
        {
            melSpectrogram = ForwardNative(phonemeTensor);
        }
        else
        {
            melSpectrogram = ForwardOnnx(phonemeTensor);
        }

        // Apply rate modification
        if (Math.Abs(effectiveRate - 1.0) > 0.01)
        {
            melSpectrogram = ModifyDuration(melSpectrogram, 1.0 / effectiveRate);
        }

        // Convert mel spectrogram to audio waveform
        Tensor<T> audio;
        if (_vocoder is not null)
        {
            audio = _vocoder.Run(melSpectrogram);
        }
        else if (_griffinLim is not null)
        {
            audio = GriffinLimSynthesize(melSpectrogram);
        }
        else
        {
            throw new InvalidOperationException("No vocoder available.");
        }

        return audio;
    }

    /// <summary>
    /// Synthesizes speech from text asynchronously.
    /// </summary>
    public Task<Tensor<T>> SynthesizeAsync(
        string text,
        string? voiceId = null,
        double speakingRate = 1.0,
        double pitch = 0.0,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Synthesize(text, voiceId, speakingRate, pitch), cancellationToken);
    }

    /// <summary>
    /// Synthesizes speech using a cloned voice from reference audio.
    /// </summary>
    public Tensor<T> SynthesizeWithVoiceCloning(
        string text,
        Tensor<T> referenceAudio,
        double speakingRate = 1.0,
        double pitch = 0.0)
    {
        throw new NotSupportedException("Voice cloning is not supported by Tacotron2. Use VITSModel for voice cloning.");
    }

    /// <summary>
    /// Synthesizes speech with emotional expression.
    /// </summary>
    public Tensor<T> SynthesizeWithEmotion(
        string text,
        string emotion,
        double emotionIntensity = 0.5,
        string? voiceId = null,
        double speakingRate = 1.0)
    {
        throw new NotSupportedException("Emotion control is not supported by Tacotron2 model.");
    }

    /// <summary>
    /// Extracts speaker embedding from reference audio.
    /// </summary>
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        throw new NotSupportedException("Speaker embedding extraction is not supported by Tacotron2.");
    }

    /// <summary>
    /// Starts a streaming synthesis session.
    /// </summary>
    public IStreamingSynthesisSession<T> StartStreamingSession(string? voiceId = null, double speakingRate = 1.0)
    {
        throw new NotSupportedException("Streaming synthesis is not supported by Tacotron2.");
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Tacotron2 takes text input, not audio
        return rawAudio;
    }

    /// <summary>
    /// Postprocesses model output.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return modelOutput;
    }

    /// <summary>
    /// Makes a prediction using the model.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!_useNativeMode)
        {
            return ForwardOnnx(input);
        }
        else
        {
            return ForwardNative(input);
        }
    }

    /// <summary>
    /// Mean squared difference between two mel tensors, as a recorded scalar the tape can follow.
    /// </summary>
    private Tensor<T> MeanSquaredDifference(Tensor<T> predicted, Tensor<T> target)
    {
        var diff = Engine.TensorSubtract(predicted, target);
        var squared = Engine.TensorMultiply(diff, diff);
        var allAxes = System.Linq.Enumerable.Range(0, squared.Shape.Length).ToArray();
        return Engine.ReduceMean(squared, allAxes, keepDims: false);
    }

    /// <summary>
    /// Installs an explicit parameter vector, as the base contract requires: the argument is the new
    /// weights, not a gradient.
    /// </summary>
    /// <remarks>
    /// This override previously treated its argument as a GRADIENT and ran an optimizer step on it,
    /// which inverted the base contract (<c>WithParameters</c> and the clone path both call this to
    /// INSTALL weights). Restoring a trained parameter vector therefore applied an Adam update on top
    /// of it, so a round-tripped clone predicted differently from the model it was copied from.
    /// Training does not go through here — it runs the optimizer via TrainWithTape — so nothing else
    /// depended on the old behaviour.
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");
        }

        SetParameters(parameters);
    }

    /// <summary>
    /// Trains the model on input data.
    /// </summary>
    // Stored target for teacher forcing during ForwardForTraining
    private Tensor<T>? _teacherForcingTarget;

    /// <summary>
    /// The decoder's mel output BEFORE the post-net residual, captured on the teacher-forced forward
    /// so the loss can supervise it directly alongside the refined output.
    /// </summary>
    private Tensor<T>? _lastPrePostnetMel;

    /// <summary>
    /// Overrides ForwardForTraining to use teacher forcing when target is available.
    /// Teacher forcing feeds ground-truth previous outputs to the decoder instead of
    /// the model's own predictions — industry standard for autoregressive training.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (_teacherForcingTarget is not null)
            return ForwardNativeWithTeacherForcing(input, _teacherForcingTarget);
        return base.ForwardForTraining(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot train in ONNX inference mode.");
        }

        if (expectedOutput is null)
            throw new ArgumentException("expectedOutput cannot be null for teacher-forced training.", nameof(expectedOutput));
        if (expectedOutput.Shape.Length < 2)
            throw new ArgumentException($"expectedOutput must have at least rank 2 [batch, time*mels], got rank {expectedOutput.Shape.Length}.", nameof(expectedOutput));
        if (expectedOutput.Shape[^1] % NumMels != 0)
            throw new ArgumentException($"expectedOutput last dimension ({expectedOutput.Shape[^1]}) must be divisible by NumMels ({NumMels}).", nameof(expectedOutput));
        int melFrameCount = expectedOutput.Shape[1] / _numMelsPerFrame;
        if (melFrameCount == 0)
            throw new ArgumentException($"expectedOutput has {expectedOutput.Shape[1]} mel values but _numMelsPerFrame is {_numMelsPerFrame}, resulting in zero frames.", nameof(expectedOutput));

        _teacherForcingTarget = expectedOutput;
        try
        {
            SetTrainingMode(true);
            // Pass the configured optimizer through. The two-argument overload left the optimizer
            // built in the constructor assigned but never read, so training silently fell back to the
            // framework default and the memorization loss came back byte-identical between step 1 and
            // step 100 (0.371638 both times) — the model was not learning at all.
            // Paper objective (Shen et al. 2018, sec. 3): "summed mean squared error (MSE) from
            // before and after the post-net". Only the refined output was supervised before, so the
            // decoder got a gradient solely THROUGH the post-net and never directly — which is the
            // convergence role the paper gives the pre-post-net term.
            TrainWithCustomLoss(
                input,
                refined =>
                {
                    var afterPostnet = MeanSquaredDifference(refined, expectedOutput);
                    if (_lastPrePostnetMel is null) return afterPostnet;
                    return Engine.TensorAdd(
                        afterPostnet, MeanSquaredDifference(_lastPrePostnetMel, expectedOutput));
                },
                _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
            _teacherForcingTarget = null;
        }
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Tacotron2",
            Description = "Attention-based sequence-to-sequence TTS model",
            FeatureCount = _vocabSize,
            Complexity = 2
        };
        metadata.AdditionalInfo["InputFormat"] = "Text/Phonemes";
        metadata.AdditionalInfo["OutputFormat"] = $"Audio ({SampleRate}Hz)";
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        metadata.AdditionalInfo["MaxDecoderSteps"] = _maxDecoderSteps.ToString();
        metadata.AdditionalInfo["HasVocoder"] = (_vocoder is not null).ToString();
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(SampleRate);
        writer.Write(NumMels);
        writer.Write(_speakingRate);
        writer.Write(_vocabSize);
        writer.Write(_embeddingDim);
        writer.Write(_encoderDim);
        writer.Write(_decoderDim);
        writer.Write(_attentionDim);
        writer.Write(_prenetDim);
        writer.Write(_postnetEmbeddingDim);
        writer.Write(_numEncoderConvLayers);
        writer.Write(_numPostnetConvLayers);
        writer.Write(_numMelsPerFrame);
        writer.Write(_maxDecoderSteps);
        writer.Write(_stopThreshold);
        writer.Write(_fftSize);
        writer.Write(_hopLength);
        writer.Write(_griffinLimIterations);
        // Added at the tail for backward compatibility with model payloads
        // written before attentionFilters was persisted.
        writer.Write(_attentionFilters);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Note: _useNativeMode is readonly and set at construction
        // Deserialized models operate in native mode
        _ = reader.ReadBoolean(); // useNativeMode (read but not assigned)

        // Restore audio configuration
        SampleRate = reader.ReadInt32();
        NumMels = reader.ReadInt32();
        _speakingRate = reader.ReadDouble();

        // Restore architecture parameters
        _vocabSize = reader.ReadInt32();
        _embeddingDim = reader.ReadInt32();
        _encoderDim = reader.ReadInt32();
        _decoderDim = reader.ReadInt32();
        _attentionDim = reader.ReadInt32();
        _prenetDim = reader.ReadInt32();
        _postnetEmbeddingDim = reader.ReadInt32();
        _numEncoderConvLayers = reader.ReadInt32();
        _numPostnetConvLayers = reader.ReadInt32();
        _numMelsPerFrame = reader.ReadInt32();
        _maxDecoderSteps = reader.ReadInt32();
        _stopThreshold = reader.ReadDouble();
        _fftSize = reader.ReadInt32();
        _hopLength = reader.ReadInt32();
        _griffinLimIterations = reader.ReadInt32();
        if (reader.BaseStream.Position < reader.BaseStream.Length)
            _attentionFilters = reader.ReadInt32();

        // Base deserialization has recreated the published Layers list by this
        // point. Rebind native component views to those restored instances.
        if (_useNativeMode)
            BindNativeLayersFromPublishedList();
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _acousticModelPath is not null)
        {
            return new Tacotron2Model<T>(
                Architecture,
                _acousticModelPath,
                _vocoderPath,
                SampleRate,
                NumMels,
                _speakingRate,
                _maxDecoderSteps,
                _stopThreshold,
                _fftSize,
                _hopLength,
                _griffinLimIterations);
        }
        else
        {
            return new Tacotron2Model<T>(
                Architecture,
                SampleRate,
                NumMels,
                _speakingRate,
                _vocabSize,
                _embeddingDim,
                _encoderDim,
                _decoderDim,
                _attentionDim,
                _attentionFilters,
                _prenetDim,
                _postnetEmbeddingDim,
                _numEncoderConvLayers,
                _numPostnetConvLayers,
                _numMelsPerFrame,
                _maxDecoderSteps,
                _stopThreshold,
                _fftSize,
                _hopLength,
                _griffinLimIterations,
                lossFunction: _lossFunction);
        }
    }

    #endregion

    #region Private Methods

    private Tensor<T> CreatePhonemeTensor(int[] phonemes)
    {
        var tensor = new Tensor<T>([1, phonemes.Length]);
        for (int i = 0; i < phonemes.Length; i++)
        {
            tensor[0, i] = NumOps.FromDouble(phonemes[i]);
        }
        return tensor;
    }

    private Tensor<T> ForwardNative(Tensor<T> phonemes)
    {
        // Embed phonemes
        var embedded = _embedding?.Forward(phonemes) ?? phonemes;

        // Encoder conv layers
        var encoderInput = embedded;
        foreach (var conv in _encoderConvLayers)
        {
            encoderInput = conv.Forward(encoderInput);
        }

        // Encoder LSTM
        var encoderOutput = _encoderLstm?.Forward(encoderInput) ?? encoderInput;

        // Autoregressive decoding
        var melFrames = new List<Tensor<T>>();
        // NVIDIA Tacotron2's pre-net consumes n_mel_channels *
        // n_frames_per_step values (the grouped previous decoder output).
        var prevMel = new Tensor<T>([1, NumMels * _numMelsPerFrame]);
        var attentionWeights = new Tensor<T>([1, phonemes.Shape[^1]]);
        var decoderState = new Tensor<T>([1, _decoderDim]);

        for (int step = 0; step < _maxDecoderSteps; step++)
        {
            // Pre-net
            var prenetOut = prevMel;
            for (int i = 0; i < 2 && i < _decoderLstmLayers.Count; i++)
            {
                prenetOut = _decoderLstmLayers[i].Forward(prenetOut);
            }

            // Attention (location-sensitive: feeds back updated weights each step)
            var (context, updatedWeights) = ComputeAttention(decoderState, encoderOutput, attentionWeights);
            attentionWeights = updatedWeights;

            // Decoder LSTM
            var lstmInput = ConcatenateTensors(prenetOut, context);
            for (int i = 2; i < _decoderLstmLayers.Count - 1; i++)
            {
                lstmInput = _decoderLstmLayers[i].Forward(lstmInput);
            }
            decoderState = lstmInput;

            // Mel output
            var decoderContext = ConcatenateTensors(decoderState, context);
            var melOutput = _decoderLstmLayers[^1].Forward(decoderContext);
            melFrames.Add(melOutput);

            // Stop token
            var stopToken = _stopTokenLayer?.Forward(decoderContext);
            if (stopToken is not null && NumOps.ToDouble(stopToken[0, 0]) > _stopThreshold)
            {
                break;
            }

            // Update previous mel for next step
            prevMel = melOutput;
        }

        // Combine mel frames
        var melSpectrogram = CombineMelFrames(melFrames);

        // Post-net refinement
        var residual = melSpectrogram;
        foreach (var postConv in _postNetLayers)
        {
            residual = postConv.Forward(residual);
        }

        // Add residual via engine op (tape-tracked)
        return Engine.TensorAdd(melSpectrogram, residual);
    }

    private Tensor<T> ForwardNativeWithTeacherForcing(Tensor<T> phonemes, Tensor<T> targetMel)
    {
        // Similar to ForwardNative but uses target mel frames as input
        var embedded = _embedding?.Forward(phonemes) ?? phonemes;

        var encoderInput = embedded;
        foreach (var conv in _encoderConvLayers)
        {
            encoderInput = conv.Forward(encoderInput);
        }

        var encoderOutput = _encoderLstm?.Forward(encoderInput) ?? encoderInput;

        int numFrames = targetMel.Rank >= 2 ? targetMel.Shape[^2] : 0;
        var melFrames = new List<Tensor<T>>();
        var attentionWeights = new Tensor<T>([1, phonemes.Shape[^1]]);
        var decoderState = new Tensor<T>([1, _decoderDim]);

        int decoderSteps = (numFrames + _numMelsPerFrame - 1) / _numMelsPerFrame;
        for (int step = 0; step < decoderSteps; step++)
        {
            // Teacher forcing: step 0 gets GO frame (zeros), step>0 gets previous ground-truth
            Tensor<T> prevMel;
            if (step == 0)
            {
                prevMel = new Tensor<T>(new[] { 1, NumMels * _numMelsPerFrame });
            }
            else
            {
                prevMel = ExtractMelFrameGroup(targetMel, (step - 1) * _numMelsPerFrame);
            }

            var prenetOut = prevMel;
            for (int i = 0; i < 2 && i < _decoderLstmLayers.Count; i++)
            {
                prenetOut = _decoderLstmLayers[i].Forward(prenetOut);
            }

            var (context, updatedWeights) = ComputeAttention(decoderState, encoderOutput, attentionWeights);
            attentionWeights = updatedWeights;
            var lstmInput = ConcatenateTensors(prenetOut, context);

            for (int i = 2; i < _decoderLstmLayers.Count - 1; i++)
            {
                lstmInput = _decoderLstmLayers[i].Forward(lstmInput);
            }
            decoderState = lstmInput;

            var decoderContext = ConcatenateTensors(decoderState, context);
            var melOutput = _decoderLstmLayers[^1].Forward(decoderContext);
            melFrames.Add(melOutput);
        }

        var melSpectrogram = CombineMelFramesTapeSafe(melFrames);

        var residual = melSpectrogram;
        foreach (var postConv in _postNetLayers)
        {
            residual = postConv.Forward(residual);
        }

        // Recorded add, matching the inference path. Writing the sum element by element produced a
        // tensor with no history on the autodiff tape, which detached the ENTIRE forward pass: no
        // gradient reached the post-net, decoder, attention or encoder, so no parameter ever moved and
        // the memorization loss came back byte-identical at step 1 and step 100 (0.325832 both times).
        // Keep the pre-post-net mel: the paper minimizes the SUMMED MSE from before AND after
        // the post-net, so both have to reach the loss (see Train).
        _lastPrePostnetMel = melSpectrogram;
        return Engine.TensorAdd(melSpectrogram, residual);
    }

    /// <summary>
    /// Assembles the decoder's per-step mel outputs into [1, steps * numMelsPerFrame, numMels] using
    /// recorded reshape/concatenate ops, so gradients flow back through every decoder step.
    /// <see cref="CombineMelFrames"/> is the fallback for a decoder that emits a different width; it
    /// is ALSO tape-safe, building its result from <c>Engine.TensorNarrow</c> and
    /// <c>Engine.TensorStack</c>. This comment used to say it assigned elements one at a time and so
    /// could not be differentiated through -- that described an implementation removed when the
    /// detached-tape bug was fixed, and left the fallback looking like a gradient-losing path when it
    /// is not.
    /// </summary>
    private Tensor<T> CombineMelFramesTapeSafe(List<Tensor<T>> frames)
    {
        int perFrame = _numMelsPerFrame * NumMels;
        foreach (var frame in frames)
        {
            // Reshape needs an exact element count. If a decoder ever emits a different width, fall
            // back to the element-wise builder rather than throwing — inference still works there,
            // and the assembly stays correct.
            if (frame.Length != perFrame)
            {
                return CombineMelFrames(frames);
            }
        }

        var reshaped = new Tensor<T>[frames.Count];
        for (int i = 0; i < frames.Count; i++)
        {
            reshaped[i] = Engine.Reshape(frames[i], new[] { 1, _numMelsPerFrame, NumMels });
        }

        return reshaped.Length == 1 ? reshaped[0] : Engine.TensorConcatenate(reshaped, axis: 1);
    }

    private Tensor<T> ForwardOnnx(Tensor<T> phonemes)
    {
        if (_acousticModel is null)
            throw new InvalidOperationException("Acoustic model not loaded.");

        return _acousticModel.Run(phonemes);
    }

    private (Tensor<T> context, Tensor<T> updatedWeights) ComputeAttention(
        Tensor<T> query, Tensor<T> keys, Tensor<T> attentionWeights)
    {
        if (_attentionLayers.Count < 4)
        {
            // Fallback: mean-pool over sequence dimension to get [1, hiddenDim] context
            var fallbackContext = Engine.ReduceMean(keys, new[] { 1 }, keepDims: false);
            return (fallbackContext, attentionWeights);
        }

        // Location-sensitive attention (Chorowski et al.):
        // e_i = v^T * tanh(W_s * s + V_h * h_j + U_f * f_j + b)
        // where f = F * α_{i-1} (location features from previous alignment)

        // [0] Query projection: W_s * s → [1, attDim]
        var projQuery = _attentionLayers[0].Forward(query);

        // [1] Key projection: V_h * h → [1, seqLen, attDim]
        var projKeys = _attentionLayers[1].Forward(keys);

        // [2] Location feature projection: U_f * f → [1, attDim]
        // In full Tacotron2 this would be Conv1D(attWeights) → DenseLayer.
        // Our architecture uses DenseLayer(attentionFilters, attentionDim) as a simplified
        // location projection. We create a fixed-size location feature from the attention weights
        // by truncating/padding to attentionFilters dimensions.
        int locDim = _attentionFilters;
        int seqLen = attentionWeights.Shape[^1];
        var locFeatures = new Tensor<T>([1, locDim]);
        int copyLen = Math.Min(seqLen, locDim);
        for (int j = 0; j < copyLen; j++)
        {
            locFeatures[0, j] = attentionWeights.Rank >= 2 ? attentionWeights[0, j] : attentionWeights[j];
        }
        var projLocation = _attentionLayers[2].Forward(locFeatures); // [1, attDim]

        // Combine: tanh(projQuery + projKeys + projLocation)
        // Broadcast query and location across sequence dimension
        var queryBroadcast = Engine.TensorBroadcastAdd(projKeys, projQuery); // [1, seqLen, attDim]
        var combined = Engine.TensorBroadcastAdd(queryBroadcast, projLocation); // [1, seqLen, attDim]
        var tanhScores = Engine.Tanh(combined); // [1, seqLen, attDim]

        // [3] Energy projection: v^T * tanh(...) → scalar per position
        // _attentionLayers[3] is DenseLayer(attentionDim, 1) applied per position
        // Reshape to [seqLen, attDim], project to [seqLen, 1], reshape back to [1, seqLen]
        var tanhFlat = Engine.Reshape(tanhScores, new[] { tanhScores.Shape[1], tanhScores.Shape[2] });
        var energyFlat = _attentionLayers[3].Forward(tanhFlat); // [seqLen, 1]
        var scores = Engine.Reshape(energyFlat, new[] { 1, tanhScores.Shape[1] }); // [1, seqLen]

        // Softmax over sequence dimension for attention weights
        var attWeights = Engine.TensorSoftmax(scores, axis: 1); // [1, seqLen]

        // Weighted sum: context = sum_t(attWeights[t] * keys[0, t, :])
        var weightsExpanded = Engine.TensorExpandDims(attWeights, 2); // [1, seqLen, 1]
        var weighted = Engine.TensorMultiply(keys, weightsExpanded); // [1, seqLen, hiddenDim]
        var context = Engine.ReduceSum(weighted, new[] { 1 }, keepDims: false); // [1, hiddenDim]

        return (context, attWeights);
    }

    private Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
    {
        // Ensure both tensors are 2D [1, dim] for concatenation along last axis
        var a2d = a.Rank == 1 ? Engine.Reshape(a, new[] { 1, a.Shape[0] }) : a;
        var b2d = b.Rank == 1 ? Engine.Reshape(b, new[] { 1, b.Shape[0] }) : b;
        return Engine.TensorConcatenate(new[] { a2d, b2d }, axis: 1);
    }

    private Tensor<T> ExtractMelFrameGroup(Tensor<T> mel, int firstFrame)
    {
        var group = new Tensor<T>([1, NumMels * _numMelsPerFrame]);
        int availableFrames = mel.Rank >= 2 ? mel.Shape[^2] : 0;
        for (int f = 0; f < _numMelsPerFrame; f++)
        {
            int frame = firstFrame + f;
            if (frame >= availableFrames)
                break;
            for (int m = 0; m < NumMels; m++)
            {
                group[0, f * NumMels + m] = mel.Rank >= 3
                    ? mel[0, frame, m]
                    : mel[frame, m];
            }
        }
        return group;
    }

    private Tensor<T> CombineMelFrames(List<Tensor<T>> frames)
    {
        if (frames.Count == 0)
            return new Tensor<T>([1, 0, NumMels]);

        // Keep decoder outputs connected to the gradient tape. The old manual
        // element copy created a fresh leaf tensor, so the mel loss had no path
        // back to any decoder/encoder parameter and every Train call was a no-op.
        var melFrames = new List<Tensor<T>>(frames.Count * _numMelsPerFrame);
        foreach (var groupedOutput in frames)
        {
            for (int f = 0; f < _numMelsPerFrame; f++)
            {
                int start = f * NumMels;
                if (start + NumMels <= groupedOutput.Shape[^1])
                    melFrames.Add(Engine.TensorNarrow(groupedOutput, groupedOutput.Rank - 1, start, NumMels));
            }
        }
        // A NON-EMPTY `frames` CAN STILL YIELD NO SLICES. The narrow above is conditional, so if every
        // grouped output is narrower than NumMels the list stays empty and TensorStack is handed a
        // zero-length array. The early return only covers `frames` being empty, which is a different
        // condition. Return the same empty mel the caller already handles rather than failing inside
        // the engine on a shape it cannot explain.
        if (melFrames.Count == 0)
            return new Tensor<T>([1, 0, NumMels]);

        return Engine.TensorStack(melFrames.ToArray(), axis: 1);
    }

    private Tensor<T> ModifyDuration(Tensor<T> melSpectrogram, double factor)
    {
        int originalFrames = melSpectrogram.Shape[1];
        int newFrames = (int)(originalFrames * factor);

        var modified = new Tensor<T>([1, newFrames, NumMels]);

        for (int f = 0; f < newFrames; f++)
        {
            double srcFrame = f / factor;
            int srcIdx = Math.Min((int)srcFrame, originalFrames - 1);

            for (int m = 0; m < NumMels; m++)
            {
                modified[0, f, m] = melSpectrogram.Rank >= 3
                    ? melSpectrogram[0, srcIdx, m]
                    : melSpectrogram[srcIdx, m];
            }
        }

        return modified;
    }

    private Tensor<T> GriffinLimSynthesize(Tensor<T> melSpectrogram)
    {
        if (_griffinLim is null)
            throw new InvalidOperationException("Griffin-Lim not available.");

        Tensor<T> mel2D;
        if (melSpectrogram.Rank == 3)
        {
            int frames = melSpectrogram.Shape[1];
            int mels = melSpectrogram.Shape[2];
            mel2D = new Tensor<T>([frames, mels]);

            for (int f = 0; f < frames; f++)
            {
                for (int m = 0; m < mels; m++)
                {
                    mel2D[f, m] = melSpectrogram[0, f, m];
                }
            }
        }
        else
        {
            mel2D = melSpectrogram;
        }

        return _griffinLim.Reconstruct(mel2D);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes the model and releases resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            _acousticModel?.Dispose();
            _vocoder?.Dispose();
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
