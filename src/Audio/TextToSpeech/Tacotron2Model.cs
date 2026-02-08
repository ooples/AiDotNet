using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
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
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

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

        // Initialize optimizer and loss (not used in ONNX mode)
        _lossFunction = new MeanSquaredErrorLoss<T>();
        _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
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
        _attentionFilters = 32;
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
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> tanhActivation = new TanhActivation<T>();
        IActivationFunction<T> sigmoidActivation = new SigmoidActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // Character/phoneme embedding
        _embedding = new EmbeddingLayer<T>(_vocabSize, _embeddingDim);

        // Encoder convolutional layers
        for (int i = 0; i < _numEncoderConvLayers; i++)
        {
            var conv = new DenseLayer<T>(_embeddingDim, _embeddingDim, reluActivation);
            _encoderConvLayers.Add(conv);
        }

        // Encoder bidirectional LSTM (approximated with dense layers)
        _encoderLstm = new DenseLayer<T>(_embeddingDim, _encoderDim * 2, tanhActivation);

        // Attention layers
        // Location-sensitive attention components
        var attentionQuery = new DenseLayer<T>(_decoderDim, _attentionDim, identityActivation);
        var attentionKey = new DenseLayer<T>(_encoderDim * 2, _attentionDim, identityActivation);
        var attentionLocation = new DenseLayer<T>(_attentionFilters, _attentionDim, identityActivation);
        var attentionValue = new DenseLayer<T>(_attentionDim, 1, identityActivation);
        _attentionLayers.Add(attentionQuery);
        _attentionLayers.Add(attentionKey);
        _attentionLayers.Add(attentionLocation);
        _attentionLayers.Add(attentionValue);

        // Decoder LSTM layers (2 layers)
        // Pre-net
        var prenet1 = new DenseLayer<T>(NumMels, _prenetDim, reluActivation);
        var prenet2 = new DenseLayer<T>(_prenetDim, _prenetDim, reluActivation);
        _decoderLstmLayers.Add(prenet1);
        _decoderLstmLayers.Add(prenet2);

        // LSTM layers (approximated)
        var lstm1 = new DenseLayer<T>(_prenetDim + _encoderDim * 2, _decoderDim, tanhActivation);
        var lstm2 = new DenseLayer<T>(_decoderDim + _encoderDim * 2, _decoderDim, tanhActivation);
        _decoderLstmLayers.Add(lstm1);
        _decoderLstmLayers.Add(lstm2);

        // Mel output projection
        var melOutput = new DenseLayer<T>(_decoderDim + _encoderDim * 2, NumMels * _numMelsPerFrame, identityActivation);
        _decoderLstmLayers.Add(melOutput);

        // Stop token prediction
        _stopTokenLayer = new DenseLayer<T>(_decoderDim + _encoderDim * 2, 1, sigmoidActivation);

        // Post-net (convolutional layers for mel refinement)
        for (int i = 0; i < _numPostnetConvLayers; i++)
        {
            var isLast = i == _numPostnetConvLayers - 1;
            var activation = isLast ? identityActivation : tanhActivation;
            var postConv = new DenseLayer<T>(
                i == 0 ? NumMels : _postnetEmbeddingDim,
                isLast ? NumMels : _postnetEmbeddingDim,
                activation);
            _postNetLayers.Add(postConv);
        }

        // Register all layers
        if (_embedding is not null)
            Layers.Add(_embedding);
        foreach (var layer in _encoderConvLayers)
            Layers.Add(layer);
        if (_encoderLstm is not null)
            Layers.Add(_encoderLstm);
        foreach (var layer in _attentionLayers)
            Layers.Add(layer);
        foreach (var layer in _decoderLstmLayers)
            Layers.Add(layer);
        if (_stopTokenLayer is not null)
            Layers.Add(_stopTokenLayer);
        foreach (var layer in _postNetLayers)
            Layers.Add(layer);
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
    public override Tensor<T> Predict(Tensor<T> input)
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
    /// Updates model parameters using the configured optimizer.
    /// </summary>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");
        }

        // Use the configured optimizer for parameter updates
        var currentParams = GetParameters();

        // Cast to gradient-based optimizer to access UpdateParameters
        if (_optimizer is IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> gradientOptimizer)
        {
            var updatedParams = gradientOptimizer.UpdateParameters(currentParams, gradients);
            SetParameters(updatedParams);
        }
        else
        {
            // Fallback: manual SGD if optimizer doesn't support gradient-based updates
            T learningRate = NumOps.FromDouble(0.001);
            for (int i = 0; i < currentParams.Length; i++)
            {
                currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
            }
            SetParameters(currentParams);
        }
    }

    /// <summary>
    /// Trains the model on input data.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot train in ONNX inference mode.");
        }

        SetTrainingMode(true);

        // Forward pass (teacher forcing - use expected mel frames as input)
        var prediction = ForwardNativeWithTeacherForcing(input, expectedOutput);

        // Calculate loss
        var flatPrediction = prediction.ToVector();
        var flatExpected = expectedOutput.ToVector();
        LastLoss = _lossFunction.CalculateLoss(flatPrediction, flatExpected);

        // Backward pass
        var lossGradient = _lossFunction.CalculateDerivative(flatPrediction, flatExpected);
        Backpropagate(Tensor<T>.FromVector(lossGradient));

        // Update parameters
        var gradients = GetParameterGradients();
        UpdateParameters(gradients);

        SetTrainingMode(false);
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
            ModelType = ModelType.NeuralNetwork,
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

        // Reinitialize layers with restored parameters if needed
        if (_useNativeMode && _encoderConvLayers.Count == 0)
        {
            InitializeLayers();
        }
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
        var prevMel = new Tensor<T>([1, NumMels]);
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

            // Attention
            var context = ComputeAttention(decoderState, encoderOutput, attentionWeights);

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
            prevMel = ExtractLastMelFrame(melOutput);
        }

        // Combine mel frames
        var melSpectrogram = CombineMelFrames(melFrames);

        // Post-net refinement
        var residual = melSpectrogram;
        foreach (var postConv in _postNetLayers)
        {
            residual = postConv.Forward(residual);
        }

        // Add residual
        var refined = new Tensor<T>(melSpectrogram.Shape);
        for (int i = 0; i < melSpectrogram.Length; i++)
        {
            refined[i] = NumOps.Add(melSpectrogram[i], residual[i]);
        }

        return refined;
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

        int numFrames = targetMel.Shape[1];
        var melFrames = new List<Tensor<T>>();
        var attentionWeights = new Tensor<T>([1, phonemes.Shape[^1]]);
        var decoderState = new Tensor<T>([1, _decoderDim]);

        for (int step = 0; step < numFrames / _numMelsPerFrame; step++)
        {
            // Use target mel frame (teacher forcing)
            var prevMel = ExtractMelFrame(targetMel, step * _numMelsPerFrame);

            var prenetOut = prevMel;
            for (int i = 0; i < 2 && i < _decoderLstmLayers.Count; i++)
            {
                prenetOut = _decoderLstmLayers[i].Forward(prenetOut);
            }

            var context = ComputeAttention(decoderState, encoderOutput, attentionWeights);
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

        var melSpectrogram = CombineMelFrames(melFrames);

        var residual = melSpectrogram;
        foreach (var postConv in _postNetLayers)
        {
            residual = postConv.Forward(residual);
        }

        var refined = new Tensor<T>(melSpectrogram.Shape);
        for (int i = 0; i < melSpectrogram.Length; i++)
        {
            refined[i] = NumOps.Add(melSpectrogram[i], residual[i]);
        }

        return refined;
    }

    private Tensor<T> ForwardOnnx(Tensor<T> phonemes)
    {
        if (_acousticModel is null)
            throw new InvalidOperationException("Acoustic model not loaded.");

        return _acousticModel.Run(phonemes);
    }

    private Tensor<T> ComputeAttention(Tensor<T> query, Tensor<T> keys, Tensor<T> attentionWeights)
    {
        if (_attentionLayers.Count < 4)
            return keys;

        // Project query and keys through attention layers
        var projQuery = _attentionLayers[0].Forward(query);
        var projKeys = _attentionLayers[1].Forward(keys);

        // Additive attention: score = tanh(projQuery + projKeys)
        int seqLen = keys.Shape[1];
        int hiddenDim = keys.Shape[^1];
        int attDim = projQuery.Shape[^1];

        // Compute attention scores for each encoder position
        var scores = new double[seqLen];
        for (int t = 0; t < seqLen; t++)
        {
            double score = 0;
            for (int d = 0; d < attDim; d++)
            {
                // Get projected key at position t
                double keyVal = NumOps.ToDouble(projKeys[0, t, d]);
                double queryVal = NumOps.ToDouble(projQuery[0, d]);
                // Additive score with tanh activation
                double combined = Math.Tanh(queryVal + keyVal);
                score += combined;
            }
            scores[t] = score;
        }

        // Softmax over scores to get attention weights
        double maxScore = scores.Max();
        double sumExp = 0;
        for (int t = 0; t < seqLen; t++)
        {
            scores[t] = Math.Exp(scores[t] - maxScore);
            sumExp += scores[t];
        }
        for (int t = 0; t < seqLen; t++)
        {
            scores[t] /= sumExp;
        }

        // Weighted sum of encoder outputs using attention weights
        var context = new Tensor<T>([1, hiddenDim]);
        for (int d = 0; d < hiddenDim; d++)
        {
            double sum = 0;
            for (int t = 0; t < seqLen; t++)
            {
                sum += scores[t] * NumOps.ToDouble(keys[0, t, d]);
            }
            context[0, d] = NumOps.FromDouble(sum);
        }

        return context;
    }

    private Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
    {
        int dimA = a.Shape[^1];
        int dimB = b.Shape[^1];
        var result = new Tensor<T>([1, dimA + dimB]);

        for (int i = 0; i < dimA; i++)
        {
            result[0, i] = a.Rank >= 2 ? a[0, i] : a[i];
        }
        for (int i = 0; i < dimB; i++)
        {
            result[0, dimA + i] = b.Rank >= 2 ? b[0, i] : b[i];
        }

        return result;
    }

    private Tensor<T> ExtractLastMelFrame(Tensor<T> melOutput)
    {
        int lastMelStart = melOutput.Shape[^1] - NumMels;
        var frame = new Tensor<T>([1, NumMels]);

        for (int m = 0; m < NumMels; m++)
        {
            frame[0, m] = melOutput.Rank >= 2
                ? melOutput[0, lastMelStart + m]
                : melOutput[lastMelStart + m];
        }

        return frame;
    }

    private Tensor<T> ExtractMelFrame(Tensor<T> mel, int frameIdx)
    {
        var frame = new Tensor<T>([1, NumMels]);

        for (int m = 0; m < NumMels; m++)
        {
            frame[0, m] = mel.Rank >= 3
                ? mel[0, frameIdx, m]
                : (mel.Rank >= 2 ? mel[frameIdx, m] : NumOps.Zero);
        }

        return frame;
    }

    private Tensor<T> CombineMelFrames(List<Tensor<T>> frames)
    {
        int totalFrames = frames.Count * _numMelsPerFrame;
        var result = new Tensor<T>([1, totalFrames, NumMels]);

        int frameIdx = 0;
        foreach (var frame in frames)
        {
            for (int f = 0; f < _numMelsPerFrame; f++)
            {
                for (int m = 0; m < NumMels; m++)
                {
                    int srcIdx = f * NumMels + m;
                    if (srcIdx < frame.Shape[^1])
                    {
                        result[0, frameIdx, m] = frame.Rank >= 2
                            ? frame[0, srcIdx]
                            : frame[srcIdx];
                    }
                }
                frameIdx++;
            }
        }

        return result;
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
