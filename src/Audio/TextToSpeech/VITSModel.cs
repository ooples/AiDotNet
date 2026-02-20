using AiDotNet.ActivationFunctions;
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
/// VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VITS is a state-of-the-art end-to-end TTS model that generates high-quality speech
/// directly from text without requiring a separate vocoder. It combines:
/// <list type="bullet">
/// <item>Variational autoencoder (VAE) for learning latent representations</item>
/// <item>Normalizing flows for improved audio quality</item>
/// <item>Adversarial training for realistic speech synthesis</item>
/// <item>Multi-speaker support with speaker embeddings</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> VITS is a modern TTS model with several advantages:
///
/// 1. End-to-end: Converts text directly to audio (no separate vocoder needed)
/// 2. Fast: Parallel generation is much faster than autoregressive models
/// 3. High quality: Produces natural-sounding speech
/// 4. Voice cloning: Can learn to speak in new voices from short audio samples
///
/// Two ways to use this class:
/// 1. ONNX Mode: Load pretrained VITS models for fast inference
/// 2. Native Mode: Train your own TTS model from scratch
///
/// ONNX Mode Example:
/// <code>
/// var vits = new VITSModel&lt;float&gt;(
///     architecture,
///     modelPath: "path/to/vits.onnx");
/// var audio = vits.Synthesize("Hello, world!");
/// </code>
///
/// Voice Cloning Example:
/// <code>
/// var audio = vits.SynthesizeWithVoiceCloning(
///     "Hello, world!",
///     referenceAudio);
/// </code>
/// </para>
/// </remarks>
public class VITSModel<T> : AudioNeuralNetworkBase<T>, ITextToSpeech<T>
{
    private readonly VITSModelOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX models (false).
    /// </summary>
    private bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// Path to the ONNX model file.
    /// </summary>
    private readonly string? _modelPath;

    /// <summary>
    /// Path to the speaker encoder ONNX model (for voice cloning).
    /// </summary>
    private readonly string? _speakerEncoderPath;

    /// <summary>
    /// ONNX speaker encoder model.
    /// </summary>
    private readonly OnnxModel<T>? _speakerEncoder;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// Text encoder layers.
    /// </summary>
    private readonly List<ILayer<T>> _textEncoderLayers = [];

    /// <summary>
    /// Duration predictor layers.
    /// </summary>
    private readonly List<ILayer<T>> _durationPredictorLayers = [];

    /// <summary>
    /// Flow layers for normalizing flows.
    /// </summary>
    private readonly List<ILayer<T>> _flowLayers = [];

    /// <summary>
    /// Decoder layers (HiFi-GAN style generator).
    /// </summary>
    private readonly List<ILayer<T>> _decoderLayers = [];

    /// <summary>
    /// Speaker embedding layer.
    /// </summary>
    private ILayer<T>? _speakerEmbedding;

    #endregion

    #region Shared Fields

    /// <summary>
    /// Text preprocessor for phoneme conversion.
    /// </summary>
    private readonly TtsPreprocessor _preprocessor;

    /// <summary>
    /// Mel spectrogram extractor for speaker encoding.
    /// </summary>
    private readonly MelSpectrogram<T> _melSpectrogram;

    /// <summary>
    /// Optimizer for training.
    /// </summary>
    private IOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

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
    /// Hidden dimension for the model.
    /// </summary>
    private int _hiddenDim;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private int _numHeads;

    /// <summary>
    /// Number of text encoder layers.
    /// </summary>
    private int _numEncoderLayers;

    /// <summary>
    /// Number of flow layers.
    /// </summary>
    private int _numFlowLayers;

    /// <summary>
    /// Speaker embedding dimension.
    /// </summary>
    private int _speakerEmbeddingDim;

    /// <summary>
    /// Number of speakers for multi-speaker model.
    /// </summary>
    private int _numSpeakers;

    /// <summary>
    /// Maximum phoneme sequence length.
    /// </summary>
    private int _maxPhonemeLength;

    /// <summary>
    /// Phoneme vocabulary size.
    /// </summary>
    private int _phonemeVocabSize;

    /// <summary>
    /// HiFi-GAN upsampling rates for the decoder.
    /// </summary>
    private int[] _upsampleRates;

    /// <summary>
    /// FFT size for audio generation.
    /// </summary>
    private int _fftSize;

    /// <summary>
    /// Hop length for audio generation.
    /// </summary>
    private int _hopLength;

    /// <summary>
    /// Speaking rate multiplier.
    /// </summary>
    private double _speakingRate;

    /// <summary>
    /// Noise scale for sampling.
    /// </summary>
    private double _noiseScale;

    /// <summary>
    /// Length scale for duration control.
    /// </summary>
    private double _lengthScale;

    #endregion

    #region ITextToSpeech Properties

    /// <summary>
    /// Gets the list of available built-in voices.
    /// </summary>
    public IReadOnlyList<VoiceInfo<T>> AvailableVoices { get; private set; }

    /// <summary>
    /// Gets whether this model supports voice cloning from reference audio.
    /// </summary>
    public bool SupportsVoiceCloning => _speakerEncoder is not null || _speakerEmbedding is not null;

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
    public bool IsReady => _useNativeMode || OnnxModel?.IsLoaded == true;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a VITS model for ONNX inference with pretrained models.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the VITS ONNX model file.</param>
    /// <param name="speakerEncoderPath">Optional path to speaker encoder for voice cloning.</param>
    /// <param name="sampleRate">Output sample rate in Hz. Default is 22050.</param>
    /// <param name="numMels">Number of mel spectrogram channels. Default is 80.</param>
    /// <param name="speakingRate">Speaking rate multiplier. Default is 1.0.</param>
    /// <param name="noiseScale">Noise scale for variational sampling. Default is 0.667.</param>
    /// <param name="lengthScale">Length scale for duration control. Default is 1.0.</param>
    /// <param name="fftSize">FFT size for audio generation. Default is 1024.</param>
    /// <param name="hopLength">Hop length for audio generation. Default is 256.</param>
    /// <param name="onnxOptions">ONNX runtime options.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor with pretrained VITS models.
    ///
    /// You can get ONNX models from:
    /// - HuggingFace: Various VITS models
    /// - Coqui TTS exports
    ///
    /// Example:
    /// <code>
    /// var vits = new VITSModel&lt;float&gt;(
    ///     architecture,
    ///     modelPath: "vits-en.onnx",
    ///     speakerEncoderPath: "speaker-encoder.onnx");  // For voice cloning
    /// </code>
    /// </para>
    /// </remarks>
    public VITSModel(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        string? speakerEncoderPath = null,
        int sampleRate = 22050,
        int numMels = 80,
        double speakingRate = 1.0,
        double noiseScale = 0.667,
        double lengthScale = 1.0,
        int fftSize = 1024,
        int hopLength = 256,
        OnnxModelOptions? onnxOptions = null,
        VITSModelOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new VITSModelOptions();
        Options = _options;
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));
        if (modelPath is null)
            throw new ArgumentNullException(nameof(modelPath));

        _useNativeMode = false;
        _modelPath = modelPath;
        _speakerEncoderPath = speakerEncoderPath;

        // Store parameters
        SampleRate = sampleRate;
        NumMels = numMels;
        _speakingRate = speakingRate;
        _noiseScale = noiseScale;
        _lengthScale = lengthScale;
        _fftSize = fftSize;
        _hopLength = hopLength;

        // Default architecture parameters
        _hiddenDim = 192;
        _numHeads = 2;
        _numEncoderLayers = 6;
        _numFlowLayers = 4;
        _speakerEmbeddingDim = 256;
        _numSpeakers = 1;
        _maxPhonemeLength = 256;
        _phonemeVocabSize = 128;
        _upsampleRates = [8, 8, 2, 2];

        // Initialize preprocessor
        _preprocessor = new TtsPreprocessor();

        // Initialize mel spectrogram for speaker encoding
        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: sampleRate,
            nMels: numMels,
            nFft: fftSize,
            hopLength: hopLength);

        MelSpec = _melSpectrogram;

        // Load ONNX models
        var onnxOpts = onnxOptions ?? new OnnxModelOptions();
        OnnxModel = new OnnxModel<T>(modelPath, onnxOpts);

        if (speakerEncoderPath is not null && speakerEncoderPath.Length > 0)
        {
            _speakerEncoder = new OnnxModel<T>(speakerEncoderPath, onnxOpts);
        }

        // Initialize available voices
        AvailableVoices = GetDefaultVoices();

        // Default loss function (MSE is standard for TTS mel-spectrogram prediction)
        _lossFunction = new MeanSquaredErrorLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Creates a VITS model for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sampleRate">Output sample rate in Hz. Default is 22050.</param>
    /// <param name="numMels">Number of mel spectrogram channels. Default is 80.</param>
    /// <param name="speakingRate">Speaking rate multiplier. Default is 1.0.</param>
    /// <param name="noiseScale">Noise scale for variational sampling. Default is 0.667.</param>
    /// <param name="lengthScale">Length scale for duration control. Default is 1.0.</param>
    /// <param name="hiddenDim">Hidden dimension. Default is 192.</param>
    /// <param name="numHeads">Number of attention heads. Default is 2.</param>
    /// <param name="numEncoderLayers">Number of text encoder layers. Default is 6.</param>
    /// <param name="numFlowLayers">Number of flow layers. Default is 4.</param>
    /// <param name="speakerEmbeddingDim">Speaker embedding dimension. Default is 256.</param>
    /// <param name="numSpeakers">Number of speakers for multi-speaker model. Default is 1.</param>
    /// <param name="maxPhonemeLength">Maximum phoneme sequence length. Default is 256.</param>
    /// <param name="phonemeVocabSize">Phoneme vocabulary size. Default is 128.</param>
    /// <param name="upsampleRates">HiFi-GAN upsampling rates. Default is [8, 8, 2, 2].</param>
    /// <param name="fftSize">FFT size for audio generation. Default is 1024.</param>
    /// <param name="hopLength">Hop length for audio generation. Default is 256.</param>
    /// <param name="optimizer">Optimizer for training. If null, uses Adam.</param>
    /// <param name="lossFunction">Loss function for training. If null, uses MSE.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train your own VITS model.
    ///
    /// Training VITS requires:
    /// 1. Large amounts of paired text-audio data
    /// 2. Significant compute resources (GPUs recommended)
    /// 3. Many training epochs
    ///
    /// Example:
    /// <code>
    /// var vits = new VITSModel&lt;float&gt;(
    ///     architecture,
    ///     numSpeakers: 10,  // Multi-speaker model
    ///     speakerEmbeddingDim: 256);
    ///
    /// // Training loop
    /// vits.Train(phonemeInput, audioOutput);
    /// </code>
    /// </para>
    /// </remarks>
    public VITSModel(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 22050,
        int numMels = 80,
        double speakingRate = 1.0,
        double noiseScale = 0.667,
        double lengthScale = 1.0,
        int hiddenDim = 192,
        int numHeads = 2,
        int numEncoderLayers = 6,
        int numFlowLayers = 4,
        int speakerEmbeddingDim = 256,
        int numSpeakers = 1,
        int maxPhonemeLength = 256,
        int phonemeVocabSize = 128,
        int[]? upsampleRates = null,
        int fftSize = 1024,
        int hopLength = 256,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        VITSModelOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new VITSModelOptions();
        Options = _options;
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));

        _useNativeMode = true;

        // Store parameters
        SampleRate = sampleRate;
        NumMels = numMels;
        _speakingRate = speakingRate;
        _noiseScale = noiseScale;
        _lengthScale = lengthScale;
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _numEncoderLayers = numEncoderLayers;
        _numFlowLayers = numFlowLayers;
        _speakerEmbeddingDim = speakerEmbeddingDim;
        _numSpeakers = numSpeakers;
        _maxPhonemeLength = maxPhonemeLength;
        Guard.Positive(phonemeVocabSize);
        _phonemeVocabSize = phonemeVocabSize;
        var rates = (upsampleRates ?? [8, 8, 2, 2]).ToArray();
        if (rates.Length == 0)
            throw new ArgumentException("Upsample rates must not be empty.", nameof(upsampleRates));
        foreach (var rate in rates)
        {
            if (rate <= 0)
                throw new ArgumentException("All upsample rates must be positive.", nameof(upsampleRates));
        }
        _upsampleRates = rates;
        _fftSize = fftSize;
        _hopLength = hopLength;

        // Initialize preprocessor
        _preprocessor = new TtsPreprocessor();

        // Initialize mel spectrogram
        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: sampleRate,
            nMels: numMels,
            nFft: fftSize,
            hopLength: hopLength);

        MelSpec = _melSpectrogram;

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
        var layers = (Architecture.Layers != null && Architecture.Layers.Count > 0)
            ? Architecture.Layers.ToList()
            : LayerHelper<T>.CreateVITSLayers(
                hiddenDim: _hiddenDim, numEncoderLayers: _numEncoderLayers,
                numHeads: _numHeads, maxPhonemeLength: _maxPhonemeLength,
                numFlowLayers: _numFlowLayers, phonemeVocabSize: _phonemeVocabSize,
                upsampleRates: _upsampleRates).ToList();

        Layers.Clear();
        _textEncoderLayers.Clear();
        _durationPredictorLayers.Clear();
        _flowLayers.Clear();
        _decoderLayers.Clear();
        Layers.AddRange(layers);

        // Distribute to internal sub-lists for forward pass
        int idx = 0;
        int textEncoderCount = 1 + _numEncoderLayers * 3;
        for (int i = 0; i < textEncoderCount && idx < layers.Count; i++)
            _textEncoderLayers.Add(layers[idx++]);
        for (int i = 0; i < 3 && idx < layers.Count; i++)
            _durationPredictorLayers.Add(layers[idx++]);
        for (int i = 0; i < _numFlowLayers && idx < layers.Count; i++)
            _flowLayers.Add(layers[idx++]);
        while (idx < layers.Count)
            _decoderLayers.Add(layers[idx++]);

        // Speaker embedding (if multi-speaker) - separate from LayerHelper
        if (_numSpeakers > 1)
        {
            _speakerEmbedding = new EmbeddingLayer<T>(_numSpeakers, _speakerEmbeddingDim);
            Layers.Add(_speakerEmbedding);
        }
    }

    private IReadOnlyList<VoiceInfo<T>> GetDefaultVoices()
    {
        var voices = new List<VoiceInfo<T>>();

        for (int i = 0; i < _numSpeakers; i++)
        {
            voices.Add(new VoiceInfo<T>
            {
                Id = $"speaker_{i}",
                Name = i == 0 ? "Default Voice" : $"Speaker {i}",
                Language = "en",
                Gender = VoiceGender.Neutral,
                Style = "neutral"
            });
        }

        return voices;
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

        // Get speaker embedding if multi-speaker
        Tensor<T>? speakerEmb = null;
        if (_numSpeakers > 1 && voiceId is not null)
        {
            int speakerId = ParseSpeakerId(voiceId);
            speakerEmb = GetSpeakerEmbedding(speakerId);
        }

        // Apply speaking rate
        double effectiveRate = Math.Abs(speakingRate - 1.0) > 0.01 ? speakingRate : _speakingRate;
        double effectiveLengthScale = _lengthScale / effectiveRate;

        // Generate audio
        Tensor<T> audio;
        if (_useNativeMode)
        {
            audio = ForwardNative(phonemeTensor, speakerEmb, effectiveLengthScale);
        }
        else
        {
            audio = ForwardOnnx(phonemeTensor, speakerEmb, effectiveLengthScale);
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
        if (!SupportsVoiceCloning)
        {
            throw new NotSupportedException("Voice cloning is not supported. Load a speaker encoder model.");
        }

        ThrowIfDisposed();

        // Extract speaker embedding from reference audio
        var speakerEmb = ExtractSpeakerEmbedding(referenceAudio);

        // Preprocess text to phonemes
        var phonemes = _preprocessor.TextToPhonemes(text);
        var phonemeTensor = CreatePhonemeTensor(phonemes);

        double effectiveRate = Math.Abs(speakingRate - 1.0) > 0.01 ? speakingRate : _speakingRate;
        double effectiveLengthScale = _lengthScale / effectiveRate;

        // Generate audio with cloned voice
        Tensor<T> audio;
        if (_useNativeMode)
        {
            audio = ForwardNative(phonemeTensor, speakerEmb, effectiveLengthScale);
        }
        else
        {
            audio = ForwardOnnx(phonemeTensor, speakerEmb, effectiveLengthScale);
        }

        return audio;
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
        throw new NotSupportedException("Emotion control is not supported by VITS model.");
    }

    /// <summary>
    /// Extracts speaker embedding from reference audio for voice cloning.
    /// </summary>
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        if (_speakerEncoder is not null)
        {
            // Use ONNX speaker encoder
            var mel = _melSpectrogram.Forward(referenceAudio);
            return _speakerEncoder.Run(mel);
        }
        else if (_speakerEmbedding is not null && _useNativeMode)
        {
            // Native mode - compute average mel features as embedding
            var mel = _melSpectrogram.Forward(referenceAudio);

            // Average across time dimension
            var embedding = new Tensor<T>([_speakerEmbeddingDim]);
            int numFrames = mel.Shape[0];
            int numMels = Math.Min(mel.Shape[1], _speakerEmbeddingDim);

            for (int m = 0; m < numMels; m++)
            {
                double sum = 0;
                for (int t = 0; t < numFrames; t++)
                {
                    sum += NumOps.ToDouble(mel[t, m]);
                }
                embedding[m] = NumOps.FromDouble(sum / numFrames);
            }

            return embedding;
        }

        throw new NotSupportedException("Speaker embedding extraction requires a speaker encoder.");
    }

    /// <summary>
    /// Starts a streaming synthesis session.
    /// </summary>
    public IStreamingSynthesisSession<T> StartStreamingSession(string? voiceId = null, double speakingRate = 1.0)
    {
        throw new NotSupportedException("Streaming synthesis is not supported by VITS model.");
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        return _melSpectrogram.Forward(rawAudio);
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
            return RunOnnxInference(input);
        }
        else
        {
            return ForwardNative(input, null, _lengthScale);
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
            // Fallback: manual SGD with VITS's smaller learning rate
            T learningRate = NumOps.FromDouble(0.0002);
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

        // Forward pass
        var prediction = Predict(input);

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
            Name = "VITS",
            Description = "Variational Inference with adversarial learning for end-to-end Text-to-Speech",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _maxPhonemeLength,
            Complexity = 3
        };
        metadata.AdditionalInfo["InputFormat"] = "Text/Phonemes";
        metadata.AdditionalInfo["OutputFormat"] = $"Audio ({SampleRate}Hz)";
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        metadata.AdditionalInfo["VoiceCloning"] = SupportsVoiceCloning.ToString();
        metadata.AdditionalInfo["NumSpeakers"] = _numSpeakers.ToString();
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
        writer.Write(_noiseScale);
        writer.Write(_lengthScale);
        writer.Write(_hiddenDim);
        writer.Write(_numHeads);
        writer.Write(_numEncoderLayers);
        writer.Write(_numFlowLayers);
        writer.Write(_speakerEmbeddingDim);
        writer.Write(_numSpeakers);
        writer.Write(_maxPhonemeLength);
        writer.Write(_fftSize);
        writer.Write(_hopLength);
        writer.Write(_phonemeVocabSize);
        writer.Write(_upsampleRates.Length);
        foreach (var rate in _upsampleRates)
        {
            writer.Write(rate);
        }
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        SampleRate = reader.ReadInt32();
        NumMels = reader.ReadInt32();
        _speakingRate = reader.ReadDouble();
        _noiseScale = reader.ReadDouble();
        _lengthScale = reader.ReadDouble();
        _hiddenDim = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _numEncoderLayers = reader.ReadInt32();
        _numFlowLayers = reader.ReadInt32();
        _speakerEmbeddingDim = reader.ReadInt32();
        _numSpeakers = reader.ReadInt32();
        _maxPhonemeLength = reader.ReadInt32();
        _fftSize = reader.ReadInt32();
        _hopLength = reader.ReadInt32();
        if (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            _phonemeVocabSize = reader.ReadInt32();
            if (_phonemeVocabSize <= 0)
                throw new InvalidDataException($"Invalid phonemeVocabSize: {_phonemeVocabSize}. Must be positive.");
            int ratesLen = reader.ReadInt32();
            if (ratesLen <= 0 || ratesLen > 64)
                throw new InvalidDataException($"Invalid upsample rates length: {ratesLen}. Expected 1-64.");
            _upsampleRates = new int[ratesLen];
            for (int i = 0; i < ratesLen; i++)
            {
                _upsampleRates[i] = reader.ReadInt32();
                if (_upsampleRates[i] <= 0)
                    throw new InvalidDataException($"Invalid upsample rate at index {i}: {_upsampleRates[i]}. Must be positive.");
            }
        }
        else
        {
            _phonemeVocabSize = 128;
            _upsampleRates = [8, 8, 2, 2];
        }
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _modelPath is not null)
        {
            return new VITSModel<T>(
                Architecture,
                _modelPath,
                _speakerEncoderPath,
                SampleRate,
                NumMels,
                _speakingRate,
                _noiseScale,
                _lengthScale,
                _fftSize,
                _hopLength);
        }
        else
        {
            return new VITSModel<T>(
                Architecture,
                SampleRate,
                NumMels,
                _speakingRate,
                _noiseScale,
                _lengthScale,
                _hiddenDim,
                _numHeads,
                _numEncoderLayers,
                _numFlowLayers,
                _speakerEmbeddingDim,
                _numSpeakers,
                _maxPhonemeLength,
                _phonemeVocabSize,
                _upsampleRates,
                _fftSize,
                _hopLength,
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

    private int ParseSpeakerId(string voiceId)
    {
        if (voiceId.StartsWith("speaker_") &&
            int.TryParse(voiceId.Substring(8), out int id))
        {
            return Math.Min(id, _numSpeakers - 1);
        }
        return 0;
    }

    private Tensor<T> GetSpeakerEmbedding(int speakerId)
    {
        if (_speakerEmbedding is not null)
        {
            var idTensor = new Tensor<T>([1]);
            idTensor[0] = NumOps.FromDouble(speakerId);
            return _speakerEmbedding.Forward(idTensor);
        }

        return new Tensor<T>([_speakerEmbeddingDim]);
    }

    private Tensor<T> ForwardNative(Tensor<T> phonemes, Tensor<T>? speakerEmb, double lengthScale)
    {
        // Text encoder
        var encoded = phonemes;
        foreach (var layer in _textEncoderLayers)
        {
            encoded = layer.Forward(encoded);
        }

        // Add speaker embedding if available
        // Requires Rank >= 3 for 3D tensor indexing [batch, time, hidden]
        if (speakerEmb is not null && encoded.Rank >= 3)
        {
            // Broadcast speaker embedding across sequence
            for (int t = 0; t < encoded.Shape[1]; t++)
            {
                for (int d = 0; d < Math.Min(encoded.Shape[^1], speakerEmb.Shape[0]); d++)
                {
                    encoded[0, t, d] = NumOps.Add(encoded[0, t, d], speakerEmb[d]);
                }
            }
        }

        // Duration prediction
        var durations = encoded;
        foreach (var layer in _durationPredictorLayers)
        {
            durations = layer.Forward(durations);
        }

        // Expand by predicted durations
        var expanded = ExpandByDurations(encoded, durations, lengthScale);

        // Flow transform
        var flowOutput = expanded;
        foreach (var layer in _flowLayers)
        {
            flowOutput = layer.Forward(flowOutput);
        }

        // Decoder - generate audio
        var audio = flowOutput;
        foreach (var layer in _decoderLayers)
        {
            audio = layer.Forward(audio);
        }

        // Flatten to 1D audio
        var flatAudio = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
        {
            flatAudio[i] = audio[i];
        }

        return flatAudio;
    }

    private Tensor<T> ForwardOnnx(Tensor<T> phonemes, Tensor<T>? speakerEmb, double lengthScale)
    {
        if (OnnxModel is null)
            throw new InvalidOperationException("ONNX model not loaded.");

        var inputs = new Dictionary<string, Tensor<T>>
        {
            ["phoneme_ids"] = phonemes
        };

        if (speakerEmb is not null)
        {
            inputs["speaker_embedding"] = speakerEmb;
        }

        var lengthScaleTensor = new Tensor<T>([1]);
        lengthScaleTensor[0] = NumOps.FromDouble(lengthScale);
        inputs["length_scale"] = lengthScaleTensor;

        var noiseScaleTensor = new Tensor<T>([1]);
        noiseScaleTensor[0] = NumOps.FromDouble(_noiseScale);
        inputs["noise_scale"] = noiseScaleTensor;

        var outputs = OnnxModel.Run(inputs);
        return outputs.Values.First();
    }

    private Tensor<T> ExpandByDurations(Tensor<T> encoded, Tensor<T> durations, double lengthScale)
    {
        // Sum durations to get output length
        int totalFrames = 0;
        int seqLen = durations.Rank >= 2 ? durations.Shape[1] : durations.Shape[0];

        for (int i = 0; i < seqLen; i++)
        {
            // Use 3D indexing only if Rank >= 3, otherwise fall back to 1D indexing
            double dur = durations.Rank >= 3
                ? NumOps.ToDouble(durations[0, i, 0])
                : NumOps.ToDouble(durations[i]);
            totalFrames += Math.Max(1, (int)(dur * lengthScale));
        }

        int hiddenDim = encoded.Rank >= 3 ? encoded.Shape[^1] : _hiddenDim;
        var expanded = new Tensor<T>([1, totalFrames, hiddenDim]);

        int frameIdx = 0;
        for (int i = 0; i < seqLen && frameIdx < totalFrames; i++)
        {
            // Use 3D indexing only if Rank >= 3, otherwise fall back to 1D indexing
            double dur = durations.Rank >= 3
                ? NumOps.ToDouble(durations[0, i, 0])
                : NumOps.ToDouble(durations[i]);
            int numFrames = Math.Max(1, (int)(dur * lengthScale));

            for (int f = 0; f < numFrames && frameIdx < totalFrames; f++)
            {
                for (int d = 0; d < hiddenDim; d++)
                {
                    expanded[0, frameIdx, d] = encoded.Rank >= 3
                        ? encoded[0, i, d]
                        : NumOps.Zero;
                }
                frameIdx++;
            }
        }

        return expanded;
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
            _speakerEncoder?.Dispose();
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
