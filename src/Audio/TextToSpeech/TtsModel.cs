using AiDotNet.LearningRateSchedulers;
using System.Diagnostics;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Text-to-speech model for synthesizing speech from text.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This TTS model uses a two-stage pipeline:
/// 1. Acoustic Model (FastSpeech2): Converts text/phonemes to mel spectrogram
/// 2. Vocoder (HiFi-GAN or Griffin-Lim): Converts mel spectrogram to audio waveform
/// </para>
/// <para><b>For Beginners:</b> Text-to-Speech works like this:
/// 1. Your text is converted to phonemes (speech sounds)
/// 2. The acoustic model predicts what the speech should "look like" (mel spectrogram)
/// 3. The vocoder makes it actually sound like speech
///
/// This class supports two modes:
/// - ONNX Mode: Load pretrained FastSpeech2/HiFi-GAN models for instant synthesis
/// - Native Mode: Train your own TTS model from scratch
///
/// Usage (ONNX Mode):
/// <code>
/// var tts = new TtsModel&lt;float&gt;(
///     architecture,
///     acousticModelPath: "path/to/fastspeech2.onnx",
///     vocoderModelPath: "path/to/hifigan.onnx");
///
/// var audio = tts.Synthesize("Hello, world!");
/// </code>
///
/// Usage (Native Training Mode):
/// <code>
/// var tts = new TtsModel&lt;float&gt;(
///     architecture,
///     optimizer: new AdamOptimizer&lt;float&gt;(),
///     lossFunction: new MeanSquaredErrorLoss&lt;float&gt;());
///
/// tts.Train(phonemeInput, expectedMelSpectrogram);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.TextToSpeech)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("FastSpeech 2: Fast and High-Quality End-to-End Text to Speech", "https://arxiv.org/abs/2006.04558", Year = 2021, Authors = "Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu")]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-9,
                Schedule = LearningRateSchedulerType.Noam, WarmupSteps = 4000,
                ReferenceBatchSize = 48,
                Source = "Ren et al. 2021, Sec. 4.1: Adam with beta1 0.9, beta2 0.98, eps 1e-9 following the learning rate schedule of Vaswani et al. 2017, batch size 48 sentences, 160k steps. No constant rate is declared because that schedule states none.")]
public partial class TtsModel<T> : AudioNeuralNetworkBase<T>, ITextToSpeech<T>
{
    private readonly TtsOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Execution Mode

    /// <summary>
    /// Whether the model is operating in native training mode.
    /// When false, the model uses ONNX for inference only.
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// Path to the acoustic model ONNX file.
    /// </summary>
    private readonly string? _acousticModelPath;

    /// <summary>
    /// Path to the vocoder model ONNX file.
    /// </summary>
    private readonly string? _vocoderModelPath;

    /// <summary>
    /// ONNX acoustic model (FastSpeech2 or similar).
    /// </summary>
    private readonly OnnxModel<T>? _acousticModel;

    /// <summary>
    /// ONNX vocoder model (HiFi-GAN or similar).
    /// </summary>
    private readonly OnnxModel<T>? _vocoder;

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
    /// Speaking rate multiplier. 1.0 = normal speed.
    /// </summary>
    private readonly double _speakingRate;

    /// <summary>
    /// Pitch shift in semitones. 0 = normal.
    /// </summary>
    private readonly double _pitchShift;

    /// <summary>
    /// Energy/volume level. 1.0 = normal.
    /// </summary>
    private readonly double _energy;

    /// <summary>
    /// Speaker ID for multi-speaker models.
    /// </summary>
    private readonly int? _speakerId;

    /// <summary>
    /// Language code for multi-lingual models.
    /// </summary>
    private readonly string? _language;

    /// <summary>
    /// Whether to use Griffin-Lim as fallback vocoder.
    /// </summary>
    private readonly bool _useGriffinLimFallback;

    /// <summary>
    /// Number of Griffin-Lim iterations.
    /// </summary>
    private readonly int _griffinLimIterations;

    /// <summary>
    /// FFT size for Griffin-Lim.
    /// </summary>
    private readonly int _fftSize;

    /// <summary>
    /// Hop length for Griffin-Lim.
    /// </summary>
    private readonly int _hopLength;

    /// <summary>
    /// Hidden dimension for the acoustic model.
    /// </summary>
    private readonly int _hiddenDim;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Number of encoder layers.
    /// </summary>
    private readonly int _numEncoderLayers;

    /// <summary>
    /// Number of decoder layers.
    /// </summary>
    private readonly int _numDecoderLayers;

    /// <summary>
    /// Maximum phoneme sequence length.
    /// </summary>
    private readonly int _maxPhonemeLength;

    #endregion

    #region Public Properties

    /// <summary>
    /// Gets whether the model is ready for synthesis.
    /// </summary>
    public bool IsReady => _acousticModel?.IsLoaded == true &&
        (_vocoder?.IsLoaded == true || _useGriffinLimFallback);

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

    #region Constructors

    /// <summary>
    /// Creates a TtsModel for ONNX inference with pretrained models.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="acousticModelPath">Required path to acoustic model ONNX file (e.g., FastSpeech2).</param>
    /// <param name="vocoderModelPath">Optional path to vocoder ONNX file (e.g., HiFi-GAN). If null, uses Griffin-Lim.</param>
    /// <param name="sampleRate">Output sample rate in Hz. Default is 22050 (standard for TTS).</param>
    /// <param name="numMels">Number of mel spectrogram channels. Default is 80.</param>
    /// <param name="speakingRate">Speaking rate multiplier. 1.0 = normal speed. Default is 1.0.</param>
    /// <param name="pitchShift">Pitch shift in semitones. 0 = normal. Default is 0.</param>
    /// <param name="energy">Energy/volume level. 1.0 = normal. Default is 1.0.</param>
    /// <param name="speakerId">Speaker ID for multi-speaker models. Default is null.</param>
    /// <param name="language">Language code for multi-lingual models. Default is null.</param>
    /// <param name="useGriffinLimFallback">Whether to use Griffin-Lim as fallback. Default is true.</param>
    /// <param name="griffinLimIterations">Number of Griffin-Lim iterations. Default is 60.</param>
    /// <param name="fftSize">FFT size for Griffin-Lim. Default is 1024.</param>
    /// <param name="hopLength">Hop length for Griffin-Lim. Default is 256.</param>
    /// <param name="onnxOptions">ONNX runtime options.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you have pretrained TTS models.
    ///
    /// You need at least an acoustic model (converts text to mel spectrogram).
    /// The vocoder (converts mel to audio) is optional - Griffin-Lim can be used as fallback.
    ///
    /// Example:
    /// <code>
    /// var tts = new TtsModel&lt;float&gt;(
    ///     architecture,
    ///     acousticModelPath: "fastspeech2.onnx",
    ///     vocoderModelPath: "hifigan.onnx");
    /// </code>
    /// </para>
    /// </remarks>
    public TtsModel(
        NeuralNetworkArchitecture<T> architecture,
        string acousticModelPath,
        string? vocoderModelPath = null,
        int sampleRate = 22050,
        int numMels = 80,
        double speakingRate = 1.0,
        double pitchShift = 0.0,
        double energy = 1.0,
        int? speakerId = null,
        string? language = null,
        bool useGriffinLimFallback = true,
        int griffinLimIterations = 60,
        int fftSize = 1024,
        int hopLength = 256,
        OnnxModelOptions? onnxOptions = null,
        TtsOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new TtsOptions();
        Options = _options;
        if (acousticModelPath is null)
            throw new ArgumentNullException(nameof(acousticModelPath));

        _useNativeMode = false;
        _acousticModelPath = acousticModelPath;
        _vocoderModelPath = vocoderModelPath;

        // Store parameters
        SampleRate = sampleRate;
        NumMels = numMels;
        _speakingRate = speakingRate;
        _pitchShift = pitchShift;
        _energy = energy;
        _speakerId = speakerId;
        _language = language;
        _useGriffinLimFallback = useGriffinLimFallback;
        _griffinLimIterations = griffinLimIterations;
        _fftSize = fftSize;
        _hopLength = hopLength;
        _hiddenDim = 256;
        _numHeads = 4;
        _numEncoderLayers = 4;
        _numDecoderLayers = 4;
        _maxPhonemeLength = 256;

        // Initialize preprocessor
        _preprocessor = new TtsPreprocessor();

        // Load ONNX models with proper cleanup on failure
        OnnxModel<T>? acousticModel = null;
        OnnxModel<T>? vocoder = null;

        try
        {
            // Load acoustic model first
            acousticModel = new OnnxModel<T>(acousticModelPath, onnxOptions ?? new OnnxModelOptions());

            // Load vocoder if path provided
            if (vocoderModelPath is not null && vocoderModelPath.Length > 0)
            {
                vocoder = new OnnxModel<T>(vocoderModelPath, onnxOptions ?? new OnnxModelOptions());
            }

            // Assign to fields only after both succeed
            _acousticModel = acousticModel;
            _vocoder = vocoder;
            if (_vocoder is not null)
            {
                OnnxModel = _vocoder;
            }
        }
        catch
        {
            // Clean up any successfully created models before rethrowing
            acousticModel?.Dispose();
            vocoder?.Dispose();
            throw;
        }

        if (useGriffinLimFallback || _vocoder is null)
        {
            _griffinLim = new GriffinLim<T>(
                nFft: fftSize,
                hopLength: hopLength,
                iterations: griffinLimIterations);
        }

        // Initialize available voices
        AvailableVoices = GetDefaultVoices();

        // Default loss function (MSE is standard for TTS mel-spectrogram prediction)
        _lossFunction = new MeanSquaredErrorLoss<T>();

        // Initialize layers (empty for ONNX mode)
        InitializeLayers();
    }

    /// <summary>
    /// Creates a TtsModel for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sampleRate">Output sample rate in Hz. Default is 22050 (standard for TTS).</param>
    /// <param name="numMels">Number of mel spectrogram channels. Default is 80.</param>
    /// <param name="speakingRate">Speaking rate multiplier. 1.0 = normal speed. Default is 1.0.</param>
    /// <param name="pitchShift">Pitch shift in semitones. 0 = normal. Default is 0.</param>
    /// <param name="energy">Energy/volume level. 1.0 = normal. Default is 1.0.</param>
    /// <param name="speakerId">Speaker ID for multi-speaker models. Default is null.</param>
    /// <param name="language">Language code for multi-lingual models. Default is null.</param>
    /// <param name="hiddenDim">Hidden dimension for acoustic model. Default is 256.</param>
    /// <param name="numHeads">Number of attention heads. Default is 4.</param>
    /// <param name="numEncoderLayers">Number of encoder layers. Default is 4.</param>
    /// <param name="numDecoderLayers">Number of decoder layers. Default is 4.</param>
    /// <param name="maxPhonemeLength">Maximum phoneme sequence length. Default is 256.</param>
    /// <param name="fftSize">FFT size for Griffin-Lim. Default is 1024.</param>
    /// <param name="hopLength">Hop length for Griffin-Lim. Default is 256.</param>
    /// <param name="griffinLimIterations">Number of Griffin-Lim iterations. Default is 60.</param>
    /// <param name="optimizer">Optimizer for training. If null, a default Adam optimizer is used.</param>
    /// <param name="lossFunction">Loss function for training. If null, MSE loss is used.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train your own TTS model.
    ///
    /// You'll need a dataset of (phoneme sequence, mel spectrogram) pairs.
    /// Training TTS from scratch requires significant data and compute resources.
    ///
    /// Example:
    /// <code>
    /// var tts = new TtsModel&lt;float&gt;(
    ///     architecture,
    ///     optimizer: new AdamOptimizer&lt;float&gt;(),
    ///     lossFunction: new MeanSquaredErrorLoss&lt;float&gt;());
    ///
    /// // Train on your dataset
    /// tts.Train(phonemeInput, expectedMelSpectrogram);
    /// </code>
    /// </para>
    /// </remarks>
    public TtsModel(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 22050,
        int numMels = 80,
        double speakingRate = 1.0,
        double pitchShift = 0.0,
        double energy = 1.0,
        int? speakerId = null,
        string? language = null,
        int hiddenDim = 256,
        int numHeads = 4,
        int numEncoderLayers = 4,
        int numDecoderLayers = 4,
        int maxPhonemeLength = 256,
        int fftSize = 1024,
        int hopLength = 256,
        int griffinLimIterations = 60,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        TtsOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new TtsOptions();
        Options = _options;
        _useNativeMode = true;
        _acousticModelPath = null;
        _vocoderModelPath = null;

        // Store parameters
        SampleRate = sampleRate;
        NumMels = numMels;
        _speakingRate = speakingRate;
        _pitchShift = pitchShift;
        _energy = energy;
        _speakerId = speakerId;
        _language = language;
        _useGriffinLimFallback = true;
        _griffinLimIterations = griffinLimIterations;
        _fftSize = fftSize;
        _hopLength = hopLength;
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _maxPhonemeLength = maxPhonemeLength;

        // Initialize preprocessor
        _preprocessor = new TtsPreprocessor();

        // Create Griffin-Lim for audio generation from mel
        _griffinLim = new GriffinLim<T>(
            nFft: fftSize,
            hopLength: hopLength,
            iterations: griffinLimIterations);

        // Initialize optimizer and loss function
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Initialize available voices
        AvailableVoices = GetDefaultVoices();

        // Initialize layers
        InitializeLayers();
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the layers for the TTS model.
    /// </summary>
    /// <remarks>
    /// Follows the golden standard pattern:
    /// 1. Check if in native mode (ONNX mode returns early)
    /// 2. Use Architecture.Layers if provided by user
    /// 3. Fall back to LayerHelper.CreateDefaultTtsLayers() otherwise
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            int maxMelFrames = (SampleRate * 30) / _hopLength; // Max 30 seconds
            int phonemeVocabSize = 128;
            Layers.AddRange(LayerHelper<T>.CreateDefaultTtsLayers(
                vocabSize: phonemeVocabSize,
                textHiddenDim: _hiddenDim,
                audioHiddenDim: _hiddenDim,
                numEncoderLayers: _numEncoderLayers,
                numDecoderLayers: _numDecoderLayers,
                numHeads: _numHeads,
                maxTextLength: _maxPhonemeLength,
                maxMelFrames: maxMelFrames,
                numMels: NumMels,
                dropoutRate: 0.0));
        }
    }

    #endregion

    #region Helper Methods

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

        var stopwatch = Stopwatch.StartNew();

        // Override options with parameters
        double effectiveRate = Math.Abs(speakingRate - 1.0) > 0.01 ? speakingRate : _speakingRate;

        // Preprocess text to phonemes
        var phonemes = _preprocessor.TextToPhonemes(text);

        // Generate mel spectrogram
        var melSpectrogram = GenerateMelSpectrogram(phonemes, voiceId);

        // Apply rate modifications
        if (Math.Abs(effectiveRate - 1.0) > 0.01)
        {
            melSpectrogram = ModifyDuration(melSpectrogram, 1.0 / effectiveRate);
        }

        // Generate audio waveform
        Tensor<T> audio;
        if (_vocoder is not null)
        {
            audio = VocoderSynthesize(melSpectrogram);
        }
        else if (_griffinLim is not null)
        {
            audio = GriffinLimSynthesize(melSpectrogram);
        }
        else
        {
            throw new InvalidOperationException("No vocoder available.");
        }

        // Apply energy/volume
        if (Math.Abs(_energy - 1.0) > 0.01)
        {
            var result = new Tensor<T>(audio._shape);
            for (int i = 0; i < audio.Length; i++)
            {
                result[i] = NumOps.Multiply(audio[i], NumOps.FromDouble(_energy));
            }
            audio = result;
        }

        stopwatch.Stop();

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
        throw new NotSupportedException("Voice cloning is not supported by this TTS model.");
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
        throw new NotSupportedException("Emotion control is not supported by this TTS model.");
    }

    /// <summary>
    /// Extracts speaker embedding from reference audio for voice cloning.
    /// </summary>
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        throw new NotSupportedException("Speaker embedding extraction is not supported by this TTS model.");
    }

    /// <summary>
    /// Starts a streaming synthesis session.
    /// </summary>
    public IStreamingSynthesisSession<T> StartStreamingSession(string? voiceId = null, double speakingRate = 1.0)
    {
        throw new NotSupportedException("Streaming synthesis is not supported by this TTS model.");
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // For TTS, input is text not audio
        // This method is used for reference audio in voice cloning scenarios
        if (MelSpec is not null)
        {
            return MelSpec.Forward(rawAudio);
        }

        return rawAudio;
    }

    /// <summary>
    /// Postprocesses model output into the final result format.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // For TTS, postprocessing is handled in Synthesize method
        return modelOutput;
    }

    /// <summary>
    /// Makes a prediction using the model.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!_useNativeMode)
        {
            // ONNX inference
            if (_acousticModel is null)
                throw new InvalidOperationException("Acoustic model not loaded.");

            return _acousticModel.Run(input);
        }
        else
        {
            // Native forward pass through unified Layers list
            Tensor<T> output = input;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }
            return output;
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <summary>
    /// Trains the model on input data.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot train in ONNX inference mode. Use the native training constructor.");
        }

        // TRY/FINALLY, NOT TWO STATEMENTS. TrainWithTape can throw -- a shape mismatch, a diverged
        // loss, an OOM part-way through the tape -- and the bare call left the model stuck in
        // training mode when it did. The next Predict then runs with dropout live and stochastic
        // batch-norm statistics, so it silently returns a different answer for the same input, and
        // nothing about that failure points back at the exception that caused it.
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "TtsModel-Native" : "TtsModel-FastSpeech2-HiFiGAN",
            Description = "Text-to-speech model using FastSpeech2 acoustic model and HiFi-GAN/Griffin-Lim vocoder",
            FeatureCount = _maxPhonemeLength,
            Complexity = 2
        };
        metadata.AdditionalInfo["InputFormat"] = "Text/Phonemes";
        metadata.AdditionalInfo["OutputFormat"] = $"Audio ({SampleRate}Hz)";
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native Training" : "ONNX Inference";
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>


    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>


    #endregion

    #region Private Methods

    private Tensor<T> GenerateMelSpectrogram(int[] phonemes, string? voiceId)
    {
        if (!_useNativeMode)
        {
            if (_acousticModel is null)
                throw new InvalidOperationException("Acoustic model not loaded.");

            // Create phoneme tensor
            var phonemeTensor = new Tensor<T>([1, phonemes.Length]);
            for (int i = 0; i < phonemes.Length; i++)
            {
                phonemeTensor[0, i] = NumOps.FromDouble(phonemes[i]);
            }

            // Add speaker ID if multi-speaker
            var inputs = new Dictionary<string, Tensor<T>>
            {
                ["phoneme_ids"] = phonemeTensor
            };

            if (_speakerId.HasValue)
            {
                var speakerTensor = new Tensor<T>([1]);
                speakerTensor[0] = NumOps.FromDouble(_speakerId.Value);
                inputs["speaker_id"] = speakerTensor;
            }

            var outputs = _acousticModel.Run(inputs);
            return outputs.Values.First();
        }
        else
        {
            // Native mode - run through encoder/decoder layers
            var phonemeTensor = new Tensor<T>([1, phonemes.Length]);
            for (int i = 0; i < phonemes.Length; i++)
            {
                phonemeTensor[0, i] = NumOps.FromDouble(phonemes[i]);
            }

            return Predict(phonemeTensor);
        }
    }

    private Tensor<T> ModifyDuration(Tensor<T> melSpectrogram, double factor)
    {
        int originalFrames = melSpectrogram.Shape[^2];
        int numMels = melSpectrogram.Shape[^1];
        int newFrames = (int)(originalFrames * factor);

        var modified = new Tensor<T>([1, newFrames, numMels]);

        for (int f = 0; f < newFrames; f++)
        {
            double srcFrame = f / factor;
            int srcIdx = Math.Min((int)srcFrame, originalFrames - 1);

            for (int m = 0; m < numMels; m++)
            {
                modified[0, f, m] = melSpectrogram.Rank == 3
                    ? melSpectrogram[0, srcIdx, m]
                    : melSpectrogram[srcIdx, m];
            }
        }

        return modified;
    }

    private Tensor<T> VocoderSynthesize(Tensor<T> melSpectrogram)
    {
        if (_vocoder is null)
            throw new InvalidOperationException("Vocoder not loaded.");

        return _vocoder.Run(melSpectrogram);
    }

    private Tensor<T> GriffinLimSynthesize(Tensor<T> melSpectrogram)
    {
        if (_griffinLim is null)
            throw new InvalidOperationException("Griffin-Lim not available.");

        // Griffin-Lim expects 2D mel spectrogram [frames, mels]
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
