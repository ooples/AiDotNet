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
public partial class Tacotron2Model<T> : AudioNeuralNetworkBase<T>, ITextToSpeech<T>
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
        Tacotron2ModelOptions? options = null,
        OnnxModelOptions? onnxOptions = null)
        : base(architecture: architecture)
    {
        _options = options ?? new Tacotron2ModelOptions();
        _options.Validate();
        Options = _options;
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));
        if (acousticModelPath is null)
            throw new ArgumentNullException(nameof(acousticModelPath));

        _useNativeMode = false;
        _acousticModelPath = acousticModelPath;
        _vocoderPath = _options.VocoderPath;

        // Store parameters
        SampleRate = _options.SampleRate;
        NumMels = _options.NumMels;
        _speakingRate = _options.SpeakingRate;
        _maxDecoderSteps = _options.MaxDecoderSteps;
        _stopThreshold = _options.StopThreshold;
        _fftSize = _options.FftSize;
        _hopLength = _options.HopLength;
        _griffinLimIterations = _options.GriffinLimIterations;

        // Default architecture parameters (standard Tacotron2)
        _vocabSize = 148; // Standard phoneme vocabulary
        _embeddingDim = _options.EmbeddingDim;
        _encoderDim = _options.EncoderDim;
        _decoderDim = _options.DecoderDim;
        _attentionDim = _options.AttentionDim;
        _attentionFilters = _options.AttentionFilters;
        _prenetDim = _options.PrenetDim;
        _postnetEmbeddingDim = _options.PostnetEmbeddingDim;
        _numEncoderConvLayers = _options.NumEncoderConvLayers;
        _numPostnetConvLayers = _options.NumPostnetConvLayers;
        _numMelsPerFrame = _options.NumMelsPerFrame;

        // Initialize preprocessor
        _preprocessor = new TtsPreprocessor();

        // Load ONNX models
        var onnxOpts = onnxOptions ?? new OnnxModelOptions();
        _acousticModel = new OnnxModel<T>(acousticModelPath, onnxOpts);

        if (_options.VocoderPath is not null && _options.VocoderPath.Length > 0)
        {
            _vocoder = new OnnxModel<T>(_options.VocoderPath, onnxOpts);
        }
        else
        {
            // Use Griffin-Lim as fallback vocoder
            _griffinLim = new GriffinLim<T>(
                nFft: _options.FftSize,
                hopLength: _options.HopLength,
                iterations: _options.GriffinLimIterations);
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
        Tacotron2ModelOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new Tacotron2ModelOptions();
        _options.Validate();
        Options = _options;
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));

        _useNativeMode = true;

        // Store parameters
        SampleRate = _options.SampleRate;
        NumMels = _options.NumMels;
        _speakingRate = _options.SpeakingRate;
        _vocabSize = _options.VocabSize;
        _embeddingDim = _options.EmbeddingDim;
        _encoderDim = _options.EncoderDim;
        _decoderDim = _options.DecoderDim;
        _attentionDim = _options.AttentionDim;
        _attentionFilters = _options.AttentionFilters;
        _prenetDim = _options.PrenetDim;
        _postnetEmbeddingDim = _options.PostnetEmbeddingDim;
        _numEncoderConvLayers = _options.NumEncoderConvLayers;
        _numPostnetConvLayers = _options.NumPostnetConvLayers;
        _numMelsPerFrame = _options.NumMelsPerFrame;
        _maxDecoderSteps = _options.MaxDecoderSteps;
        _stopThreshold = _options.StopThreshold;
        _fftSize = _options.FftSize;
        _hopLength = _options.HopLength;
        _griffinLimIterations = _options.GriffinLimIterations;

        // Initialize preprocessor
        _preprocessor = new TtsPreprocessor();

        // Create Griffin-Lim vocoder
        _griffinLim = new GriffinLim<T>(
            nFft: _options.FftSize,
            hopLength: _options.HopLength,
            iterations: _options.GriffinLimIterations);

        // Initialize available voices
        AvailableVoices = GetDefaultVoices();

        // Initialize training components
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        // Paper training configuration (Shen et al. 2018, sec. 3): "Adam optimizer with beta1 = 0.9,
        // beta2 = 0.999, eps = 10^-6 and a learning rate of 10^-3 exponentially decaying to 10^-5".
        // Fix the published coefficients, disable AiDotNet's adaptive Adam extensions, and apply the
        // paper's coupled L2 weight of 10^-6. Callers can still supply their own optimizer.
        _optimizer = optimizer ?? CreateFixedAdamOptimizer(
            initialLearningRate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-6,
            l2RegularizationStrength: 1e-6,
            // §3.1: 1e-3 "exponentially decaying to 1e-5 starting after 50,000 iterations". The paper gives no rate;
            // halving every 50k steps to the 1e-5 floor is Rayhane-mamah/Tacotron-2's reading of the same sentence.
            learningRateScheduler: new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(1e-3,
                step => step < 50000 ? 1.0 : Math.Max(1e-2, Math.Pow(0.5, (step - 50000) / 50000.0))),
            // Gradient-norm clip 1 (NVIDIA tacotron2 grad_clip_thresh).
            maxGradientNorm: 1.0);

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


    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.


    /// <summary>
    /// Parameters cannot be written while the model is backed by a loaded ONNX graph: the weights
    /// belong to that graph, not to this instance.
    /// </summary>
    /// <remarks>
    /// Replaces a hand-written throw that used to sit inside UpdateParameters. The base checks this
    /// on every mutating entry point rather than the one member the throw happened to guard, and
    /// reading -- ParameterCount and GetParameters -- stays available either way.
    /// </remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <summary>
    /// Trains the model on input data.
    /// </summary>
    // Stored target for teacher forcing during ForwardForTraining




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


    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>


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




    private Tensor<T> ForwardOnnx(Tensor<T> phonemes)
    {
        if (_acousticModel is null)
            throw new InvalidOperationException("Acoustic model not loaded.");

        return _acousticModel.Run(phonemes);
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
