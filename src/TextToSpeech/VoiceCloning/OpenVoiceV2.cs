using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>OpenVoiceV2: OpenVoice V2: Improved Instant Voice Cloning.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "OpenVoice V2: Improved Instant Voice Cloning" (Qin et al., 2024)</item></list></para><para><b>For Beginners:</b> OpenVoiceV2: OpenVoice V2: Improved Instant Voice Cloning.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create an OpenVoice V2 model for improved instant voice cloning
/// // with enhanced tone color conversion and multilingual support
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new OpenVoiceV2&lt;double&gt;(architecture, "openvoicev2.onnx");
///
/// // Training mode with native layers
/// var trainModel = new OpenVoiceV2&lt;double&gt;(architecture, new OpenVoiceV2Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "OpenVoice V2: Improved Instant Voice Cloning",
    "https://arxiv.org/abs/2312.01479",
    Year = 2024,
    Authors = "Qin et al."
)]
public partial class OpenVoiceV2<T> : TtsModelBase<T>, IEndToEndTts<T>, IVoiceCloner<T>
{
    private readonly OpenVoiceV2Options _options;
    private OpenVoiceNative<T>? _native;
    private ITtsModel<T>? _baseSynthesizer;

    /// <summary>The separate native MeloTTS base voice used by the V2 release.</summary>
    public ITtsModel<T> BaseSynthesizer
    {
        get => _baseSynthesizer ??= new EndToEnd.MeloTTS<T>(Architecture);
        set => _baseSynthesizer = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>Loads the original V2 converter checkpoint, including its reference encoder.</summary>
    public void LoadConverterCheckpoint(string path)
    {
        ThrowIfDisposed();
        if (_native is null) throw new InvalidOperationException("A native converter is required.");
        _native.LoadCheckpoint(path);
    }

    /// <summary>Converts mono 22050 Hz source audio using source and target speaker embeddings.</summary>
    public Tensor<T> ConvertVoice(Tensor<T> audio, Tensor<T> sourceEmbedding, Tensor<T> targetEmbedding,
        double tau = .3, Tensor<T>? noise = null)
    {
        ThrowIfDisposed();
        if (_native is null) throw new InvalidOperationException("A native converter is required.");
        return _native.Convert(audio, sourceEmbedding, targetEmbedding, tau, noise);
    }

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public OpenVoiceV2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        OpenVoiceV2Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new OpenVoiceV2Options();
        Options = _options;
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public OpenVoiceV2(
        NeuralNetworkArchitecture<T> architecture,
        OpenVoiceV2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new OpenVoiceV2Options();
        Options = _options;
        _useNativeMode = true;
        // Honor the model's paper-derived optimizer options rather than AdamW's generic defaults.
        // The helper also keeps the released no-clipping policy consistent with the MeloTTS path.
        _optimizer = optimizer ?? CreatePaperOptimizer();
        _usesDefaultOptimizer = optimizer is null;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int HiddenDim => _options.HiddenDim;
    public int NumFlowSteps => 4;

    /// <summary>
    /// Synthesizes speech from text.
    /// Per MyShell (2024): VITS-based base TTS + enhanced tone color converter for instant cloning.
    /// </summary>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        var synthesizer = BaseSynthesizer;
        return OpenVoiceNative<T>.Resample(synthesizer.Synthesize(text), synthesizer.SampleRate);
    }

    public double MinReferenceDuration => 3.0;
    public int SpeakerEmbeddingDim => _options.SpeakerEmbeddingDim;

    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();
        var baseAudio = Synthesize(text);
        return ConvertVoice(baseAudio, ExtractSpeakerEmbedding(baseAudio), ExtractSpeakerEmbedding(referenceAudio));
    }

    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();
        if (_native is null) throw new InvalidOperationException("A native reference encoder is required.");
        return _native.Embedding(referenceAudio);
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (_options.SampleRate != 22050 || _options.HopSize != 256)
            throw new ArgumentException("The released OpenVoice converter uses 22050 Hz audio and hop size 256.");
        _native = new OpenVoiceNative<T>(Engine, _options.FftSize, _options.HopSize,
            _options.HiddenDim, _options.DecoderDim, _options.SpeakerEmbeddingDim, true);
        Layers.AddRange(_native.Layers);
        if (_options.Training.IncludeTrainingComponents) EnsurePaperTraining();
    }


    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (_native is null) throw new InvalidOperationException("A native converter is required.");
        var speaker = _native.Embedding(input);
        return _native.Convert(input, speaker, speaker, tau: 0);
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer ?? base.GetOrCreateBaseOptimizer();

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer()
        => new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                Beta1 = _options.Beta1,
                Beta2 = _options.Beta2,
                Epsilon = _options.Epsilon,
                WeightDecay = _options.WeightDecay,
                UseAMSGrad = false,
                UseAdaptiveBetas = false,
                // MeloTTS/OpenVoice only records the gradient norm here; it does not
                // clip the generator update in the released training loop.
                EnableGradientClipping = false,
            });

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "OpenVoiceV2-Native" : "OpenVoiceV2-ONNX",
            Description = "OpenVoiceV2 TTS",
            FeatureCount = _options.HiddenDim,
        };
        m.AdditionalInfo["Architecture"] = "OpenVoiceV2";
        m.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        m.AdditionalInfo["HiddenDim"] = base.HiddenDim;
        m.AdditionalInfo["SampleRate"] = base.SampleRate;
        m.AdditionalInfo["MelChannels"] = base.MelChannels;
        m.AdditionalInfo["HopSize"] = base.HopSize;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(OpenVoiceV2<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
