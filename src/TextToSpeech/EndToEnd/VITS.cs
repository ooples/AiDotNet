using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// VITS: end-to-end TTS with conditional VAE, normalizing flows, and adversarial training for parallel high-quality synthesis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" (Kim et al., 2021)</item></list></para>
/// <para><b>For Beginners:</b> VITS (Variational Inference with adversarial learning for end-to-end
/// Text-to-Speech) was a breakthrough model that unified the entire TTS pipeline into a single
/// end-to-end architecture. Previous systems had separate components (text encoder, acoustic model,
/// vocoder) that were trained independently and could introduce errors at each boundary.
/// VITS combines three key techniques:
/// (1) A Conditional Variational Autoencoder (CVAE) that learns a latent representation of speech,
/// (2) Normalizing flows that transform simple distributions into complex speech distributions,
/// (3) Adversarial training (like GANs) that ensures the generated speech sounds natural.
/// During training, it uses Monotonic Alignment Search (MAS) to learn text-to-speech alignment
/// without external alignment tools. At inference, it generates high-quality speech in parallel
/// (all at once), making it fast while maintaining natural prosody.</para>
/// <example>
/// <code>
/// // Create a VITS model for end-to-end TTS with conditional VAE,
/// // normalizing flows, and adversarial training for parallel synthesis
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new VITS&lt;double&gt;(architecture, "vits.onnx");
///
/// // Training mode with native layers
/// var trainModel = new VITS&lt;double&gt;(architecture, new VITSOptions());
/// </code>
/// </example>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech",
    "https://arxiv.org/abs/2106.06103",
    Year = 2021,
    Authors = "Kim et al."
)]
public partial class VITS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly VITSOptions _options;
    // Not readonly: a restore rewrites _options, and the default optimizer is BUILT FROM
    // those options, so it has to be rebuilt afterwards or the model keeps running on the
    // coefficients it happened to be constructed with.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <summary>
    /// Whether <see cref="_optimizer"/> is the one this model built from its options rather than
    /// one the caller supplied. Only the former may be rebuilt when a restore rewrites those
    /// options; substituting AdamW for a caller's optimizer would change how the model trains.
    /// </summary>
    private readonly bool _usesDefaultOptimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="VITS{T}"/> class in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional model configuration options.</param>
    public VITS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        VITSOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new VITSOptions();
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

    /// <summary>
    /// Initializes a new instance of the <see cref="VITS{T}"/> class in native training/inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Optional model configuration options.</param>
    /// <param name="optimizer">Optional gradient-based optimizer for training.</param>
    public VITS(
        NeuralNetworkArchitecture<T> architecture,
        VITSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new VITSOptions();
        _useNativeMode = true;
        _usesDefaultOptimizer = optimizer is null;
        _optimizer = optimizer ?? CreatePaperOptimizer();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;

    /// <summary>
    /// Gets the hidden dimension size. Intentionally hides base HiddenDim to expose options-driven value.
    /// </summary>
    public new int HiddenDim => _options.HiddenDim;

    /// <summary>
    /// Gets the number of normalizing flow steps used to transform between prior and posterior distributions.
    /// </summary>
    public int NumFlowSteps => _options.NumFlowSteps;

    /// <summary>
    /// Synthesizes speech from text using VITS' VAE + normalizing flow + HiFi-GAN decoder pipeline.
    /// </summary>
    /// <param name="text">The input text to synthesize.</param>
    /// <returns>A tensor containing the generated waveform.</returns>
    /// <remarks>
    /// <para>Per the paper (Kim et al., 2021), the inference pipeline:</para>
    /// <para>(1) Text encoder: transformer encoder produces text hidden states h_text.</para>
    /// <para>(2) Stochastic duration predictor: predicts phoneme durations via a flow-based model.</para>
    /// <para>(3) Expand: h_text is expanded according to predicted durations to match audio frame rate.</para>
    /// <para>(4) Prior normalizing flow: transforms the expanded text features into latent variable z.</para>
    /// <para>(5) HiFi-GAN decoder: converts z into a raw audio waveform.</para>
    /// <para><b>For Beginners:</b> At inference time, VITS works in five steps: (1) encode the text into
    /// hidden representations, (2) predict how long each sound should last, (3) stretch the text
    /// representations to match those durations, (4) use normalizing flows to transform these into
    /// a rich latent representation that captures the nuances of speech, and (5) use the HiFi-GAN
    /// neural vocoder to convert that latent representation directly into an audio waveform.
    /// All of this happens in parallel (not one sample at a time), making VITS fast at inference.</para>
    /// </remarks>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var output = Predict(input);
        return PostprocessAudio(output);
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
    {
        int len = Math.Min(text.Length, _options.MaxTextLength);
        var t = new Tensor<T>([len]);
        for (int i = 0; i < len; i++)
            t[i] = NumOps.FromDouble(text[i] / 128.0);
        return t;
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    /// <summary>
    /// Builds the optimizer the paper prescribes: AdamW at the configured learning rate with
    /// beta = (Beta1, Beta2) and decoupled weight decay. Kim et al. 2021, section 4.1: AdamW, beta = (0.8, 0.99), weight decay 0.01, lr 2e-4.
    /// </summary>
    /// <remarks>
    /// Constructing AdamW with no options at all took the library defaults -- lr 1e-3 and
    /// beta = (0.9, 0.999) -- rather than the published recipe, and the resulting steps drove the
    /// loss UP on this stack across the conformance budget. Every coefficient stays a caller-visible
    /// option, and passing an explicit optimizer still bypasses this entirely.
    /// </remarks>
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
            });

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultVITSLayers(
                    _options.HiddenDim,
                    _options.InterChannels,
                    _options.FilterChannels,
                    _options.NumEncoderLayers,
                    _options.NumFlowSteps,
                    _options.NumDecoderLayers,
                    _options.NumHeads,
                    _options.DropoutRate,
                    inputFeatures: _options.MelChannels
                )
            );
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
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
    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = _useNativeMode ? "VITS-Native" : "VITS-ONNX",
            Description =
                "VITS: Conditional VAE with Adversarial Learning for End-to-End TTS (Kim et al., 2021)",
            FeatureCount = _options.HiddenDim,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["HiddenDim"] = _options.HiddenDim,
                ["Mode"] = _useNativeMode ? "Native" : "ONNX",
            },
        };
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(VITS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
