using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>StyleTTS: style-based generative model for expressive TTS with style diffusion and adversarial training.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis" (Li et al., 2022)</item></list></para><para><b>For Beginners:</b> StyleTTS: style-based generative model for expressive TTS with style diffusion and adversarial training.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a StyleTTS model for style-based expressive speech synthesis
/// // with style diffusion and adversarial training for diverse prosody
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new StyleTTS&lt;double&gt;(architecture, "styletts.onnx");
///
/// // Training mode with native layers
/// var trainModel = new StyleTTS&lt;double&gt;(architecture, new StyleTTSOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis",
    "https://arxiv.org/abs/2205.15439",
    Year = 2022,
    Authors = "Li et al."
)]
public partial class StyleTTS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly StyleTTSOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public StyleTTS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        StyleTTSOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new StyleTTSOptions();
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

    public StyleTTS(
        NeuralNetworkArchitecture<T> architecture,
        StyleTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new StyleTTSOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int HiddenDim => _options.HiddenDim;
    public int NumFlowSteps => _options.NumStyleDiffusionSteps;

    /// Synthesizes speech using StyleTTS's style-based generation pipeline.
    /// Per the paper (Li et al., 2022):
    /// (1) Style encoder: extracts style vector from reference or diffusion prior,
    /// (2) Text encoder: processes input phonemes,
    /// (3) Acoustic decoder: style-conditioned mel generation,
    /// (4) HiFi-GAN decoder: mel → waveform.
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var output = Predict(input);
        return PostprocessAudio(output);
    }

    protected override Tensor<T> PreprocessText(string text)
    {
        int len = Math.Min(text.Length, _options.MaxTextLength);
        var t = new Tensor<T>([len]);
        for (int i = 0; i < len; i++)
            t[i] = NumOps.FromDouble(text[i] / 128.0);
        return t;
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultStyleTTSLayers(
                    _options.HiddenDim,
                    _options.StyleDim,
                    _options.MelChannels,
                    _options.NumEncoderLayers,
                    _options.NumStyleDiffusionSteps,
                    _options.NumDecoderLayers,
                    _options.NumHeads,
                    _options.DropoutRate,
                    inputFeatureDim: _options.MelChannels
                )
            );
    }

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
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "StyleTTS-Native" : "StyleTTS-ONNX",
            Description = "StyleTTS: Style-Based Generative TTS (Li et al., 2022)",
            FeatureCount = _options.HiddenDim,
        };
        m.AdditionalInfo["Architecture"] = "StyleTTS";
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
            throw new ObjectDisposedException(GetType().FullName ?? nameof(StyleTTS<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
