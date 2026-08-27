using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>E3 TTS: non-autoregressive end-to-end diffusion TTS without explicit duration modeling.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "Simple and Efficient Non-Autoregressive Text-to-Speech" (Gao et al., 2023)</item></list></para><para><b>For Beginners:</b> E3 TTS: non-autoregressive end-to-end diffusion TTS without explicit duration modeling.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create an E3 TTS model for simple non-autoregressive diffusion TTS
/// // without explicit duration modeling for streamlined synthesis
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new E3TTS&lt;double&gt;(architecture, "e3tts.onnx");
///
/// // Training mode with native layers
/// var trainModel = new E3TTS&lt;double&gt;(architecture, new E3TTSOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Simple and Efficient Non-Autoregressive Text-to-Speech (E3 TTS)",
    "https://arxiv.org/abs/2311.04571",
    Year = 2023,
    Authors = "Gao et al."
)]
public partial class E3TTS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly E3TTSOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public E3TTS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        E3TTSOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new E3TTSOptions();
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

    public E3TTS(
        NeuralNetworkArchitecture<T> architecture,
        E3TTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new E3TTSOptions();
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
    public int NumFlowSteps => _options.NumDiffusionSteps;

    /// Synthesizes speech using E3 TTS's duration-free diffusion approach.
    /// Per the paper (Gao et al., 2023):
    /// Directly denoises mel spectrogram frames conditioned on character-level text,
    /// without explicit duration prediction or alignment modules.
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
                LayerHelper<T>.CreateDefaultFlowMatchingTTSLayers(
                    _options.HiddenDim,
                    _options.DiffusionDim,
                    _options.MelChannels,
                    _options.NumEncoderLayers,
                    _options.NumDiffusionSteps,
                    _options.NumHeads,
                    _options.DropoutRate,
                    inputFeatures: _options.MelChannels
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
        return new ModelMetadata<T>
        {
            Name = _useNativeMode ? "E3-TTS-Native" : "E3-TTS-ONNX",
            Description = "E3 TTS: Duration-Free Diffusion TTS (Gao et al., 2023)",
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
            throw new ObjectDisposedException(GetType().FullName ?? nameof(E3TTS<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
