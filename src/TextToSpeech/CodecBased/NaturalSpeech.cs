using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>NaturalSpeech: fully end-to-end TTS with VAE, normalizing flow, and bidirectional prior/posterior for human-level quality.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality" (Tan et al., 2022)</item></list></para><para><b>For Beginners:</b> NaturalSpeech: fully end-to-end TTS with VAE, normalizing flow, and bidirectional prior/posterior for human-level quality.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a NaturalSpeech model for human-level end-to-end TTS
/// // with VAE, normalizing flow, and bidirectional prior/posterior
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new NaturalSpeech&lt;double&gt;(architecture, "naturalspeech.onnx");
///
/// // Training mode with native layers
/// var trainModel = new NaturalSpeech&lt;double&gt;(architecture, new NaturalSpeechOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality",
    "https://arxiv.org/abs/2205.04421",
    Year = 2022,
    Authors = "Tan et al."
)]
public partial class NaturalSpeech<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly NaturalSpeechOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public NaturalSpeech(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        NaturalSpeechOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new NaturalSpeechOptions();
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

    public NaturalSpeech(
        NeuralNetworkArchitecture<T> architecture,
        NaturalSpeechOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new NaturalSpeechOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay
            });
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int HiddenDim => _options.HiddenDim;
    public int NumFlowSteps => _options.NumFlowSteps;

    /// Synthesizes speech using NaturalSpeech's enhanced VITS pipeline.
    /// Per the paper (Tan et al., 2022):
    /// Adds large-scale pre-training, phoneme pre-training, differentiable duration modeling,
    /// and bidirectional prior/posterior modules to VITS for human-parity quality.
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

    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer ?? throw new InvalidOperationException("A native NaturalSpeech optimizer is not available in ONNX mode.");

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = _useNativeMode ? "NaturalSpeech-Native" : "NaturalSpeech-ONNX",
            Description = "NaturalSpeech: Human-Level End-to-End TTS (Tan et al., 2022)",
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
            throw new ObjectDisposedException(GetType().FullName ?? nameof(NaturalSpeech<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
