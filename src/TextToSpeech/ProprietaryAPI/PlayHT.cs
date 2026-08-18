using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>PlayHT: AI voice generation with 2.0 turbo model for ultra-realistic speech synthesis.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> PlayHT provides ultra-realistic AI voice generation using their
/// 2.0 Turbo model. It supports instant voice cloning from short audio samples and offers
/// low-latency streaming synthesis ideal for real-time applications like chatbots and
/// voice assistants. This local implementation provides offline inference.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Generation,
///     inputSize: 256, outputSize: 24000);
///
/// var model = new PlayHT&lt;float&gt;(architecture, "playht.onnx");
/// Tensor&lt;float&gt; audio = model.Synthesize("Hello from PlayHT!");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ResearchPaper("PlayHT", "https://play.ht")]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public partial class PlayHT<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly PlayHTOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public PlayHT(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        PlayHTOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new PlayHTOptions();
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

    public PlayHT(
        NeuralNetworkArchitecture<T> architecture,
        PlayHTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new PlayHTOptions();
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
    public int NumFlowSteps => _options.NumFlowSteps;

    /// <summary>Synthesizes speech using PlayHT's API-compatible local inference pipeline.</summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return PostprocessAudio(OnnxModel.Run(input));
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
                LayerHelper<T>.CreateDefaultProprietaryTTSLayers(
                    _options.HiddenDim,
                    _options.HiddenDim,
                    _options.NumEncoderLayers,
                    _options.NumDecoderLayers,
                    _options.NumHeads,
                    _options.DropoutRate
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
        ThrowIfDisposed();
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

    /// <summary>
    /// Refuses parameter work on a disposed model, on every entry point rather than one.
    /// </summary>
    /// <remarks>
    /// This check used to live inside UpdateParameters, which meant ParameterCount, GetParameters
    /// and SetParameters reached a disposed model unguarded. The base calls this hook from all of
    /// them, so moving it here widens the guard and lets the hand-written UpdateParameters -- whose
    /// only other content was a walk the base already performs -- be deleted.
    /// </remarks>
    protected override void EnsureParametersReady()
    {
        ThrowIfDisposed();
        base.EnsureParametersReady();
    }

    // UpdateParameters folded one enumeration the base already folds. Removed under AIDN082.
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "PlayHT-Native" : "PlayHT-ONNX",
            Description = "PlayHT 2.0 Turbo TTS",
            FeatureCount = _options.HiddenDim,
        };
        m.AdditionalInfo["Architecture"] = "PlayHT";
        m.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        m.AdditionalInfo["HiddenDim"] = _options.HiddenDim;
        m.AdditionalInfo["SampleRate"] = base.SampleRate;
        m.AdditionalInfo["MelChannels"] = base.MelChannels;
        m.AdditionalInfo["HopSize"] = base.HopSize;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(PlayHT<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
