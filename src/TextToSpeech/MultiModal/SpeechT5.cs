using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>SpeechT5: SpeechT5: Unified-Modal Encoder-Decoder Pre-Training.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "SpeechT5: Unified-Modal Encoder-Decoder Pre-Training" (Ao et al., 2022)</item></list></para><para><b>For Beginners:</b> SpeechT5: SpeechT5: Unified-Modal Encoder-Decoder Pre-Training.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a SpeechT5 model for unified-modal speech processing
/// // with encoder-decoder pre-training for TTS and other tasks
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new SpeechT5&lt;double&gt;(architecture, "speecht5.onnx");
///
/// // Training mode with native layers
/// var trainModel = new SpeechT5&lt;double&gt;(architecture, new SpeechT5Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing",
    "https://arxiv.org/abs/2110.07205",
    Year = 2022,
    Authors = "Ao et al."
)]
public class SpeechT5<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly SpeechT5Options _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public SpeechT5(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SpeechT5Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new SpeechT5Options();
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

    public SpeechT5(
        NeuralNetworkArchitecture<T> architecture,
        SpeechT5Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new SpeechT5Options();
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

    /// <summary>
    /// Synthesizes speech from text.
    /// Per Ao et al. (2022): Shared encoder-decoder for ASR/TTS/voice-conversion with task-specific pre/post-nets.
    /// </summary>
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
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        Layers.AddRange(BuildSpeechT5Layers());
    }

    /// <summary>
    /// Builds the paper-faithful SpeechT5 layer stack (Ao et al. 2022, "SpeechT5:
    /// Unified-Modal Encoder-Decoder Pre-Training"). SpeechT5 is a shared
    /// Transformer encoder-decoder with modality-specific pre/post-nets — NOT a
    /// VITS flow/HiFi-GAN model. Each Transformer block is a canonical Pre-LN
    /// residual block (y = x + SelfAttn(LN(x)); z = y + FFN(LN(y)); Vaswani 2017
    /// §3.1). The residual connections are what keep deep-stack training stable;
    /// the previous VITS-layer approximation used a residual-LESS flat
    /// MHA→Norm→FFN→Norm stack that washed out the signal and diverged with more
    /// training (the #1380 collapse mechanism).
    /// </summary>
    private System.Collections.Generic.IEnumerable<ILayer<T>> BuildSpeechT5Layers()
    {
        int d = _options.EncoderDim > 0 ? _options.EncoderDim : 192;
        int mel = _options.MelChannels > 0 ? _options.MelChannels : 80;
        int heads = _options.NumHeads > 0 ? _options.NumHeads : 8;
        int ffn = d * 4;
        int blocks =
            (_options.NumEncoderLayers > 0 ? _options.NumEncoderLayers : 6)
            + (_options.NumDecoderLayers > 0 ? _options.NumDecoderLayers : 6);

        // Speech pre-net: project input mel features → model dimension.
        yield return new DenseLayer<T>(d, new IdentityActivation<T>() as IActivationFunction<T>);
        // Shared Transformer encoder-decoder stack (Pre-LN residual blocks).
        for (int i = 0; i < blocks; i++)
            yield return new TransformerEncoderBlock<T>(
                hiddenSize: d,
                numHeads: heads,
                ffnDim: ffn,
                dropoutRate: _options.DropoutRate
            );
        // Speech post-net: project model dimension → mel channels.
        yield return new DenseLayer<T>(mel, new IdentityActivation<T>() as IActivationFunction<T>);
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
            TrainWithTape(input, expected);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers)
        {
            int c = (int)l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SpeechT5-Native" : "SpeechT5-ONNX",
            Description = "SpeechT5 TTS",
            FeatureCount = _options.HiddenDim,
        };
        m.AdditionalInfo["Architecture"] = "SpeechT5";
        m.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        m.AdditionalInfo["HiddenDim"] = base.HiddenDim;
        m.AdditionalInfo["SampleRate"] = base.SampleRate;
        m.AdditionalInfo["MelChannels"] = base.MelChannels;
        m.AdditionalInfo["HopSize"] = base.HopSize;
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.SampleRate);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.EncoderDim);
        writer.Write(_options.HiddenDim);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumFlowSteps);
        writer.Write(_options.NumHeads);
        writer.Write(_options.MelChannels);
        writer.Write(_options.HopSize);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.SampleRate = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _options.EncoderDim = reader.ReadInt32();
        _options.HiddenDim = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumEncoderLayers = reader.ReadInt32();
        _options.NumFlowSteps = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.MelChannels = reader.ReadInt32();
        _options.HopSize = reader.ReadInt32();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new SpeechT5<T>(Architecture, mp, _options);
        return new SpeechT5<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SpeechT5<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
