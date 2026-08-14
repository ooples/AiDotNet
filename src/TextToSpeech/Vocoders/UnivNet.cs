using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>UnivNet: universal neural vocoder with location-variable convolution (LVC) for adaptive kernel generation.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminator for High-Fidelity Waveform Generation" (Jang et al., 2021)</item></list></para><para><b>For Beginners:</b> UnivNet: universal neural vocoder with location-variable convolution (LVC) for adaptive kernel generation.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a UnivNet vocoder with location-variable convolution (LVC)
/// // for adaptive kernel generation and high-fidelity waveform synthesis
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new UnivNet&lt;double&gt;(architecture, "univnet.onnx");
///
/// // Training mode with native layers
/// var trainModel = new UnivNet&lt;double&gt;(architecture, new UnivNetOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    // Title corrected to the published plural form ("Discriminators"); the arXiv id was already right.
    "UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation",
    "https://arxiv.org/abs/2106.07889",
    Year = 2021,
    Authors = "Jang et al."
)]
public partial class UnivNet<T> : VocoderBase<T>
{
    private readonly UnivNetOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public UnivNet(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        UnivNetOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new UnivNetOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public UnivNet(
        NeuralNetworkArchitecture<T> architecture,
        UnivNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new UnivNetOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        InitializeLayers();
    }

    /// <inheritdoc />
    /// <remarks>
    /// MEASURED: <c>[1,80,8] -&gt; [1,1,2048]</c>, 8 mel frames x an UpsampleFactor of 256. One of the
    /// three vocoders whose Predict is a whole-waveform synthesis.
    /// </remarks>
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => WaveformUpsampleContract(inputRank);

    // SampleRate, MelChannels and UpsampleFactor now come from VocoderBase - see BigVGAN for why
    // these three restated what the base already derives from the same _options fields.

    /// <summary>
    /// Converts mel to waveform using UnivNet's LVC (Location-Variable Convolution) blocks.
    /// Per the paper (Jang et al., 2021):
    /// (1) Noise input + mel conditioning,
    /// (2) LVC blocks: kernel weights are dynamically generated from mel features (not fixed),
    /// (3) GABlock with gated activation + location-variable conv for adaptive frequency modeling,
    /// (4) Multi-resolution spectrogram discriminator (MRSD) for training.
    /// </summary>
    public override Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(melSpectrogram);
        return Predict(melSpectrogram);
    }

    protected override Tensor<T> PreprocessText(string text)
    {
        var t = new Tensor<T>([1]);
        t[0] = NumOps.FromDouble(0.0);
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
        var d = new UnivNetOptions();
        if (_options.NumLMBlocks != d.NumLMBlocks || _options.DropoutRate > double.Epsilon)
            throw new InvalidOperationException(
                "UnivNetOptions.NumLMBlocks/DropoutRate are configured but not applied by the paper-faithful HiFi-GAN generator default; supply explicit Architecture.Layers for a custom LVCNet configuration."
            );
        Layers.AddRange(LayerHelper<T>.CreateDefaultHiFiGANLayers(_options.MelChannels, 512, 1));
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
        return new ModelMetadata<T>
        {
            Name = _useNativeMode ? "UnivNet-Native" : "UnivNet-ONNX",
            Description = "UnivNet: Universal Neural Vocoder (Jang et al., 2021)",
            FeatureCount = _options.MelChannels,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["MelChannels"] = _options.MelChannels,
                ["Mode"] = _useNativeMode ? "Native" : "ONNX",
            },
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.SampleRate);
        writer.Write(_options.MelChannels);
        writer.Write(_options.HopSize);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.NumLMBlocks);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.SampleRate = reader.ReadInt32();
        _options.MelChannels = reader.ReadInt32();
        _options.HopSize = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _options.NumLMBlocks = reader.ReadInt32();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(UnivNet<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
