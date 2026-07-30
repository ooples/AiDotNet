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

/// <summary>Vocos: ConvNeXt-based vocoder that reconstructs waveform from Fourier coefficients (STFT magnitude + phase via ISTFT) instead of time-domain upsampling.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "Vocos: Closing the Gap between Time-Domain and Fourier-Based Neural Vocoders for High-Quality Audio Synthesis" (Siuzdak, 2023)</item></list></para><para><b>For Beginners:</b> Vocos: ConvNeXt-based vocoder that reconstructs waveform from Fourier coefficients (STFT magnitude + phase via ISTFT) instead of time-domain upsampling.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a Vocos vocoder with ConvNeXt backbone
/// // reconstructing waveforms from Fourier coefficients via ISTFT
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new Vocos&lt;double&gt;(architecture, "vocos.onnx");
///
/// // Training mode with native layers
/// var trainModel = new Vocos&lt;double&gt;(architecture, new VocosOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Vocos: Closing the Gap between Time-Domain and Fourier-Based Neural Vocoders for High-Quality Audio Synthesis",
    "https://arxiv.org/abs/2306.00814",
    Year = 2023,
    Authors = "Siuzdak"
)]
public class Vocos<T> : TtsModelBase<T>, IVocoder<T>
{
    private readonly VocosOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public Vocos(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        VocosOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new VocosOptions();
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

    public Vocos(
        NeuralNetworkArchitecture<T> architecture,
        VocosOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new VocosOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                UseAdaptiveLearningRate = false,
            });
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        InitializeLayers();
    }

    int IVocoder<T>.SampleRate => _options.SampleRate;
    int IVocoder<T>.MelChannels => _options.MelChannels;
    public int UpsampleFactor => _options.HopSize;

    /// <summary>
    /// Converts mel to waveform using Vocos' ConvNeXt backbone predicting STFT coefficients.
    /// Per the paper (Siuzdak, 2023): ConvNeXt V2 backbone processes mel features at mel-spectrogram resolution (no upsampling). Output heads predict STFT magnitude and instantaneous frequency (phase derivative). Waveform reconstructed via iSTFT. Achieves HiFi-GAN quality at 3x fewer parameters and faster inference.
    /// </summary>
    public Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(melSpectrogram);
        return ForwardNative(melSpectrogram);
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
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultVocosLayers(
                    _options.MelChannels,
                    _options.ConvNeXtDim,
                    _options.NumBackboneBlocks,
                    _options.IntermediateDim,
                    _options.FftSize / 2 + 1,
                    _options.DropoutRate,
                    _options.HopSize
                )
            );
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        return ForwardNative(input);
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
        return new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Vocos-Native" : "Vocos-ONNX",
            Description = "Vocos: ConvNeXt Fourier-Based Neural Vocoder (Siuzdak, 2023)",
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
        writer.Write(_options.FftSize);
        writer.Write(_options.ConvNeXtDim);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.NumBackboneBlocks);
        writer.Write(_options.IntermediateDim);
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
        _options.FftSize = reader.ReadInt32();
        _options.ConvNeXtDim = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _options.NumBackboneBlocks = reader.ReadInt32();
        _options.IntermediateDim = reader.ReadInt32();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new Vocos<T>(Architecture, mp, new VocosOptions(_options));
        return new Vocos<T>(Architecture, new VocosOptions(_options));
    }

    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var output = input;
        foreach (var layer in Layers)
            output = layer.Forward(output);
        return output;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(Vocos<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
