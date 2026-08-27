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
public partial class Vocos<T> : VocoderBase<T>
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

    // SampleRate, MelChannels and UpsampleFactor now come from VocoderBase - see BigVGAN for why
    // these three restated what the base already derives from the same _options fields.

    /// <summary>
    /// Converts mel to waveform using Vocos' ConvNeXt backbone predicting STFT coefficients.
    /// Per the paper (Siuzdak, 2023): a ConvNeXt backbone processes mel features at mel-spectrogram resolution (no learnable upsampling). The output head predicts STFT magnitude and wrapped phase, and the waveform is reconstructed via iSTFT.
    /// </summary>
    public override Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
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

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
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
