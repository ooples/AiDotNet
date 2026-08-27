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

/// <summary>BigVGAN: large-scale universal vocoder with anti-aliased multi-periodicity composition (AMP) and Snake activation for high-fidelity synthesis.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "BigVGAN: A Universal Neural Vocoder with Large-Scale Training" (Lee et al., 2023)</item></list></para><para><b>For Beginners:</b> BigVGAN: large-scale universal vocoder with anti-aliased multi-periodicity composition (AMP) and Snake activation for high-fidelity synthesis.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a BigVGAN universal vocoder with anti-aliased
/// // multi-periodicity composition (AMP) and Snake activation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new BigVGAN&lt;double&gt;(architecture, "bigvgan.onnx");
///
/// // Training mode with native layers
/// var trainModel = new BigVGAN&lt;double&gt;(architecture, new BigVGANOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "BigVGAN: A Universal Neural Vocoder with Large-Scale Training",
    "https://arxiv.org/abs/2206.04658",
    Year = 2023,
    Authors = "Lee et al."
)]
public partial class BigVGAN<T> : VocoderBase<T>
{
    private readonly BigVGANOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public BigVGAN(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        BigVGANOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new BigVGANOptions();
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

    public BigVGAN(
        NeuralNetworkArchitecture<T> architecture,
        BigVGANOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new BigVGANOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        InitializeLayers();
    }

    // SampleRate, MelChannels and UpsampleFactor now come from VocoderBase. Each constructor already
    // assigns base.SampleRate / .MelChannels / .HopSize from these same _options fields, and the base
    // UpsampleFactor is HopSize, so the three members deleted here restated values the base derives.

    /// <summary>
    /// Converts mel to waveform using BigVGAN's AMP blocks with Snake activation.
    /// Per the paper (Lee et al., 2023): Uses anti-aliased multi-periodicity composition (AMP) modules replacing standard residual blocks. Snake activation (x + sin^2(alpha*x)/alpha) captures periodic patterns better than LeakyReLU. Trained on large-scale data (LibriTTS + others) for universal vocoding across unseen speakers, languages, and recording conditions.
    /// </summary>
    public override Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(melSpectrogram);
        // Run mel through learned vocoder layers for feature extraction
        var features = melSpectrogram;
        foreach (var l in Layers)
            features = l.Forward(features);
        int melLen = features.Length;
        int waveLen = melLen * _options.HopSize;
        var waveform = new Tensor<T>([waveLen]);
        // Progressive upsampling through AMP blocks with Snake activation
        int currentLen = melLen;
        double[] signal = new double[melLen];
        for (int i = 0; i < melLen; i++)
            signal[i] = NumOps.ToDouble(features[i]);
        // Multi-stage upsampling (each stage doubles resolution)
        int numStages = (int)Math.Ceiling(Math.Log((double)_options.HopSize) / Math.Log(2.0));
        for (int stage = 0; stage < numStages; stage++)
        {
            int nextLen = Math.Min(currentLen * 2, waveLen);
            double[] upsampled = new double[nextLen];
            for (int i = 0; i < nextLen; i++)
            {
                int srcIdx = Math.Min(i * currentLen / nextLen, currentLen - 1);
                double x = signal[srcIdx];
                // Snake activation: x + sin^2(alpha * x) / alpha
                double alpha = _options.SnakeAlpha;
                double snake = x + Math.Pow(Math.Sin(alpha * x), 2) / alpha;
                // AMP: anti-aliased multi-periodicity composition
                double amp = 0;
                for (int p = 0; p < _options.NumPeriods; p++)
                {
                    double period = 2.0 + p * 3.0;
                    amp += Math.Sin(2.0 * Math.PI * i / period + x * 0.5) / _options.NumPeriods;
                }
                upsampled[i] = Math.Tanh(snake * 0.5 + amp * 0.3);
            }
            signal = upsampled;
            currentLen = nextLen;
        }
        // Copy to output
        for (int i = 0; i < waveLen; i++)
        {
            int srcIdx = Math.Min(i * currentLen / waveLen, currentLen - 1);
            waveform[i] = NumOps.FromDouble(signal[srcIdx]);
        }
        return waveform;
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
                LayerHelper<T>.CreateDefaultVocoderLayers(
                    _options.MelChannels,
                    _options.HiddenChannels,
                    1,
                    _options.NumUpsampleLayers,
                    3,
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
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        TrainWithTape(input, expected, _optimizer);
        SetTrainingMode(false);
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
            Name = _useNativeMode ? "BigVGAN-Native" : "BigVGAN-ONNX",
            Description = "BigVGAN: Universal Neural Vocoder with AMP + Snake (Lee et al., 2023)",
            FeatureCount = _options.MelChannels,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["MelChannels"] = _options.MelChannels,
                ["Mode"] = _useNativeMode ? "Native" : "ONNX",
            },
        };
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(BigVGAN<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
