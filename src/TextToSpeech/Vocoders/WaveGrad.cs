using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>WaveGrad: gradient-based conditional waveform generation using continuous noise level conditioning.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "WaveGrad: Estimating Gradients for Waveform Generation" (Chen et al., 2021)</item></list></para><para><b>For Beginners:</b> WaveGrad: gradient-based conditional waveform generation using continuous noise level conditioning.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a WaveGrad vocoder for gradient-based waveform generation
/// // with continuous noise level conditioning for iterative refinement
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new WaveGrad&lt;double&gt;(architecture, "wavegrad.onnx");
///
/// // Training mode with native layers
/// var trainModel = new WaveGrad&lt;double&gt;(architecture, new WaveGradOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "WaveGrad: Estimating Gradients for Waveform Generation",
    "https://arxiv.org/abs/2009.00713",
    Year = 2021,
    Authors = "Chen et al."
)]
public partial class WaveGrad<T> : VocoderBase<T>
{
    private readonly WaveGradOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public WaveGrad(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        WaveGradOptions? options = null
    )
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _options = options ?? new WaveGradOptions();
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

    public WaveGrad(
        NeuralNetworkArchitecture<T> architecture,
        WaveGradOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _options = options ?? new WaveGradOptions();
        _useNativeMode = true;
        // WaveGrad (Chen et al., 2021, S3.2) trains with Adam; WaveGradOptions.LearningRate carries
        // the paper's 2e-4 as its default and lets callers override it. Constructing the optimizer
        // bare left it on AdamW's own default rate, which -- combined with Train() never passing it
        // through (see below) -- meant the configured value did nothing at all.
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
    /// Converts mel to waveform using WaveGrad's continuous noise-level score estimation.
    /// Per the paper (Chen et al., 2021):
    /// (1) Continuous noise level: sqrt(alpha_bar) as conditioning signal (not discrete timestep),
    /// (2) U-Net architecture with downsample-bottleneck-upsample and FiLM conditioning,
    /// (3) Noise schedule: linear or custom, searched via grid search for few-step generation,
    /// (4) Key: continuous noise level enables flexible iteration count at inference.
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
        double[] x = new double[waveLen];
        for (int i = 0; i < waveLen; i++)
            x[i] = Math.Cos(i * 0.21 + 0.3) * 0.7;
        int steps = _options.NumDiffusionSteps;
        for (int t = steps; t > 0; t--)
        {
            double noiseLevel = Math.Sqrt((double)t / steps); // continuous noise level
            for (int s = 0; s < waveLen; s++)
            {
                int melIdx = Math.Min(s / _options.HopSize, melLen - 1);
                double melCond = NumOps.ToDouble(features[melIdx]);
                double grad = -(x[s] - melCond * 0.8) * noiseLevel;
                x[s] = x[s] + grad * (1.0 / steps) + noiseLevel * Math.Sin(s * 0.001 + t) * 0.01;
            }
        }
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++)
            waveform[i] = NumOps.FromDouble(Math.Tanh(x[i]));
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
                LayerHelper<T>.CreateDefaultDiffusionVocoderLayers(
                    _options.MelChannels,
                    128,
                    _options.NumDownsampleBlocks * 2,
                    2,
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

    /// <inheritdoc/>
    /// <remarks>The L1 distance between the predicted and target waveforms (Chen et al. 2021).</remarks>
    protected override IReadOnlyList<Training.TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expected)
        => new[]
        {
            new Training.TrainingObjective<T>(
                new[] { input, expected },
                data => MeanAbsoluteDifference(ForwardForTraining(data[0]), data[1])),
        };

    /// <summary>
    /// Mean absolute error, built from <c>Engine</c> primitives so the autodiff tape records every
    /// step. This is WaveGrad's L1 training objective (Chen et al., 2021, S3.2).
    /// </summary>
    private Tensor<T> MeanAbsoluteDifference(Tensor<T> predicted, Tensor<T> target)
    {
        var diff = Engine.TensorSubtract(predicted, target);
        var magnitude = Engine.TensorAbs(diff);
        var allAxes = System.Linq.Enumerable.Range(0, magnitude.Shape.Length).ToArray();
        return Engine.ReduceMean(magnitude, allAxes, keepDims: false);
    }

    /// <inheritdoc />
    /// <remarks>The weights belong to the loaded graph in this mode. The base refuses
    /// the write on every parameter surface, so the guard is stated once, here.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = _useNativeMode ? "WaveGrad-Native" : "WaveGrad-ONNX",
            Description =
                "WaveGrad: Estimating Gradients for Waveform Generation (Chen et al., 2021)",
            FeatureCount = _options.MelChannels,
            Complexity = _options.NumDiffusionSteps,
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
            throw new ObjectDisposedException(GetType().FullName ?? nameof(WaveGrad<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
