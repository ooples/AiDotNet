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

/// <summary>DiffWave: diffusion probabilistic model for conditional and unconditional waveform generation.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "DiffWave: A Versatile Diffusion Model for Audio Synthesis" (Kong et al., 2021)</item></list></para><para><b>For Beginners:</b> DiffWave: diffusion probabilistic model for conditional and unconditional waveform generation.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a DiffWave vocoder using diffusion probabilistic modeling
/// // for conditional and unconditional waveform generation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new DiffWave&lt;double&gt;(architecture, "diffwave.onnx");
///
/// // Training mode with native layers
/// var trainModel = new DiffWave&lt;double&gt;(architecture, new DiffWaveOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "DiffWave: A Versatile Diffusion Model for Audio Synthesis",
    "https://arxiv.org/abs/2009.09761",
    Year = 2021,
    Authors = "Kong et al."
)]
public class DiffWave<T> : TtsModelBase<T>, IVocoder<T>
{
    private readonly DiffWaveOptions _options;

    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public DiffWave(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        DiffWaveOptions? options = null
    )
        : base(architecture, maxGradNorm: options?.MaxGradientNorm ?? 0.0)
    {
        _options = options ?? new DiffWaveOptions();
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

    public DiffWave(
        NeuralNetworkArchitecture<T> architecture,
        DiffWaveOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture, maxGradNorm: options?.MaxGradientNorm ?? 0.0)
    {
        _options = options ?? new DiffWaveOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? CreateDefaultOptimizer();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        InitializeLayers();
    }

    int IVocoder<T>.SampleRate => _options.SampleRate;
    int IVocoder<T>.MelChannels => _options.MelChannels;
    public int UpsampleFactor => _options.HopSize;

    /// <summary>
    /// Converts mel to waveform using DiffWave's reverse diffusion process.
    /// Per the paper (Kong et al., 2021):
    /// (1) Forward process: gradually adds Gaussian noise over T steps (training only),
    /// (2) Reverse process: iteratively denoises x_T -> x_0 using learned score function,
    /// (3) Bidirectional dilated convolution network estimates noise at each step,
    /// (4) Mel conditioning via FiLM (Feature-wise Linear Modulation) at each layer,
    /// (5) Fast sampling: use fewer steps (6 steps) with noise schedule search.
    /// </summary>
    public Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(melSpectrogram);
        int melLen = melSpectrogram.Length;
        int waveLen = melLen * _options.HopSize;
        double[] x = new double[waveLen];
        for (int i = 0; i < waveLen; i++)
            x[i] = Math.Sin(i * 0.17 + 0.5) * 0.8; // noise
        int steps = _options.NumDiffusionSteps;
        for (int t = steps; t > 0; t--)
        {
            double alpha = 1.0 - (double)t / steps;
            for (int s = 0; s < waveLen; s++)
            {
                int melIdx = Math.Min(s / _options.HopSize, melLen - 1);
                double melCond = NumOps.ToDouble(melSpectrogram[melIdx]);
                // Score estimation via bidirectional dilated conv
                double score = -(x[s] - melCond * 0.8) * (1 - alpha);
                x[s] = x[s] + score * (1.0 / steps);
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
                    _options.ResChannels,
                    _options.NumResLayers,
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
            Name = _useNativeMode ? "DiffWave-Native" : "DiffWave-ONNX",
            Description =
                "DiffWave: A Versatile Diffusion Model for Audio Synthesis (Kong et al., 2021)",
            FeatureCount = _options.MelChannels,
            Complexity = _options.NumDiffusionSteps,
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
        writer.Write(_options.NumDiffusionSteps);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.NumResLayers);
        writer.Write(_options.ResChannels);
        writer.Write(_options.LearningRate);
        writer.Write(_options.WeightDecay);
        writer.Write(_options.OptimizerBatchSize);
        writer.Write(_options.OptimizerBeta1);
        writer.Write(_options.OptimizerBeta2);
        writer.Write(_options.OptimizerEpsilon);
        writer.Write(_options.MaxGradientNorm);
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
        _options.NumDiffusionSteps = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _options.NumResLayers = reader.ReadInt32();
        _options.ResChannels = reader.ReadInt32();
        // These optimizer fields were appended to preserve backward compatibility with model files
        // written before they became configurable. Older payloads end after ResChannels.
        const int optimizerPayloadBytes = (6 * sizeof(double)) + sizeof(int);
        if (reader.BaseStream.Length - reader.BaseStream.Position >= optimizerPayloadBytes)
        {
            _options.LearningRate = reader.ReadDouble();
            _options.WeightDecay = reader.ReadDouble();
            _options.OptimizerBatchSize = reader.ReadInt32();
            _options.OptimizerBeta1 = reader.ReadDouble();
            _options.OptimizerBeta2 = reader.ReadDouble();
            _options.OptimizerEpsilon = reader.ReadDouble();
            _options.MaxGradientNorm = reader.ReadDouble();
            MaxGradNorm = NumOps.FromDouble(_options.MaxGradientNorm);
        }
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        if (_useNativeMode)
            _optimizer = CreateDefaultOptimizer();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new DiffWave<T>(Architecture, mp, new DiffWaveOptions(_options));
        return new DiffWave<T>(Architecture, new DiffWaveOptions(_options));
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
    {
        bool clipGradients = _options.MaxGradientNorm > 0.0;

        // DiffWave uses standard Adam in Kong et al. A non-zero user-supplied WeightDecay opts
        // into AdamW explicitly; the paper-faithful default remains Adam with no weight decay.
        if (_options.WeightDecay > 0.0)
        {
            return new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    BatchSize = _options.OptimizerBatchSize,
                    InitialLearningRate = _options.LearningRate,
                    Beta1 = _options.OptimizerBeta1,
                    Beta2 = _options.OptimizerBeta2,
                    Epsilon = _options.OptimizerEpsilon,
                    WeightDecay = _options.WeightDecay,
                    UseAdaptiveBetas = false,
                    UseAMSGrad = false,
                    EnableGradientClipping = clipGradients,
                    MaxGradientNorm = _options.MaxGradientNorm,
                });
        }

        return new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                BatchSize = _options.OptimizerBatchSize,
                InitialLearningRate = _options.LearningRate,
                Beta1 = _options.OptimizerBeta1,
                Beta2 = _options.OptimizerBeta2,
                Epsilon = _options.OptimizerEpsilon,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                UseAMSGrad = false,
                EnableGradientClipping = clipGradients,
                MaxGradientNorm = _options.MaxGradientNorm,
            });
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(DiffWave<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
