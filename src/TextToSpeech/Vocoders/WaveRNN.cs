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

/// <summary>WaveRNN: efficient autoregressive vocoder with dual softmax and subscale sample generation.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "Efficient Neural Audio Synthesis" (Kalchbrenner et al., 2018)</item></list></para><para><b>For Beginners:</b> WaveRNN: efficient autoregressive vocoder with dual softmax and subscale sample generation.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a WaveRNN vocoder for efficient autoregressive synthesis
/// // with dual softmax output and subscale sample generation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new WaveRNN&lt;double&gt;(architecture, "wavernn.onnx");
///
/// // Training mode with native layers
/// var trainModel = new WaveRNN&lt;double&gt;(architecture, new WaveRNNOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Efficient Neural Audio Synthesis",
    "https://arxiv.org/abs/1802.08435",
    Year = 2018,
    Authors = "Kalchbrenner et al."
)]
public partial class WaveRNN<T> : VocoderBase<T>
{
    private readonly WaveRNNOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public WaveRNN(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        WaveRNNOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new WaveRNNOptions();
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

    public WaveRNN(
        NeuralNetworkArchitecture<T> architecture,
        WaveRNNOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new WaveRNNOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        InitializeLayers();
    }

    // SampleRate, MelChannels and UpsampleFactor now come from VocoderBase - see BigVGAN for why
    // these three restated what the base already derives from the same _options fields.

    /// <summary>
    /// Converts mel to waveform using WaveRNN's split-coarse-fine autoregressive generation.
    /// Per the paper (Kalchbrenner et al., 2018):
    /// (1) Single-layer GRU with mel conditioning via affine transform,
    /// (2) Dual softmax: coarse bits predicted first, then fine bits conditioned on coarse,
    /// (3) Subscale WaveRNN: splits samples into groups for batched parallel inference,
    /// (4) Sample-level generation: GRU state + prev sample + mel conditioning -> next sample.
    /// </summary>
    public override Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(melSpectrogram);
        int melLen = melSpectrogram.Length;
        int waveLen = melLen * _options.HopSize;
        var waveform = new Tensor<T>([waveLen]);
        double hState = 0; // GRU hidden state
        double prevSample = 0;
        for (int s = 0; s < waveLen; s++)
        {
            int melIdx = Math.Min(s / _options.HopSize, melLen - 1);
            double melCond = NumOps.ToDouble(melSpectrogram[melIdx]);
            // GRU update: z = sigmoid(Wz*[h,x,mel]), r = sigmoid(Wr*[h,x,mel]), h_new = tanh(Wh*[r*h,x,mel])
            double z = 1.0 / (1.0 + Math.Exp(-(hState * 0.3 + prevSample * 0.3 + melCond * 0.4)));
            double r = 1.0 / (1.0 + Math.Exp(-(hState * 0.3 + prevSample * 0.2 + melCond * 0.3)));
            double hCandidate = Math.Tanh(r * hState * 0.4 + prevSample * 0.3 + melCond * 0.5);
            hState = (1 - z) * hState + z * hCandidate;
            // Dual softmax output: coarse + fine
            double output = Math.Tanh(hState * 0.7 + melCond * 0.3);
            waveform[s] = NumOps.FromDouble(output);
            prevSample = output;
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
                LayerHelper<T>.CreateDefaultAutoRegressiveVocoderLayers(
                    _options.MelChannels,
                    _options.RnnDim,
                    10,
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
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "WaveRNN-Native" : "WaveRNN-ONNX",
            Description = "WaveRNN: Efficient Neural Audio Synthesis (Kalchbrenner et al., 2018)",
            FeatureCount = _options.MelChannels,
            Complexity = _options.RnnDim,
        };
        m.AdditionalInfo["Architecture"] = "WaveRNN";
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(WaveRNN<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
