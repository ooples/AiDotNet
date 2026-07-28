using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>APNet2: improved amplitude-phase network with ResNet backbone and multi-resolution STFT loss for higher-quality waveform reconstruction.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "APNet 2: High-Quality and High-Efficiency Neural Vocoder with Direct Prediction of Amplitude and Phase Spectra" (Du et al., 2023)</item></list></para><para><b>For Beginners:</b> APNet2: improved amplitude-phase network with ResNet backbone and multi-resolution STFT loss for higher-quality waveform reconstruction.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create an APNet2 vocoder with ResNet backbone
/// // and multi-resolution STFT loss for high-quality waveform reconstruction
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new APNet2&lt;double&gt;(architecture, "apnet2.onnx");
///
/// // Training mode with native layers
/// var trainModel = new APNet2&lt;double&gt;(architecture, new APNet2Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "APNet 2: High-Quality and High-Efficiency Neural Vocoder with Direct Prediction of Amplitude and Phase Spectra",
    "https://arxiv.org/abs/2311.11545",
    Year = 2023,
    Authors = "Du et al."
)]
public class APNet2<T> : TtsModelBase<T>, IVocoder<T>
{
    private readonly APNet2Options _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>Amplitude spectrum predictor (ASP) branch.</summary>
    private readonly List<ILayer<T>> _amplitudeLayers = new();

    /// <summary>Phase spectrum predictor (PSP) branch, ending in the parallel estimator.</summary>
    private readonly List<ILayer<T>> _phaseLayers = new();

    public APNet2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        APNet2Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new APNet2Options();
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

    public APNet2(
        NeuralNetworkArchitecture<T> architecture,
        APNet2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new APNet2Options();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        InitializeLayers();
    }

    int IVocoder<T>.SampleRate => _options.SampleRate;
    int IVocoder<T>.MelChannels => _options.MelChannels;
    public int UpsampleFactor => _options.HopSize;

    /// <summary>
    /// Converts mel to waveform using APNet2's improved ResNet backbone with multi-resolution STFT.
    /// Per the paper (Du et al., 2023): Replaces APNet's simple convolution backbone with ResNet blocks for deeper feature extraction. Uses multi-resolution STFT loss (at 3 different STFT configs) for better spectral fidelity. Adds phase loss with instantaneous frequency constraint. 2x faster than APNet with better MOS.
    /// </summary>
    public Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(melSpectrogram);
        // ASP and PSP are PARALLEL branches over the same mel input, per Du et al. 2023
        // (arXiv:2311.11545). ASP emits log-amplitude coefficients; PSP emits the pseudo-real
        // and pseudo-imaginary pair that the phase calculation formula turns into a wrapped
        // phase. The waveform is then reconstructed by inverse STFT.
        //
        // This replaces a fabricated inference path that ignored the network entirely: it
        // derived amplitude as Math.Exp(feat * (1 - freqRatio * 0.4) + 0.3) * 0.15 and phase as
        // omega * t + feat * 0.15 from hand-tuned constants, then summed only the first 32 bins
        // through a hand-rolled cosine transform. No amount of training could have changed its
        // output.
        var amplitudeFeatures = melSpectrogram;
        foreach (var l in _amplitudeLayers)
            amplitudeFeatures = l.Forward(amplitudeFeatures);

        var phaseFeatures = melSpectrogram;
        foreach (var l in _phaseLayers)
            phaseFeatures = l.Forward(phaseFeatures);

        int fftBins = (_options.FftSize / 2) + 1;
        int frames = amplitudeFeatures.Length / fftBins;
        if (frames <= 0)
            throw new InvalidOperationException(
                $"APNet2 amplitude branch produced {amplitudeFeatures.Length} values, which is not a whole number of {fftBins}-bin frames.");

        var spectrogram = new Tensor<Complex<T>>(new[] { frames, fftBins });
        for (int t = 0; t < frames; t++)
        {
            for (int f = 0; f < fftBins; f++)
            {
                // ASP predicts LOG amplitude, so exponentiate to recover the magnitude.
                double logAmplitude = NumOps.ToDouble(amplitudeFeatures[(t * fftBins) + f]);
                double magnitude = Math.Exp(logAmplitude);

                // Phase parallel estimation: two heads give a pseudo-real and pseudo-imaginary
                // component, and Phi = atan2(imag, real) yields a phase already wrapped into
                // (-pi, pi] with no unwrapping step.
                int phaseBase = t * 2 * fftBins;
                double real = NumOps.ToDouble(phaseFeatures[phaseBase + f]);
                double imaginary = NumOps.ToDouble(phaseFeatures[phaseBase + fftBins + f]);
                double phi = Math.Atan2(imaginary, real);

                spectrogram[(t * fftBins) + f] = new Complex<T>(
                    NumOps.FromDouble(magnitude * Math.Cos(phi)),
                    NumOps.FromDouble(magnitude * Math.Sin(phi)));
            }
        }

        int waveLen = frames * _options.HopSize;
        var stft = new ShortTimeFourierTransform<T>(
            nFft: _options.FftSize,
            hopLength: _options.HopSize,
            windowLength: _options.WindowLength);

        var waveform = stft.Inverse(spectrogram, waveLen);
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
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        if (_options.DropoutRate > double.Epsilon)
            throw new InvalidOperationException(
                "APNet2Options.DropoutRate is configured but the paper's ConvNeXt v2 backbone (Du et al., 2023) applies no dropout; leave DropoutRate at 0 for native mode or supply explicit Architecture.Layers."
            );

        // APNet2 is a DUAL-BRANCH predictor: an amplitude spectrum predictor (ASP) and a phase
        // spectrum predictor (PSP), both fed the same mel input and combined through an inverse
        // STFT. This previously used CreateDefaultHiFiGANLayers — a time-domain upsampling
        // generator, which is precisely the architecture the paper replaces.
        _amplitudeLayers.AddRange(LayerHelper<T>.CreateDefaultAPNet2AmplitudeLayers(
            numMels: _options.MelChannels,
            channels: _options.ConvNeXtChannels,
            intermediateChannels: _options.ConvNeXtIntermediateChannels,
            numBlocks: _options.NumConvNeXtBlocks,
            kernelSize: _options.DepthwiseKernelSize,
            fftSize: _options.FftSize));

        _phaseLayers.AddRange(LayerHelper<T>.CreateDefaultAPNet2PhaseLayers(
            numMels: _options.MelChannels,
            channels: _options.ConvNeXtChannels,
            intermediateChannels: _options.ConvNeXtIntermediateChannels,
            numBlocks: _options.NumConvNeXtBlocks,
            kernelSize: _options.DepthwiseKernelSize,
            fftSize: _options.FftSize));

        // Both branches go into Layers so parameter enumeration, serialization and device
        // transfer see every weight. They are executed as parallel branches in MelToWaveform,
        // not as one sequential stack.
        Layers.AddRange(_amplitudeLayers);
        Layers.AddRange(_phaseLayers);
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);

        SetTrainingMode(false);
        return ForwardDualBranch(input);
    }

    /// <summary>
    /// Runs the ASP and PSP branches in parallel over the same mel input and returns the
    /// concatenated [amplitude | real | imaginary] prediction.
    /// </summary>
    /// <remarks>
    /// ASP and PSP are PARALLEL branches over the same mel input. <see cref="Layers"/> holds
    /// both of them concatenated — that is only so parameter enumeration, serialization and
    /// device transfer see every weight — so the inherited "chain everything in Layers" pass
    /// would feed the amplitude branch's log-amplitude output into the phase branch and return
    /// something meaningless. Both the inference path and the training path route through here
    /// so they cannot drift apart.
    /// </remarks>
    private Tensor<T> ForwardDualBranch(Tensor<T> input)
    {
        var amplitude = input;
        foreach (var l in _amplitudeLayers)
            amplitude = l.Forward(amplitude);

        var phase = input;
        foreach (var l in _phaseLayers)
            phase = l.Forward(phase);

        return Engine.TensorConcatenate(
            new[] { amplitude, phase },
            axis: amplitude.Shape.Length - 1);
    }

    /// <summary>
    /// Trains through the same parallel dual-branch forward that inference uses.
    /// </summary>
    /// <remarks>
    /// The base implementation chains every layer in <see cref="Layers"/> sequentially. For a
    /// dual-branch model that produces the phase branch's output alone — 2 * fftBins values
    /// instead of the 3 * fftBins the model actually predicts — so the tape loss compared a
    /// [1, 80, 1026] prediction against a [1, 80, 1539] target and training threw before a
    /// single gradient was taken.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        // Subclasses that bypass the base forward must seed stochastic layers themselves,
        // otherwise dropout masks vary run-to-run and the trajectory invariants flake.
        EnsureLayerRandomSeedsWired();
        return ForwardDualBranch(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        TrainWithTape(input, expected, _optimizer);
        SetTrainingMode(false);
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
            Name = _useNativeMode ? "APNet2-Native" : "APNet2-ONNX",
            Description = "APNet 2: Improved Amplitude-Phase Network (Du et al., 2023)",
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
        writer.Write(_options.DropoutRate);
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
        _options.DropoutRate = reader.ReadDouble();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new APNet2<T>(Architecture, mp, _options);
        return new APNet2<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(APNet2<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
