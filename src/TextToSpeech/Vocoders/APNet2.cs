using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
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
public class APNet2<T> : VocoderBase<T>
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

    private int _amplitudeLayerCount;
    private int _phaseLayerCount;

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

    /// <param name="architecture">The network architecture.</param>
    /// <param name="options">Vocoder options; defaults to the paper's configuration.</param>
    /// <param name="optimizer">Optimizer; defaults to the paper's AdamW settings.</param>
    /// <param name="lossFunction">
    /// Training objective. Defaults to <see cref="APNet2GeneratorLoss{T}"/>, the paper's
    /// <c>lambda_A * L_A + lambda_P * L_P + lambda_S * L_S</c>. The inherited squared-error default
    /// penalised phase as though it were an ordinary number, which the anti-wrapping losses exist
    /// precisely to avoid.
    /// </param>
    public APNet2(
        NeuralNetworkArchitecture<T> architecture,
        APNet2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null
    )
        : base(architecture, lossFunction ?? new APNet2GeneratorLoss<T>(
            (options ?? new APNet2Options()).AmplitudeLossWeight,
            (options ?? new APNet2Options()).PhaseLossWeight,
            (options ?? new APNet2Options()).StftLossWeight))
    {
        _options = options ?? new APNet2Options();
        _useNativeMode = true;
        // AdamW at the paper's 2e-4 (Du et al., 2023). Constructing it with no options left it
        // at AdamW's own 1e-3 default, five times the paper's rate.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                // Du et al. 2023 §4: AdamW, beta1 = 0.8, beta2 = 0.99, weight decay 0.01,
                // initial learning rate 2e-4 decayed by 0.999 each epoch.
                InitialLearningRate = _options.LearningRate,
                Beta1 = 0.8,
                Beta2 = 0.99,
                WeightDecay = 0.01,
                // "the exponential decay strategy with a decreasing factor of 0.999 per epoch".
                // Without it the rate stays at its initial value for the whole run, so late
                // training keeps taking early-training-sized steps.
                LearningRateScheduler = new ExponentialLRScheduler(_options.LearningRate, gamma: 0.999),
                SchedulerStepMode = SchedulerStepMode.StepPerEpoch
            });
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        InitializeLayers();
    }

    // SampleRate, MelChannels and UpsampleFactor now come from VocoderBase - see BigVGAN for why
    // these three restated what the base already derives from the same _options fields.

    /// <summary>
    /// Converts mel to waveform using APNet2's improved ResNet backbone with multi-resolution STFT.
    /// Per the paper (Du et al., 2023): Replaces APNet's simple convolution backbone with ResNet blocks for deeper feature extraction. Uses multi-resolution STFT loss (at 3 different STFT configs) for better spectral fidelity. Adds phase loss with instantaneous frequency constraint. 2x faster than APNet with better MOS.
    /// </summary>
    public override Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
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

        // Resolve each branch's lazy layers from the model's input shape. Both branches are fed
        // the SAME mel input, so both chains are rooted at it.
        //
        // Without this the ConvNeXt v2 blocks stay unresolved until their first forward, and a
        // freshly constructed instance understates its ParameterCount. That breaks the
        // serialization round-trip: NeuralNetworkBase slices the flat parameter vector using
        // each layer's ParameterCount, so the offsets are computed from the understated sizes
        // and every layer after the first lazy one receives the wrong slice -- before the
        // layer's own SetParameters (which does resolve) ever runs. It surfaced as a clone
        // predicting differently from the model it was cloned from, even untrained.
        var rootShape = Architecture.GetInputShape();
        if (rootShape is { Length: > 0 })
        {
            LayerHelper<T>.ResolveChain(_amplitudeLayers, rootShape);
            LayerHelper<T>.ResolveChain(_phaseLayers, rootShape);
        }

        // Both branches go into Layers so parameter enumeration, serialization and device
        // transfer see every weight. They are executed as parallel branches in MelToWaveform,
        // not as one sequential stack.
        Layers.AddRange(_amplitudeLayers);
        Layers.AddRange(_phaseLayers);

        // Deserialization rebuilds Layers with fresh objects; remember where the branches sit so
        // RebindBranchLayers can re-point the private lists at the restored ones.
        _amplitudeLayerCount = _amplitudeLayers.Count;
        _phaseLayerCount = _phaseLayers.Count;
    }

    /// <summary>
    /// Re-points the branch lists at the layers deserialization just rebuilt.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Both branches are appended to <c>Layers</c>, so their weights serialize and restore
    /// correctly. But the forward pass reads <c>_amplitudeLayers</c> / <c>_phaseLayers</c>, and
    /// those still referenced the objects this instance built in its own constructor —
    /// deserialization replaces the contents of <c>Layers</c> without touching them. The restored
    /// weights therefore landed in layers the model never evaluated, and a clone predicted from
    /// its initialisation values while reporting success.
    /// </para>
    /// </remarks>
    private void RebindBranchLayers()
    {
        int branchTotal = _amplitudeLayerCount + _phaseLayerCount;
        if (branchTotal == 0 || Layers.Count < branchTotal) return;

        int start = Layers.Count - branchTotal;

        _amplitudeLayers.Clear();
        _phaseLayers.Clear();
        for (int i = 0; i < _amplitudeLayerCount; i++) _amplitudeLayers.Add(Layers[start + i]);
        for (int i = 0; i < _phaseLayerCount; i++) _phaseLayers.Add(Layers[start + _amplitudeLayerCount + i]);
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
            new[] { amplitude, NormalizePhasePair(phase) },
            axis: amplitude.Shape.Length - 1);
    }

    /// <summary>
    /// Rescales the PSP's pseudo-real/pseudo-imaginary pair to unit length per frequency bin.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Per Du et al. 2023 (arXiv:2311.11545), R and I are INTERMEDIATE representations: the
    /// paper's phase losses are applied to the phase derived from them,
    /// Phi(R, I) = arctan(I/R) - (pi/2) * Sgn*(I) * [Sgn*(R) - 1], and never to R and I
    /// directly. Nothing in the paper's objective constrains the pair's magnitude.
    /// </para>
    /// <para>
    /// That matters because Phi is scale-invariant — atan2(kI, kR) = atan2(I, R) for every
    /// k &gt; 0 — so the radius of (R, I) is a completely free direction. Emitting the raw pair
    /// let the training objective regress it directly, which is ill-posed along that direction:
    /// nothing bounded the magnitude and one optimizer step drove the parameters to NaN
    /// (OptimizerStep_ParamL2_DoesNotExplode measured an L2 of 372.5 collapsing to NaN in a
    /// single step).
    /// </para>
    /// <para>
    /// Normalising to (cos Phi, sin Phi) removes exactly that free direction while preserving
    /// all the phase information, since the unit vector is bijective with Phi. It is also better
    /// behaved than emitting Phi itself would be: a plain squared-error loss on an angle is
    /// discontinuous at the +/-pi wrap, which is the very problem the paper's anti-wrapping
    /// function f_AW(x) = |x - 2*pi*round(x / (2*pi))| exists to solve. The unit vector has no
    /// wrap.
    /// </para>
    /// <para>
    /// Inference is unaffected: <see cref="MelToWaveform"/> recovers the phase with
    /// Math.Atan2, which is invariant to this rescaling.
    /// </para>
    /// </remarks>
    private Tensor<T> NormalizePhasePair(Tensor<T> phase)
    {
        int lastAxis = phase.Shape.Length - 1;
        int fftBins = (_options.FftSize / 2) + 1;

        if (phase.Shape[lastAxis] != 2 * fftBins)
            throw new InvalidOperationException(
                $"APNet2 phase branch produced {phase.Shape[lastAxis]} values per frame, expected {2 * fftBins} (a pseudo-real and pseudo-imaginary value for each of {fftBins} bins).");

        var real = Engine.TensorNarrow(phase, dim: lastAxis, start: 0, length: fftBins);
        var imaginary = Engine.TensorNarrow(phase, dim: lastAxis, start: fftBins, length: fftBins);

        var sumSquares = Engine.TensorAdd(
            Engine.TensorMultiply(real, real),
            Engine.TensorMultiply(imaginary, imaginary));

        // Offset inside the root: d/du sqrt(u) is infinite at u = 0, and a bin whose predicted
        // real and imaginary parts are both zero gives exactly that.
        var epsilon = new Tensor<T>(sumSquares.Shape.ToArray());
        for (int i = 0; i < epsilon.Length; i++) epsilon[i] = NumOps.FromDouble(1e-12);

        var magnitude = Engine.TensorSqrt(Engine.TensorAdd(sumSquares, epsilon));

        return Engine.TensorConcatenate(
            new[] { Engine.TensorDivide(real, magnitude), Engine.TensorDivide(imaginary, magnitude) },
            axis: lastAxis);
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

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
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
        // The ConvNeXt v2 backbone geometry decides the parameter count, so it has to survive
        // the round-trip: restoring into a model rebuilt at different widths silently
        // misaligns every slice of the flat parameter vector.
        writer.Write(_options.ConvNeXtChannels);
        writer.Write(_options.ConvNeXtIntermediateChannels);
        writer.Write(_options.NumConvNeXtBlocks);
        writer.Write(_options.DepthwiseKernelSize);
        writer.Write(_options.WindowLength);
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
        _options.ConvNeXtChannels = reader.ReadInt32();
        _options.ConvNeXtIntermediateChannels = reader.ReadInt32();
        _options.NumConvNeXtBlocks = reader.ReadInt32();
        _options.DepthwiseKernelSize = reader.ReadInt32();
        _options.WindowLength = reader.ReadInt32();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    
        RebindBranchLayers();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new APNet2<T>(Architecture, mp, _options);

        // Carry the objective and the optimizer across explicitly. Rebuilding from architecture and
        // options alone silently re-derives them, so a model trained under a caller-supplied loss
        // came back as a clone trained under a different one: the more-data invariant trains the
        // original for a few steps and its CLONE for more, and the clone's parameters were
        // bit-identical no matter which objective the original had been given.
        return new APNet2<T>(Architecture, _options, _optimizer, LossFunction);
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
