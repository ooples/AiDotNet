using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// iSTFTNet: a fast, lightweight mel-spectrogram vocoder — HiFi-GAN with its output-side upsampling replaced by an
/// inverse STFT of a small magnitude and phase spectrogram it predicts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time
/// Fourier Transform" (Kaneko et al., ICASSP 2022).</para>
/// <para>
/// After two ×8 upsamplings the output convolution gives (f/2 + 1) × 2 channels; an exponential makes the first half a
/// linear magnitude and a sine makes the second half the phase (§3.3), and iSTFT(16, 4, 16) turns them into the waveform
/// (Eq. 1). Training is HiFi-GAN's — its discriminators, LSGAN, feature matching (×2) and mel L1 (×45) — with Adam
/// (β = 0.5, 0.9) at 2e-4 (§4.1).
/// </para>
/// <para><b>For Beginners:</b> Instead of building every audio sample with neural layers, the network predicts a tiny
/// spectrogram and a fixed mathematical transform turns it into sound, which is much faster.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform",
    "https://arxiv.org/abs/2203.02395",
    Year = 2022,
    Authors = "Kaneko et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4, Beta1 = 0.5, Beta2 = 0.9, DecayRate = 0.999, ReferenceBatchSize = 16,
                Source = "Kaneko et al. 2022, Sec. 4.1: Adam with an initial learning rate of 0.0002 and momentum terms 0.5 and "
                        + "0.9; the HiFi-GAN configuration's 0.999 per-epoch decay and batch size 16.")]
public partial class ISTFTNet<T> : GanVocoderBase<T>
{
    private HiFiGanGenerator<T>? _generator;
    private InverseStft<T>? _istft;
    private HiFiGanDiscriminators<T>? _discriminators;
    private DifferentiableMel<T>? _inputMel;
    private DifferentiableMel<T>? _lossMel;

    /// <summary>Creates an iSTFTNet that runs an exported ONNX graph.</summary>
    public ISTFTNet(NeuralNetworkArchitecture<T> architecture, string modelPath, ISTFTNetOptions? options = null)
        : base(architecture, modelPath, options ?? new ISTFTNetOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable iSTFTNet.</summary>
    public ISTFTNet(NeuralNetworkArchitecture<T> architecture, ISTFTNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new ISTFTNetOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private ISTFTNetOptions PaperOptions => (ISTFTNetOptions)VocoderSettings;

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.UpsampleRates.Aggregate(1, (a, b) => a * b) * PaperOptions.InverseHopSize;

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.SegmentSize;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateGenerator()
    {
        var o = PaperOptions;
        if (UpsampleFactor != o.HopSize)
            throw new ArgumentException($"The upsampling ({string.Join("x", o.UpsampleRates)}) times the inverse STFT hop ({o.InverseHopSize}) must equal the hop ({o.HopSize}).");
        int bins = o.InverseFftSize / 2 + 1;
        _generator = new HiFiGanGenerator<T>(Engine, o.MelChannels, o.UpsampleInitialChannels, o.UpsampleRates, o.UpsampleKernelSizes,
            o.ResblockKernelSizes, o.ResblockDilationSizes, o.ResblockType == 1, outputChannels: 2 * bins, tanhOutput: false, outputReflectPad: 1,
            initialization: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 7));
        _istft = new InverseStft<T>(Engine, o.InverseFftSize, o.InverseHopSize, o.InverseWindowSize);
        _inputMel = new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, 0.0, o.MelMaxFrequency);
        _lossMel = new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, 0.0, o.SampleRate / 2.0);
        return _generator.Layers;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        var o = PaperOptions;
        _discriminators = new HiFiGanDiscriminators<T>(Engine, o.DiscriminatorPeriods, 3, true, o.DiscriminatorWidthDivisor);
        return _discriminators.Layers;
    }

    /// <inheritdoc />
    /// <remarks>magnitude = exp(first half), phase = sin(second half), waveform = iSTFT(magnitude, phase).</remarks>
    protected override Tensor<T> Generate(Tensor<T> mel)
    {
        var spectrum = _generator!.Forward(mel);                                         // [1, 2·bins, frames + 1]
        int bins = PaperOptions.InverseFftSize / 2 + 1, frames = spectrum.Shape[2];
        var magnitude = Engine.TensorExp(Engine.TensorSlice(spectrum, new[] { 0, 0, 0 }, new[] { 1, bins, frames }));
        var phase = Engine.TensorSin(Engine.TensorSlice(spectrum, new[] { 0, bins, 0 }, new[] { 1, bins, frames }));
        var wave = _istft!.Forward(magnitude, phase);
        return Engine.Reshape(wave, new[] { 1, 1, wave.Length });
    }

    /// <inheritdoc />
    /// <remarks>HiFi-GAN's <c>mel_spectrogram</c> (natural log, floor 1e-5), up to
    /// <see cref="ISTFTNetOptions.MelMaxFrequency"/>.</remarks>
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio)
    {
        var rows = _inputMel!.Forward(audio);
        return Engine.Reshape(Engine.TensorTranspose(rows), new[] { 1, PaperOptions.MelChannels, rows.Shape[0] });
    }

    /// <inheritdoc />
    protected override Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var realScores = _discriminators!.Forward(real);
        var fakeScores = _discriminators.Forward(generated);
        return Sum(Enumerable.Range(0, realScores.Count).Select(k => LeastSquaresDiscriminator(realScores[k].Score, fakeScores[k].Score)));
    }

    /// <inheritdoc />
    protected override Tensor<T> GeneratorLoss(Tensor<T> mel, Tensor<T> real, bool adversarial)
    {
        var generated = Flat(Generate(mel));
        var fake = _discriminators!.Forward(generated);
        List<(Tensor<T> Score, List<Tensor<T>> Features)> realOut;
        using (new NoGradScope<T>()) realOut = _discriminators.Forward(real);
        var adversarialTerm = Sum(fake.Select(f => LeastSquaresGenerator(f.Score)));
        var featureMatching = Sum(Enumerable.Range(0, fake.Count).Select(k => FeatureMatching(realOut[k].Features, fake[k].Features)));
        Tensor<T> realMel;
        using (new NoGradScope<T>()) realMel = Detached(_lossMel!.Forward(real));
        var melLoss = Mean(Engine.TensorAbs(Engine.TensorSubtract(_lossMel.Forward(generated), realMel)));
        return Engine.TensorAdd(Engine.TensorAdd(adversarialTerm,
                Engine.TensorMultiplyScalar(featureMatching, NumOps.FromDouble(PaperOptions.FeatureMatchingWeight))),
            Engine.TensorMultiplyScalar(melLoss, NumOps.FromDouble(PaperOptions.MelLossWeight)));
    }

    /// <inheritdoc />
    /// <remarks>The mel reconstruction term (×45) of the generator objective.</remarks>
    protected override Tensor<T> ReconstructionObjective(Tensor<T> mel, Tensor<T> real)
    {
        var (generated, target) = GeneratedAndReal(mel, real);
        return Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(_lossMel!.Forward(generated), _lossMel.Forward(target)))),
            NumOps.FromDouble(PaperOptions.MelLossWeight));
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group)
    {
        var o = PaperOptions;
        int perEpoch = Math.Max(1, o.UpdatesPerEpoch);
        var scheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.LearningRate, step => Math.Pow(o.LearningRateDecay, step / perEpoch));
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate,
                Beta1 = o.Beta1,
                Beta2 = o.Beta2,
                UseAdaptiveBetas = false,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            }));
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "ISTFTNet-ONNX" : "ISTFTNet-Native",
            Description = "iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse STFT (Kaneko et al., 2022)",
            FeatureCount = o.MelChannels,
            Complexity = o.UpsampleRates.Length + o.ResblockKernelSizes.Length,
        };
        m.AdditionalInfo["Architecture"] = "ISTFTNet";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
