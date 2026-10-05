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
/// Vocos: a GAN vocoder that generates Fourier spectral coefficients — magnitude and phase — at frame rate with a
/// ConvNeXt backbone and turns them into audio with the inverse STFT.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality
/// audio synthesis" (Siuzdak, ICLR 2024) and gemelo-ai/vocos for what the paper leaves unstated.</para>
/// <para>
/// The head's magnitude is <c>exp(m)</c> and its phase the point <c>(cos p, sin p)</c> on the unit circle, wrapping
/// implicitly (§3.2). The discriminators train on the hinge loss <c>1/K Σ_k max(0, 1 − D_k(x)) + max(0, 1 + D_k(x̂))</c> and
/// the generator on <c>1/K Σ_k max(0, 1 − D_k(x̂))</c> plus feature matching and the mel L1 (§3.3), the multi-resolution
/// discriminator's terms weighted 0.1 and the mel term 45 as the reference trains it. AdamW (0.9, 0.999) at 2e-4 with a
/// cosine decay (§4.1).
/// </para>
/// <para><b>For Beginners:</b> Rather than building audio sample by sample, Vocos predicts how loud and in what phase each
/// frequency is in every frame and lets an inverse Fourier transform assemble the sound, which is very fast.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis",
    "https://arxiv.org/abs/2306.00814",
    Year = 2024,
    Authors = "Siuzdak"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.9, Beta2 = 0.999, WeightDecay = 0.01, ReferenceBatchSize = 16,
                Source = "Siuzdak 2024, Sec. 4.1: AdamW with an initial learning rate of 2e-4 and betas (0.9, 0.999), decayed by "
                        + "a cosine schedule; 1M iterations per network, batch size 16.")]
public partial class Vocos<T> : GanVocoderBase<T>
{
    private VocosGenerator<T>? _generator;
    private HiFiGanDiscriminators<T>? _periods;
    private MultiResolutionSpectrogramDiscriminators<T>? _resolutions;
    private CenteredLogMel<T>? _features;

    /// <summary>Creates a Vocos that runs an exported ONNX graph.</summary>
    public Vocos(NeuralNetworkArchitecture<T> architecture, string modelPath, VocosOptions? options = null)
        : base(architecture, modelPath, options ?? new VocosOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable Vocos.</summary>
    public Vocos(NeuralNetworkArchitecture<T> architecture, VocosOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new VocosOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private VocosOptions PaperOptions => (VocosOptions)VocoderSettings;

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.HopSize;

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.SegmentSize;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateGenerator()
    {
        var o = PaperOptions;
        _generator = new VocosGenerator<T>(Engine, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 7),
            o.MelChannels, o.ConvNeXtDim, o.IntermediateDim, o.NumBackboneBlocks, o.FftSize, o.HopSize);
        _features = new CenteredLogMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.FftSize, o.MelChannels, 0.0, o.SampleRate / 2.0, 1e-7,
            htkScale: true, naturalLog: true);
        return _generator.Layers;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        var o = PaperOptions;
        _periods = new HiFiGanDiscriminators<T>(Engine, o.DiscriminatorPeriods, 0, useScaleDiscriminator: false, widthDivisor: o.DiscriminatorWidthDivisor);
        _resolutions = new MultiResolutionSpectrogramDiscriminators<T>(Engine, o.ResolutionFftSizes, o.ResolutionHopSizes, o.ResolutionWindowSizes,
            o.ResolutionDiscriminatorChannels, 0.2);
        return _periods.Layers.Concat(_resolutions.Layers).ToList();
    }

    /// <inheritdoc />
    /// <remarks>A mel spectrogram of F frames (the centred analysis of (F − 1) · hop samples) gives (F − 1) · hop samples.</remarks>
    protected override Tensor<T> Generate(Tensor<T> mel)
    {
        var wave = _generator!.Forward(mel);
        return Engine.Reshape(wave, new[] { 1, 1, wave.Length });
    }

    /// <inheritdoc />
    /// <remarks>The reference's <c>MelSpectrogramFeatures</c>: torchaudio's centred magnitude mel (HTK scale, no filter
    /// normalization) and <c>log(max(x, 1e-7))</c>.</remarks>
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio) => _features!.Forward(audio);

    // The generator of F frames produces (F − 1) · hop samples, so a segment's features keep their extra centred frame.
    /// <inheritdoc />
    protected override bool KeepsCentredFrame => true;

    private Tensor<T> Hinge(Tensor<T> x, double sign)
        => Mean(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(sign)), NumOps.One)));

    /// <inheritdoc />
    protected override Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        Tensor<T> Family(List<(Tensor<T> Score, List<Tensor<T>> Features)> r, List<(Tensor<T> Score, List<Tensor<T>> Features)> g)
            => Engine.TensorMultiplyScalar(Sum(Enumerable.Range(0, r.Count).Select(k => Engine.TensorAdd(Hinge(r[k].Score, -1), Hinge(g[k].Score, 1)))),
                NumOps.FromDouble(1.0 / r.Count));
        var periodic = Family(_periods!.Forward(real), _periods.Forward(generated));
        var resolution = Family(_resolutions!.Forward(real), _resolutions.Forward(generated));
        return Engine.TensorAdd(periodic, Engine.TensorMultiplyScalar(resolution, NumOps.FromDouble(PaperOptions.ResolutionLossWeight)));
    }

    /// <inheritdoc />
    protected override Tensor<T> GeneratorLoss(Tensor<T> mel, Tensor<T> real, bool adversarial)
    {
        var o = PaperOptions;
        var generated = Flat(Generate(mel));
        var target = Flat(real);
        int n = Math.Min(generated.Length, target.Length);
        generated = Engine.TensorSlice(generated, new[] { 0 }, new[] { n });
        target = Engine.TensorSlice(target, new[] { 0 }, new[] { n });

        Tensor<T> realMel;
        using (new NoGradScope<T>()) realMel = Detached(_features!.Forward(target));
        var loss = Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(_features!.Forward(generated), realMel))),
            NumOps.FromDouble(o.MelLossWeight));
        if (!adversarial) return loss;

        (Tensor<T> Adversarial, Tensor<T> Matching) Family(List<(Tensor<T> Score, List<Tensor<T>> Features)> fake,
            List<(Tensor<T> Score, List<Tensor<T>> Features)> realOut, int skip)
        {
            var adversarialTerm = Engine.TensorMultiplyScalar(Sum(fake.Select(f => Hinge(f.Score, -1))), NumOps.FromDouble(1.0 / fake.Count));
            var matching = Engine.TensorMultiplyScalar(Sum(Enumerable.Range(0, fake.Count).Select(k =>
                    FeatureMatching(realOut[k].Features.Skip(skip).ToList(), fake[k].Features.Skip(skip).ToList()))),
                NumOps.FromDouble(1.0 / fake.Count));
            return (adversarialTerm, matching);
        }
        List<(Tensor<T> Score, List<Tensor<T>> Features)> realPeriods, realResolutions;
        using (new NoGradScope<T>())
        {
            realPeriods = _periods!.Forward(target);
            realResolutions = _resolutions!.Forward(target);
        }
        // The reference's period discriminators leave their first layer out of the feature maps.
        var (periodAdversarial, periodMatching) = Family(_periods!.Forward(generated), realPeriods, 1);
        var (resolutionAdversarial, resolutionMatching) = Family(_resolutions!.Forward(generated), realResolutions, 0);
        var weight = NumOps.FromDouble(o.ResolutionLossWeight);
        return Sum(new[]
        {
            loss, periodAdversarial, periodMatching,
            Engine.TensorMultiplyScalar(resolutionAdversarial, weight), Engine.TensorMultiplyScalar(resolutionMatching, weight),
        });
    }

    /// <inheritdoc />
    /// <remarks>The mel reconstruction term (×45).</remarks>
    protected override Tensor<T> ReconstructionObjective(Tensor<T> mel, Tensor<T> real)
    {
        var (generated, target) = GeneratedAndReal(mel, real);
        return Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(_features!.Forward(generated), _features.Forward(target)))),
            NumOps.FromDouble(PaperOptions.MelLossWeight));
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group)
    {
        var o = PaperOptions;
        int total = Math.Max(1, o.CosineSteps);
        var scheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.LearningRate,
            step => 0.5 * (1 + Math.Cos(Math.PI * Math.Min(step, total) / total)));
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate,
                Beta1 = o.Beta1,
                Beta2 = o.Beta2,
                WeightDecay = o.WeightDecay,
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
            Name = IsOnnxMode ? "Vocos-ONNX" : "Vocos-Native",
            Description = "Vocos: Fourier-based neural vocoder with a ConvNeXt backbone (Siuzdak, 2024)",
            FeatureCount = o.MelChannels,
            Complexity = o.NumBackboneBlocks,
        };
        m.AdditionalInfo["Architecture"] = "Vocos";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
