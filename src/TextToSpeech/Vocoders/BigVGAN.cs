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
/// BigVGAN: a universal GAN vocoder — HiFi-GAN with anti-aliased periodic (Snake) activations, trained at scale against
/// multi-period and multi-resolution discriminators.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "BigVGAN: A Universal Neural Vocoder with Large-Scale Training" (Lee et al., ICLR 2023) and
/// NVIDIA/BigVGAN for what the paper leaves unstated.</para>
/// <para>
/// The generator replaces HiFi-GAN's leaky ReLUs with Snake, made anti-aliased by 2× sinc up- and down-sampling around it
/// (the AMP module, §3.2–3.3). Training is HiFi-GAN's with the multi-scale discriminator replaced by UnivNet's
/// multi-resolution discriminator (§3.1): LSGAN, feature matching (×2) and mel L1 (×45), AdamW (0.8, 0.99) at 1e-4
/// decayed 0.999 per epoch, gradient norms clipped at 1000 — the generator's, the period discriminators' and the
/// resolution discriminators' separately (§3.4, reference <c>train.py</c>).
/// </para>
/// <para><b>For Beginners:</b> BigVGAN turns spectrograms into audio for any voice, language or instrument; its
/// activations ripple periodically like sound does, and a filter keeps them from adding high-frequency artefacts.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "BigVGAN: A Universal Neural Vocoder with Large-Scale Training",
    "https://arxiv.org/abs/2206.04658",
    Year = 2023,
    Authors = "Lee et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.8, Beta2 = 0.99, WeightDecay = 0.01, DecayRate = 0.999,
                ReferenceBatchSize = 32,
                Source = "Lee et al. 2023, Sec. 3.4 and 4.2: batch 32, learning rate 1e-4, gradient norm clipped at 1000; the "
                        + "optimizer (AdamW 0.8/0.99) and 0.999 per-epoch decay of HiFi-GAN's official configuration.")]
public partial class BigVGAN<T> : GanVocoderBase<T>
{
    private BigVganGenerator<T>? _generator;
    private HiFiGanDiscriminators<T>? _periods;
    private MultiResolutionSpectrogramDiscriminators<T>? _resolutions;
    private DifferentiableMel<T>? _inputMel;
    private DifferentiableMel<T>? _lossMel;

    /// <summary>Creates a BigVGAN that runs an exported ONNX graph.</summary>
    public BigVGAN(NeuralNetworkArchitecture<T> architecture, string modelPath, BigVGANOptions? options = null)
        : base(architecture, modelPath, options ?? new BigVGANOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable BigVGAN.</summary>
    public BigVGAN(NeuralNetworkArchitecture<T> architecture, BigVGANOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new BigVGANOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private BigVGANOptions PaperOptions => (BigVGANOptions)VocoderSettings;

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.UpsampleRates.Aggregate(1, (a, b) => a * b);

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.SegmentSize;

    /// <inheritdoc />
    protected override double GradientClipNorm(string group) => PaperOptions.GradientClipNorm;

    /// <inheritdoc />
    /// <remarks>The period and the resolution discriminators are clipped separately.</remarks>
    protected override IReadOnlyList<IReadOnlyList<LayerBase<T>>>? ClippingLayerGroups(string group)
        => group == "discriminator" ? new[] { _periods!.Layers, _resolutions!.Layers } : null;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateGenerator()
    {
        var o = PaperOptions;
        _generator = new BigVganGenerator<T>(Engine, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 7),
            o.MelChannels, o.UpsampleInitialChannels, o.UpsampleRates, o.UpsampleKernelSizes, o.ResblockKernelSizes, o.ResblockDilationSizes);
        _inputMel = new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, 0.0, o.MelMaxFrequency);
        _lossMel = new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, 0.0, o.SampleRate / 2.0);
        return _generator.Layers;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        var o = PaperOptions;
        _periods = new HiFiGanDiscriminators<T>(Engine, o.DiscriminatorPeriods, 0, useScaleDiscriminator: false, widthDivisor: o.DiscriminatorWidthDivisor);
        _resolutions = new MultiResolutionSpectrogramDiscriminators<T>(Engine, o.ResolutionFftSizes, o.ResolutionHopSizes, o.ResolutionWindowSizes,
            o.ResolutionDiscriminatorChannels, 0.1);
        return _periods.Layers.Concat(_resolutions.Layers).ToList();
    }

    /// <inheritdoc />
    protected override Tensor<T> Generate(Tensor<T> mel) => _generator!.Forward(mel);

    /// <inheritdoc />
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio)
    {
        var rows = _inputMel!.Forward(audio);
        return Engine.Reshape(Engine.TensorTranspose(rows), new[] { 1, PaperOptions.MelChannels, rows.Shape[0] });
    }

    private List<(Tensor<T> Score, List<Tensor<T>> Features)> Discriminate(Tensor<T> audio)
        => _periods!.Forward(audio).Concat(_resolutions!.Forward(audio)).ToList();

    /// <inheritdoc />
    protected override Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var realScores = Discriminate(real);
        var fakeScores = Discriminate(generated);
        return Sum(Enumerable.Range(0, realScores.Count).Select(k => LeastSquaresDiscriminator(realScores[k].Score, fakeScores[k].Score)));
    }

    /// <inheritdoc />
    protected override Tensor<T> GeneratorLoss(Tensor<T> mel, Tensor<T> real, bool adversarial)
    {
        var generated = Flat(Generate(mel));
        var fake = Discriminate(generated);
        List<(Tensor<T> Score, List<Tensor<T>> Features)> realOut;
        using (new NoGradScope<T>()) realOut = Discriminate(real);
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
            Name = IsOnnxMode ? "BigVGAN-ONNX" : "BigVGAN-Native",
            Description = "BigVGAN: A Universal Neural Vocoder with Large-Scale Training (Lee et al., 2023)",
            FeatureCount = o.MelChannels,
            Complexity = o.UpsampleRates.Length + o.ResblockKernelSizes.Length,
        };
        m.AdditionalInfo["Architecture"] = "BigVGAN";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
