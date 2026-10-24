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
/// HiFi-GAN: a GAN vocoder whose generator upsamples a mel spectrogram to a waveform through transposed convolutions and
/// multi-receptive-field residual blocks, trained against multi-period and multi-scale discriminators.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech
/// Synthesis" (Kong et al., NeurIPS 2020) and its reference implementation (jik876/hifi-gan) for what the paper leaves
/// unstated.</para>
/// <para>
/// Each training step (§2.4, reference <c>train.py</c>) crops an 8192-sample segment, computes its log-mel input, runs
/// the generator, updates the discriminators on the LSGAN loss <c>Σ_k (D_k(x) − 1)² + D_k(G(s))²</c> (Eq. 1, 6) with
/// the generation detached, then updates the generator on <c>Σ_k (D_k(G(s)) − 1)² + 2 L_FM + 45 L_Mel</c> (Eq. 2–5, 7),
/// where L_FM is the mean L1 distance of every discriminator feature map and L_Mel the L1 distance of log-mel
/// spectrograms (full band). Generator and discriminators each use AdamW (β = 0.8, 0.99, weight decay 0.01) at 2e-4,
/// decayed by 0.999 every epoch. The input mel is the reference's <c>mel_spectrogram</c> of the audio
/// (<see cref="ComputeMel"/>); synthesis expects the same features.
/// </para>
/// <para><b>For Beginners:</b> A generator turns a spectrogram into audio while two "critics" — one looking at periodic
/// patterns, one at different time scales — learn to tell its audio from real recordings; competing makes the audio
/// realistic.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis",
    "https://arxiv.org/abs/2010.05646",
    Year = 2020,
    Authors = "Kong et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.8, Beta2 = 0.99, WeightDecay = 0.01, DecayRate = 0.999,
                ReferenceBatchSize = 16,
                Source = "Kong et al. 2020, Sec. 3: AdamW with beta1 0.8, beta2 0.99 and weight decay 0.01, an initial "
                        + "learning rate of 2e-4 decayed by a factor of 0.999 every epoch, batch size 16.")]
public partial class HiFiGAN<T> : GanVocoderBase<T>
{
    private HiFiGanGenerator<T>? _generator;
    private HiFiGanDiscriminators<T>? _discriminators;
    private DifferentiableMel<T>? _inputMel;
    private DifferentiableMel<T>? _lossMel;

    /// <summary>Creates a HiFi-GAN that runs an exported ONNX graph.</summary>
    public HiFiGAN(NeuralNetworkArchitecture<T> architecture, string modelPath, HiFiGANOptions? options = null)
        : base(architecture, modelPath, options ?? new HiFiGANOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable HiFi-GAN.</summary>
    public HiFiGAN(NeuralNetworkArchitecture<T> architecture, HiFiGANOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new HiFiGANOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private HiFiGANOptions PaperOptions => (HiFiGANOptions)VocoderSettings;

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.UpsampleRates.Aggregate(1, (a, b) => a * b);

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.SegmentSize;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateGenerator()
    {
        var o = PaperOptions;
        _generator = new HiFiGanGenerator<T>(Engine, o.MelChannels, o.UpsampleInitialChannels, o.UpsampleRates, o.UpsampleKernelSizes,
            o.ResblockKernelSizes, o.ResblockDilationSizes, o.ResblockType == 1,
            initialization: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 7));
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
    protected override Tensor<T> Generate(Tensor<T> mel) => _generator!.Forward(mel);

    /// <inheritdoc />
    /// <remarks>The reference <c>mel_spectrogram</c>: reflect padding, Hann window, Slaney mel up to
    /// <see cref="HiFiGANOptions.MelMaxFrequency"/>, <c>log(max(x, 1e-5))</c>.</remarks>
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio)
    {
        var rows = _inputMel!.Forward(audio);
        return Engine.Reshape(Engine.TensorTranspose(rows), new[] { 1, PaperOptions.MelChannels, rows.Shape[0] });
    }

    /// <inheritdoc />
    /// <remarks>Σ_k (D_k(x) − 1)² + D_k(G(s))² over the period and scale discriminators (Eq. 1, 6).</remarks>
    protected override Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var realScores = _discriminators!.Forward(real);
        var fakeScores = _discriminators.Forward(generated);
        return Sum(Enumerable.Range(0, realScores.Count).Select(k => LeastSquaresDiscriminator(realScores[k].Score, fakeScores[k].Score)));
    }

    /// <inheritdoc />
    /// <remarks>Σ_k (D_k(G(s)) − 1)² + 2 L_FM + 45 L_Mel (Eq. 2–5, 7).</remarks>
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
        var melLoss = Mean(Engine.TensorAbs(Engine.TensorSubtract(_lossMel!.Forward(generated), realMel)));
        return Engine.TensorAdd(Engine.TensorAdd(adversarialTerm,
                Engine.TensorMultiplyScalar(featureMatching, NumOps.FromDouble(PaperOptions.FeatureMatchingWeight))),
            Engine.TensorMultiplyScalar(melLoss, NumOps.FromDouble(PaperOptions.MelLossWeight)));
    }

    /// <inheritdoc />
    /// <remarks>The reconstruction term λ_mel L_Mel of the generator objective.</remarks>
    protected override Tensor<T> ReconstructionObjective(Tensor<T> mel, Tensor<T> real)
    {
        var (generated, target) = GeneratedAndReal(mel, real);
        return Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(_lossMel!.Forward(generated), _lossMel.Forward(target)))),
            NumOps.FromDouble(PaperOptions.MelLossWeight));
    }

    /// <inheritdoc />
    /// <remarks>AdamW (β = 0.8, 0.99, weight decay 0.01) at 2e-4 for each network, decayed by
    /// <see cref="HiFiGANOptions.LearningRateDecay"/> once per epoch of <see cref="HiFiGANOptions.UpdatesPerEpoch"/>
    /// steps.</remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group)
    {
        var o = PaperOptions;
        int perEpoch = Math.Max(1, o.UpdatesPerEpoch);
        var scheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.LearningRate,
            step => Math.Pow(o.LearningRateDecay, step / perEpoch));
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate,
                Beta1 = 0.8,
                Beta2 = 0.99,
                WeightDecay = 0.01,
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
            Name = IsOnnxMode ? "HiFiGAN-ONNX" : "HiFiGAN-Native",
            Description = "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (Kong et al., 2020)",
            FeatureCount = o.MelChannels,
            Complexity = o.UpsampleRates.Length + o.ResblockKernelSizes.Length,
        };
        m.AdditionalInfo["Architecture"] = "HiFiGAN";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
