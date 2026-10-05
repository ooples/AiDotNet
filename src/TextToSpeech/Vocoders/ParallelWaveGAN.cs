using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// Parallel WaveGAN: a non-autoregressive WaveNet that turns Gaussian noise into speech conditioned on a mel spectrogram,
/// trained jointly with a multi-resolution STFT loss and an adversarial loss.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks
/// with multi-resolution spectrogram" (Yamamoto et al., ICASSP 2020) and kan-bayashi/ParallelWaveGAN for what the paper
/// leaves unstated.</para>
/// <para>
/// The generator is trained on <c>L_aux + λ_adv · mean((1 − D(G(z)))²)</c> (Eq. 1, 7) with λ_adv = 4, where L_aux averages
/// the spectral-convergence and log-magnitude losses over three STFT resolutions (Eq. 3–6); for the first 100k steps the
/// discriminator is fixed and the generator trains on L_aux alone. The discriminator then trains on
/// <c>mean((1 − D(x))²) + mean(D(G(z))²)</c> (Eq. 2) after each generator step, on a fresh generation. Both use RAdam
/// (ε = 1e-6), at 1e-4 and 5e-5, halved every 200k steps.
/// </para>
/// <para><b>For Beginners:</b> Parallel WaveGAN shapes random noise into speech in a single pass, guided by the
/// spectrogram; matching spectrograms at several resolutions teaches it quickly, and a critic polishes the result.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram",
    "https://arxiv.org/abs/1910.11480",
    Year = 2020,
    Authors = "Yamamoto et al."
)]
[PaperOptimizer(OptimizerKind.RAdam, LearningRate = 1e-4, Epsilon = 1e-6, ReferenceBatchSize = 8,
                Source = "Yamamoto et al. 2020, Sec. 4.1.2: RAdam with epsilon 1e-6, initial learning rates 1e-4 (generator) and "
                        + "5e-5 (discriminator) halved every 200K steps, batch size 8 of 1-second clips.")]
public partial class ParallelWaveGAN<T> : GanVocoderBase<T>
{
    private ParallelWaveGanGenerator<T>? _generator;
    private ParallelWaveGanDiscriminator<T>? _discriminator;
    private CenteredLogMel<T>? _features;
    private MultiResolutionStftLoss<T>? _stftLoss;
    private Random _noise;

    /// <summary>Creates a Parallel WaveGAN that runs an exported ONNX graph.</summary>
    public ParallelWaveGAN(NeuralNetworkArchitecture<T> architecture, string modelPath, ParallelWaveGANOptions? options = null)
        : base(architecture, modelPath, options ?? new ParallelWaveGANOptions(), options?.SamplingSeed ?? 0)
    {
        _noise = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(PaperOptions.SamplingSeed + 1);
    }

    /// <summary>Creates a trainable Parallel WaveGAN.</summary>
    public ParallelWaveGAN(NeuralNetworkArchitecture<T> architecture, ParallelWaveGANOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new ParallelWaveGANOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
        _noise = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(PaperOptions.SamplingSeed + 1);
    }

    private ParallelWaveGANOptions PaperOptions => (ParallelWaveGANOptions)VocoderSettings;

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.UpsampleRates.Aggregate(1, (a, b) => a * b);

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.SegmentSize;

    /// <inheritdoc />
    protected override long DiscriminatorStartStep => PaperOptions.DiscriminatorStartStep;

    /// <inheritdoc />
    /// <remarks>The reference updates the generator first, then the discriminator on a fresh generation.</remarks>
    protected override bool DiscriminatorStepFirst => false;

    /// <inheritdoc />
    protected override double GradientClipNorm(string group)
        => group == "discriminator" ? PaperOptions.DiscriminatorGradientClipNorm : PaperOptions.GeneratorGradientClipNorm;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateGenerator()
    {
        var o = PaperOptions;
        if (UpsampleFactor != o.HopSize)
            throw new ArgumentException($"The upsampling scales ({string.Join("x", o.UpsampleRates)}) must multiply to the hop ({o.HopSize}).");
        var init = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 2);
        _generator = new ParallelWaveGanGenerator<T>(Engine, init, o.MelChannels, o.UpsampleRates, o.AuxContextWindow,
            o.NumLayers, o.NumStacks, o.ResidualChannels, o.GateChannels, o.SkipChannels, o.KernelSize);
        _features = new CenteredLogMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, o.MelMinFrequency, o.MelMaxFrequency, 1e-10);
        _stftLoss = new MultiResolutionStftLoss<T>(Engine, o.StftFftSizes, o.StftHopSizes, o.StftWindowSizes);
        return _generator.Layers;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        var o = PaperOptions;
        var init = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 3);
        _discriminator = new ParallelWaveGanDiscriminator<T>(Engine, init, o.DiscriminatorLayers, o.DiscriminatorChannels, o.KernelSize);
        return _discriminator.Layers;
    }

    /// <inheritdoc />
    /// <remarks>Fresh Gaussian noise at every training forward; at synthesis, noise seeded by
    /// <see cref="ParallelWaveGANOptions.SamplingSeed"/> so a mel spectrogram always gives the same waveform.</remarks>
    protected override Tensor<T> Generate(Tensor<T> mel)
    {
        int samples = mel.Shape[2] * UpsampleFactor;
        var random = IsTrainingMode ? _noise : AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(PaperOptions.SamplingSeed);
        var noise = new Tensor<T>(new[] { 1, 1, samples });
        for (int i = 0; i < samples; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            noise[i] = NumOps.FromDouble(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return _generator!.Forward(noise, mel);
    }

    /// <inheritdoc />
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio) => _features!.Forward(audio, PaperOptions.MelMean, PaperOptions.MelScale);

    /// <inheritdoc />
    protected override Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
        => LeastSquaresDiscriminator(_discriminator!.Forward(real), _discriminator.Forward(generated));

    /// <inheritdoc />
    protected override Tensor<T> GeneratorLoss(Tensor<T> mel, Tensor<T> real, bool adversarial)
    {
        var generated = Flat(Generate(mel));
        var loss = StftLoss(generated, real);
        if (!adversarial) return loss;
        var adversarialTerm = LeastSquaresGenerator(_discriminator!.Forward(generated));
        return Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(adversarialTerm, NumOps.FromDouble(PaperOptions.AdversarialWeight)));
    }

    private Tensor<T> StftLoss(Tensor<T> generated, Tensor<T> real)
    {
        int n = Math.Min(generated.Length, real.Length);
        var (sc, mag) = _stftLoss!.Forward(new[] { Engine.TensorSlice(generated, new[] { 0 }, new[] { n }) },
            new[] { Engine.TensorSlice(Flat(real), new[] { 0 }, new[] { n }) });
        return Engine.TensorAdd(sc, mag);
    }

    /// <inheritdoc />
    /// <remarks>L_aux, the multi-resolution STFT loss (Eq. 6).</remarks>
    protected override Tensor<T> ReconstructionObjective(Tensor<T> mel, Tensor<T> real)
    {
        var (generated, target) = GeneratedAndReal(mel, real);
        return StftLoss(generated, target);
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group)
    {
        var o = PaperOptions;
        double rate = group == "discriminator" ? o.DiscriminatorLearningRate : o.LearningRate;
        int halving = Math.Max(1, o.LearningRateHalvingSteps);
        var scheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(rate, step => Math.Pow(0.5, step / halving));
        var optimizer = new RAdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new RAdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = rate,
            Epsilon = o.Epsilon,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
        });
        // The declared recipe is the generator's; the discriminator's rate is the paper's other one.
        return group == "discriminator" ? optimizer : PaperOptimizerFactory.VerifyHandBuilt(this, optimizer);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "ParallelWaveGAN-ONNX" : "ParallelWaveGAN-Native",
            Description = "Parallel WaveGAN: GAN waveform generation with multi-resolution spectrogram loss (Yamamoto et al., 2020)",
            FeatureCount = o.MelChannels,
            Complexity = o.NumLayers,
        };
        m.AdditionalInfo["Architecture"] = "ParallelWaveGAN";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
