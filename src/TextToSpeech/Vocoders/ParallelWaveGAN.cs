using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;

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
    private bool _recordingContext;
    private Tensor<T>? _trainingNoise;

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
        if (o.UpsampleRates.Length == 0 || o.UpsampleRates.Any(s => s < 1))
            throw new ArgumentException("Conditioning upsampling scales must be positive.", nameof(o.UpsampleRates));
        if (o.SegmentSize < o.HopSize) throw new ArgumentOutOfRangeException(nameof(o.SegmentSize));
        if (o.AuxContextWindow < 0) throw new ArgumentOutOfRangeException(nameof(o.AuxContextWindow));
        if (double.IsNaN(o.WeightDecay) || double.IsInfinity(o.WeightDecay) || o.WeightDecay < 0)
            throw new ArgumentOutOfRangeException(nameof(o.WeightDecay));
        if (double.IsNaN(o.Epsilon) || double.IsInfinity(o.Epsilon) || o.Epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(o.Epsilon));
        if ((o.MelMean is null) != (o.MelScale is null))
            throw new ArgumentException("Mel normalization requires both fitted means and scales.");
        if (o.MelMean is not null && (o.MelMean.Length != o.MelChannels || o.MelMean.Any(x => double.IsNaN(x) || double.IsInfinity(x))))
            throw new ArgumentException("Each mel channel needs a finite fitted mean.", nameof(o.MelMean));
        if (o.MelScale is not null && (o.MelScale.Length != o.MelChannels || o.MelScale.Any(x => double.IsNaN(x) || double.IsInfinity(x) || x <= 0)))
            throw new ArgumentException("Each mel channel needs a positive finite fitted scale.", nameof(o.MelScale));
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
    /// <remarks>One Gaussian input per training batch, reused for the post-update discriminator generation;
    /// at synthesis, noise seeded by
    /// <see cref="ParallelWaveGANOptions.SamplingSeed"/> so a mel spectrogram always gives the same waveform.</remarks>
    protected override Tensor<T> Generate(Tensor<T> mel)
    {
        int frames = mel.Shape[2] - (_recordingContext ? 2 * PaperOptions.AuxContextWindow : 0);
        int samples = frames * UpsampleFactor;
        var noise = IsTrainingMode && _trainingNoise is not null ? _trainingNoise
            : DrawNoise(samples, IsTrainingMode ? _noise : AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(PaperOptions.SamplingSeed));
        return _generator!.Forward(noise, mel, hasContext: _recordingContext);
    }

    private Tensor<T> DrawNoise(int samples, Random random)
    {
        var noise = new Tensor<T>(new[] { 1, 1, samples });
        for (int i = 0; i < samples; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            noise[i] = NumOps.FromDouble(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return noise;
    }

    /// <inheritdoc />
    protected override void PrepareTrainingStep(Tensor<T> mel, Tensor<T> audio)
        => _trainingNoise = DrawNoise((mel.Shape[2] - (_recordingContext ? 2 * PaperOptions.AuxContextWindow : 0)) * UpsampleFactor, _noise);

    /// <inheritdoc />
    protected override void CompleteTrainingStep() => _trainingNoise = null;

    /// <inheritdoc />
    protected override T TrainOnSample(AiDotNet.TextToSpeech.TtsTrainingSample<T> sample)
    {
        bool previous = _recordingContext;
        _recordingContext = true;
        try { return base.TrainOnSample(sample); }
        finally { _recordingContext = previous; }
    }

    /// <inheritdoc />
    protected override (Tensor<T> Mel, Tensor<T> Audio) RecordingPair(Tensor<T> audio, Random random)
        => PrepareRecording(audio, random);

    // Released Collater: round the segment down to complete hops, choose a frame in the
    // interior, retain real left/right conditioning context, and crop its aligned waveform.
    internal (Tensor<T> Mel, Tensor<T> Audio) PrepareRecording(Tensor<T> audio, Random random)
    {
        var o = PaperOptions;
        var mel = ComputeMel(audio);
        int frames = o.SegmentSize / o.HopSize, context = o.AuxContextWindow;
        if (mel.Shape[2] <= frames + 2 * context)
            throw new ArgumentException("The recording is too short for the training segment and its real conditioning context.", nameof(audio));
        int start = random.Next(context, mel.Shape[2] - frames - context);
        return (Engine.TensorSlice(mel, new[] { 0, 0, start - context }, new[] { 1, o.MelChannels, frames + 2 * context }),
            Engine.TensorSlice(Flat(audio), new[] { start * o.HopSize }, new[] { frames * o.HopSize }));
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
        var scheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(rate, step => Math.Pow(0.5, Math.Truncate(step / (double)halving)));
        var settings = new RAdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = rate,
            Epsilon = o.Epsilon,
            EnableGradientClipping = false,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
        };
        if (o.WeightDecay > 0)
            settings.Regularization = new L2Regularization<T, Tensor<T>, Tensor<T>>(
                new RegularizationOptions { Strength = o.WeightDecay, L1Ratio = 0.0 });
        var optimizer = new RAdamOptimizer<T, Tensor<T>, Tensor<T>>(this, settings);
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
