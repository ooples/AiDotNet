using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// UnivNet: a GAN vocoder whose generator uses location-variable convolutions with kernels predicted from the mel
/// spectrogram, trained against multi-resolution spectrogram and multi-period waveform discriminators.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity
/// Waveform Generation" (Jang et al., Interspeech 2021) and maum-ai/univnet for what the paper leaves unstated.</para>
/// <para>
/// The generator shapes Gaussian noise into a waveform through three LVC residual stacks (§3.1). It first trains alone
/// for 200k steps on λ · L_aux, the multi-resolution STFT loss (§3.3, §4.3); afterwards each step updates the generator on
/// λ · L_aux plus the LSGAN generator loss averaged over the three resolution and five period discriminators (§3.2), and
/// the discriminators on the averaged LSGAN loss against that same generation. Adam (β = 0.5, 0.9) at 1e-4.
/// </para>
/// <para><b>For Beginners:</b> Every short stretch of the spectrogram designs its own small filter, which the network uses
/// to shape noise into the matching stretch of audio; critics looking at spectrograms of several resolutions keep the
/// result sharp.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation",
    "https://arxiv.org/abs/2106.07889",
    Year = 2021,
    Authors = "Jang et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta1 = 0.5, Beta2 = 0.9, ReferenceBatchSize = 32,
                Source = "Jang et al. 2021, Sec. 4.3: Adam with beta1 0.5, beta2 0.9 and a learning rate of 1e-4; maum-ai/univnet "
                        + "trains with batch size 32.")]
public partial class UnivNet<T> : GanVocoderBase<T>
{
    private UnivNetGenerator<T>? _generator;
    private MultiResolutionSpectrogramDiscriminators<T>? _resolutions;
    private HiFiGanDiscriminators<T>? _periods;
    private DifferentiableMel<T>? _features;
    private MultiResolutionStftLoss<T>? _stftLoss;
    private Random _noise;

    /// <summary>Creates a UnivNet that runs an exported ONNX graph.</summary>
    public UnivNet(NeuralNetworkArchitecture<T> architecture, string modelPath, UnivNetOptions? options = null)
        : base(architecture, modelPath, options ?? new UnivNetOptions(), options?.SamplingSeed ?? 0)
    {
        _noise = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(PaperOptions.SamplingSeed + 1);
    }

    /// <summary>Creates a trainable UnivNet.</summary>
    public UnivNet(NeuralNetworkArchitecture<T> architecture, UnivNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new UnivNetOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
        _noise = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(PaperOptions.SamplingSeed + 1);
    }

    private UnivNetOptions PaperOptions => (UnivNetOptions)VocoderSettings;

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.UpsampleRates.Aggregate(1, (a, b) => a * b);

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.SegmentSize;

    /// <inheritdoc />
    protected override long DiscriminatorStartStep => PaperOptions.PretrainingSteps;

    /// <inheritdoc />
    /// <remarks>The reference updates the generator first, then the discriminators on that same generation.</remarks>
    protected override bool DiscriminatorStepFirst => false;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateGenerator()
    {
        var o = PaperOptions;
        if (UpsampleFactor != o.HopSize)
            throw new ArgumentException($"The strides ({string.Join("x", o.UpsampleRates)}) must multiply to the hop ({o.HopSize}).");
        _generator = new UnivNetGenerator<T>(Engine, o.MelChannels, o.NoiseDim, o.ChannelSize, o.UpsampleRates, o.Dilations,
            o.LeakyReluSlope, o.KernelPredictorHidden, o.KernelPredictorConvSize);
        _features = new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, 0.0, o.MelMaxFrequency);
        _stftLoss = new MultiResolutionStftLoss<T>(Engine, o.StftFftSizes, o.StftHopSizes, o.StftWindowSizes);
        return _generator.Layers;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        var o = PaperOptions;
        _resolutions = new MultiResolutionSpectrogramDiscriminators<T>(Engine, o.StftFftSizes, o.StftHopSizes, o.StftWindowSizes,
            o.ResolutionDiscriminatorChannels, o.LeakyReluSlope);
        _periods = new HiFiGanDiscriminators<T>(Engine, o.DiscriminatorPeriods, 0, useScaleDiscriminator: false,
            widthDivisor: o.DiscriminatorWidthDivisor, periodChannels: o.PeriodDiscriminatorChannels, slope: o.LeakyReluSlope);
        return _resolutions.Layers.Concat(_periods.Layers).ToList();
    }

    /// <inheritdoc />
    /// <remarks>Fresh noise <c>[1, 64, frames]</c> at every training forward; at synthesis, noise seeded by
    /// <see cref="UnivNetOptions.SamplingSeed"/>.</remarks>
    protected override Tensor<T> Generate(Tensor<T> mel)
    {
        var o = PaperOptions;
        var random = IsTrainingMode ? _noise : AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
        var noise = new Tensor<T>(new[] { 1, o.NoiseDim, mel.Shape[2] });
        for (int i = 0; i < noise.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            noise[i] = NumOps.FromDouble(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return _generator!.Forward(mel, noise);
    }

    /// <inheritdoc />
    /// <remarks>The reference's <c>inference</c>: ten frames of log(1e-5) appended, their samples trimmed.</remarks>
    protected override Tensor<T> GenerateForInference(Tensor<T> mel)
    {
        int extra = PaperOptions.InferencePaddingFrames;
        if (extra <= 0) return Generate(mel);
        var silence = new Tensor<T>(new[] { 1, mel.Shape[1], extra });
        for (int i = 0; i < silence.Length; i++) silence[i] = NumOps.FromDouble(Math.Log(1e-5));
        var wave = Flat(Generate(Engine.TensorConcatenate(new[] { mel, silence }, 2)));
        int keep = mel.Shape[2] * UpsampleFactor;
        return Engine.Reshape(Engine.TensorSlice(wave, new[] { 0 }, new[] { keep }), new[] { 1, 1, keep });
    }

    /// <inheritdoc />
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio)
    {
        var rows = _features!.Forward(audio);
        return Engine.Reshape(Engine.TensorTranspose(rows), new[] { 1, PaperOptions.MelChannels, rows.Shape[0] });
    }

    private List<(Tensor<T> Score, List<Tensor<T>> Features)> Discriminate(Tensor<T> audio)
        => _resolutions!.Forward(audio).Concat(_periods!.Forward(audio)).ToList();

    /// <inheritdoc />
    protected override Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var realScores = Discriminate(real);
        var fakeScores = Discriminate(generated);
        var total = Sum(Enumerable.Range(0, realScores.Count).Select(k => LeastSquaresDiscriminator(realScores[k].Score, fakeScores[k].Score)));
        return Engine.TensorMultiplyScalar(total, NumOps.FromDouble(1.0 / realScores.Count));
    }

    /// <inheritdoc />
    protected override Tensor<T> GeneratorLoss(Tensor<T> mel, Tensor<T> real, bool adversarial)
    {
        var generated = Flat(Generate(mel));
        if (adversarial) RememberGeneration(generated);
        var loss = Engine.TensorMultiplyScalar(StftLoss(generated, real), NumOps.FromDouble(PaperOptions.StftLossWeight));
        if (!adversarial) return loss;
        var fake = Discriminate(generated);
        var score = Engine.TensorMultiplyScalar(Sum(fake.Select(f => LeastSquaresGenerator(f.Score))), NumOps.FromDouble(1.0 / fake.Count));
        return Engine.TensorAdd(loss, score);
    }

    private Tensor<T> StftLoss(Tensor<T> generated, Tensor<T> real)
    {
        int n = Math.Min(generated.Length, real.Length);
        var (sc, mag) = _stftLoss!.Forward(new[] { Engine.TensorSlice(generated, new[] { 0 }, new[] { n }) },
            new[] { Engine.TensorSlice(Flat(real), new[] { 0 }, new[] { n }) });
        return Engine.TensorAdd(sc, mag);
    }

    /// <inheritdoc />
    /// <remarks>λ · L_aux, the weighted multi-resolution STFT loss.</remarks>
    protected override Tensor<T> ReconstructionObjective(Tensor<T> mel, Tensor<T> real)
    {
        var (generated, target) = GeneratedAndReal(mel, real);
        return Engine.TensorMultiplyScalar(StftLoss(generated, target), NumOps.FromDouble(PaperOptions.StftLossWeight));
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group)
        => PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = PaperOptions.LearningRate,
                Beta1 = PaperOptions.Beta1,
                Beta2 = PaperOptions.Beta2,
                UseAdaptiveBetas = false,
            }));

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "UnivNet-ONNX" : "UnivNet-Native",
            Description = "UnivNet: Neural Vocoder with Multi-Resolution Spectrogram Discriminators (Jang et al., 2021)",
            FeatureCount = o.MelChannels,
            Complexity = o.UpsampleRates.Length * o.Dilations.Length,
        };
        m.AdditionalInfo["Architecture"] = "UnivNet";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
