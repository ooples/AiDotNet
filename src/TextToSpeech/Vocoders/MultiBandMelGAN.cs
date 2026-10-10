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
/// Multi-band MelGAN: a MelGAN generator that predicts PQMF sub-bands of the waveform at a quarter of its rate, trained
/// with full- and sub-band multi-resolution STFT losses and multi-scale discriminators.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech" (Yang et al.,
/// SLT 2021) and kan-bayashi/ParallelWaveGAN for what the paper leaves unstated.</para>
/// <para>
/// The generator predicts four sub-band signals; the PQMF synthesis filter merges them into the full-band waveform the
/// discriminators see (§2.2, §3.2). The generator first trains alone for 200k steps on
/// <c>L_mr_stft = ½ (L_full + L_sub)</c> (Eq. 9; sub-band targets from the PQMF analysis filter), then the
/// discriminators train on the LSGAN loss (Eq. 1) and the generator on <c>λ_adv Σ_k (D_k(G(s)) − 1)² + L_mr_stft</c>
/// (Eq. 8). Both use Adam at 1e-4, halved every 100k steps to 1e-6.
/// </para>
/// <para><b>For Beginners:</b> Instead of generating every audio sample, the network generates four narrow frequency
/// bands at a quarter of the sample rate and a fixed filter bank merges them, which makes it several times faster.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech",
    "https://arxiv.org/abs/2005.05106",
    Year = 2021,
    Authors = "Yang et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, ReferenceBatchSize = 128,
                Source = "Yang et al. 2021, Sec. 3.2: Adam with an initial learning rate of 1e-4 for G and D, halved every "
                        + "100K steps until 1e-6; batch size 128 for MB-MelGAN, one second of audio per example.")]
public partial class MultiBandMelGAN<T> : GanVocoderBase<T>
{
    private MelGanGenerator<T>? _generator;
    private PseudoQmf<T>? _pqmf;
    private MelGanDiscriminators<T>? _discriminators;
    private CenteredLogMel<T>? _features;
    private MultiResolutionStftLoss<T>? _fullBand;
    private MultiResolutionStftLoss<T>? _subBand;

    /// <summary>Creates a Multi-band MelGAN that runs an exported ONNX graph.</summary>
    public MultiBandMelGAN(NeuralNetworkArchitecture<T> architecture, string modelPath, MultiBandMelGANOptions? options = null)
        : base(architecture, modelPath, options ?? new MultiBandMelGANOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable Multi-band MelGAN.</summary>
    public MultiBandMelGAN(NeuralNetworkArchitecture<T> architecture, MultiBandMelGANOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new MultiBandMelGANOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private MultiBandMelGANOptions PaperOptions => (MultiBandMelGANOptions)VocoderSettings;

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.NumBands * PaperOptions.UpsampleRates.Aggregate(1, (a, b) => a * b);

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.SegmentSize;

    /// <inheritdoc />
    protected override long DiscriminatorStartStep => PaperOptions.PretrainingSteps;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateGenerator()
    {
        var o = PaperOptions;
        if (o.UpsampleRates.Aggregate(1, (a, b) => a * b) * o.NumBands != o.HopSize)
            throw new ArgumentException($"The bands ({o.NumBands}) times the upsampling ({string.Join("x", o.UpsampleRates)}) must equal the hop ({o.HopSize}).");
        _generator = new MelGanGenerator<T>(Engine, o.MelChannels, o.UpsampleInitialChannels, o.UpsampleRates, o.ResidualLayers, o.NumBands);
        _pqmf = new PseudoQmf<T>(Engine, o.NumBands, o.PqmfTaps, o.PqmfCutoffRatio, o.PqmfBeta);
        _features = new CenteredLogMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, o.MelMinFrequency, o.MelMaxFrequency, 1e-10);
        _fullBand = new MultiResolutionStftLoss<T>(Engine, o.FullBandFftSizes, o.FullBandHopSizes, o.FullBandWindowSizes);
        _subBand = new MultiResolutionStftLoss<T>(Engine, o.SubBandFftSizes, o.SubBandHopSizes, o.SubBandWindowSizes);
        return _generator.Layers;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        var o = PaperOptions;
        _discriminators = new MelGanDiscriminators<T>(Engine, o.NumDiscriminators, o.DiscriminatorChannels, o.DiscriminatorWidthDivisor);
        return _discriminators.Layers;
    }

    /// <summary>The sub-bands <c>[1, bands, frames · Π r]</c> the generator predicts for <paramref name="mel"/>.</summary>
    private Tensor<T> SubBands(Tensor<T> mel) => _generator!.Forward(mel);

    /// <inheritdoc />
    protected override Tensor<T> Generate(Tensor<T> mel) => _pqmf!.Synthesis(SubBands(mel));

    /// <inheritdoc />
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio) => _features!.Forward(audio, PaperOptions.MelMean, PaperOptions.MelScale);

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
        var bands = SubBands(mel);
        var generated = Flat(_pqmf!.Synthesis(bands));
        var loss = StftLoss(bands, generated, real);
        if (!adversarial) return loss;
        var fake = _discriminators!.Forward(generated);
        var adversarialTerm = Sum(fake.Select(f => LeastSquaresGenerator(f.Score)));
        return Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(adversarialTerm, NumOps.FromDouble(PaperOptions.AdversarialWeight)));
    }

    // Eq. 9: ½ (full-band + sub-band multi-resolution STFT losses), each the sum of its spectral convergence and
    // log-magnitude terms.
    private Tensor<T> StftLoss(Tensor<T> bands, Tensor<T> generated, Tensor<T> real)
    {
        int n = Math.Min(generated.Length, real.Length);
        var g = Engine.TensorSlice(generated, new[] { 0 }, new[] { n });
        var r = Engine.TensorSlice(Flat(real), new[] { 0 }, new[] { n });
        var (fullSc, fullMag) = _fullBand!.Forward(new[] { g }, new[] { r });

        Tensor<T> realBands;
        using (new NoGradScope<T>()) realBands = Detached(_pqmf!.Analysis(Engine.Reshape(r, new[] { 1, 1, n })));
        int k = PaperOptions.NumBands, length = Math.Min(bands.Shape[2], realBands.Shape[2]);
        var generatedBands = Enumerable.Range(0, k).Select(b => Engine.Reshape(Engine.TensorSlice(bands, new[] { 0, b, 0 }, new[] { 1, 1, length }), new[] { length })).ToList();
        var targetBands = Enumerable.Range(0, k).Select(b => Engine.Reshape(Engine.TensorSlice(realBands, new[] { 0, b, 0 }, new[] { 1, 1, length }), new[] { length })).ToList();
        var (subSc, subMag) = _subBand!.Forward(generatedBands, targetBands);
        return Engine.TensorMultiplyScalar(Engine.TensorAdd(Engine.TensorAdd(fullSc, fullMag), Engine.TensorAdd(subSc, subMag)), NumOps.FromDouble(0.5));
    }

    /// <inheritdoc />
    /// <remarks>The multi-resolution STFT loss (Eq. 9) of the whole generation.</remarks>
    protected override Tensor<T> ReconstructionObjective(Tensor<T> mel, Tensor<T> real)
    {
        var bands = SubBands(mel);
        return StftLoss(bands, Flat(_pqmf!.Synthesis(bands)), real);
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group)
    {
        var o = PaperOptions;
        int halving = Math.Max(1, o.LearningRateHalvingSteps);
        var scheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.LearningRate,
            step => Math.Max(Math.Pow(0.5, step / halving), o.MinimumLearningRate / o.LearningRate));
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
                // Adam's options adapt the betas during training by default; the paper's stay fixed.
                UseAdaptiveBetas = false,
            }));
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "MultiBandMelGAN-ONNX" : "MultiBandMelGAN-Native",
            Description = "Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech (Yang et al., 2021)",
            FeatureCount = o.MelChannels,
            Complexity = o.UpsampleRates.Length * o.ResidualLayers,
        };
        m.AdditionalInfo["Architecture"] = "MultiBandMelGAN";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
