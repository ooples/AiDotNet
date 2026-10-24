using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// APNet2: a high-quality, high-efficiency vocoder that predicts the amplitude and phase spectra directly with ConvNeXt v2
/// networks and reconstructs the waveform by an inverse STFT, trained with linear anti-wrapping phase losses and a hinge
/// GAN loss against a multi-period and a multi-resolution discriminator.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "APNet2: High-quality and High-efficiency Neural Vocoder with Direct Prediction of Amplitude
/// and Phase Spectra" (Du et al., NCMMSC 2023) and redmist328/APNet2 for what the paper leaves unstated.</para>
/// <para>
/// APNet's structure and losses (§3) with ConvNeXt v2 predictors (§3.1), the anti-wrapping function
/// <c>f_AW(x) = |x − 2π round(x / 2π)|</c> (Eq. 4) in the phase losses, and L_W's hinge GAN terms averaged over the
/// sub-discriminators of the MPD and the MRD (Eq. 2–3).
/// </para>
/// <para><b>For Beginners:</b> Like APNet, APNet2 predicts how loud each frequency is and where each wave is in its cycle
/// for every frame and turns that into audio with an inverse Fourier transform; a stronger backbone lets it work at a
/// higher sampling rate and a longer frame shift.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "APNet2: High-quality and High-efficiency Neural Vocoder with Direct Prediction of Amplitude and Phase Spectra",
    "https://arxiv.org/abs/2311.11545",
    Year = 2023,
    Authors = "Du et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.8, Beta2 = 0.99, WeightDecay = 0.01, DecayRate = 0.999,
                ReferenceBatchSize = 16,
                Source = "Du et al. 2023, Sec. 4.1: AdamW with beta1 0.8, beta2 0.99 and weight decay 0.01, a learning rate of 2e-4 "
                        + "decayed by 0.999 every epoch, batch 16.")]
public partial class APNet2<T> : AmplitudePhaseVocoderBase<T>
{
    private ConvNeXtV2Predictor<T>? _amplitude;
    private ConvNeXtV2Predictor<T>? _phase;
    private HiFiGanDiscriminators<T>? _periods;
    private ApNet2ResolutionDiscriminators<T>? _resolutions;

    /// <summary>Creates an APNet2 that runs an exported ONNX graph.</summary>
    public APNet2(NeuralNetworkArchitecture<T> architecture, string modelPath, APNet2Options? options = null)
        : base(architecture, modelPath, options ?? new APNet2Options(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable APNet2.</summary>
    public APNet2(NeuralNetworkArchitecture<T> architecture, APNet2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new APNet2Options(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private APNet2Options PaperOptions => (APNet2Options)VocoderSettings;

    /// <inheritdoc />
    protected override void CreatePredictors(List<LayerBase<T>> layers, Random initialization)
    {
        var o = PaperOptions;
        int bins = o.FftSize / 2 + 1;
        _amplitude = new ConvNeXtV2Predictor<T>(Engine, initialization, o.MelChannels, o.ConvNeXtChannels, o.ConvNeXtIntermediateChannels,
            o.NumConvNeXtBlocks, o.DepthwiseKernelSize, o.InputKernelSize, o.OutputKernelSize, bins, 1, layers);
        _phase = new ConvNeXtV2Predictor<T>(Engine, initialization, o.MelChannels, o.ConvNeXtChannels, o.ConvNeXtIntermediateChannels,
            o.NumConvNeXtBlocks, o.DepthwiseKernelSize, o.InputKernelSize, o.OutputKernelSize, bins, 2, layers);
    }

    /// <inheritdoc />
    protected override (Tensor<T> LogAmplitude, Tensor<T> Real, Tensor<T> Imaginary) PredictComponents(Tensor<T> mel)
    {
        var parts = _phase!.Forward(mel);
        return (_amplitude!.Forward(mel)[0], parts[0], parts[1]);
    }

    /// <inheritdoc />
    /// <remarks><c>f_AW(x) = |x − 2π round(x / 2π)|</c> (Eq. 4); the rounding is a constant step, so the gradient is
    /// sign(x − 2π round(x / 2π)).</remarks>
    protected override Tensor<T> PhaseError(Tensor<T> difference)
    {
        Tensor<T> wraps;
        using (new NoGradScope<T>())
        {
            var turns = new Tensor<T>(difference._shape);
            for (int i = 0; i < turns.Length; i++)
                turns[i] = NumOps.FromDouble(2 * Math.PI * Math.Round(NumOps.ToDouble(difference[i]) / (2 * Math.PI), MidpointRounding.ToEven));
            wraps = turns;
        }
        return Engine.TensorAbs(Engine.TensorSubtract(difference, wraps));
    }

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        var o = PaperOptions;
        _periods = new HiFiGanDiscriminators<T>(Engine, o.DiscriminatorPeriods, 0, useScaleDiscriminator: false, widthDivisor: o.DiscriminatorWidthDivisor);
        _resolutions = new ApNet2ResolutionDiscriminators<T>(Engine, o.ResolutionFftSizes, o.ResolutionHopSizes, o.ResolutionWindowSizes,
            Math.Max(1, o.ResolutionDiscriminatorChannels / o.DiscriminatorWidthDivisor));
        return _periods.Layers.Concat(_resolutions.Layers).ToList();
    }

    private List<(Tensor<T> Score, List<Tensor<T>> Features)> Discriminate(Tensor<T> audio)
        => _periods!.Forward(audio).Concat(_resolutions!.Forward(audio)).ToList();

    private Tensor<T> Hinge(Tensor<T> x, double sign)
        => Mean(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(sign)), NumOps.One)));

    /// <inheritdoc />
    /// <remarks><c>(1/L) Σ_l max(0, 1 − D_l(x)) + max(0, 1 + D_l(x̂))</c> over the MPD's and MRD's sub-discriminators (Eq. 3).</remarks>
    protected override Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var r = Discriminate(real);
        var g = Discriminate(generated);
        return Engine.TensorMultiplyScalar(Sum(Enumerable.Range(0, r.Count).Select(k => Engine.TensorAdd(Hinge(r[k].Score, -1), Hinge(g[k].Score, 1)))),
            NumOps.FromDouble(1.0 / r.Count));
    }

    /// <inheritdoc />
    /// <remarks><c>(1/L) Σ_l max(0, 1 − D_l(x̂))</c> (Eq. 2) plus the feature-matching loss, the MRD's maps weighted
    /// 0.1 (reference).</remarks>
    protected override Tensor<T> AdversarialTerms(Tensor<T> generated, Tensor<T> real)
    {
        var o = PaperOptions;
        var fake = Discriminate(generated);
        List<(Tensor<T> Score, List<Tensor<T>> Features)> realOut;
        using (new NoGradScope<T>()) realOut = Discriminate(real);
        int periods = o.DiscriminatorPeriods.Length;
        var adversarial = Engine.TensorMultiplyScalar(Sum(fake.Select(f => Hinge(f.Score, -1))), NumOps.FromDouble(1.0 / fake.Count));
        var featureMatching = Sum(Enumerable.Range(0, fake.Count).Select(k =>
            Engine.TensorMultiplyScalar(FeatureMatching(realOut[k].Features, fake[k].Features), NumOps.FromDouble(k < periods ? 1.0 : o.ResolutionFeatureMatchingWeight))));
        return Engine.TensorAdd(adversarial, Engine.TensorMultiplyScalar(featureMatching, NumOps.FromDouble(o.FeatureMatchingWeight)));
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "APNet2-ONNX" : "APNet2-Native",
            Description = "APNet2: High-quality and High-efficiency Neural Vocoder with Direct Prediction of Amplitude and Phase Spectra (Du et al., 2023)",
            FeatureCount = o.MelChannels,
            Complexity = o.NumConvNeXtBlocks,
        };
        m.AdditionalInfo["Architecture"] = "APNet2";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
