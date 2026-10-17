using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// APNet: an all-frame-level neural vocoder that predicts the log amplitude spectrum (ASP) and the phase spectrum (PSP)
/// directly from the mel spectrogram with residual convolution networks and reconstructs the waveform by an inverse STFT.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "APNet: An All-Frame-Level Neural Vocoder Incorporating Direct Prediction of Amplitude and
/// Phase Spectra" (Ai and Ling, IEEE/ACM TASLP 2023) and yangai520/APNet for what the paper leaves unstated.</para>
/// <para>
/// The ASP and PSP are residual convolution networks (§III-A/B, Fig. 3); the phase losses use the negative cosine as the
/// anti-wrapping function (Eq. 11, 17, 22); L_W is HiFi-GAN's least-squares GAN loss against the MPD and MSD, twice the
/// feature-matching loss and λ_Mel times the mel loss (§III-C4).
/// </para>
/// <para><b>For Beginners:</b> Rather than drawing the waveform sample by sample, APNet predicts how loud each frequency
/// is (amplitude) and where each wave is in its cycle (phase) for every frame, and an inverse Fourier transform turns
/// that into audio, which is very fast.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "APNet: An All-Frame-Level Neural Vocoder Incorporating Direct Prediction of Amplitude and Phase Spectra",
    "https://arxiv.org/abs/2305.07952",
    Year = 2023,
    Authors = "Ai and Ling"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.8, Beta2 = 0.99, WeightDecay = 0.01, DecayRate = 0.999,
                ReferenceBatchSize = 16,
                Source = "Ai and Ling 2023, Sec. IV-A: AdamW with beta1 0.8 and beta2 0.99, an initial learning rate of 0.0002 decayed "
                        + "by 0.999 every epoch, batch 16 (weight decay: torch.optim.AdamW's default in the reference).")]
public partial class APNet<T> : AmplitudePhaseVocoderBase<T>
{
    private ApNetPredictor<T>? _amplitude;
    private ApNetPredictor<T>? _phase;
    private HiFiGanDiscriminators<T>? _discriminators;

    /// <summary>Creates an APNet that runs an exported ONNX graph.</summary>
    public APNet(NeuralNetworkArchitecture<T> architecture, string modelPath, APNetOptions? options = null)
        : base(architecture, modelPath, options ?? new APNetOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable APNet.</summary>
    public APNet(NeuralNetworkArchitecture<T> architecture, APNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new APNetOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private APNetOptions PaperOptions => (APNetOptions)VocoderSettings;

    /// <inheritdoc />
    protected override void CreatePredictors(List<LayerBase<T>> layers, Random initialization)
    {
        var o = PaperOptions;
        int bins = o.FftSize / 2 + 1;
        _amplitude = new ApNetPredictor<T>(Engine, initialization, o.MelChannels, o.Channels, o.ResblockKernelSizes, o.ResblockDilationSizes,
            o.InputKernelSize, o.OutputKernelSize, bins, 1, layers);
        _phase = new ApNetPredictor<T>(Engine, initialization, o.MelChannels, o.Channels, o.ResblockKernelSizes, o.ResblockDilationSizes,
            o.InputKernelSize, o.OutputKernelSize, bins, 2, layers);
    }

    /// <inheritdoc />
    protected override (Tensor<T> LogAmplitude, Tensor<T> Real, Tensor<T> Imaginary) PredictComponents(Tensor<T> mel)
    {
        var parts = _phase!.Forward(mel);
        return (_amplitude!.Forward(mel)[0], parts[0], parts[1]);
    }

    /// <inheritdoc />
    /// <remarks>The negative cosine (§III-C2): even, 2π-periodic and increasing on [0, π].</remarks>
    protected override Tensor<T> PhaseError(Tensor<T> difference) => Engine.TensorNegate(Engine.TensorCos(difference));

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        var o = PaperOptions;
        _discriminators = new HiFiGanDiscriminators<T>(Engine, o.DiscriminatorPeriods, 3, true, o.DiscriminatorWidthDivisor);
        return _discriminators.Layers;
    }

    /// <inheritdoc />
    /// <remarks>Σ_k (D_k(x) − 1)² + D_k(x̂)² over the MPD and MSD (reference <c>discriminator_loss</c>).</remarks>
    protected override Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var realScores = _discriminators!.Forward(real);
        var fakeScores = _discriminators.Forward(generated);
        return Sum(Enumerable.Range(0, realScores.Count).Select(k => LeastSquaresDiscriminator(realScores[k].Score, fakeScores[k].Score)));
    }

    /// <inheritdoc />
    /// <remarks>Σ_k (D_k(x̂) − 1)² + 2 L_FM (reference <c>generator_loss</c>, <c>feature_loss</c>).</remarks>
    protected override Tensor<T> AdversarialTerms(Tensor<T> generated, Tensor<T> real)
    {
        var fake = _discriminators!.Forward(generated);
        List<(Tensor<T> Score, List<Tensor<T>> Features)> realOut;
        using (new NoGradScope<T>()) realOut = _discriminators.Forward(real);
        return Engine.TensorAdd(Sum(fake.Select(f => LeastSquaresGenerator(f.Score))),
            Engine.TensorMultiplyScalar(Sum(Enumerable.Range(0, fake.Count).Select(k => FeatureMatching(realOut[k].Features, fake[k].Features))),
                NumOps.FromDouble(PaperOptions.FeatureMatchingWeight)));
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "APNet-ONNX" : "APNet-Native",
            Description = "APNet: An All-Frame-Level Neural Vocoder Incorporating Direct Prediction of Amplitude and Phase Spectra (Ai and Ling, 2023)",
            FeatureCount = o.MelChannels,
            Complexity = o.Channels,
        };
        m.AdditionalInfo["Architecture"] = "APNet";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
