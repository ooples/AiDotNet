using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// MelGAN: a non-autoregressive, fully convolutional GAN vocoder — transposed-convolution upsampling with dilated residual
/// stacks — trained against multi-scale window-based discriminators with the hinge loss and feature matching.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis" (Kumar et al.,
/// NeurIPS 2019) and its released code (descriptinc/melgan-neurips) for what the paper leaves unstated.</para>
/// <para>
/// Each step crops an 8192-sample segment, updates the three discriminators on
/// <c>Σ_k mean(max(0, 1 − D_k(x))) + mean(max(0, 1 + D_k(G(s))))</c> (Eq. 1) with the generation detached, then the
/// generator on <c>Σ_k −mean(D_k(G(s))) + λ Σ_k Σ_i mean|D_k⁽ⁱ⁾(x) − D_k⁽ⁱ⁾(G(s))|</c> with λ = 10 (Eq. 2–4) over every
/// intermediate discriminator layer; no loss is taken in the audio or spectrogram domain (§2.3). Both networks use Adam
/// (β = 0.5, 0.9) at 1e-4.
/// </para>
/// <para><b>For Beginners:</b> A fast generator turns a spectrogram into audio while three critics, listening at
/// different time scales, learn to tell its audio from real speech.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis",
    "https://arxiv.org/abs/1910.06711",
    Year = 2019,
    Authors = "Kumar et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta1 = 0.5, Beta2 = 0.9, ReferenceBatchSize = 16,
                Source = "Kumar et al. 2019, App. B: Adam with learning rate 0.0001, beta1 0.5 and beta2 0.9 for both the "
                        + "generator and the discriminator, batch size 16.")]
public partial class MelGAN<T> : GanVocoderBase<T>
{
    private MelGanGenerator<T>? _generator;
    private MelGanDiscriminators<T>? _discriminators;
    private DifferentiableMel<T>? _mel;

    /// <summary>Creates a MelGAN that runs an exported ONNX graph.</summary>
    public MelGAN(NeuralNetworkArchitecture<T> architecture, string modelPath, MelGANOptions? options = null)
        : base(architecture, modelPath, options ?? new MelGANOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable MelGAN.</summary>
    public MelGAN(NeuralNetworkArchitecture<T> architecture, MelGANOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new MelGANOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private MelGANOptions PaperOptions => (MelGANOptions)VocoderSettings;

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.UpsampleRates.Aggregate(1, (a, b) => a * b);

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.SegmentSize;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateGenerator()
    {
        var o = PaperOptions;
        _generator = new MelGanGenerator<T>(Engine, o.MelChannels, o.Ngf * (1 << o.UpsampleRates.Length), o.UpsampleRates, o.ResidualLayers);
        _mel = new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, UpsampleFactor, o.WindowSize, o.MelChannels, 0.0, o.SampleRate / 2.0);
        return _generator.Layers;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateDiscriminators()
    {
        _discriminators = new MelGanDiscriminators<T>(Engine, PaperOptions.NumDiscriminators, new[] { 64, 256, 1024, 1024 }, PaperOptions.DiscriminatorWidthDivisor);
        return _discriminators.Layers;
    }

    /// <inheritdoc />
    protected override Tensor<T> Generate(Tensor<T> mel) => _generator!.Forward(mel);

    /// <inheritdoc />
    /// <remarks>The reference's <c>Audio2Mel</c>: reflect padding of (n_fft − hop)/2, a Hann-windowed magnitude
    /// spectrogram, Slaney mel bands from 0 Hz to Nyquist and <c>log10(max(x, 1e-5))</c>.</remarks>
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio) => LogMel10(audio, out _);

    private Tensor<T> LogMel10(Tensor<T> audio, out int frames)
    {
        var rows = Engine.TensorMultiplyScalar(_mel!.Forward(audio), NumOps.FromDouble(1.0 / Math.Log(10.0)));   // [frames, mel]
        frames = rows.Shape[0];
        return Engine.Reshape(Engine.TensorTranspose(rows), new[] { 1, PaperOptions.MelChannels, frames });
    }

    /// <inheritdoc />
    protected override Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var realScores = _discriminators!.Forward(real);
        var fakeScores = _discriminators.Forward(generated);
        return Sum(Enumerable.Range(0, realScores.Count).Select(k => HingeDiscriminator(realScores[k].Score, fakeScores[k].Score)));
    }

    /// <inheritdoc />
    protected override Tensor<T> GeneratorLoss(Tensor<T> mel, Tensor<T> real, bool adversarial)
    {
        var fake = _discriminators!.Forward(Flat(Generate(mel)));
        List<(Tensor<T> Score, List<Tensor<T>> Features)> realOut;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) realOut = _discriminators.Forward(real);
        var adversarialTerm = Sum(fake.Select(f => HingeGenerator(f.Score)));
        var featureMatching = Sum(Enumerable.Range(0, fake.Count).Select(k => FeatureMatching(realOut[k].Features, fake[k].Features)));
        return Engine.TensorAdd(adversarialTerm, Engine.TensorMultiplyScalar(featureMatching, NumOps.FromDouble(PaperOptions.FeatureMatchingWeight)));
    }

    /// <inheritdoc />
    /// <remarks>The mean L1 distance between the log10-mel features of the generated and real audio — a measure of
    /// reconstruction, not a term MelGAN trains on (§2.3 takes no loss outside the discriminators).</remarks>
    protected override Tensor<T> ReconstructionObjective(Tensor<T> mel, Tensor<T> real)
    {
        var (generated, target) = GeneratedAndReal(mel, real);
        return Mean(Engine.TensorAbs(Engine.TensorSubtract(LogMel10(generated, out _), LogMel10(target, out _))));
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group)
        => PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = PaperOptions.LearningRate,
                Beta1 = PaperOptions.Beta1,
                Beta2 = PaperOptions.Beta2,
                // Adam's options adapt the betas during training by default; the paper's stay fixed.
                UseAdaptiveBetas = false,
            }));

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "MelGAN-ONNX" : "MelGAN-Native",
            Description = "MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis (Kumar et al., 2019)",
            FeatureCount = o.MelChannels,
            Complexity = o.UpsampleRates.Length * o.ResidualLayers,
        };
        m.AdditionalInfo["Architecture"] = "MelGAN";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
