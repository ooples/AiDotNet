using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.StyleTransfer;

/// <summary>
/// InstantStyle model for zero-shot style transfer using IP-Adapter with style-content separation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// InstantStyle achieves high-quality style transfer in a single forward pass by
/// selectively injecting IP-Adapter image features into only the style-relevant
/// attention layers, preventing content leakage from the style reference.
/// </para>
/// <para>
/// <b>For Beginners:</b> InstantStyle transfers artistic styles instantly — no training
/// or fine-tuning needed per style. It smartly injects style information into specific
/// parts of the model to capture the look without copying the content of the style image.
/// </para>
/// <para>
/// Reference: Wang et al., "InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation", 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new DiffusionModelOptions&lt;float&gt; { LatentChannels = 4, DefaultInferenceSteps = 30 };
/// var model = new InstantStyleModel&lt;float&gt;(options: options);
/// var styleRef = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 4, 64, 64 });
/// var stylized = model.Predict(styleRef);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.StyleTransfer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation",
    "https://arxiv.org/abs/2404.02733",
    Year = 2024,
    Authors = "Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, Anthony Chen")]
public partial class InstantStyleModel<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_predictor);
        RegisterParameterComponent(_vae);
    }

    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 7.5;

    /// <summary>
    /// Number of transformer blocks in the SDXL UNet: "There are 11 transformer blocks with SDXL,
    /// 4 for downsample blocks, 1 for middle block, 6 for upsample blocks."
    /// </summary>
    public const int TransformerBlockCount = 11;

    /// <summary>
    /// The STYLE block, 1-based within <see cref="TransformerBlockCount"/>. The paper locates it as
    /// <c>up_blocks.0.attentions.1</c>, which "capture[s] style (color, material, atmosphere)", and
    /// notes it is "the 6th" for easy understanding. "Most time, the 6th block is enough to capture
    /// style."
    /// </summary>
    public const int StyleBlockIndex = 6;

    /// <summary>
    /// The LAYOUT block, 1-based. The paper locates it as <c>down_blocks.2.attentions.1</c>, which
    /// captures "spatial layout (structure, composition)", and notes it is "the 4th". It "matters
    /// only when the layout is a part of style in some cases", so it is off by default.
    /// </summary>
    public const int LayoutBlockIndex = 4;

    /// <summary>
    /// Whether reference features are also injected into the layout block
    /// (<see cref="LayoutBlockIndex"/>). Default <c>false</c>, per "most time, the 6th block is
    /// enough to capture style; the 4th matters only when the layout is a part of style".
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Turn this on when the reference image's composition — where things sit
    /// in the frame — is part of the look you want, not just its colours and texture.
    /// </remarks>
    public bool InjectLayoutBlock { get; set; }

    /// <summary>
    /// The blocks that receive reference image features, 1-based and ascending. Everything else in
    /// the UNet receives none — that exclusivity IS the method.
    /// </summary>
    public IReadOnlyList<int> InjectionBlocks =>
        InjectLayoutBlock ? new[] { LayoutBlockIndex, StyleBlockIndex } : new[] { StyleBlockIndex };

    /// <summary>
    /// Whether block <paramref name="oneBasedBlockIndex"/> receives reference image features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// "As we have located style blocks, we can inject our image features into these blocks only to
    /// achieve style transfer seamlessly. Furthermore, since the number of parameters of the adapter
    /// is greatly reduced, the text control ability is also enhanced."
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Rather than pushing the style reference into every layer of the network
    /// — which drags the reference's actual CONTENT along with it — it is pushed into just the one
    /// or two layers that were found to encode style. That is why this needs no per-image weight
    /// tuning.
    /// </para>
    /// </remarks>
    public bool IsInjectionBlock(int oneBasedBlockIndex)
    {
        if (oneBasedBlockIndex == StyleBlockIndex) return true;
        return InjectLayoutBlock && oneBasedBlockIndex == LayoutBlockIndex;
    }

    /// <summary>
    /// Per-block injection scale for the whole UNet: 1 at an injection block, 0 everywhere else.
    /// </summary>
    /// <remarks>
    /// Index 0 corresponds to block 1. A caller wiring an IP-Adapter multiplies each block's
    /// image-feature contribution by this, so non-style blocks contribute exactly nothing.
    /// </remarks>
    public double[] BuildBlockInjectionScales()
    {
        var scales = new double[TransformerBlockCount];
        for (int i = 0; i < TransformerBlockCount; i++) scales[i] = IsInjectionBlock(i + 1) ? 1.0 : 0.0;
        return scales;
    }

    /// <summary>
    /// Decouples style from content by subtracting content-text features from reference-image
    /// features in the shared CLIP space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The paper's first strategy, in full: "we can use CLIP's text encoder to extract the
    /// characteristics of the content text as content representation. At the same time, we use
    /// CLIP's image encoder to extract the features of the reference image... after subtracting the
    /// content text features from the image features, the style and content can be explicitly
    /// decoupled." It rests on the assumption "that features within the same space can be either
    /// added to or subtracted from one another", which is why both operands must come from the same
    /// CLIP embedding space.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Describe in words what is IN the reference picture ("a dog"), encode
    /// that text, and subtract it from the picture's own encoding. What is left is roughly the
    /// picture's look with its subject removed — so the model copies the style without also copying
    /// the dog. The paper calls this "quite effective in mitigating content leakage".
    /// </para>
    /// </remarks>
    /// <param name="referenceImageFeatures">CLIP image-encoder features of the style reference.</param>
    /// <param name="contentTextFeatures">CLIP text-encoder features of the content description.</param>
    /// <returns>Style-only features, the same shape as the inputs.</returns>
    /// <exception cref="ArgumentNullException">Either argument is null.</exception>
    /// <exception cref="ArgumentException">The two feature tensors are not the same shape, which
    /// would mean they are not in a shared space and cannot be subtracted.</exception>
    public Tensor<T> DecoupleStyleFromContent(Tensor<T> referenceImageFeatures, Tensor<T> contentTextFeatures)
    {
        if (referenceImageFeatures is null) throw new ArgumentNullException(nameof(referenceImageFeatures));
        if (contentTextFeatures is null) throw new ArgumentNullException(nameof(contentTextFeatures));
        if (!referenceImageFeatures.Shape.ToArray().SequenceEqual(contentTextFeatures.Shape.ToArray()))
        {
            throw new ArgumentException(
                $"Content subtraction requires both feature sets in the SAME CLIP space; got image " +
                $"[{string.Join(",", referenceImageFeatures.Shape)}] and text " +
                $"[{string.Join(",", contentTextFeatures.Shape)}].",
                nameof(contentTextFeatures));
        }

        return Engine.TensorSubtract(referenceImageFeatures, contentTextFeatures);
    }

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    public override INoisePredictor<T> NoisePredictor => _predictor;
    public override IVAEModel<T> VAE => _vae;
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => LATENT_CHANNELS;

    public InstantStyleModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: 4, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2 }, contextDim: 2048, seed: seed);
        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocksPerLevel: 2, seed: seed);
    }



    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "InstantStyle", Version = "1.0",
            Description = "Zero-shot IP-Adapter style transfer with content-style separation",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount };
        m.SetProperty("architecture", "ip-adapter-style-injection");
        m.SetProperty("base_model", "Stable Diffusion XL");
        m.SetProperty("text_encoder", "CLIP ViT-L/14 + OpenCLIP ViT-bigG/14");
        m.SetProperty("context_dim", 2048);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        return m;
    }
}
