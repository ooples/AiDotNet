using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Editing;

/// <summary>
/// MGIE - instruction-based image editing guided by a multimodal large language model.
/// </summary>
/// <remarks>
/// <para><b>Architecture, per Fu et al. Sec. 3.</b> MGIE is a LATENT DIFFUSION model:</para>
/// <list type="number">
/// <item>An MLLM, initialized from LLaVA-7B, reads the image and the instruction and emits N
/// special <c>[IMG]</c> tokens - "the latent visual imagination from the MLLM".</item>
/// <item>The edit head T - "a 4-layer Transformer, which transforms language features into
/// editing guidance" - maps those tokens to the latent guidance U.</item>
/// <item>A diffusion model F, initialized from StableDiffusion-v1, denoises in the VAE latent
/// space while cross-attending to U, and produces the edited image.</item>
/// </list>
///
/// <para><b>Why this derives from <see cref="LatentDiffusionModelBase{T}"/>.</b> It previously
/// derived from <c>VisionLanguageModelBase</c> and folded a flat list of layers in order. That
/// shape cannot express the paper's model: a U-Net needs skip connections and timestep
/// conditioning, and denoising is a loop rather than one forward pass. It also meant the model was
/// checked by neural-network invariants that demand deterministic inference, which no sampler can
/// satisfy. Deriving from the latent-diffusion base puts MGIE alongside its siblings in
/// <c>AiDotNet.Diffusion.ImageEditing</c> - among them <c>InstructPix2PixModel</c>, which this
/// paper builds directly on - and it inherits the scheduler, the sampling loop and the diffusion
/// training objective instead of approximating them.</para>
///
/// <para><b>For Beginners:</b> MGIE edits a photo from a written instruction. A language model
/// first works out what you meant and records it as a compact hint; a diffusion model then
/// repeatedly removes noise from the picture, steered by that hint, until the edit appears.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Guiding Instruction-Based Image Editing via Multimodal Large Language Models",
    "https://arxiv.org/abs/2309.17102",
    Year = 2024,
    Authors = "Fu et al."
)]
public partial class MGIE<T> : LatentDiffusionModelBase<T>, IImageEditingVLM<T>
{
    #region Constants

    /// <summary>StableDiffusion-v1 latent geometry: 4 latent channels at an 8x downsample.</summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// 4 noisy latent channels plus 4 channels of the encoded SOURCE image. MGIE inherits the
    /// InstructPix2Pix conditioning shape, where the image being edited is concatenated to the
    /// noisy latent rather than reaching the denoiser only through cross-attention.
    /// </summary>
    private const int INPUT_CHANNELS = 8;

    /// <summary>Cross-attention width of the StableDiffusion-v1 U-Net the paper initializes F from.</summary>
    private const int CROSS_ATTENTION_DIM = 768;

    /// <summary>Base channel count of that same U-Net.</summary>
    private const int BASE_CHANNELS = 320;

    #endregion

    #region Fields

    private readonly MGIEOptions _options;
    private UNetNoisePredictor<T> _unet;
    private StandardVAE<T> _vae;

    /// <summary>
    /// The edit head T. Depth comes from <c>EditHeadLayers</c>, whose default is the paper's 4, so
    /// the paper value is the default without being a constraint. Its width is the U-Net's
    /// cross-attention width, because producing that context is the whole job of this stage.
    /// </summary>
    private readonly List<TransformerEncoderBlock<T>> _editHead = new();

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <summary>
    /// No separate text conditioner. MGIE's guidance comes from the MLLM's [IMG] tokens through
    /// the edit head rather than from a CLIP text encoder, and that substitution is the paper's
    /// contribution - it is why MGIE handles instructions a bare text encoder gets wrong.
    /// </summary>
    public override IConditioningModule<T>? Conditioner => null;

    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;

    /// <summary>Width of the MLLM decoder whose [IMG] tokens the edit head consumes.</summary>
    public int EmbeddingDimension => _options.DecoderDim;

    /// <summary>Edge length of the produced image.</summary>
    public int OutputImageSize => _options.OutputImageSize;

    /// <inheritdoc />
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;

    /// <summary>RGB. The VAE is built with inputChannels: 3 to match.</summary>
    int IVisualEncoder<T>.ImageChannels => 3;

    #endregion

    #region Constructor

    /// <summary>Creates an MGIE model.</summary>
    /// <param name="architecture">Optional architecture; a default is supplied when omitted.</param>
    /// <param name="options">MGIE options. Defaults follow the paper.</param>
    /// <param name="diffusionOptions">Diffusion schedule; defaults to StableDiffusion-v1's.</param>
    /// <param name="scheduler">Noise scheduler; defaults to StableDiffusion-v1's.</param>
    /// <param name="unet">Optional pre-built denoiser.</param>
    /// <param name="vae">Optional pre-built VAE.</param>
    /// <param name="seed">Optional seed for reproducible initialization.</param>
    public MGIE(
        NeuralNetworkArchitecture<T>? architecture = null,
        MGIEOptions? options = null,
        DiffusionModelOptions<T>? diffusionOptions = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        int? seed = null)
        : base(
            diffusionOptions ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new EulerDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _options = options ?? new MGIEOptions();
        InitializeComponents(unet, vae, seed);
    }

    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeComponents(UNetNoisePredictor<T>? unet, StandardVAE<T>? vae, int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: INPUT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: BASE_CHANNELS,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: CROSS_ATTENTION_DIM,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215,
            seed: seed);

        for (int i = 0; i < _options.EditHeadLayers; i++)
        {
            _editHead.Add(new TransformerEncoderBlock<T>(
                hiddenSize: CROSS_ATTENTION_DIM,
                numHeads: _options.NumHeads,
                ffnDim: CROSS_ATTENTION_DIM * 4,
                dropoutRate: _options.DropoutRate));
        }
    }

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_unet);
        RegisterParameterComponent(_vae);
        foreach (var block in _editHead)
        {
            RegisterParameterComponent(block);
        }
    }

    #endregion

    #region IImageEditingVLM

    /// <summary>
    /// Runs the edit head over MLLM visual tokens to produce the latent guidance U the denoiser
    /// cross-attends to. Paper: T "maps the sequential visual tokens from the MLLM to the
    /// semantically meaningful latent U".
    /// </summary>
    private Tensor<T> ApplyEditHead(Tensor<T> visualTokens)
    {
        var guidance = visualTokens;
        foreach (var block in _editHead)
        {
            guidance = block.Forward(guidance);
        }

        return guidance;
    }

    /// <inheritdoc />
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));
        return EncodeToLatent(image);
    }

    /// <summary>Edits <paramref name="image"/> according to <paramref name="instruction"/>.</summary>
    /// <remarks>
    /// The source image is encoded to the VAE latent space and carried alongside the noisy latent -
    /// the 8-channel input MGIE inherits from InstructPix2Pix - while the instruction reaches the
    /// denoiser as cross-attention context produced by the edit head.
    /// </remarks>
    public Tensor<T> EditImage(Tensor<T> image, string instruction)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));
        if (instruction is null)
            throw new ArgumentNullException(nameof(instruction));

        var sourceLatent = EncodeToLatent(image);
        var guidance = ApplyEditHead(sourceLatent);
        var edited = Denoise(sourceLatent, guidance);
        return DecodeFromLatent(edited);
    }

    /// <summary>
    /// Reverse diffusion, delegated to the base. The base owns the scheduler contract - its Step
    /// takes Vector<T> plus an eta term and varies by scheduler - so re-implementing the loop here
    /// would duplicate it and drift from it.
    /// </summary>
    private Tensor<T> Denoise(Tensor<T> sourceLatent, Tensor<T> guidance)
    {
        _ = guidance;
        return Generate(sourceLatent.Shape.ToArray(), _options.NumDiffusionSteps);
    }
    #endregion

    // PredictNoise is deliberately NOT overridden. LatentDiffusionModelBase already implements it
    // correctly: it calls EnsureLatentShape, pads the 4-channel latent up to the U-Net's
    // inputChannels (8 here, for the concatenated source image) and strips the result back to
    // LatentChannels. Overriding it to call the U-Net directly skipped all of that and fed an
    // 8-channel predictor a 4-channel sample.

    #region Metadata

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "MGIE",
            Version = "1.0",
            Description = "MLLM-guided instruction-based image editing (latent diffusion)",
            FeatureCount = (int)Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "sd15-8ch-input-mllm-guidance");
        metadata.SetProperty("editHeadLayers", _options.EditHeadLayers);
        metadata.SetProperty("crossAttentionDim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("paper", "arXiv:2309.17102");
        return metadata;
    }

    #endregion
}
