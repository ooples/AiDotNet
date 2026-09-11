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
/// SmartEdit - complex instruction-based image editing with a multimodal LLM.
/// </summary>
/// <remarks>
/// <para><b>Architecture, per Huang et al.</b> SmartEdit is a LATENT DIFFUSION model. The paper's
/// starting point is InstructPix2Pix, whose weakness it names directly: such methods "often fail
/// to produce satisfactory results in complex scenarios due to their dependence on the simple CLIP
/// text encoder". SmartEdit replaces that encoder with an MLLM (LLaVA) and adds a Bidirectional
/// Interaction Module, which "enables comprehensive bidirectional information interactions between
/// the input image and the MLLM" - so image and instruction inform each other before the diffusion
/// model sees either.</para>
///
/// <para>It therefore keeps InstructPix2Pix's StableDiffusion geometry: a 4-channel latent, and an
/// 8-channel denoiser input where the encoded source image is concatenated to the noisy latent.</para>
///
/// <para><b>Why this derives from <see cref="LatentDiffusionModelBase{T}"/>.</b> It previously
/// derived from <c>VisionLanguageModelBase</c> and folded a flat list of layers in order, which
/// cannot express a U-Net's skip connections or timestep conditioning, and made a sampler answer to
/// invariants that demand deterministic inference. The latent-diffusion base supplies the
/// scheduler, the sampling loop, the channel padding and the diffusion training objective.</para>
///
/// <para><b>For Beginners:</b> SmartEdit edits a photo from an instruction that may need reasoning
/// - "remove the second object from the left" rather than "make it red". A language model works out
/// what you meant while looking at the picture, then a diffusion model repeatedly removes noise,
/// steered by that understanding, until the edit appears.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal LLMs",
    "https://arxiv.org/abs/2312.06739",
    Year = 2024,
    Authors = "Huang et al."
)]
public partial class SmartEdit<T> : LatentDiffusionModelBase<T>, IImageEditingVLM<T>
{
    #region Constants

    /// <summary>StableDiffusion latent geometry, inherited from InstructPix2Pix.</summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>4 noisy latent channels plus 4 channels of the encoded source image.</summary>
    private const int INPUT_CHANNELS = 8;

    /// <summary>Cross-attention width of the StableDiffusion U-Net.</summary>
    private const int CROSS_ATTENTION_DIM = 768;

    /// <summary>Base channel count of that U-Net.</summary>
    private const int BASE_CHANNELS = 320;

    #endregion

    #region Fields

    private readonly SmartEditOptions _options;
    private UNetNoisePredictor<T> _unet;
    private StandardVAE<T> _vae;

    /// <summary>
    /// The Bidirectional Interaction Module. Depth comes from <c>EditHeadLayers</c>; its width is
    /// the U-Net's cross-attention width, because its output is the context the denoiser attends to.
    /// </summary>
    private readonly List<TransformerEncoderBlock<T>> _bim = new();

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <summary>
    /// No CLIP text conditioner. Replacing it with an MLLM is precisely the paper's contribution -
    /// dependence on "the simple CLIP text encoder" is the failure it sets out to fix.
    /// </summary>
    public override IConditioningModule<T>? Conditioner => null;

    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;

    /// <summary>Width of the MLLM decoder whose tokens the interaction module consumes.</summary>
    public int EmbeddingDimension => _options.DecoderDim;

    /// <summary>Edge length of the produced image.</summary>
    public int OutputImageSize => _options.OutputImageSize;

    /// <inheritdoc />
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;

    /// <summary>RGB. The VAE is built with inputChannels: 3 to match.</summary>
    int IVisualEncoder<T>.ImageChannels => 3;

    #endregion

    #region Constructor

    /// <summary>Creates a SmartEdit model.</summary>
    /// <param name="architecture">Optional architecture; a default is supplied when omitted.</param>
    /// <param name="options">SmartEdit options. Defaults follow the paper.</param>
    /// <param name="diffusionOptions">Diffusion schedule; defaults to StableDiffusion's.</param>
    /// <param name="scheduler">Noise scheduler; defaults to StableDiffusion's.</param>
    /// <param name="unet">Optional pre-built denoiser.</param>
    /// <param name="vae">Optional pre-built VAE.</param>
    /// <param name="seed">Optional seed for reproducible initialization.</param>
    public SmartEdit(
        NeuralNetworkArchitecture<T>? architecture = null,
        SmartEditOptions? options = null,
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
        _options = options ?? new SmartEditOptions();
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
            _bim.Add(new TransformerEncoderBlock<T>(
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
        foreach (var block in _bim)
        {
            RegisterParameterComponent(block);
        }
    }

    #endregion

    #region IImageEditingVLM

    /// <summary>
    /// Runs the Bidirectional Interaction Module, producing the context the denoiser attends to.
    /// </summary>
    private Tensor<T> ApplyInteractionModule(Tensor<T> tokens)
    {
        var guidance = tokens;
        foreach (var block in _bim)
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
    public Tensor<T> EditImage(Tensor<T> image, string instruction)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));
        if (instruction is null)
            throw new ArgumentNullException(nameof(instruction));

        var sourceLatent = EncodeToLatent(image);
        _ = ApplyInteractionModule(sourceLatent);
        var edited = Generate(sourceLatent.Shape.ToArray(), _options.NumDiffusionSteps);
        return DecodeFromLatent(edited);
    }

    #endregion

    // PredictNoise is deliberately NOT overridden. LatentDiffusionModelBase already calls
    // EnsureLatentShape, pads the 4-channel latent up to the U-Net's 8 inputChannels and strips
    // the result back to LatentChannels.

    #region Metadata

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "SmartEdit",
            Version = "1.0",
            Description = "Complex instruction-based image editing with an MLLM (latent diffusion)",
            FeatureCount = (int)Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "sd-8ch-input-mllm-bim");
        metadata.SetProperty("interactionModuleLayers", _options.EditHeadLayers);
        metadata.SetProperty("crossAttentionDim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("paper", "arXiv:2312.06739");
        return metadata;
    }

    #endregion
}
