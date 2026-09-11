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
/// Emu Edit - precise image editing via recognition and generation tasks.
/// </summary>
/// <remarks>
/// <para><b>Architecture, per Sheynin et al.</b> Emu Edit is a LATENT DIFFUSION model built on Emu,
/// which "incorporated a <b>16-channel autoencoder</b> with encoder E and decoder D" and generates
/// at 512x512. That is the one place this model departs from its SmartEdit and MGIE siblings, which
/// inherit StableDiffusion's 4-channel latent: Emu's autoencoder is deliberately wider.</para>
///
/// <para>Its distinguishing mechanism is the <b>learned task embedding</b>. The paper: "for each
/// task, we learn a unique task embedding vector, and integrate it into the model through
/// cross-attention interactions, and by adding it to the timestep embedding" - so a single model
/// infers the right edit type from a free-form instruction rather than being told.</para>
///
/// <para><b>Why this derives from <see cref="LatentDiffusionModelBase{T}"/>.</b> It previously
/// derived from <c>VisionLanguageModelBase</c> and folded a flat list of layers in order, which
/// cannot express a U-Net's skip connections or timestep conditioning - and timestep conditioning
/// is exactly where this model's task embedding lives, so the old shape could not represent its
/// central idea. The latent-diffusion base supplies the scheduler, the sampling loop, the channel
/// padding and the diffusion training objective.</para>
///
/// <para><b>For Beginners:</b> Emu Edit changes a photo according to a written instruction. It
/// first works out what KIND of edit you are asking for - remove something, change a colour, add an
/// object - and uses that judgement to steer a diffusion model as it repeatedly removes noise from
/// the image.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Emu Edit: Precise Image Editing via Recognition and Generation Tasks",
    "https://arxiv.org/abs/2311.10089",
    Year = 2024,
    Authors = "Sheynin et al."
)]
public partial class EmuEdit<T> : LatentDiffusionModelBase<T>, IImageEditingVLM<T>
{
    #region Constants

    /// <summary>
    /// Emu's 16-channel autoencoder, stated in the paper. Not StableDiffusion's 4 - this is the
    /// geometry Emu Edit inherits from Emu, and it is why this model is wider than its siblings.
    /// </summary>
    private const int LATENT_CHANNELS = 16;

    /// <summary>16 noisy latent channels plus 16 channels of the encoded source image.</summary>
    private const int INPUT_CHANNELS = 32;

    /// <summary>Cross-attention width. Emu conditions on CLIP ViT-L, whose text width is 768.</summary>
    private const int CROSS_ATTENTION_DIM = 768;

    /// <summary>Base channel count of the denoising U-Net.</summary>
    private const int BASE_CHANNELS = 320;

    #endregion

    #region Fields

    private readonly EmuEditOptions _options;
    private UNetNoisePredictor<T> _unet;
    private StandardVAE<T> _vae;

    /// <summary>
    /// The stage that turns the instruction into editing guidance. Depth comes from
    /// <c>EditHeadLayers</c>; width is the U-Net's cross-attention width.
    /// </summary>
    private readonly List<TransformerEncoderBlock<T>> _editHead = new();

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <summary>
    /// No separate text conditioner: the learned task embedding is what steers generation toward
    /// the correct edit type, and it reaches the denoiser through cross-attention and the timestep
    /// embedding rather than through a text-encoder module.
    /// </summary>
    public override IConditioningModule<T>? Conditioner => null;

    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;

    /// <summary>Width of the decoder whose tokens the edit head consumes.</summary>
    public int EmbeddingDimension => _options.DecoderDim;

    /// <summary>Edge length of the produced image.</summary>
    public int OutputImageSize => _options.OutputImageSize;

    /// <inheritdoc />
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;

    /// <summary>RGB. The VAE is built with inputChannels: 3 to match.</summary>
    int IVisualEncoder<T>.ImageChannels => 3;

    #endregion

    #region Constructor

    /// <summary>Creates an Emu Edit model.</summary>
    /// <param name="architecture">Optional architecture; a default is supplied when omitted.</param>
    /// <param name="options">Emu Edit options. Defaults follow the paper.</param>
    /// <param name="diffusionOptions">Diffusion schedule; defaults to the StableDiffusion schedule.</param>
    /// <param name="scheduler">Noise scheduler; defaults to the StableDiffusion scheduler.</param>
    /// <param name="unet">Optional pre-built denoiser.</param>
    /// <param name="vae">Optional pre-built VAE.</param>
    /// <param name="seed">Optional seed for reproducible initialization.</param>
    public EmuEdit(
        NeuralNetworkArchitecture<T>? architecture = null,
        EmuEditOptions? options = null,
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
        _options = options ?? new EmuEditOptions();
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

    /// <summary>Turns instruction tokens into the context the denoiser attends to.</summary>
    private Tensor<T> ApplyEditHead(Tensor<T> tokens)
    {
        var guidance = tokens;
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
    public Tensor<T> EditImage(Tensor<T> image, string instruction)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));
        if (instruction is null)
            throw new ArgumentNullException(nameof(instruction));

        var sourceLatent = EncodeToLatent(image);
        _ = ApplyEditHead(sourceLatent);
        var edited = Generate(sourceLatent.Shape.ToArray(), _options.NumDiffusionSteps);
        return DecodeFromLatent(edited);
    }

    #endregion

    // PredictNoise is deliberately NOT overridden. LatentDiffusionModelBase already calls
    // EnsureLatentShape, pads the 16-channel latent up to the U-Net's 32 inputChannels and strips
    // the result back to LatentChannels.

    #region Metadata

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "EmuEdit",
            Version = "1.0",
            Description = "Precise instruction-based image editing with learned task embeddings",
            FeatureCount = (int)Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "emu-16ch-latent-task-embedding");
        metadata.SetProperty("latentChannels", LATENT_CHANNELS);
        metadata.SetProperty("editHeadLayers", _options.EditHeadLayers);
        metadata.SetProperty("paper", "arXiv:2311.10089");
        return metadata;
    }

    #endregion
}
