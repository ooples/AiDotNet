using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Control;

/// <summary>
/// IP-Adapter FaceID Plus model for face-identity-preserving generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Combines IP-Adapter's image prompting with face recognition embeddings
/// (FaceID) for identity-preserving face generation. Supports both face swapping
/// and face-consistent character creation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Give this model a photo of someone's face, and it will
/// generate new images that look like the same person in different poses, styles,
/// or settings. It preserves the person's identity using face recognition technology.
/// </para>
/// <para>
/// Reference: Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter", 2023
/// </para>
/// </remarks>
public class IPAdapterFaceIDPlusModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const int FACE_EMBED_DIM = 512;

    private UNetNoisePredictor<T> _baseUNet;
    private StandardVAE<T> _vae;
    private DenseLayer<T> _faceProjection;
    private DenseLayer<T> _imageProjection;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly double _faceIdScale;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _baseUNet;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _baseUNet.ParameterCount + _faceProjection.ParameterCount + _imageProjection.ParameterCount;

    /// <summary>
    /// Initializes a new IP-Adapter FaceID Plus model.
    /// </summary>
    public IPAdapterFaceIDPlusModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? baseUNet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        double faceIdScale = 0.7,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        _faceIdScale = faceIdScale;
        InitializeLayers(baseUNet, vae, seed);
    }

    [MemberNotNull(nameof(_baseUNet), nameof(_vae), nameof(_faceProjection), nameof(_imageProjection))]
    private void InitializeLayers(UNetNoisePredictor<T>? baseUNet, StandardVAE<T>? vae, int? seed)
    {
        _baseUNet = baseUNet ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 }, contextDim: 768, seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS, baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocksPerLevel: 2, seed: seed);

        _faceProjection = new DenseLayer<T>(FACE_EMBED_DIM, 768);
        _imageProjection = new DenseLayer<T>(1024, 768);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        AddParams(allParams, _baseUNet.GetParameters());
        AddParams(allParams, _faceProjection.GetParameters());
        AddParams(allParams, _imageProjection.GetParameters());
        return new Vector<T>(allParams.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        var c1 = _baseUNet.ParameterCount;
        var a1 = new T[c1];
        for (int i = 0; i < c1; i++) a1[i] = parameters[offset + i];
        _baseUNet.SetParameters(new Vector<T>(a1));
        offset += c1;

        var c2 = _faceProjection.ParameterCount;
        var a2 = new T[c2];
        for (int i = 0; i < c2; i++) a2[i] = parameters[offset + i];
        _faceProjection.SetParameters(new Vector<T>(a2));
        offset += c2;

        var c3 = _imageProjection.ParameterCount;
        var a3 = new T[c3];
        for (int i = 0; i < c3; i++) a3[i] = parameters[offset + i];
        _imageProjection.SetParameters(new Vector<T>(a3));
    }

    private static void AddParams(List<T> allParams, Vector<T> p)
    {
        for (int i = 0; i < p.Length; i++) allParams.Add(p[i]);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new IPAdapterFaceIDPlusModel<T>(conditioner: _conditioner, faceIdScale: _faceIdScale, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "IP-Adapter-FaceID-Plus", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Face-identity-preserving image prompt adapter combining FaceID + CLIP embeddings",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "unet-faceid-ip-adapter");
        metadata.SetProperty("base_model", "Stable Diffusion 1.5");
        metadata.SetProperty("text_encoder", "CLIP ViT-L/14");
        metadata.SetProperty("face_encoder", "InsightFace (ArcFace)");
        metadata.SetProperty("image_encoder", "CLIP ViT-H/14");
        metadata.SetProperty("context_dim", 768);
        metadata.SetProperty("face_embed_dim", FACE_EMBED_DIM);
        metadata.SetProperty("face_id_scale", _faceIdScale);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("guidance_scale", 7.5);
        return metadata;
    }
}
