using AiDotNet.Attributes;
using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// Configuration of <see cref="SyncDreamerModel{T}"/>: the fixed multiview camera layout, the noise predictor, the
/// 3D-aware feature attention (spatial feature volume, view-frustum volumes and depth-wise attention), the input-view
/// CLIP and VAE conditioning, the synchronized DDIM sampler and the NeuS reconstruction, with the values of the
/// official SyncDreamer release.
/// </summary>
/// <remarks>
/// <para>
/// Every default is the value of the released SyncDreamer code (liuyuan-pal/SyncDreamer: <c>configs/syncdreamer.yaml</c>,
/// <c>ldm/models/diffusion/sync_dreamer.py</c>, <c>sync_dreamer_network.py</c>, <c>sync_dreamer_attention.py</c>,
/// <c>generate.py</c> and <c>configs/neus.yaml</c>) that Liu et al. 2023 (arXiv 2309.03453, Sec. 3 and 4) describe:
/// 16 target views at a fixed 30 degree elevation spaced 22.5 degrees in azimuth, 1.5 units from the origin, seen
/// through a 256 x 256 pinhole camera with a 280 pixel focal length; a 32^3 spatial feature volume over
/// [-0.5, 0.5]^3 built from all views' noisy latents; 48-deep view-frustum volumes; depth-wise attention in the
/// Zero123-initialized U-Net; CLIP ViT-L/14 image conditioning; and DDIM with 50 steps, eta 1 and guidance 2.
/// </para>
/// <para>
/// <b>For Beginners:</b> The defaults build the full-size network (CLIP ViT-L/14, the Stable Diffusion 1.5 sized
/// U-Net and the 3D volume networks; weights are allocated lazily). Shrink the widths for experiments and tests; the
/// architecture stays the same.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var tiny = new SyncDreamerOptions
/// {
///     ViewCount = 4, ImageSize = 32, FocalLength = 35, UNetBaseChannels = 32, UNetChannelMultipliers = new[] { 1, 2 },
///     UNetNumResBlocks = 1, UNetAttentionLevels = new[] { 1 }, UNetHeads = 2, CrossAttentionDim = 16,
///     SpatialVolumeSize = 4, SpatialVolumeDims = new[] { 8, 8, 8, 8 }, FrustumVolumeDepth = 8,
///     FrustumVolumeDims = new[] { 8, 8, 8, 8 }, TimeEmbedDim = 8, ClipImageSize = 16, ClipPatchSize = 8,
///     ClipWidth = 16, ClipLayers = 1, ClipHeads = 2, Seed = 42
/// };
/// var model = new SyncDreamerModel&lt;float&gt;(tiny);
/// </code>
/// </example>
public class SyncDreamerOptions : ModelOptions
{
    private const string Repo = "liuyuan-pal/SyncDreamer";

    #region Camera layout

    /// <summary>The number of target views denoised together (<c>view_num</c>).</summary>
    [PaperDefault(16, Source = Repo + " configs/syncdreamer.yaml: view_num 16; paper Sec. 4.1")]
    public int ViewCount { get; set; } = 16;

    /// <summary>The elevation of every target view, in degrees (<c>elevation_target = 30</c>).</summary>
    [PaperDefault(30.0, Source = Repo + " sync_dreamer.py get_viewpoint_embedding: elevation_target = -deg2rad(30)")]
    public double TargetElevationDegrees { get; set; } = 30.0;

    /// <summary>
    /// The elevation of the input view, in degrees. The user supplies it in the official code (<c>--elevation</c>);
    /// its default there is 30.
    /// </summary>
    [PaperDefault(30.0, Source = Repo + " generate.py: --elevation default 30")]
    public double InputElevationDegrees { get; set; } = 30.0;

    /// <summary>The distance of every camera from the origin (<c>default_origin_depth</c>, the render distance).</summary>
    [PaperDefault(1.5, Source = Repo + " sync_dreamer_network.py SpatialVolumeNet: default_origin_depth 1.5")]
    public double CameraDistance { get; set; } = 1.5;

    /// <summary>The focal length in pixels at <see cref="ImageSize"/> (<c>meta_info/camera-16.pkl</c>, K = diag(280, 280, 1) at 256).</summary>
    [PaperDefault(280.0, Source = Repo + " meta_info/camera-16.pkl: fx = fy = 280 at 256 x 256")]
    public double FocalLength { get; set; } = 280.0;

    /// <summary>The side of the square input and output images (<c>image_size</c>).</summary>
    [PaperDefault(256, Source = Repo + " configs/syncdreamer.yaml: image_size 256")]
    public int ImageSize { get; set; } = 256;

    /// <summary>The spatial downsampling of the VAE: latents are <see cref="ImageSize"/> / this on a side.</summary>
    [PaperDefault(8, Source = "Stable Diffusion KL-f8 autoencoder (Rombach et al. 2022)")]
    public int VaeDownsampleFactor { get; set; } = 8;

    /// <summary>The VAE latent scale factor (<c>first_stage_scale_factor</c>).</summary>
    [PaperDefault(0.18215, Source = Repo + " sync_dreamer.py: first_stage_scale_factor 0.18215")]
    public double LatentScaleFactor { get; set; } = 0.18215;

    #endregion

    #region Noise predictor (Zero123-initialized U-Net)

    /// <summary>The U-Net base width (<c>model_channels</c>).</summary>
    [PaperDefault(320, Source = Repo + " configs/syncdreamer.yaml unet_config: model_channels 320")]
    public int UNetBaseChannels { get; set; } = 320;

    /// <summary>The U-Net width multipliers per level (<c>channel_mult</c>).</summary>
    [PaperDefault(new[] { 1, 2, 4, 4 }, Source = Repo + " configs/syncdreamer.yaml unet_config: channel_mult [1, 2, 4, 4]")]
    public int[] UNetChannelMultipliers { get; set; } = new[] { 1, 2, 4, 4 };

    /// <summary>The residual blocks per U-Net level (<c>num_res_blocks</c>).</summary>
    [PaperDefault(2, Source = Repo + " configs/syncdreamer.yaml unet_config: num_res_blocks 2")]
    public int UNetNumResBlocks { get; set; } = 2;

    /// <summary>
    /// The U-Net levels with self- and cross-attention. The reference's <c>attention_resolutions [4, 2, 1]</c> are
    /// downsampling rates 1, 2 and 4, which are levels 0, 1 and 2.
    /// </summary>
    [PaperDefault(new[] { 0, 1, 2 }, Source = Repo + " configs/syncdreamer.yaml unet_config: attention_resolutions [4, 2, 1]")]
    public int[] UNetAttentionLevels { get; set; } = new[] { 0, 1, 2 };

    /// <summary>The U-Net attention heads (<c>num_heads</c>).</summary>
    [PaperDefault(8, Source = Repo + " configs/syncdreamer.yaml unet_config: num_heads 8")]
    public int UNetHeads { get; set; } = 8;

    /// <summary>The cross-attention width: the CLIP image embedding the U-Net attends to (<c>context_dim</c>).</summary>
    [PaperDefault(768, Source = Repo + " configs/syncdreamer.yaml unet_config: context_dim 768")]
    public int CrossAttentionDim { get; set; } = 768;

    #endregion

    #region 3D-aware feature attention

    /// <summary>The width of the timestep embedding the volume networks read (<c>time_embed_dim</c>).</summary>
    [PaperDefault(256, Source = Repo + " sync_dreamer.py _init_time_step_embedding: time_embed_dim 256")]
    public int TimeEmbedDim { get; set; } = 256;

    /// <summary>The working width of the noisy target-view encoder (<c>NoisyTargetViewEncoder.run_dim</c>).</summary>
    [PaperDefault(16, Source = Repo + " sync_dreamer_network.py NoisyTargetViewEncoder: run_dim 16")]
    public int TargetEncoderWidth { get; set; } = 16;

    /// <summary>The per-view feature width the encoder unprojects into the volume (<c>output_dim</c>).</summary>
    [PaperDefault(16, Source = Repo + " sync_dreamer_network.py SpatialVolumeNet: NoisyTargetViewEncoder(output_dim=16)")]
    public int TargetEncoderOutputDim { get; set; } = 16;

    /// <summary>The group count of every group normalization in the volume networks and depth transformers.</summary>
    [PaperDefault(8, Source = Repo + " sync_dreamer_network.py / sync_dreamer_attention.py: nn.GroupNorm(8, c)")]
    public int NormGroups { get; set; } = 8;

    /// <summary>The spatial volume's resolution per axis (<c>spatial_volume_size</c>).</summary>
    [PaperDefault(32, Source = Repo + " sync_dreamer_network.py SpatialVolumeNet: spatial_volume_size 32")]
    public int SpatialVolumeSize { get; set; } = 32;

    /// <summary>The spatial volume's half extent: it spans [-this, this]^3 (<c>spatial_volume_length</c>).</summary>
    [PaperDefault(0.5, Source = Repo + " sync_dreamer_network.py SpatialVolumeNet: spatial_volume_length 0.5")]
    public double SpatialVolumeLength { get; set; } = 0.5;

    /// <summary>The widths of the spatial volume's 3D U-Net (<c>SpatialTime3DNet dims</c>).</summary>
    [PaperDefault(new[] { 64, 128, 256, 512 }, Source = Repo + " sync_dreamer_network.py SpatialVolumeNet: dims (64, 128, 256, 512)")]
    public int[] SpatialVolumeDims { get; set; } = new[] { 64, 128, 256, 512 };

    /// <summary>The number of depth samples of each view-frustum volume (<c>frustum_volume_depth</c>).</summary>
    [PaperDefault(48, Source = Repo + " sync_dreamer_network.py SpatialVolumeNet: frustum_volume_depth 48")]
    public int FrustumVolumeDepth { get; set; } = 48;

    /// <summary>
    /// The half depth range of each frustum: depths run from <see cref="CameraDistance"/> minus this to plus this
    /// (<c>frustum_volume_length</c>, sqrt(3)/2).
    /// </summary>
    [PaperDefault(0.86603, Source = Repo + " sync_dreamer_network.py SpatialVolumeNet: frustum_volume_length 0.86603")]
    public double FrustumVolumeLength { get; set; } = 0.86603;

    /// <summary>The widths of the frustum volume's 3D U-Net (<c>FrustumTV3DNet dims</c>), one per depth-attention scale.</summary>
    [PaperDefault(new[] { 64, 128, 256, 512 }, Source = Repo + " sync_dreamer_network.py SpatialVolumeNet: dims (64, 128, 256, 512)")]
    public int[] FrustumVolumeDims { get; set; } = new[] { 64, 128, 256, 512 };

    /// <summary>The heads of every depth-wise attention (<c>DepthTransformer(ch, 4, d // 2)</c>).</summary>
    [PaperDefault(4, Source = Repo + " sync_dreamer_attention.py DepthWiseAttention: DepthTransformer(ch, 4, d // 2)")]
    public int DepthAttentionHeads { get; set; } = 4;

    #endregion

    #region Input-view CLIP conditioning

    /// <summary>The CLIP input side (ViT-L/14 at 224).</summary>
    [PaperDefault(224, Source = Repo + " sync_dreamer.py: clip ViT-L-14.pt")]
    public int ClipImageSize { get; set; } = 224;

    /// <summary>The CLIP patch side (ViT-L/14).</summary>
    [PaperDefault(14, Source = Repo + " sync_dreamer.py: clip ViT-L-14.pt")]
    public int ClipPatchSize { get; set; } = 14;

    /// <summary>The CLIP transformer width (ViT-L: 1024).</summary>
    [PaperDefault(1024, Source = "OpenAI CLIP ViT-L/14: vision width 1024")]
    public int ClipWidth { get; set; } = 1024;

    /// <summary>The CLIP transformer depth (ViT-L: 24).</summary>
    [PaperDefault(24, Source = "OpenAI CLIP ViT-L/14: vision layers 24")]
    public int ClipLayers { get; set; } = 24;

    /// <summary>The CLIP attention heads (ViT-L: 16).</summary>
    [PaperDefault(16, Source = "OpenAI CLIP ViT-L/14: vision heads 16")]
    public int ClipHeads { get; set; } = 16;

    #endregion

    #region Sampler

    /// <summary>The DDIM steps (<c>--sample_steps</c>).</summary>
    [PaperDefault(50, Source = Repo + " generate.py: --sample_steps default 50")]
    public int SamplingSteps { get; set; } = 50;

    /// <summary>The classifier-free guidance scale (<c>--cfg_scale</c>).</summary>
    [PaperDefault(2.0, Source = Repo + " generate.py: --cfg_scale default 2.0")]
    public double GuidanceScale { get; set; } = 2.0;

    /// <summary>The DDIM stochasticity eta (<c>SyncDDIMSampler(ddim_eta=1.0)</c>).</summary>
    [PaperDefault(1.0, Source = Repo + " sync_dreamer.py SyncDDIMSampler: ddim_eta 1.0")]
    public double DdimEta { get; set; } = 1.0;

    /// <summary>The training timesteps of the DDPM schedule.</summary>
    [PaperDefault(1000, Source = Repo + " configs/syncdreamer.yaml scheduler: 1000 timesteps")]
    public int TrainTimesteps { get; set; } = 1000;

    /// <summary>The first beta of the scaled-linear schedule (<c>linear_start</c>).</summary>
    [PaperDefault(0.00085, Source = Repo + " sync_dreamer.py _init_schedule: linear_start 0.00085")]
    public double BetaStart { get; set; } = 0.00085;

    /// <summary>The last beta of the scaled-linear schedule (<c>linear_end</c>).</summary>
    [PaperDefault(0.012, Source = Repo + " sync_dreamer.py _init_schedule: linear_end 0.0120")]
    public double BetaEnd { get; set; } = 0.012;

    #endregion

    #region NeuS reconstruction

    /// <summary>The NeuS optimization steps over the generated views (<c>total_steps</c>).</summary>
    [PaperDefault(2000, Source = Repo + " configs/neus.yaml: total_steps 2000")]
    public int ReconstructionIterations { get; set; } = 2000;

    /// <summary>The NeuS learning-rate warm-up steps (<c>warm_up_steps</c>).</summary>
    [PaperDefault(100, Source = Repo + " configs/neus.yaml: warm_up_steps 100")]
    public int ReconstructionWarmUpIterations { get; set; } = 100;

    /// <summary>The marching-cubes grid resolution the NeuS field is meshed at.</summary>
    [PaperDefault(512, Source = Repo + " renderer/renderer.py extract_mesh: resolution 512")]
    public int MeshResolution { get; set; } = 512;

    /// <summary>The number of surface points a point cloud keeps.</summary>
    public int PointCount { get; set; } = 4096;

    #endregion
}
