using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// Configuration of <see cref="Wonder3DModel{T}"/>: the cross-domain multi-view diffusion, its sampler, and the
/// geometry-aware normal fusion that reconstructs the mesh, with the values of the released Wonder3D.
/// </summary>
/// <remarks>
/// <para>
/// The diffusion defaults are the released checkpoint's (<c>flamehaze1115/wonder3d-v1.0</c>,
/// <c>configs/mvdiffusion-joint-ortho-6views.yaml</c> of xxlong0/Wonder3D): an SD image-variations U-Net
/// (320 channels, multipliers 1-2-4-4, two res blocks per level, attention at the first three levels and in the
/// middle, 8 heads, 768-wide CLIP image context) widened to 8 input channels for the conditioning latent, sample size
/// 32 (256x256 images), a 10-wide domain-and-camera class embedding, multi-view self-attention over the 6 views and
/// cross-domain attention between the normal and color branches in every transformer block. Sampling is DDIM with
/// 50 steps, eta 1 and classifier-free guidance 3 against a zero image embedding and zero conditioning latent.
/// </para>
/// <para>
/// The reconstruction defaults are <c>instant-nsr-pl/configs/neuralangelo-ortho-wmask.yaml</c>: 3000 iterations of
/// NeuS on the 6 orthographic views with the view weights 1, 0.8, 0.2, 1, 0.4, 0.7, and the loss weights rgb-mse 0.5,
/// mask 1, eikonal 0.2, normal 1 (geometry-aware, keeping the best 80% of rays), 3D normal smoothness 1 and
/// sparsity 0.5, then marching cubes at 192.
/// </para>
/// <para>
/// <b>For Beginners:</b> The defaults build the full-size network (about a billion weights, allocated lazily). For
/// experiments or tests, shrink the widths, the image size and the reconstruction budget; the architecture stays the
/// same.
/// </para>
/// </remarks>
public class Wonder3DOptions : ModelOptions
{
    private const string UNetConfig = "Wonder3D unet/config.json (flamehaze1115/wonder3d-v1.0)";
    private const string RunConfig = "Wonder3D configs/mvdiffusion-joint-ortho-6views.yaml";
    private const string NeuSConfig = "Wonder3D instant-nsr-pl/configs/neuralangelo-ortho-wmask.yaml";
    private const string SchedulerConfigSource = "Wonder3D scheduler/scheduler_config.json (flamehaze1115/wonder3d-v1.0)";

    #region Views

    /// <summary>The number of generated views (front, front-right, right, back, left, front-left).</summary>
    [PaperDefault(6, Source = UNetConfig + ": num_views 6")]
    public int NumViews { get; set; } = 6;

    /// <summary>
    /// Each view's azimuth in degrees around the object, measured from the front toward the right (the
    /// reconstruction cameras; the released fixed poses are 0, 45, 90, 180, 270 and 315).
    /// </summary>
    [PaperDefault(new[] { 0.0, 45.0, 90.0, 180.0, 270.0, 315.0 }, Source = NeuSConfig + ": datasets/fixed_poses/000_*_RT.txt")]
    public double[] ViewAzimuthsDegrees { get; set; } = { 0.0, 45.0, 90.0, 180.0, 270.0, 315.0 };

    /// <summary>
    /// The camera part of each view's class embedding, three values per view (conditioning elevation, elevation,
    /// azimuth, in radians), exactly the table the released pipeline feeds the U-Net (<c>MVDiffusionImagePipeline.camera_embedding</c>).
    /// </summary>
    [PaperDefault(new[]
    {
        0.0, 0.0, 0.0,
        0.0, -0.2362, 0.8125,
        0.0, -0.1686, 1.6934,
        0.0, 0.5220, 3.1406,
        0.0, 0.6904, 4.8359,
        0.0, 0.3733, 5.5859,
    }, Source = "Wonder3D mvdiffusion/pipelines/pipeline_mvdiffusion_image.py: camera_embedding")]
    public double[] ViewCameraEmbedding { get; set; } =
    {
        0.0, 0.0, 0.0,
        0.0, -0.2362, 0.8125,
        0.0, -0.1686, 1.6934,
        0.0, 0.5220, 3.1406,
        0.0, 0.6904, 4.8359,
        0.0, 0.3733, 5.5859,
    };

    /// <summary>The side of the square input and output images.</summary>
    [PaperDefault(256, Source = RunConfig + ": validation_dataset img_wh [256, 256]")]
    public int ImageSize { get; set; } = 256;

    #endregion

    #region Diffusion network

    /// <summary>The VAE's latent channels (the U-Net's input is twice this: the noisy latent and the image latent).</summary>
    [PaperDefault(4, Source = UNetConfig + ": out_channels 4, in_channels 8")]
    public int LatentChannels { get; set; } = 4;

    /// <summary>The U-Net's base width.</summary>
    [PaperDefault(320, Source = UNetConfig + ": block_out_channels [320, 640, 1280, 1280]")]
    public int BaseChannels { get; set; } = 320;

    /// <summary>The U-Net's width multiplier per level.</summary>
    [PaperDefault(new[] { 1, 2, 4, 4 }, Source = UNetConfig + ": block_out_channels [320, 640, 1280, 1280]")]
    public int[] ChannelMultipliers { get; set; } = { 1, 2, 4, 4 };

    /// <summary>Residual blocks per U-Net level.</summary>
    [PaperDefault(2, Source = UNetConfig + ": layers_per_block 2")]
    public int ResBlocksPerLevel { get; set; } = 2;

    /// <summary>The U-Net levels with transformer (attention) blocks; the middle block always has one.</summary>
    [PaperDefault(new[] { 0, 1, 2 }, Source = UNetConfig + ": down_block_types CrossAttnDownBlockMV2D x3, DownBlock2D")]
    public int[] AttentionLevels { get; set; } = { 0, 1, 2 };

    /// <summary>Attention heads.</summary>
    [PaperDefault(8, Source = UNetConfig + ": attention_head_dim 8")]
    public int AttentionHeads { get; set; } = 8;

    /// <summary>The width of the CLIP image embedding the U-Net cross-attends to.</summary>
    [PaperDefault(768, Source = UNetConfig + ": cross_attention_dim 768")]
    public int CrossAttentionDim { get; set; } = 768;

    /// <summary>The VAE's base width (Stable Diffusion's AutoencoderKL).</summary>
    public int VaeBaseChannels { get; set; } = 128;

    /// <summary>The VAE's width multiplier per level; the latent is the image downscaled 2^(levels − 1) times.</summary>
    public int[] VaeChannelMultipliers { get; set; } = { 1, 2, 4, 4 };

    /// <summary>The VAE's residual blocks per level.</summary>
    public int VaeResBlocksPerLevel { get; set; } = 2;

    /// <summary>The VAE latent scaling factor.</summary>
    public double VaeScaleFactor { get; set; } = 0.18215;

    #endregion

    #region Sampling

    /// <summary>DDIM steps.</summary>
    [PaperDefault(50, Source = "Wonder3D pipeline_mvdiffusion_image.py: num_inference_steps 50")]
    public int InferenceSteps { get; set; } = 50;

    /// <summary>The classifier-free guidance scale against a zero image embedding and zero conditioning latent.</summary>
    [PaperDefault(3.0, Source = RunConfig + ": validation_guidance_scales [1.0, 3.0]")]
    public double GuidanceScale { get; set; } = 3.0;

    /// <summary>The DDIM eta (1 is DDPM-like stochastic sampling).</summary>
    [PaperDefault(1.0, Source = RunConfig + ": pipe_validation_kwargs eta 1.0")]
    public double Eta { get; set; } = 1.0;

    /// <summary>How DDIM spreads its timesteps over the 1000 training steps (50 leading steps: 981, 961, …, 1).</summary>
    [PaperDefault(TimestepSpacing.Leading, Source = SchedulerConfigSource + ": timestep_spacing leading")]
    public TimestepSpacing SchedulerTimestepSpacing { get; set; } = TimestepSpacing.Leading;

    /// <summary>The offset added to every leading timestep.</summary>
    [PaperDefault(1, Source = SchedulerConfigSource + ": steps_offset 1")]
    public int SchedulerStepsOffset { get; set; } = 1;

    /// <summary>Whether the final DDIM step uses alpha_cumprod = 1 instead of the first training step's.</summary>
    [PaperDefault(false, Source = SchedulerConfigSource + ": set_alpha_to_one false")]
    public bool SchedulerSetAlphaToOne { get; set; } = false;

    #endregion

    #region Reconstruction

    /// <summary>NeuS iterations of the geometry-aware normal fusion.</summary>
    [PaperDefault(3000, Source = NeuSConfig + ": trainer max_steps 3000")]
    public int ReconstructionIterations { get; set; } = 3000;

    /// <summary>The starting rays per NeuS iteration (the dynamic batch's initial size).</summary>
    [PaperDefault(256, Source = NeuSConfig + ": model train_num_rays 256")]
    public int ReconstructionBatchSize { get; set; } = 256;

    /// <summary>
    /// Whether the ray batch adapts to keep <c>ReconstructionBatchSize · SamplesPerRayBudget</c> samples per iteration.
    /// </summary>
    [PaperDefault(true, Source = NeuSConfig + ": model dynamic_ray_sampling true")]
    public bool DynamicRaySampling { get; set; } = true;

    /// <summary>The largest dynamic ray batch.</summary>
    [PaperDefault(8192, Source = NeuSConfig + ": model max_train_num_rays 8192")]
    public int ReconstructionMaxBatchSize { get; set; } = 8192;

    /// <summary>The samples per ray the dynamic batch budgets for.</summary>
    [PaperDefault(1024, Source = NeuSConfig + ": model num_samples_per_ray 1024")]
    public int SamplesPerRayBudget { get; set; } = 1024;

    /// <summary>The per-view weights of the mask and normal terms.</summary>
    [PaperDefault(new[] { 1.0, 0.8, 0.2, 1.0, 0.4, 0.7 }, Source = NeuSConfig + ": view_weights")]
    public double[] ViewWeights { get; set; } = { 1.0, 0.8, 0.2, 1.0, 0.4, 0.7 };

    /// <summary>Half the width of the orthographic image plane in world units (rays from −1 to 1).</summary>
    public double OrthographicHalfExtent { get; set; } = 1.0;

    /// <summary>The camera distance from the origin (it only places the orthographic image planes).</summary>
    public double CameraDistance { get; set; } = 1.3;

    /// <summary>A pixel is background where all three channels of its generated normal map exceed this (white).</summary>
    public double BackgroundThreshold { get; set; } = 250.0 / 255.0;

    /// <summary>The geometry's (hash grid and SDF network) AdamW learning rate.</summary>
    [PaperDefault(1e-3, Source = NeuSConfig + ": optimizer params geometry lr 0.001")]
    public double ReconstructionLearningRate { get; set; } = 1e-3;

    /// <summary>The color network's AdamW learning rate.</summary>
    [PaperDefault(1e-2, Source = NeuSConfig + ": optimizer params texture lr 0.01")]
    public double ColorLearningRate { get; set; } = 1e-2;

    /// <summary>The S-density variance's AdamW learning rate.</summary>
    [PaperDefault(1e-3, Source = NeuSConfig + ": optimizer params variance lr 0.001")]
    public double VarianceLearningRate { get; set; } = 1e-3;

    /// <summary>AdamW's beta2.</summary>
    [PaperDefault(0.99, Source = NeuSConfig + ": optimizer args betas [0.9, 0.99]")]
    public double AdamBeta2 { get; set; } = 0.99;

    /// <summary>AdamW's epsilon.</summary>
    [PaperDefault(1e-15, Source = NeuSConfig + ": optimizer args eps 1e-15")]
    public double AdamEpsilon { get; set; } = 1e-15;

    /// <summary>AdamW's decoupled weight decay (the config sets none, so PyTorch's default applies).</summary>
    [PaperDefault(0.01, Source = "torch.optim.AdamW default weight_decay 0.01")]
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>The iterations at the full learning rate before the exponential decay.</summary>
    [PaperDefault(500, Source = NeuSConfig + ": system constant_steps 500")]
    public int ConstantLearningRateIterations { get; set; } = 500;

    /// <summary>The learning-rate fraction the exponential decay reaches at the last iteration.</summary>
    [PaperDefault(0.1, Source = NeuSConfig + ": ExponentialLR gamma calc_exp_lr_decay_rate(0.1, max_steps - constant_steps)")]
    public double ReconstructionLearningRateAlpha { get; set; } = 0.1;

    /// <summary>The initial S-density variance parameter (sharpness exp(10 · v)).</summary>
    [PaperDefault(0.3, Source = NeuSConfig + ": model variance init_val 0.3")]
    public double InitialVariance { get; set; } = 0.3;

    /// <summary>The SDF slope annealing length in iterations.</summary>
    [PaperDefault(20000, Source = NeuSConfig + ": cos_anneal_end 20000")]
    public int ReconstructionAnnealIterations { get; set; } = 20000;

    /// <summary>The squared color term's weight.</summary>
    [PaperDefault(0.5, Source = NeuSConfig + ": lambda_rgb_mse 0.5")]
    public double ColorMseWeight { get; set; } = 0.5;

    /// <summary>The mask cross-entropy weight.</summary>
    [PaperDefault(1.0, Source = NeuSConfig + ": lambda_mask 1.0")]
    public double MaskWeight { get; set; } = 1.0;

    /// <summary>The Eikonal weight.</summary>
    [PaperDefault(0.2, Source = NeuSConfig + ": lambda_eikonal 0.2")]
    public double EikonalWeight { get; set; } = 0.2;

    /// <summary>The normal term's weight.</summary>
    [PaperDefault(1.0, Source = NeuSConfig + ": lambda_normal 1.0")]
    public double NormalWeight { get; set; } = 1.0;

    /// <summary>The 3D normal-smoothness weight.</summary>
    [PaperDefault(1.0, Source = NeuSConfig + ": lambda_3d_normal_smooth 1.0")]
    public double NormalSmoothWeight { get; set; } = 1.0;

    /// <summary>The sparsity weight.</summary>
    [PaperDefault(0.5, Source = NeuSConfig + ": lambda_sparsity 0.5")]
    public double SparsityWeight { get; set; } = 0.5;

    /// <summary>The sparsity sharpness.</summary>
    [PaperDefault(100.0, Source = NeuSConfig + ": sparsity_scale 100.0")]
    public double SparsityScale { get; set; } = 100.0;

    /// <summary>Whether the normal term is geometry-aware.</summary>
    [PaperDefault(true, Source = NeuSConfig + ": geo_aware true")]
    public bool GeometryAwareNormals { get; set; } = true;

    /// <summary>The fraction of color rays the color term keeps.</summary>
    [PaperDefault(0.8, Source = NeuSConfig + ": rgb_p_ratio 0.8")]
    public double ColorKeepRatio { get; set; } = 0.8;

    /// <summary>The fraction of normal rays the normal term keeps.</summary>
    [PaperDefault(0.8, Source = NeuSConfig + ": normal_p_ratio 0.8")]
    public double NormalKeepRatio { get; set; } = 0.8;

    /// <summary>The fraction of rays the mask term keeps.</summary>
    [PaperDefault(0.9, Source = NeuSConfig + ": mask_p_ratio 0.9")]
    public double MaskKeepRatio { get; set; } = 0.9;

    /// <summary>The largest ray-to-normal cosine for a ray to supervise normals.</summary>
    public double NormalVisibleCosine { get; set; } = -0.1;

    /// <summary>The hash grid's level count.</summary>
    [PaperDefault(10, Source = NeuSConfig + ": xyz_encoding_config n_levels 10")]
    public int HashGridLevels { get; set; } = 10;

    /// <summary>The hash grid's features per level.</summary>
    [PaperDefault(2, Source = NeuSConfig + ": xyz_encoding_config n_features_per_level 2")]
    public int HashGridFeaturesPerLevel { get; set; } = 2;

    /// <summary>The base-2 logarithm of the hash grid's largest table.</summary>
    [PaperDefault(19, Source = NeuSConfig + ": xyz_encoding_config log2_hashmap_size 19")]
    public int HashGridLog2TableSize { get; set; } = 19;

    /// <summary>The hash grid's coarsest resolution.</summary>
    [PaperDefault(32, Source = NeuSConfig + ": xyz_encoding_config base_resolution 32")]
    public int HashGridBaseResolution { get; set; } = 32;

    /// <summary>The hash grid's per-level scale.</summary>
    [PaperDefault(1.3195079107728942, Source = NeuSConfig + ": xyz_encoding_config per_level_scale 1.3195079107728942")]
    public double HashGridPerLevelScale { get; set; } = 1.3195079107728942;

    /// <summary>The hash grid levels active at the start (ProgressiveBandHashGrid).</summary>
    [PaperDefault(4, Source = NeuSConfig + ": xyz_encoding_config start_level 4")]
    public int HashGridStartLevel { get; set; } = 4;

    /// <summary>The iterations between switching on successive hash grid levels.</summary>
    [PaperDefault(1000, Source = NeuSConfig + ": xyz_encoding_config update_steps 1000")]
    public int HashGridUpdateIterations { get; set; } = 1000;

    /// <summary>The SDF network's hidden width.</summary>
    [PaperDefault(64, Source = NeuSConfig + ": geometry mlp_network_config n_neurons 64")]
    public int SdfHiddenDim { get; set; } = 64;

    /// <summary>The SDF network's hidden layers.</summary>
    [PaperDefault(1, Source = NeuSConfig + ": geometry mlp_network_config n_hidden_layers 1")]
    public int SdfHiddenLayers { get; set; } = 1;

    /// <summary>The SDF network's geometry feature width (its output is the SDF and these features).</summary>
    [PaperDefault(12, Source = NeuSConfig + ": geometry feature_dim 13 (the SDF and 12 features)")]
    public int SdfFeatureDim { get; set; } = 12;

    /// <summary>The color network's hidden width.</summary>
    [PaperDefault(64, Source = NeuSConfig + ": texture mlp_network_config n_neurons 64")]
    public int ColorHiddenDim { get; set; } = 64;

    /// <summary>The color network's hidden layers.</summary>
    [PaperDefault(2, Source = NeuSConfig + ": texture mlp_network_config n_hidden_layers 2")]
    public int ColorHiddenLayers { get; set; } = 2;

    /// <summary>The view-direction embedding's frequency count.</summary>
    public int ViewFrequencies { get; set; } = 4;

    /// <summary>Uniform samples per ray.</summary>
    public int RaySamples { get; set; } = 64;

    /// <summary>Importance samples per ray.</summary>
    public int ImportanceSamples { get; set; } = 64;

    /// <summary>Hierarchical up-sampling rounds.</summary>
    public int UpSampleSteps { get; set; } = 4;

    /// <summary>The marching-cubes grid resolution.</summary>
    [PaperDefault(192, Source = NeuSConfig + ": geometry isosurface resolution 192")]
    public int MeshResolution { get; set; } = 192;

    #endregion
}
