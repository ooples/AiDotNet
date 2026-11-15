using AiDotNet.Diffusion.ThreeD.Meshing;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// Settings of the DreamGaussian pipeline (Tang et al., "DreamGaussian: Generative Gaussian Splatting for Efficient
/// 3D Content Creation", ICLR 2024). Every default is the official code's (<c>configs/image.yaml</c>,
/// <c>configs/text.yaml</c>, <c>main.py</c>, <c>main2.py</c>, <c>gs_renderer.py</c>).
/// </summary>
/// <remarks>
/// <para>
/// Where the two configs differ the setting comes in an image and a text variant (camera radius, centre learning-rate
/// horizon and the densification schedule). The paper-sized defaults render at up to 512 x 512 for 500 + 50 steps;
/// small values make the pipeline cheap enough to test.
/// </para>
/// <para><b>For Beginners:</b> leave everything at its default for the paper's behaviour. Lower
/// <see cref="Iterations"/>, <see cref="InitialGaussianCount"/> and the resolutions for a quick preview.</para>
/// </remarks>
public class DreamGaussianOptions
{
    // ---- Stage 1: generative Gaussian splatting --------------------------------------------------------------

    /// <summary>Initial number of Gaussians (<c>num_pts</c>). Default: 5000.</summary>
    public int InitialGaussianCount { get; set; } = 5000;

    /// <summary>Radius of the ball the initial centres are drawn in (<c>Renderer.initialize(radius)</c>). Default: 0.5.</summary>
    public double InitialRadius { get; set; } = 0.5;

    /// <summary>Initial opacity of every Gaussian (<c>create_from_pcd</c>). Default: 0.1.</summary>
    public double InitialOpacity { get; set; } = 0.1;

    /// <summary>Stage-1 optimisation steps (<c>iters</c>). Default: 500.</summary>
    public int Iterations { get; set; } = 500;

    /// <summary>Novel views rendered per step (<c>batch_size</c>). Default: 1.</summary>
    public int BatchSize { get; set; } = 1;

    /// <summary>Anneal the SDS timestep linearly from T to 0 (<c>anneal_timestep</c>). Default: true.</summary>
    public bool AnnealTimestep { get; set; } = true;

    /// <summary>Ramp the reference-view losses in with the step ratio (<c>warmup_rgb_loss</c>). Default: true.</summary>
    public bool WarmupReferenceLoss { get; set; } = true;

    /// <summary>Weight of the reference-view RGB MSE (<c>10000</c> in <c>main.py</c>). Default: 10000.</summary>
    public double ReferenceRgbLossWeight { get; set; } = 10000.0;

    /// <summary>Weight of the reference-view mask MSE (<c>1000</c> in <c>main.py</c>). Default: 1000.</summary>
    public double ReferenceMaskLossWeight { get; set; } = 1000.0;

    /// <summary>Weight of the Zero-1-to-3 guidance in image-to-3D (<c>lambda_zero123</c>). Default: 1.</summary>
    public double ImageGuidanceWeight { get; set; } = 1.0;

    /// <summary>Weight of the Stable Diffusion guidance in text-to-3D (<c>lambda_sd</c>). Default: 1.</summary>
    public double TextGuidanceWeight { get; set; } = 1.0;

    /// <summary>Zero-1-to-3 classifier-free guidance scale (<c>train_step(guidance_scale=5)</c>). Default: 5.</summary>
    public double ImageGuidanceScale { get; set; } = 5.0;

    /// <summary>Stable Diffusion classifier-free guidance scale (<c>train_step(guidance_scale=100)</c>). Default: 100.</summary>
    public double TextGuidanceScale { get; set; } = 100.0;

    /// <summary>Zero-1-to-3 working resolution. Default: 256.</summary>
    public int ImageGuidanceResolution { get; set; } = 256;

    /// <summary>Stable Diffusion working resolution. Default: 512.</summary>
    public int TextGuidanceResolution { get; set; } = 512;

    /// <summary>Training camera vertical field of view in degrees (<c>fovy</c>, Zero-1-to-3's renders). Default: 49.1.</summary>
    public double FovYDegrees { get; set; } = 49.1;

    /// <summary>Lowest sampled elevation offset in degrees (<c>min_ver</c>). Default: -30.</summary>
    public double MinElevation { get; set; } = -30.0;

    /// <summary>Highest sampled elevation offset in degrees (<c>max_ver</c>). Default: 30.</summary>
    public double MaxElevation { get; set; } = 30.0;

    /// <summary>Estimated elevation of the input image in degrees (<c>elevation</c>). Default: 0.</summary>
    public double ReferenceElevation { get; set; }

    /// <summary>Resolution of the reference view (<c>ref_size</c>). Default: 256.</summary>
    public int ReferenceSize { get; set; } = 256;

    /// <summary>Probability that a novel view is rendered on black instead of white (<c>invert_bg_prob</c>). Default: 0.5.</summary>
    public double InvertBackgroundProbability { get; set; } = 0.5;

    /// <summary>
    /// Novel-view render resolutions by training progress (<c>128 if ratio &lt; 0.3 else 256 if ratio &lt; 0.6 else
    /// 512</c>). Default: { 128, 256, 512 }.
    /// </summary>
    public int[] RenderResolutions { get; set; } = { 128, 256, 512 };

    /// <summary>Step-ratio thresholds between <see cref="RenderResolutions"/>. Default: { 0.3, 0.6 }.</summary>
    public double[] RenderResolutionThresholds { get; set; } = { 0.3, 0.6 };

    /// <summary>Camera radius for image-to-3D (<c>radius</c> in image.yaml). Default: 2.</summary>
    public double ImageCameraRadius { get; set; } = 2.0;

    /// <summary>Camera radius for text-to-3D (<c>radius</c> in text.yaml). Default: 2.5.</summary>
    public double TextCameraRadius { get; set; } = 2.5;

    /// <summary>Initial centre learning rate (<c>position_lr_init</c>). Default: 0.001.</summary>
    public double PositionLearningRateInit { get; set; } = 0.001;

    /// <summary>Final centre learning rate (<c>position_lr_final</c>). Default: 0.00002.</summary>
    public double PositionLearningRateFinal { get; set; } = 0.00002;

    /// <summary>Centre learning-rate delay multiplier (<c>position_lr_delay_mult</c>). Default: 0.02.</summary>
    public double PositionLearningRateDelayMultiplier { get; set; } = 0.02;

    /// <summary>Centre learning-rate decay horizon for image-to-3D (<c>position_lr_max_steps</c>). Default: 500.</summary>
    public int ImagePositionLearningRateMaxSteps { get; set; } = 500;

    /// <summary>Centre learning-rate decay horizon for text-to-3D (<c>position_lr_max_steps</c>). Default: 300.</summary>
    public int TextPositionLearningRateMaxSteps { get; set; } = 300;

    /// <summary>Colour learning rate (<c>feature_lr</c>). Default: 0.01.</summary>
    public double FeatureLearningRate { get; set; } = 0.01;

    /// <summary>Opacity learning rate (<c>opacity_lr</c>). Default: 0.05.</summary>
    public double OpacityLearningRate { get; set; } = 0.05;

    /// <summary>Scale learning rate (<c>scaling_lr</c>). Default: 0.005.</summary>
    public double ScalingLearningRate { get; set; } = 0.005;

    /// <summary>Rotation learning rate (<c>rotation_lr</c>). Default: 0.005.</summary>
    public double RotationLearningRate { get; set; } = 0.005;

    /// <summary>Clone/split boundary as a fraction of the scene extent (<c>percent_dense</c>). Default: 0.01.</summary>
    public double PercentDense { get; set; } = 0.01;

    /// <summary>First densification step for image-to-3D (<c>density_start_iter</c>). Default: 100.</summary>
    public int ImageDensifyStartIteration { get; set; } = 100;

    /// <summary>First densification step for text-to-3D (<c>density_start_iter</c>). Default: 0.</summary>
    public int TextDensifyStartIteration { get; set; }

    /// <summary>Last densification step (<c>density_end_iter</c>). Default: 3000.</summary>
    public int DensifyEndIteration { get; set; } = 3000;

    /// <summary>Densification interval for image-to-3D (<c>densification_interval</c>). Default: 100.</summary>
    public int ImageDensificationInterval { get; set; } = 100;

    /// <summary>Densification interval for text-to-3D (<c>densification_interval</c>). Default: 50.</summary>
    public int TextDensificationInterval { get; set; } = 50;

    /// <summary>Opacity reset interval (<c>opacity_reset_interval</c>). Default: 700.</summary>
    public int OpacityResetInterval { get; set; } = 700;

    /// <summary>Densification gradient threshold (<c>densify_grad_threshold</c>). Default: 0.01.</summary>
    public double DensifyGradientThreshold { get; set; } = 0.01;

    /// <summary>Opacity below which Gaussians are pruned (<c>min_opacity</c>). Default: 0.01.</summary>
    public double PruneMinOpacity { get; set; } = 0.01;

    /// <summary>Scene extent for densification (<c>extent</c>). Default: 4.</summary>
    public double SceneExtent { get; set; } = 4.0;

    /// <summary>Screen-size prune limit in pixels (<c>max_screen_size</c>). Default: 1.</summary>
    public double MaxScreenSize { get; set; } = 1.0;

    /// <summary>The diffusion timestep range as fractions of T (<c>t_range</c>). Default: 0.02.</summary>
    public double MinTimestepRatio { get; set; } = 0.02;

    /// <summary>The diffusion timestep range as fractions of T (<c>t_range</c>). Default: 0.98.</summary>
    public double MaxTimestepRatio { get; set; } = 0.98;

    // ---- Mesh extraction ----------------------------------------------------------------------------------------

    /// <summary>The density-grid mesh extraction (<c>extract_mesh</c>); its defaults are the reference's.</summary>
    public GaussianMeshExtractionOptions MeshExtraction { get; set; } = new GaussianMeshExtractionOptions();

    // ---- Stage 2: UV texture -------------------------------------------------------------------------------------

    /// <summary>Whether to bake and refine a UV texture after extracting the mesh. Default: true.</summary>
    public bool RefineTexture { get; set; } = true;

    /// <summary>Texture resolution (<c>texture_size</c>). Default: 1024.</summary>
    public int TextureSize { get; set; } = 1024;

    /// <summary>Empty texels kept around each UV chart. Default: 2.</summary>
    public int UvPaddingTexels { get; set; } = 2;

    /// <summary>Resolution of the views baked into the texture (<c>render_resolution</c> of <c>save_model</c>). Default: 512.</summary>
    public int BakeRenderResolution { get; set; } = 512;

    /// <summary>Coarsest level of the mipmapped back-projection (<c>min_resolution</c>). Default: 256.</summary>
    public int BakeMinResolution { get; set; } = 256;

    /// <summary>Bake-view elevations (<c>[0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]</c>).</summary>
    public double[] BakeElevations { get; set; } =
    {
        0, 0, 0, 0, 0, 0, 0, 0, -45, -45, -45, -45, -45, -45, -45, -45, 45, 45, 45, 45, 45, 45, 45, 45, -89.9, 89.9,
    };

    /// <summary>Bake-view azimuths (<c>[0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]</c>).</summary>
    public double[] BakeAzimuths { get; set; } =
    {
        0, 45, -45, 90, -90, 135, -135, 180, 0, 45, -45, 90, -90, 135, -135, 180, 0, 45, -45, 90, -90, 135, -135, 180, 0, 0,
    };

    /// <summary>Surfaces seen at a cosine below this are not baked or supervised (<c>viewcos &gt; 0.5</c>). Default: 0.5.</summary>
    public double ViewCosineThreshold { get; set; } = 0.5;

    /// <summary>Texels within this many dilation steps of the baked region are inpainted. Default: 32.</summary>
    public int InpaintDilation { get; set; } = 32;

    /// <summary>Erosion steps that select the border texels inpainting copies from. Default: 3.</summary>
    public int InpaintSearchErosion { get; set; } = 3;

    /// <summary>Stage-2 refinement steps (<c>iters_refine</c>). Default: 50.</summary>
    public int RefineIterations { get; set; } = 50;

    /// <summary>Texture learning rate (<c>texture_lr</c>). Default: 0.2.</summary>
    public double TextureLearningRate { get; set; } = 0.2;

    /// <summary>Novel-view resolution of stage 2 (<c>render_resolution = 512</c>). Default: 512.</summary>
    public int RefineRenderResolution { get; set; } = 512;

    /// <summary>SDEdit strength at the first refinement step (<c>0.8</c>). Default: 0.8.</summary>
    public double RefineStrengthStart { get; set; } = 0.8;

    /// <summary>SDEdit strength increase over the refinement (<c>step_ratio * 0.15</c>). Default: 0.15.</summary>
    public double RefineStrengthRange { get; set; } = 0.15;

    /// <summary>Denoising schedule length of each refinement (<c>refine(steps=50)</c>). Default: 50.</summary>
    public int RefineDenoisingSteps { get; set; } = 50;

    /// <summary>Texture Adam beta1 (<c>torch.optim.Adam</c>'s default, as <c>main2.py</c> uses). Default: 0.9.</summary>
    public double TextureAdamBeta1 { get; set; } = 0.9;

    /// <summary>Texture Adam beta2 (<c>torch.optim.Adam</c>'s default). Default: 0.999.</summary>
    public double TextureAdamBeta2 { get; set; } = 0.999;

    /// <summary>Texture Adam epsilon (<c>torch.optim.Adam</c>'s default). Default: 1e-8.</summary>
    public double TextureAdamEpsilon { get; set; } = 1e-8;

    internal double CameraRadius(DreamGaussianMode mode) => mode == DreamGaussianMode.ImageTo3D ? ImageCameraRadius : TextCameraRadius;

    internal int PositionLearningRateMaxSteps(DreamGaussianMode mode)
        => mode == DreamGaussianMode.ImageTo3D ? ImagePositionLearningRateMaxSteps : TextPositionLearningRateMaxSteps;

    internal int DensifyStartIteration(DreamGaussianMode mode)
        => mode == DreamGaussianMode.ImageTo3D ? ImageDensifyStartIteration : TextDensifyStartIteration;

    internal int DensificationInterval(DreamGaussianMode mode)
        => mode == DreamGaussianMode.ImageTo3D ? ImageDensificationInterval : TextDensificationInterval;

    internal double GuidanceWeight(DreamGaussianMode mode) => mode == DreamGaussianMode.ImageTo3D ? ImageGuidanceWeight : TextGuidanceWeight;

    /// <summary>The novel-view render resolution at a step ratio.</summary>
    internal int RenderResolution(double stepRatio)
    {
        var resolutions = RenderResolutions;
        if (resolutions is null || resolutions.Length == 0)
            throw new InvalidOperationException("At least one render resolution is needed.");
        var thresholds = RenderResolutionThresholds ?? new double[0];
        for (int i = 0; i < resolutions.Length - 1 && i < thresholds.Length; i++)
        {
            if (stepRatio < thresholds[i])
                return resolutions[i];
        }

        return resolutions[Math.Min(resolutions.Length - 1, thresholds.Length)];
    }

    /// <summary>A member-wise copy (arrays and <see cref="MeshExtraction"/> shared).</summary>
    internal DreamGaussianOptions ShallowCopy() => (DreamGaussianOptions)MemberwiseClone();

    /// <summary>Checks the settings the pipeline cannot run without.</summary>
    internal void Validate()
    {
        if (InitialGaussianCount < 1) throw new InvalidOperationException("InitialGaussianCount must be positive.");
        if (Iterations < 0 || RefineIterations < 0) throw new InvalidOperationException("Iteration counts cannot be negative.");
        if (BatchSize < 1) throw new InvalidOperationException("BatchSize must be positive.");
        if (ReferenceSize < 1 || BakeRenderResolution < 1 || RefineRenderResolution < 1 || TextureSize < 2)
            throw new InvalidOperationException("Resolutions must be positive and the texture at least 2 texels wide.");
        if (BakeElevations is null || BakeAzimuths is null || BakeElevations.Length != BakeAzimuths.Length)
            throw new InvalidOperationException("BakeElevations and BakeAzimuths must have the same length.");
        if (MeshExtraction is null) throw new InvalidOperationException("MeshExtraction is required.");
        if (RenderResolutions is null || RenderResolutions.Length == 0)
            throw new InvalidOperationException("At least one render resolution is needed.");
    }
}

/// <summary>Which DreamGaussian pipeline runs: image-to-3D (Zero-1-to-3 guidance) or text-to-3D (Stable Diffusion).</summary>
public enum DreamGaussianMode
{
    /// <summary>Image-to-3D: Zero-1-to-3 guidance plus the reference-view losses (<c>configs/image.yaml</c>).</summary>
    ImageTo3D,

    /// <summary>Text-to-3D: Stable Diffusion guidance only (<c>configs/text.yaml</c>).</summary>
    TextTo3D,
}
