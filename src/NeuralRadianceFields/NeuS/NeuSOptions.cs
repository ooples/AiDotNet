namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>Which inputs NeuS's color network conditions on (reference <c>RenderingNetwork.mode</c>).</summary>
internal enum NeuSRenderingMode
{
    /// <summary>Point, view direction, normal and feature (the IDR rendering network; the paper's setting).</summary>
    Idr,

    /// <summary>Point, normal and feature.</summary>
    NoViewDirection,

    /// <summary>Point, view direction and feature.</summary>
    NoNormal,
}

/// <summary>Which NeuS renderer's sampling region and section rules a field follows.</summary>
internal enum NeuSRendererVariant
{
    /// <summary>
    /// The NeuS reference (<c>models/renderer.py</c>): sections valid when either end is inside the unit sphere, the
    /// Eikonal term over samples inside radius 1.2, every sample contributing opacity, the last section
    /// <c>2 / n_samples</c> long, and the color network reading the ray direction.
    /// </summary>
    Reference,

    /// <summary>
    /// SyncDreamer's renderer (<c>renderer/renderer.py NeuSRenderer</c>, Liu et al. 2023): only samples inside the
    /// cube <c>|x|, |y|, |z| &lt;= CubeHalfSide</c> get opacity, color and the Eikonal term (normalized by
    /// <c>sum(inner + 1e-5)</c>); up-sampling sections are valid when either end is inside the cube; the last section
    /// repeats the previous one; and the color network reads the negated ray direction.
    /// </summary>
    SyncDreamer,
}

/// <summary>
/// The NeuS configuration (Wang et al. 2021, "NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view
/// Reconstruction", arXiv 2106.10689). The defaults are the reference's <c>confs/wmask.conf</c>: object-centric
/// reconstruction with foreground masks and no NeRF++ background model, the setting multi-view-diffusion pipelines use.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> NeuS learns a 3D shape as a signed distance function from photos taken around an object.
/// These settings control the size of its networks, how many points it samples along each camera ray, and how it trains.
/// The defaults match the paper; small values are only useful for tests.</para>
/// </remarks>
internal sealed class NeuSOptions
{
    // ---- SDF network (sdf_network) ----

    /// <summary>The SDF network's hidden width (<c>d_hidden</c> = 256).</summary>
    public int SdfHiddenDim { get; set; } = 256;

    /// <summary>The SDF network's hidden layer count (<c>n_layers</c> = 8).</summary>
    public int SdfHiddenLayers { get; set; } = 8;

    /// <summary>The geometry feature width passed to the color network (<c>d_out</c> − 1 = 256).</summary>
    public int SdfFeatureDim { get; set; } = 256;

    /// <summary>The layers that take the embedded input again (<c>skip_in</c> = [4]).</summary>
    public int[] SdfSkipLayers { get; set; } = new[] { 4 };

    /// <summary>The point embedding's frequency count (<c>multires</c> = 6).</summary>
    public int SdfFrequencies { get; set; } = 6;

    /// <summary>The geometric-initialization sphere radius (<c>bias</c> = 0.5).</summary>
    public double SdfInitialSphereRadius { get; set; } = 0.5;

    /// <summary>The input scale (<c>scale</c> = 1).</summary>
    public double SdfScale { get; set; } = 1.0;

    /// <summary>Whether to use the geometric initialization (<c>geometric_init</c> = true).</summary>
    public bool GeometricInit { get; set; } = true;

    /// <summary>Whether the cameras look out from inside the surface (<c>inside_outside</c> = false).</summary>
    public bool InsideOutside { get; set; }

    /// <summary>The softplus sharpness β (100).</summary>
    public double SoftplusBeta { get; set; } = 100.0;

    /// <summary>Whether every MLP layer is weight-normalized (<c>weight_norm</c> = true).</summary>
    public bool WeightNorm { get; set; } = true;

    // ---- Rendering network ----

    /// <summary>The color network's inputs (<c>mode</c> = idr).</summary>
    public NeuSRenderingMode RenderingMode { get; set; } = NeuSRenderingMode.Idr;

    /// <summary>The color network's hidden width (<c>d_hidden</c> = 256).</summary>
    public int ColorHiddenDim { get; set; } = 256;

    /// <summary>The color network's hidden layer count (<c>n_layers</c> = 4).</summary>
    public int ColorHiddenLayers { get; set; } = 4;

    /// <summary>The view-direction embedding's frequency count (<c>multires_view</c> = 4).</summary>
    public int ViewFrequencies { get; set; } = 4;

    /// <summary>Whether the color passes through a sigmoid (<c>squeeze_out</c> = true).</summary>
    public bool SqueezeColor { get; set; } = true;

    // ---- Variance network ----

    /// <summary>The initial log-variance parameter; the S-density's sharpness is <c>s = exp(10 · v)</c> (<c>init_val</c> = 0.3).</summary>
    public double InitialVariance { get; set; } = 0.3;

    // ---- Renderer (neus_renderer) ----

    /// <summary>Uniform samples per ray (<c>n_samples</c> = 64).</summary>
    public int Samples { get; set; } = 64;

    /// <summary>Importance samples per ray added by hierarchical up-sampling (<c>n_importance</c> = 64).</summary>
    public int ImportanceSamples { get; set; } = 64;

    /// <summary>Up-sampling rounds; round i uses the fixed sharpness 64·2ⁱ (<c>up_sample_steps</c> = 4).</summary>
    public int UpSampleSteps { get; set; } = 4;

    /// <summary>Whether training rays get a random shift of the uniform samples (<c>perturb</c> = 1).</summary>
    public bool Perturb { get; set; } = true;

    /// <summary>A constant background color composited behind the surface (<c>background_rgb</c>), or null for none.</summary>
    public double[]? BackgroundColor { get; set; }

    // ---- Training (train) ----

    /// <summary>The peak learning rate (<c>learning_rate</c> = 5e-4).</summary>
    public double LearningRate { get; set; } = 5e-4;

    /// <summary>The cosine schedule's final learning-rate fraction (<c>learning_rate_alpha</c> = 0.05).</summary>
    public double LearningRateAlpha { get; set; } = 0.05;

    /// <summary>The linear warm-up length in iterations (<c>warm_up_end</c> = 5000).</summary>
    public int WarmUpIterations { get; set; } = 5000;

    /// <summary>The total iteration count (<c>end_iter</c> = 300000).</summary>
    public int Iterations { get; set; } = 300000;

    /// <summary>Rays per iteration (<c>batch_size</c> = 512).</summary>
    public int BatchSize { get; set; } = 512;

    /// <summary>The cosine-annealing length; 0 disables the annealing (<c>anneal_end</c> = 0 with masks, 50000 without).</summary>
    public int AnnealIterations { get; set; }

    /// <summary>
    /// The weight of NeuS's L1 color term, <c>Σ|ĉ − c|</c> per ray averaged over the color rays (NeuS: 1; Wonder3D's
    /// <c>lambda_rgb_l1</c> = 0).
    /// </summary>
    public double ColorL1Weight { get; set; } = 1.0;

    /// <summary>
    /// The weight of a squared color term, <c>Σ(ĉ − c)²</c> per ray averaged over the color rays (not part of NeuS, so
    /// 0; Wonder3D's <c>lambda_rgb_mse</c> = 0.5).
    /// </summary>
    public double ColorMseWeight { get; set; }

    /// <summary>
    /// Whether the color terms count only foreground rays (NeuS: true). Wonder3D composites its targets onto the
    /// background color and supervises every ray (false).
    /// </summary>
    public bool ColorOnForegroundOnly { get; set; } = true;

    /// <summary>
    /// The fraction of the color rays, smallest error first, the color terms keep; the rest are dropped as outliers
    /// (1 keeps all, as NeuS; Wonder3D's <c>rgb_p_ratio</c> = 0.8).
    /// </summary>
    public double ColorKeepRatio { get; set; } = 1.0;

    /// <summary>The Eikonal loss weight (<c>igr_weight</c> = 0.1).</summary>
    public double EikonalWeight { get; set; } = 0.1;

    /// <summary>The mask cross-entropy weight (<c>mask_weight</c> = 0.1; 0 trains without masks).</summary>
    public double MaskWeight { get; set; } = 0.1;

    /// <summary>
    /// The fraction of rays, smallest cross-entropy first, the mask term keeps (1 keeps all, as NeuS; Wonder3D's
    /// <c>mask_p_ratio</c> = 0.9).
    /// </summary>
    public double MaskKeepRatio { get; set; } = 1.0;

    /// <summary>
    /// The weight of a normal-map term <c>1 − cos(n̂_rendered, n_target)</c> over foreground rays, for views that carry
    /// normals (Wonder3D's geometry fusion; not part of NeuS, so 0 by default).
    /// </summary>
    public double NormalWeight { get; set; }

    /// <summary>
    /// The fraction of the normal rays, smallest error first, the normal term keeps; the rest are dropped as outliers
    /// (1 keeps all; Wonder3D's <c>normal_p_ratio</c> = 0.8).
    /// </summary>
    public double NormalKeepRatio { get; set; } = 1.0;

    /// <summary>
    /// Wonder3D's geometry-aware normal loss (<c>geo_aware</c>): each ray's error is weighted by
    /// <c>exp(|cos θ|) / Σ exp(|cos θ|)</c>, θ the angle between the ray and its target normal, and the kept errors
    /// are summed rather than averaged. Off by default.
    /// </summary>
    public bool GeometryAwareNormals { get; set; }

    /// <summary>
    /// The largest <c>cos θ</c> between a ray and its target normal for the ray to supervise normals, or null for no
    /// limit. Wonder3D uses −0.1: a surface seen at a grazing angle or from behind gives an unreliable normal, so
    /// those rays are dropped and their <c>cos θ</c> is zeroed in the geometry-aware weights.
    /// </summary>
    public double? NormalVisibleCosine { get; set; }

    /// <summary>
    /// The weight of Wonder3D's sparsity term <c>mean(exp(−SparsityScale · |f(p)|))</c> over random points of the
    /// unit cube, which pushes empty space away from the surface (<c>lambda_sparsity</c>; 0 in NeuS, 0.5 in Wonder3D).
    /// </summary>
    public double SparsityWeight { get; set; }

    /// <summary>The sparsity term's sharpness (<c>sparsity_scale</c> = 100).</summary>
    public double SparsityScale { get; set; } = 100.0;

    /// <summary>
    /// The weight of Wonder3D's 3D normal-smoothness term <c>mean|∇f(p) − ∇f(p + ε)|</c>, ε ~ N(0, σ²) per coordinate,
    /// over the same random points (<c>lambda_3d_normal_smooth</c>; 0 in NeuS, 1 in Wonder3D).
    /// </summary>
    public double NormalSmoothWeight { get; set; }

    /// <summary>The perturbation σ of the normal-smoothness term (0.01).</summary>
    public double NormalSmoothNoise { get; set; } = 0.01;

    /// <summary>The random points per iteration of the sparsity and smoothness terms (Wonder3D: 2048).</summary>
    public int RegularizationPoints { get; set; } = 2048;

    /// <summary>Points per query when sampling a dense grid (NeuS's <c>extract_fields</c> uses 64³-point blocks).</summary>
    public int GridChunkSize { get; set; } = 64 * 64 * 64;

    /// <summary>The seed of every random draw (initialization, ray batches, perturbation); null draws a secure seed.</summary>
    public int? Seed { get; set; }

    /// <summary>The renderer's sampling-region and section rules (the NeuS reference by default).</summary>
    public NeuSRendererVariant RendererVariant { get; set; } = NeuSRendererVariant.Reference;

    /// <summary>
    /// The radius of the sphere whose chord gives each ray's near and far depths (<c>near_far_from_sphere</c>: 1 in
    /// NeuS, <c>sqrt(3) / 2</c> in SyncDreamer).
    /// </summary>
    public double NearFarRadius { get; set; } = 1.0;

    /// <summary>The half side of <see cref="NeuSRendererVariant.SyncDreamer"/>'s region of interest (<c>DEFAULT_SIDE_LENGTH</c> = 0.6).</summary>
    public double CubeHalfSide { get; set; } = 0.6;

    /// <summary>Validates the configuration.</summary>
    public void Validate()
    {
        if (SdfHiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(SdfHiddenDim));
        if (SdfHiddenLayers <= 0) throw new ArgumentOutOfRangeException(nameof(SdfHiddenLayers));
        if (SdfFeatureDim < 0) throw new ArgumentOutOfRangeException(nameof(SdfFeatureDim));
        if (SdfFrequencies < 0) throw new ArgumentOutOfRangeException(nameof(SdfFrequencies));
        if (!(SdfScale > 0)) throw new ArgumentOutOfRangeException(nameof(SdfScale));
        if (!(SoftplusBeta > 0)) throw new ArgumentOutOfRangeException(nameof(SoftplusBeta));
        if (ColorHiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(ColorHiddenDim));
        if (ColorHiddenLayers < 0) throw new ArgumentOutOfRangeException(nameof(ColorHiddenLayers));
        if (ViewFrequencies < 0) throw new ArgumentOutOfRangeException(nameof(ViewFrequencies));
        if (Samples < 2) throw new ArgumentOutOfRangeException(nameof(Samples), "At least two samples per ray are needed.");
        if (ImportanceSamples < 0) throw new ArgumentOutOfRangeException(nameof(ImportanceSamples));
        if (UpSampleSteps < 1) throw new ArgumentOutOfRangeException(nameof(UpSampleSteps));
        if (ImportanceSamples % UpSampleSteps != 0)
            throw new ArgumentException("ImportanceSamples must be divisible by UpSampleSteps.", nameof(ImportanceSamples));
        if (BackgroundColor is not null && BackgroundColor.Length != 3)
            throw new ArgumentException("The background color needs three channels.", nameof(BackgroundColor));
        if (!(LearningRate > 0)) throw new ArgumentOutOfRangeException(nameof(LearningRate));
        if (LearningRateAlpha < 0 || LearningRateAlpha > 1) throw new ArgumentOutOfRangeException(nameof(LearningRateAlpha));
        if (WarmUpIterations < 0) throw new ArgumentOutOfRangeException(nameof(WarmUpIterations));
        if (Iterations <= 0) throw new ArgumentOutOfRangeException(nameof(Iterations));
        if (BatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(BatchSize));
        if (AnnealIterations < 0) throw new ArgumentOutOfRangeException(nameof(AnnealIterations));
        if (EikonalWeight < 0) throw new ArgumentOutOfRangeException(nameof(EikonalWeight));
        if (MaskWeight < 0) throw new ArgumentOutOfRangeException(nameof(MaskWeight));
        if (NormalWeight < 0) throw new ArgumentOutOfRangeException(nameof(NormalWeight));
        if (ColorL1Weight < 0) throw new ArgumentOutOfRangeException(nameof(ColorL1Weight));
        if (ColorMseWeight < 0) throw new ArgumentOutOfRangeException(nameof(ColorMseWeight));
        if (!(ColorKeepRatio > 0 && ColorKeepRatio <= 1)) throw new ArgumentOutOfRangeException(nameof(ColorKeepRatio));
        if (!(MaskKeepRatio > 0 && MaskKeepRatio <= 1)) throw new ArgumentOutOfRangeException(nameof(MaskKeepRatio));
        if (!(NormalKeepRatio > 0 && NormalKeepRatio <= 1)) throw new ArgumentOutOfRangeException(nameof(NormalKeepRatio));
        if (NormalVisibleCosine is double limit && (double.IsNaN(limit) || limit < -1 || limit > 1))
            throw new ArgumentOutOfRangeException(nameof(NormalVisibleCosine));
        if (SparsityWeight < 0) throw new ArgumentOutOfRangeException(nameof(SparsityWeight));
        if (!(SparsityScale > 0)) throw new ArgumentOutOfRangeException(nameof(SparsityScale));
        if (NormalSmoothWeight < 0) throw new ArgumentOutOfRangeException(nameof(NormalSmoothWeight));
        if (!(NormalSmoothNoise > 0)) throw new ArgumentOutOfRangeException(nameof(NormalSmoothNoise));
        if (RegularizationPoints <= 0) throw new ArgumentOutOfRangeException(nameof(RegularizationPoints));
        if (GridChunkSize <= 0) throw new ArgumentOutOfRangeException(nameof(GridChunkSize));
        if (!(NearFarRadius > 0)) throw new ArgumentOutOfRangeException(nameof(NearFarRadius));
        if (!(CubeHalfSide > 0)) throw new ArgumentOutOfRangeException(nameof(CubeHalfSide));
    }

    /// <summary>
    /// The learning-rate factor at <paramref name="iteration"/>: linear warm-up, then cosine decay to
    /// <see cref="LearningRateAlpha"/> (reference <c>update_learning_rate</c>).
    /// </summary>
    /// <param name="iteration">The iteration (0-based).</param>
    public double LearningRateFactor(int iteration)
    {
        if (iteration < WarmUpIterations) return (double)iteration / WarmUpIterations;
        double span = Math.Max(1, Iterations - WarmUpIterations);
        double progress = Math.Min(1.0, (iteration - WarmUpIterations) / span);
        return (Math.Cos(Math.PI * progress) + 1.0) * 0.5 * (1.0 - LearningRateAlpha) + LearningRateAlpha;
    }

    /// <summary>The cosine-annealing ratio at <paramref name="iteration"/> (reference <c>get_cos_anneal_ratio</c>).</summary>
    /// <param name="iteration">The iteration.</param>
    public double CosineAnnealRatio(int iteration)
        => AnnealIterations == 0 ? 1.0 : Math.Min(1.0, (double)iteration / AnnealIterations);
}
