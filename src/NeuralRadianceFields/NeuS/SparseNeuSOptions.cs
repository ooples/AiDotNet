namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// The SparseNeuS configuration (Long et al. 2022, "SparseNeuS: Fast Generalizable Neural Surface Reconstruction from
/// Sparse Views", arXiv 2206.05737). The defaults are the reference's coarse level, <c>confs/general_lod0.conf</c>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SparseNeuS reconstructs a surface from a handful of photos in one forward pass, using
/// what it learned from many training scenes. Without trained weights it produces a shape, but not a meaningful one:
/// load a trained checkpoint (One-2-3-45 ships one) before relying on its output.</para>
/// </remarks>
internal sealed class SparseNeuSOptions
{
    /// <summary>The geometry volume's voxel counts per axis (<c>vol_dims</c> = 96³; each divisible by 8).</summary>
    public int[] VolumeDims { get; set; } = new[] { 96, 96, 96 };

    /// <summary>The world position of voxel (0, 0, 0) (<c>partial_vol_origin</c> = (−1, −1, −1)).</summary>
    public double[] VolumeOrigin { get; set; } = new[] { -1.0, -1.0, -1.0 };

    /// <summary>The voxel edge length (<c>voxel_size</c> = 2/95, so 96 voxels span [−1, 1]).</summary>
    public double VoxelSize { get; set; } = 2.0 / 95.0;

    /// <summary>The compressed per-view feature width before aggregation (<c>d_pyramid_feature_compress</c> = 16).</summary>
    public int CompressedFeatureDim { get; set; } = 16;

    /// <summary>The geometry encoding width (<c>regnet_d_out</c> = 16).</summary>
    public int GeometryFeatureDim { get; set; } = 16;

    /// <summary>The SDF MLP's hidden width (<c>hidden_dim</c> = 128).</summary>
    public int SdfHiddenDim { get; set; } = 128;

    /// <summary>The SDF MLP's <c>num_sdf_layers</c> (4: three linear layers).</summary>
    public int SdfLayers { get; set; } = 4;

    /// <summary>The point embedding's frequency count (<c>multires</c> = 6).</summary>
    public int SdfFrequencies { get; set; } = 6;

    /// <summary>The geometric-initialization sphere radius (0.5).</summary>
    public double SdfInitialSphereRadius { get; set; } = 0.5;

    /// <summary>The softplus sharpness β (100).</summary>
    public double SoftplusBeta { get; set; } = 100.0;

    /// <summary>The initial log-variance (<c>variance_network.init_val</c> = 0.2).</summary>
    public double InitialVariance { get; set; } = 0.2;

    /// <summary>Whether the color network uses anti-alias pooling (true).</summary>
    public bool AntiAliasPooling { get; set; } = true;

    /// <summary>Uniform samples per ray (<c>n_samples_lod0</c> = 64).</summary>
    public int Samples { get; set; } = 64;

    /// <summary>Importance samples per ray (<c>n_importance_lod0</c> = 64), added in four up-sampling rounds.</summary>
    public int ImportanceSamples { get; set; } = 64;

    /// <summary>Up-sampling rounds (4, fixed in the reference).</summary>
    public int UpSampleSteps { get; set; } = 4;

    /// <summary>Whether training rays get stratified jitter (<c>perturb</c> = 1).</summary>
    public bool Perturb { get; set; } = true;

    /// <summary>The learning rate (<c>learning_rate</c> = 2e-4).</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>The iterations at which the learning rate is multiplied by <see cref="LearningRateFactor"/> (100k, 150k, 200k).</summary>
    public int[] LearningRateMilestones { get; set; } = new[] { 100000, 150000, 200000 };

    /// <summary>The learning-rate step factor (0.5).</summary>
    public double LearningRateFactor { get; set; } = 0.5;

    /// <summary>Rays per training iteration (<c>N_rays</c> = 512).</summary>
    public int BatchSize { get; set; } = 512;

    /// <summary>The cosine-annealing and sparse-loss ramp length (<c>anneal_end</c> = 25000).</summary>
    public int AnnealIterations { get; set; } = 25000;

    /// <summary>The Eikonal loss weight (<c>sdf_igr_weight</c> = 0.1).</summary>
    public double EikonalWeight { get; set; } = 0.1;

    /// <summary>The sparseness loss weight (<c>sdf_sparse_weight</c> = 0.02), ramped over twice <see cref="AnnealIterations"/>.</summary>
    public double SparseWeight { get; set; } = 0.02;

    /// <summary>The sparseness loss decay <c>exp(−decay·|f|)</c> (<c>sdf_decay_param</c> = 100).</summary>
    public double SparseDecay { get; set; } = 100.0;

    /// <summary>Random points per iteration for the sparseness loss (1024).</summary>
    public int SparseLossPoints { get; set; } = 1024;

    /// <summary>The background-opacity weight (<c>fg_bg_weight</c> = 0.01), applied after <see cref="ForegroundLossStart"/>.</summary>
    public double ForegroundWeight { get; set; } = 0.01;

    /// <summary>The iteration from which the background-opacity loss applies (50000).</summary>
    public int ForegroundLossStart { get; set; } = 50000;

    /// <summary>Points per query when sampling a dense grid.</summary>
    public int GridChunkSize { get; set; } = 32768;

    /// <summary>The seed of every random draw; null draws a secure seed.</summary>
    public int? Seed { get; set; }

    /// <summary>Validates the configuration.</summary>
    public void Validate()
    {
        if (VolumeDims is null || VolumeDims.Length != 3) throw new ArgumentException("VolumeDims needs three values.", nameof(VolumeDims));
        foreach (int d in VolumeDims)
            if (d <= 0 || d % 8 != 0) throw new ArgumentException("Each volume dimension must be a positive multiple of 8.", nameof(VolumeDims));
        if (VolumeOrigin is null || VolumeOrigin.Length != 3) throw new ArgumentException("VolumeOrigin needs three values.", nameof(VolumeOrigin));
        if (!(VoxelSize > 0)) throw new ArgumentOutOfRangeException(nameof(VoxelSize));
        if (CompressedFeatureDim <= 0) throw new ArgumentOutOfRangeException(nameof(CompressedFeatureDim));
        if (GeometryFeatureDim <= 0) throw new ArgumentOutOfRangeException(nameof(GeometryFeatureDim));
        if (SdfHiddenDim <= GeometryFeatureDim) throw new ArgumentOutOfRangeException(nameof(SdfHiddenDim), "The hidden width must exceed the latent width.");
        if (SdfLayers < 3) throw new ArgumentOutOfRangeException(nameof(SdfLayers));
        if (SdfFrequencies < 0) throw new ArgumentOutOfRangeException(nameof(SdfFrequencies));
        if (Samples < 2) throw new ArgumentOutOfRangeException(nameof(Samples));
        if (ImportanceSamples < 0) throw new ArgumentOutOfRangeException(nameof(ImportanceSamples));
        if (UpSampleSteps < 1 || ImportanceSamples % UpSampleSteps != 0)
            throw new ArgumentException("ImportanceSamples must be divisible by UpSampleSteps.", nameof(UpSampleSteps));
        if (!(LearningRate > 0)) throw new ArgumentOutOfRangeException(nameof(LearningRate));
        if (BatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(BatchSize));
        if (AnnealIterations < 0) throw new ArgumentOutOfRangeException(nameof(AnnealIterations));
        if (SparseLossPoints <= 0) throw new ArgumentOutOfRangeException(nameof(SparseLossPoints));
        if (GridChunkSize <= 0) throw new ArgumentOutOfRangeException(nameof(GridChunkSize));
    }

    /// <summary>The cosine-annealing ratio at <paramref name="iteration"/> (<c>alpha_inter_ratio</c>).</summary>
    public double CosineAnnealRatio(int iteration)
        => AnnealIterations == 0 ? 1.0 : Math.Min(1.0, (double)iteration / AnnealIterations);

    /// <summary>The reference's <c>get_weight</c> ramp: <c>min(1, it / (2·anneal_end))·weight</c>.</summary>
    public double Ramp(int iteration, double weight)
        => AnnealIterations == 0 ? weight : Math.Min(1.0, iteration / (2.0 * AnnealIterations)) * weight;

    /// <summary>The learning-rate multiplier at <paramref name="iteration"/> (multi-step decay).</summary>
    public double LearningRateMultiplier(int iteration)
    {
        double factor = 1.0;
        foreach (int milestone in LearningRateMilestones ?? Array.Empty<int>())
            if (iteration >= milestone) factor *= LearningRateFactor;
        return factor;
    }
}
