namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Instant-NGP models.
/// </summary>
public class InstantNGPOptions<T> : ModelOptions
{
    /// <summary>Number of entries in each level's hash table (T in Müller et al. 2022).</summary>
    public int HashTableSize { get; set; } = 524288;
    /// <summary>Number of resolution levels of the multiresolution hash encoding (L in Müller et al. 2022).</summary>
    public int NumLevels { get; set; } = 16;
    /// <summary>Number of features stored per hash-table entry (F in Müller et al. 2022).</summary>
    public int FeaturesPerLevel { get; set; } = 2;
    /// <summary>Grid resolution of the finest hash-encoding level (N_max in Müller et al. 2022).</summary>
    public int FinestResolution { get; set; } = 2048;
    /// <summary>Grid resolution of the coarsest hash-encoding level (N_min in Müller et al. 2022).</summary>
    public int CoarsestResolution { get; set; } = 16;
    /// <summary>Hidden width of the density MLP.</summary>
    public int MlpHiddenDim { get; set; } = 64;
    /// <summary>Number of hidden layers of the density MLP.</summary>
    public int MlpNumLayers { get; set; } = 2;
    /// <summary>Size of the geometric feature vector the density MLP passes to the color MLP.</summary>
    public int FeatureDim { get; set; } = 16;
    /// <summary>Hidden width of the color MLP.</summary>
    public int ColorHiddenDim { get; set; } = 64;
    /// <summary>Number of hidden layers of the color MLP.</summary>
    public int ColorNumLayers { get; set; } = 2;
    /// <summary>Whether to skip empty space with an occupancy grid when sampling rays.</summary>
    public bool UseOccupancyGrid { get; set; } = true;
    /// <summary>Number of cells along each axis of the occupancy grid.</summary>
    public int OccupancyGridResolution { get; set; } = 128;
    /// <summary>Learning rate of the optimizer.</summary>
    public double LearningRate { get; set; } = 1e-2;
    /// <summary>Factor by which stored occupancy decays at each update before new samples are merged.</summary>
    public double OccupancyDecay { get; set; } = 0.95;
    /// <summary>Density below which a grid cell is treated as empty and skipped when rendering.</summary>
    public double OccupancyThreshold { get; set; } = 0.01;
    /// <summary>Number of training steps between occupancy-grid updates.</summary>
    public int OccupancyUpdateInterval { get; set; } = 16;
    /// <summary>Number of density samples per grid cell at each occupancy update.</summary>
    public int OccupancySamplesPerCell { get; set; } = 1;
    /// <summary>Fraction of a grid cell by which occupancy sample positions are jittered.</summary>
    public double OccupancyJitter { get; set; } = 1.0;
    /// <summary>Number of samples taken along each rendered ray.</summary>
    public int RenderSamples { get; set; } = 64;
    /// <summary>Near bound of each ray, in normalized scene units.</summary>
    public double RenderNearBound { get; set; } = 0.0;
    /// <summary>Far bound of each ray, in normalized scene units.</summary>
    public double RenderFarBound { get; set; } = 1.0;
    /// <summary>Lower corner of the scene's axis-aligned bounding box; null uses the default bounds.</summary>
    public Vector<T>? SceneMin { get; set; }
    /// <summary>Upper corner of the scene's axis-aligned bounding box; null uses the default bounds.</summary>
    public Vector<T>? SceneMax { get; set; }
}
