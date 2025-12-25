namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Instant-NGP models.
/// </summary>
public class InstantNGPOptions<T> : ModelOptions
{
    public int HashTableSize { get; set; } = 524288;
    public int NumLevels { get; set; } = 16;
    public int FeaturesPerLevel { get; set; } = 2;
    public int FinestResolution { get; set; } = 2048;
    public int CoarsestResolution { get; set; } = 16;
    public int MlpHiddenDim { get; set; } = 64;
    public int MlpNumLayers { get; set; } = 2;
    public int FeatureDim { get; set; } = 16;
    public int ColorHiddenDim { get; set; } = 64;
    public int ColorNumLayers { get; set; } = 2;
    public bool UseOccupancyGrid { get; set; } = true;
    public int OccupancyGridResolution { get; set; } = 128;
    public double LearningRate { get; set; } = 1e-2;
    public double OccupancyDecay { get; set; } = 0.95;
    public double OccupancyThreshold { get; set; } = 0.01;
    public int OccupancyUpdateInterval { get; set; } = 16;
    public int OccupancySamplesPerCell { get; set; } = 1;
    public double OccupancyJitter { get; set; } = 1.0;
    public int RenderSamples { get; set; } = 64;
    public double RenderNearBound { get; set; } = 0.0;
    public double RenderFarBound { get; set; } = 1.0;
    public Vector<T>? SceneMin { get; set; }
    public Vector<T>? SceneMax { get; set; }
}
