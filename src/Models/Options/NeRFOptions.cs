namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for NeRF models.
/// </summary>
public class NeRFOptions : ModelOptions
{
    /// <summary>Number of frequency bands of the positional encoding of the 3D position.</summary>
    public int PositionEncodingLevels { get; set; } = 10;
    /// <summary>Number of frequency bands of the positional encoding of the viewing direction.</summary>
    public int DirectionEncodingLevels { get; set; } = 4;
    /// <summary>Hidden width of the density MLP.</summary>
    public int HiddenDim { get; set; } = 256;
    /// <summary>Number of fully connected layers of the density MLP.</summary>
    public int NumLayers { get; set; } = 8;
    /// <summary>Hidden width of the view-dependent color head.</summary>
    public int ColorHiddenDim { get; set; } = 128;
    /// <summary>Number of hidden layers of the view-dependent color head.</summary>
    public int ColorNumLayers { get; set; } = 1;
    /// <summary>Whether to draw fine samples where the coarse pass found density (hierarchical volume sampling).</summary>
    public bool UseHierarchicalSampling { get; set; } = true;
    /// <summary>Number of stratified coarse samples per ray (N_c in Mildenhall et al. 2020).</summary>
    public int RenderSamples { get; set; } = 64;
    /// <summary>Number of additional fine samples per ray drawn from the coarse density (N_f in Mildenhall et al. 2020).</summary>
    public int HierarchicalSamples { get; set; } = 128;
    /// <summary>Near bound of each ray, in scene units.</summary>
    public double RenderNearBound { get; set; } = 2.0;
    /// <summary>Far bound of each ray, in scene units.</summary>
    public double RenderFarBound { get; set; } = 6.0;
    /// <summary>Learning rate of the optimizer.</summary>
    public double LearningRate { get; set; } = 5e-4;
}
