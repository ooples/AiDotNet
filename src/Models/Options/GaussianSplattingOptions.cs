namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Gaussian Splatting models.
/// </summary>
public class GaussianSplattingOptions : ModelOptions
{
    public bool UseSphericalHarmonics { get; set; } = true;
    public int ShDegree { get; set; } = 3;
    public bool EnableDensification { get; set; } = true;
    public int DensificationInterval { get; set; } = 100;
    public double PruneOpacityThreshold { get; set; } = 0.01;
    public double SplitGradientThreshold { get; set; } = 0.1;
    public double SplitPositionJitter { get; set; } = 0.25;
    public double SplitScaleFactor { get; set; } = 0.7;
    public double SplitOpacityFactor { get; set; } = 0.5;
    public double SplitOpacityMax { get; set; } = 0.99;
    public int MaxGaussians { get; set; } = 2000000;
    public double PositionLearningRate { get; set; } = 1e-3;
    public double ColorLearningRate { get; set; } = 1e-2;
    public double OpacityLearningRate { get; set; } = 1e-2;
    public double ScaleLearningRate { get; set; } = 1e-3;
    public double RotationLearningRate { get; set; } = 1e-3;
    public int TileSize { get; set; } = 16;
    public bool EnableSpatialIndex { get; set; } = true;
    public int SpatialIndexRadius { get; set; } = 1;
    public double InitialNeighborSearchScale { get; set; } = 4.0;
    public double InitialScaleMultiplier { get; set; } = 0.5;
    public double DefaultPointSpacing { get; set; } = 0.05;
    public double MinScale { get; set; } = 1e-6;
}
