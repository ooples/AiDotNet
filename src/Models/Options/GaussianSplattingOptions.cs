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
    // -----------------------------------------------------------------------
    // Densification schedule knobs added in #1835. Every field is nullable and
    // defaults to the 3DGS paper value (Kerbl et al. 2023) when null, so a
    // caller who doesn't touch these gets paper behavior; a caller who wants
    // to tune only ONE parameter can set it and leave the rest at the paper
    // default. All defaults documented in <see cref="EffectiveDensification"/>.
    // -----------------------------------------------------------------------

    /// <summary>
    /// Iteration index at which the split/prune loop first activates. Paper: 500.
    /// Skipping the first N iterations lets the initial random cloud stabilize
    /// under gradient descent before we start splitting Gaussians. Nullable —
    /// null uses the paper default.
    /// </summary>
    public int? DensificationStartIteration { get; set; }

    /// <summary>
    /// Iteration index after which densification stops (freeze the cloud for
    /// final convergence). Paper: 15000. Nullable — null uses the paper default.
    /// </summary>
    public int? DensificationEndIteration { get; set; }

    /// <summary>
    /// Gradient magnitude threshold above which a Gaussian is deemed to cover too
    /// much space and is SPLIT into two children. Paper: τ_pos = 0.0002.
    /// Nullable — null uses the paper default.
    /// </summary>
    public double? GradientNormThreshold { get; set; }

    /// <summary>
    /// Opacity below which a Gaussian is culled during pruning (its rendered
    /// contribution has collapsed to near-zero). Paper: 0.005. Nullable — null
    /// uses the paper default; falls back to <see cref="PruneOpacityThreshold"/>
    /// if that legacy field is set.
    /// </summary>
    public double? OpacityPruneThreshold { get; set; }

    /// <summary>
    /// Hard ceiling on total Gaussian count during densification (OOM guard).
    /// Nullable — null falls back to <see cref="MaxGaussians"/>. Kept separate so
    /// existing callers using MaxGaussians as a soft budget see identical
    /// behavior; new callers can opt in to the beyond-industry ceiling variant.
    /// </summary>
    public int? MaxGaussianCount { get; set; }

    /// <summary>
    /// Number of iterations over which the mean gradient norm is averaged before
    /// deciding to split (avoids spurious splits from a single high-gradient
    /// step). Paper: 100 (a full densification interval). Nullable — null uses
    /// the paper default.
    /// </summary>
    public int? GradientAccumulationWindow { get; set; }

    public int TileSize { get; set; } = 16;
    public bool EnableSpatialIndex { get; set; } = true;
    public int SpatialIndexRadius { get; set; } = 1;
    public double InitialNeighborSearchScale { get; set; } = 4.0;
    public double InitialScaleMultiplier { get; set; } = 0.5;
    public double DefaultPointSpacing { get; set; } = 0.05;
    public double MinScale { get; set; } = 1e-6;
}
