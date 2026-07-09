using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Gaussian Splatting models (Kerbl et al. 2023, "3D Gaussian
/// Splatting for Real-Time Radiance Field Rendering").
/// </summary>
/// <remarks>
/// <para>
/// Two families of settings live on this class: paper defaults for the model itself (opacity /
/// scale / rotation initialization, per-attribute learning rates, spatial-index tuning) and
/// densification-schedule knobs added by #1835 for finer-grained control over the split/prune
/// loop that grows the initial random cloud into millions of specialized Gaussians during
/// training.
/// </para>
/// <para><b>For Beginners:</b> If you're new to 3DGS: leave every field at its default and
/// you'll get paper-quality behavior. As you learn the model, override individual fields
/// (e.g. bump <see cref="MaxGaussians"/> for a larger scene, lower
/// <see cref="PositionLearningRate"/> for a more delicate initial cloud). Every field has a
/// paper-anchored default so no configuration is required to start.
/// </para>
/// </remarks>
public class GaussianSplattingOptions : ModelOptions
{
    /// <summary>Whether to use spherical-harmonics color evaluation (view-dependent color).</summary>
    /// <value>true (paper default — enables view-dependent lighting).</value>
    /// <remarks>Disable for models where every Gaussian's color is view-independent.</remarks>
    public bool UseSphericalHarmonics { get; set; } = true;

    /// <summary>Spherical-harmonics degree (0-3). Higher = more view-dependent detail.</summary>
    /// <value>3 (paper default — 16 SH coefficients per Gaussian).</value>
    public int ShDegree { get; set; } = 3;

    /// <summary>Whether the split/prune densification loop runs during training.</summary>
    /// <value>true (paper default — the mechanism that grows the cloud from ~100k to ~1M+ Gaussians).</value>
    public bool EnableDensification { get; set; } = true;

    /// <summary>Iterations between fixed-interval densification firings.</summary>
    /// <value>100 (paper default).</value>
    public int DensificationInterval { get; set; } = 100;

    /// <summary>Legacy opacity threshold below which a Gaussian is pruned. Superseded by <see cref="OpacityPruneThreshold"/>.</summary>
    /// <value>0.01 (legacy default).</value>
    public double PruneOpacityThreshold { get; set; } = 0.01;

    /// <summary>Legacy gradient threshold above which a Gaussian is split. Superseded by <see cref="GradientNormThreshold"/>.</summary>
    /// <value>0.1 (legacy default; paper τ_pos = 0.0002).</value>
    public double SplitGradientThreshold { get; set; } = 0.1;

    /// <summary>Position jitter magnitude applied to split children.</summary>
    /// <value>0.25 (fraction of the source Gaussian's scale).</value>
    public double SplitPositionJitter { get; set; } = 0.25;

    /// <summary>Scale factor applied to split children.</summary>
    /// <value>0.7 (paper default — children are 70% the size of the parent).</value>
    public double SplitScaleFactor { get; set; } = 0.7;

    /// <summary>Opacity factor applied to split children.</summary>
    /// <value>0.5 (paper default — children start at half the parent's opacity).</value>
    public double SplitOpacityFactor { get; set; } = 0.5;

    /// <summary>Maximum post-sigmoid opacity a split child can start at.</summary>
    /// <value>0.99 (avoids fully-opaque children that dominate rendering).</value>
    public double SplitOpacityMax { get; set; } = 0.99;

    /// <summary>Hard ceiling on total Gaussian count during training (OOM guard).</summary>
    /// <value>2000000 (2M Gaussians — comfortably fits scenes on a 24GB GPU).</value>
    public int MaxGaussians { get; set; } = 2000000;

    /// <summary>Base learning rate for Gaussian positions.</summary>
    /// <value>1e-3 (compatible legacy default; #1833 IHyperparameterAware routing derives the paper 1.6e-4 anchor when the caller sets a base LR).</value>
    public double PositionLearningRate { get; set; } = 1e-3;

    /// <summary>Base learning rate for Gaussian colors (or SH coefficients when SH is enabled).</summary>
    /// <value>1e-2.</value>
    public double ColorLearningRate { get; set; } = 1e-2;

    /// <summary>Base learning rate for Gaussian opacities.</summary>
    /// <value>1e-2.</value>
    public double OpacityLearningRate { get; set; } = 1e-2;

    /// <summary>Base learning rate for Gaussian anisotropic scales.</summary>
    /// <value>1e-3.</value>
    public double ScaleLearningRate { get; set; } = 1e-3;

    /// <summary>Base learning rate for Gaussian rotations (quaternion updates).</summary>
    /// <value>1e-3.</value>
    public double RotationLearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Iteration index at which the split/prune loop first activates.
    /// </summary>
    /// <value>Nullable. Null defaults to 500 iterations (or a MaxIterations-scaled value for short runs). Skipping the first N iterations lets the initial random cloud stabilize before we start splitting.</value>
    /// <remarks>Paper: 500 (of ~30000 total). #1835 auto-scales this down proportionally for shorter runs where 500 &gt; MaxIterations would mean densification never fires.</remarks>
    public int? DensificationStartIteration { get; set; }

    /// <summary>
    /// Iteration index after which densification stops.
    /// </summary>
    /// <value>Nullable. Null defaults to 15000 (or an 85%-of-MaxIterations value for shorter runs). Freezing the cloud in the last portion of training lets final iterations converge with a stable topology.</value>
    /// <remarks>Paper: 15000 (of ~30000 total).</remarks>
    public int? DensificationEndIteration { get; set; }

    /// <summary>
    /// Gradient magnitude threshold above which a Gaussian is deemed to cover too much space and is SPLIT into two children.
    /// </summary>
    /// <value>Nullable. When null, the effective threshold falls back to the legacy <see cref="SplitGradientThreshold"/> field. To opt into the paper default (τ_pos = 0.0002), set explicitly.</value>
    /// <remarks>Paper: 0.0002.</remarks>
    public double? GradientNormThreshold { get; set; }

    /// <summary>
    /// Opacity below which a Gaussian is culled during pruning.
    /// </summary>
    /// <value>Nullable. When null, the effective threshold falls back to the legacy <see cref="PruneOpacityThreshold"/> field. To opt into the paper default (0.005), set explicitly.</value>
    /// <remarks>Paper: 0.005.</remarks>
    public double? OpacityPruneThreshold { get; set; }

    /// <summary>Hard ceiling on total Gaussian count during densification.</summary>
    /// <value>Nullable. Null falls back to <see cref="MaxGaussians"/>.</value>
    /// <remarks>Kept separate so existing callers using <see cref="MaxGaussians"/> as a soft budget see identical behavior; new callers can opt into the beyond-industry ceiling variant.</remarks>
    public int? MaxGaussianCount { get; set; }

    /// <summary>Iterations of gradient magnitude averaged before a split decision.</summary>
    /// <value>Nullable. Null defaults to 100 (one densification interval).</value>
    /// <remarks>Paper: 100. Averaging avoids spurious splits from a single high-gradient step.</remarks>
    public int? GradientAccumulationWindow { get; set; }

    /// <summary>Swappable densification-schedule strategy.</summary>
    /// <value>Nullable. When null, the model uses <see cref="AiDotNet.NeuralRadianceFields.Data.FixedIntervalDensificationSchedule"/> (reference-impl behavior).</value>
    /// <remarks>#1835 excellence goal #2. Set to <see cref="AiDotNet.NeuralRadianceFields.Data.AdaptiveDensificationSchedule"/> to fire based on observed loss / gradient signals instead of a fixed clock.</remarks>
    public AiDotNet.NeuralRadianceFields.Data.IDensificationSchedule? DensificationSchedule { get; set; }

    /// <summary>Run a post-training compression pass (prune + merge + SH quantize).</summary>
    /// <value>false (default — industry-standard: caller runs their own post-processing).</value>
    /// <remarks>#1835 excellence goal #3. When true, <c>AiModelBuilder.BuildAsync</c> automatically runs the compression pass at the end of image-space training.</remarks>
    public bool CompressOnBuildComplete { get; set; }

    /// <summary>Per-technique on/off + tuning for the post-training compression pass.</summary>
    /// <value>Nullable. Null uses defaults (all techniques enabled at paper thresholds).</value>
    public GaussianCompressionOptions? CompressionOptions { get; set; }

    /// <summary>Per-attribute LR multipliers for split children.</summary>
    /// <value>Nullable. Null uses paper defaults (position: 0.7, scale: 1.5, others: 1.0).</value>
    /// <remarks>#1835 excellence goal #4. Applied at split time as a one-shot perturbation adjustment (stateless approximation of the paper's per-Gaussian LR state).</remarks>
    public SplitChildLearningRateScales? SplitChildLearningRateScales { get; set; }

    /// <summary>Splat rendering tile size (pixels per tile side).</summary>
    /// <value>16 (paper default).</value>
    public int TileSize { get; set; } = 16;

    /// <summary>Whether to maintain a spatial index for accelerated per-pixel Gaussian queries.</summary>
    /// <value>true.</value>
    public bool EnableSpatialIndex { get; set; } = true;

    /// <summary>Neighborhood radius (in tiles) for spatial-index lookups.</summary>
    /// <value>1.</value>
    public int SpatialIndexRadius { get; set; } = 1;

    /// <summary>Initial scale factor for neighbor search during construction.</summary>
    /// <value>4.0.</value>
    public double InitialNeighborSearchScale { get; set; } = 4.0;

    /// <summary>Initial scale multiplier for newly-created Gaussians.</summary>
    /// <value>0.5.</value>
    public double InitialScaleMultiplier { get; set; } = 0.5;

    /// <summary>Default 3D spacing for cloud initialization when no point cloud is supplied.</summary>
    /// <value>0.05 (scene-relative).</value>
    public double DefaultPointSpacing { get; set; } = 0.05;

    /// <summary>Minimum Gaussian scale — floor to prevent collapse during optimization.</summary>
    /// <value>1e-6.</value>
    public double MinScale { get; set; } = 1e-6;

    /// <summary>Creates a new instance with paper-quality defaults.</summary>
    public GaussianSplattingOptions() { }

    /// <summary>Deep-copies every field from <paramref name="other"/>.</summary>
    /// <exception cref="ArgumentNullException"><paramref name="other"/> is null.</exception>
    public GaussianSplattingOptions(GaussianSplattingOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        UseSphericalHarmonics = other.UseSphericalHarmonics;
        ShDegree = other.ShDegree;
        EnableDensification = other.EnableDensification;
        DensificationInterval = other.DensificationInterval;
        PruneOpacityThreshold = other.PruneOpacityThreshold;
        SplitGradientThreshold = other.SplitGradientThreshold;
        SplitPositionJitter = other.SplitPositionJitter;
        SplitScaleFactor = other.SplitScaleFactor;
        SplitOpacityFactor = other.SplitOpacityFactor;
        SplitOpacityMax = other.SplitOpacityMax;
        MaxGaussians = other.MaxGaussians;
        PositionLearningRate = other.PositionLearningRate;
        ColorLearningRate = other.ColorLearningRate;
        OpacityLearningRate = other.OpacityLearningRate;
        ScaleLearningRate = other.ScaleLearningRate;
        RotationLearningRate = other.RotationLearningRate;
        DensificationStartIteration = other.DensificationStartIteration;
        DensificationEndIteration = other.DensificationEndIteration;
        GradientNormThreshold = other.GradientNormThreshold;
        OpacityPruneThreshold = other.OpacityPruneThreshold;
        MaxGaussianCount = other.MaxGaussianCount;
        GradientAccumulationWindow = other.GradientAccumulationWindow;
        DensificationSchedule = other.DensificationSchedule;
        CompressOnBuildComplete = other.CompressOnBuildComplete;
        CompressionOptions = other.CompressionOptions;
        SplitChildLearningRateScales = other.SplitChildLearningRateScales;
        TileSize = other.TileSize;
        EnableSpatialIndex = other.EnableSpatialIndex;
        SpatialIndexRadius = other.SpatialIndexRadius;
        InitialNeighborSearchScale = other.InitialNeighborSearchScale;
        InitialScaleMultiplier = other.InitialScaleMultiplier;
        DefaultPointSpacing = other.DefaultPointSpacing;
        MinScale = other.MinScale;
    }
}
