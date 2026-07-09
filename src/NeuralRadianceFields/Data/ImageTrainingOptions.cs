namespace AiDotNet.NeuralRadianceFields.Data;

/// <summary>
/// Options bundle for image-space (photometric) radiance-field training (#1834 excellence
/// goals #3 + #4). Every field is nullable — null means "use the default": the facade auto-
/// derives scene bounds from the pose set (via <see cref="SceneBoundsEstimator"/>), applies
/// the paper's progressive coarse→fine schedule (<see cref="ProgressiveSamplingSchedule.Paper"/>),
/// and leaves the per-view <see cref="ImageView{T}.Prior"/> to control single-image
/// reconstruction. Callers can override any subset.
/// </summary>
public sealed class ImageTrainingOptions
{
    /// <summary>
    /// Explicit scene bounds + near/far. When null, the model auto-derives them from the
    /// loader's pose set via <see cref="SceneBoundsEstimator.EstimateFromViews"/>. Reference
    /// NeRF impls make callers supply these by hand.
    /// </summary>
    public SceneBounds? SceneBounds { get; set; }

    /// <summary>
    /// Progressive coarse→fine schedule. When null, uses <see cref="ProgressiveSamplingSchedule.Paper"/>
    /// (64 coarse → 128 fine over 5000 iters). Reference impls hard-code this — surfacing it as
    /// a first-class option is beyond industry.
    /// </summary>
    public ProgressiveSamplingSchedule? Schedule { get; set; }

    /// <summary>
    /// Global override for the confidence weight applied to per-view <see cref="ImageView{T}.Prior"/>
    /// hallucinated novel views. When null, uses each prior's own <see cref="LearnedPrior{T}.Confidence"/>
    /// as-is. Set to 0 to disable priors globally; set to 1 to force full trust.
    /// </summary>
    public double? PriorConfidenceOverride { get; set; }
}
