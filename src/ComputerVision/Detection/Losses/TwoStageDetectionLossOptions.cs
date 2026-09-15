namespace AiDotNet.ComputerVision.Detection.Losses;

/// <summary>Sampling and weighting for training two-stage (R-CNN family) detectors.</summary>
/// <remarks>
/// <para>
/// Defaults follow the published recipes. Region proposal network (Ren et al. 2015, Sec. 3.1.2-3.1.3): an anchor
/// is positive when its IoU with an object exceeds 0.7 or it is an object's best anchor, negative below 0.3;
/// 256 anchors per image are sampled with up to half positive; the classification term is averaged over the
/// sample and the regression term is weighted by lambda = 10 and normalized by the number of anchor locations.
/// Region-of-interest head (Girshick 2015, Sec. 2.3): 64 RoIs per image, 25% from proposals with IoU of at least
/// 0.5, the rest from IoU in [0.1, 0.5), with smooth-L1 box regression weighted 1. Cascade R-CNN (Cai and
/// Vasconcelos 2018, Sec. 5.1) trains three stages at IoU thresholds 0.5, 0.6 and 0.7, each with that loss.
/// </para>
/// <para><b>For Beginners:</b> a two-stage detector first proposes regions that might hold objects, then classifies
/// and refines them. These settings decide which proposals count as objects or background while training, how
/// many of each are used, and how strongly box positions are corrected.</para>
/// </remarks>
public sealed class TwoStageDetectionLossOptions
{
    /// <summary>IoU above which an anchor is a positive proposal example.</summary>
    /// <remarks><para><b>For Beginners:</b> how closely an anchor must overlap an object to count as one.</para></remarks>
    public double RpnPositiveIoU { get; set; } = 0.7;

    /// <summary>IoU below which an anchor is a negative (background) proposal example.</summary>
    /// <remarks><para><b>For Beginners:</b> anchors between the two thresholds are ignored while training.</para></remarks>
    public double RpnNegativeIoU { get; set; } = 0.3;

    /// <summary>Anchors sampled per image for the proposal loss.</summary>
    /// <remarks><para><b>For Beginners:</b> using a fixed, balanced sample stops background anchors dominating.</para></remarks>
    public int RpnBatchSizePerImage { get; set; } = 256;

    /// <summary>Largest fraction of the anchor sample that may be positive.</summary>
    /// <remarks><para><b>For Beginners:</b> 0.5 means at most one object anchor per background anchor.</para></remarks>
    public double RpnPositiveFraction { get; set; } = 0.5;

    /// <summary>Weight lambda of the proposal box-regression term.</summary>
    /// <remarks><para><b>For Beginners:</b> balances box correction against object-versus-background scoring.</para></remarks>
    public double RpnRegressionWeight { get; set; } = 10.0;

    /// <summary>Regions of interest sampled per image for the detection-head loss.</summary>
    /// <remarks><para><b>For Beginners:</b> how many proposals per image teach the final classifier.</para></remarks>
    public int RoiBatchSizePerImage { get; set; } = 64;

    /// <summary>Largest fraction of the RoI sample drawn from foreground proposals.</summary>
    /// <remarks><para><b>For Beginners:</b> 0.25 keeps three background examples per object example.</para></remarks>
    public double RoiForegroundFraction { get; set; } = 0.25;

    /// <summary>Lower bound of the background IoU interval [low, foreground threshold).</summary>
    /// <remarks><para><b>For Beginners:</b> proposals overlapping nothing at all are skipped as too easy.</para></remarks>
    public double RoiBackgroundIoULow { get; set; } = 0.1;

    /// <summary>Weight of the detection-head box-regression term.</summary>
    /// <remarks><para><b>For Beginners:</b> balances box correction against class scoring in the final head.</para></remarks>
    public double RoiRegressionWeight { get; set; } = 1.0;

    /// <summary>Foreground IoU threshold of each detection stage; Faster R-CNN uses the first.</summary>
    /// <remarks><para><b>For Beginners:</b> later cascade stages demand tighter boxes before calling them objects.</para></remarks>
    public double[] StageForegroundIoU { get; set; } = { 0.5, 0.6, 0.7 };

    /// <summary>Weight of each detection stage's loss in the total objective.</summary>
    /// <remarks><para><b>For Beginners:</b> Cascade R-CNN sums its stages equally by default.</para></remarks>
    public double[] StageLossWeights { get; set; } = { 1.0, 1.0, 1.0 };

    internal TwoStageDetectionLossOptions Snapshot(int stages)
    {
        var copy = (TwoStageDetectionLossOptions)MemberwiseClone();
        copy.StageForegroundIoU = (double[])(StageForegroundIoU ?? throw new ArgumentNullException(nameof(StageForegroundIoU))).Clone();
        copy.StageLossWeights = (double[])(StageLossWeights ?? throw new ArgumentNullException(nameof(StageLossWeights))).Clone();
        copy.Validate(stages);
        return copy;
    }

    private void Validate(int stages)
    {
        RequireProbability(RpnPositiveIoU, nameof(RpnPositiveIoU));
        RequireProbability(RpnNegativeIoU, nameof(RpnNegativeIoU));
        if (RpnNegativeIoU > RpnPositiveIoU)
            throw new ArgumentOutOfRangeException(nameof(RpnNegativeIoU), "The negative IoU threshold cannot exceed the positive one.");
        if (RpnBatchSizePerImage < 1) throw new ArgumentOutOfRangeException(nameof(RpnBatchSizePerImage));
        RequireProbability(RpnPositiveFraction, nameof(RpnPositiveFraction));
        RequireNonnegative(RpnRegressionWeight, nameof(RpnRegressionWeight));
        if (RoiBatchSizePerImage < 1) throw new ArgumentOutOfRangeException(nameof(RoiBatchSizePerImage));
        RequireProbability(RoiForegroundFraction, nameof(RoiForegroundFraction));
        RequireProbability(RoiBackgroundIoULow, nameof(RoiBackgroundIoULow));
        RequireNonnegative(RoiRegressionWeight, nameof(RoiRegressionWeight));
        if (StageForegroundIoU.Length < stages)
            throw new ArgumentException($"StageForegroundIoU needs one threshold for each of the {stages} detection stages.", nameof(StageForegroundIoU));
        if (StageLossWeights.Length < stages)
            throw new ArgumentException($"StageLossWeights needs one weight for each of the {stages} detection stages.", nameof(StageLossWeights));
        for (int stage = 0; stage < stages; stage++)
        {
            RequireProbability(StageForegroundIoU[stage], nameof(StageForegroundIoU));
            if (RoiBackgroundIoULow > StageForegroundIoU[stage])
                throw new ArgumentOutOfRangeException(nameof(RoiBackgroundIoULow), "The background interval must end at or above its lower bound.");
            RequireNonnegative(StageLossWeights[stage], nameof(StageLossWeights));
        }
    }

    private static void RequireProbability(double value, string name)
    {
        if (double.IsNaN(value) || value < 0 || value > 1)
            throw new ArgumentOutOfRangeException(name, "Thresholds and fractions must be in [0, 1].");
    }

    private static void RequireNonnegative(double value, string name)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value < 0)
            throw new ArgumentOutOfRangeException(name, "Loss weights must be finite and nonnegative.");
    }
}
