using AiDotNet.Enums;

namespace AiDotNet.ComputerVision.Detection.Losses;

/// <summary>Weights and classification form for a DETR-family set prediction loss.</summary>
/// <remarks>
/// <para>
/// Matching costs and loss weights are separate, because the published recipes differ between them:
/// DINO (Zhang et al. 2022, Table 8) and RT-DETR (Zhao et al. 2023, Table A) both match with class
/// cost 2, L1 cost 5 and GIoU cost 2, but train with class weight 1, L1 weight 5 and GIoU weight 2.
/// The factory methods return each paper's defaults; every value can be overridden.
/// </para>
/// <para><b>For Beginners:</b> Training a set detector happens in two steps. First each real object
/// is paired with one prediction using the matching costs. Then the paired predictions are pushed
/// toward their objects using the loss weights. These options control both steps.</para>
/// </remarks>
public sealed class DetrSetLossOptions
{
    /// <summary>How candidate class scores are trained.</summary>
    /// <remarks><para><b>For Beginners:</b> Must match how the detector's class head is decoded:
    /// softmax heads include a no-object class, sigmoid heads do not.</para></remarks>
    public SetPredictionClassificationLoss ClassificationLoss { get; set; } = SetPredictionClassificationLoss.SoftmaxCrossEntropy;

    /// <summary>Weight of the classification loss.</summary>
    /// <remarks><para><b>For Beginners:</b> Larger values make correct class labels matter more.</para></remarks>
    public double ClassLossWeight { get; set; } = 1.0;

    /// <summary>Weight of the matched center-format L1 box loss.</summary>
    /// <remarks><para><b>For Beginners:</b> Larger values push box coordinates harder.</para></remarks>
    public double L1LossWeight { get; set; } = 5.0;

    /// <summary>Weight of the matched generalized IoU box loss.</summary>
    /// <remarks><para><b>For Beginners:</b> Larger values push box overlap harder.</para></remarks>
    public double GIoULossWeight { get; set; } = 2.0;

    /// <summary>Weight of the classification term in the matching cost.</summary>
    /// <remarks><para><b>For Beginners:</b> Larger values pair objects with confident predictions.</para></remarks>
    public double ClassCostWeight { get; set; } = 1.0;

    /// <summary>Weight of the L1 box distance in the matching cost.</summary>
    /// <remarks><para><b>For Beginners:</b> Larger values pair objects with nearby predictions.</para></remarks>
    public double L1CostWeight { get; set; } = 5.0;

    /// <summary>Weight of the negative generalized IoU in the matching cost.</summary>
    /// <remarks><para><b>For Beginners:</b> Larger values pair objects with overlapping predictions.</para></remarks>
    public double GIoUCostWeight { get; set; } = 2.0;

    /// <summary>Relative weight of the no-object class in softmax cross-entropy (DETR uses 0.1).</summary>
    /// <remarks><para><b>For Beginners:</b> Most predictions are empty, so this keeps them from
    /// dominating training. Only used by <see cref="SetPredictionClassificationLoss.SoftmaxCrossEntropy"/>.</para></remarks>
    public double NoObjectWeight { get; set; } = 0.1;

    /// <summary>The alpha of the focal or varifocal classification loss.</summary>
    /// <remarks><para><b>For Beginners:</b> Balances object and background examples. Focal loss
    /// uses 0.25 (Lin et al. 2017); varifocal loss in RT-DETR uses 0.75.</para></remarks>
    public double FocalAlpha { get; set; } = 0.25;

    /// <summary>The gamma (focusing exponent) of the focal or varifocal classification loss.</summary>
    /// <remarks><para><b>For Beginners:</b> Larger values ignore easy examples more (2 in both papers).</para></remarks>
    public double FocalGamma { get; set; } = 2.0;

    /// <summary>The alpha of the focal classification matching cost used with sigmoid heads.</summary>
    /// <remarks><para><b>For Beginners:</b> Only affects which prediction is paired with each object.</para></remarks>
    public double MatchingFocalAlpha { get; set; } = 0.25;

    /// <summary>The gamma of the focal classification matching cost used with sigmoid heads.</summary>
    /// <remarks><para><b>For Beginners:</b> Only affects which prediction is paired with each object.</para></remarks>
    public double MatchingFocalGamma { get; set; } = 2.0;

    /// <summary>DETR defaults: softmax cross-entropy, no-object weight 0.1, costs and weights 1/5/2.</summary>
    /// <remarks><para><b>For Beginners:</b> Use with detectors whose class head has a no-object class.</para></remarks>
    public static DetrSetLossOptions ForDetr() => new();

    /// <summary>DINO defaults: sigmoid focal loss (alpha 0.25, gamma 2), costs 2/5/2, weights 1/5/2.</summary>
    /// <remarks><para><b>For Beginners:</b> The published DINO recipe, Table 8 of Zhang et al. 2022.</para></remarks>
    public static DetrSetLossOptions ForDino() => new()
    {
        ClassificationLoss = SetPredictionClassificationLoss.SigmoidFocal,
        ClassCostWeight = 2.0
    };

    /// <summary>RT-DETR defaults: varifocal loss (alpha 0.75, gamma 2), costs 2/5/2, weights 1/5/2.</summary>
    /// <remarks><para><b>For Beginners:</b> The published RT-DETR recipe, Table A of Zhao et al. 2023.</para></remarks>
    public static DetrSetLossOptions ForRtDetr() => new()
    {
        ClassificationLoss = SetPredictionClassificationLoss.VariFocal,
        ClassCostWeight = 2.0,
        FocalAlpha = 0.75
    };

    internal DetrSetLossOptions Snapshot()
    {
        var copy = (DetrSetLossOptions)MemberwiseClone();
        copy.Validate();
        return copy;
    }

    internal void Validate()
    {
        if (!Enum.IsDefined(typeof(SetPredictionClassificationLoss), ClassificationLoss))
            throw new ArgumentOutOfRangeException(nameof(ClassificationLoss));
        RequireNonnegative(ClassLossWeight, nameof(ClassLossWeight));
        RequireNonnegative(L1LossWeight, nameof(L1LossWeight));
        RequireNonnegative(GIoULossWeight, nameof(GIoULossWeight));
        RequireNonnegative(ClassCostWeight, nameof(ClassCostWeight));
        RequireNonnegative(L1CostWeight, nameof(L1CostWeight));
        RequireNonnegative(GIoUCostWeight, nameof(GIoUCostWeight));
        RequireNonnegative(NoObjectWeight, nameof(NoObjectWeight));
        RequireNonnegative(FocalGamma, nameof(FocalGamma));
        RequireNonnegative(MatchingFocalGamma, nameof(MatchingFocalGamma));
        RequireProbability(FocalAlpha, nameof(FocalAlpha));
        RequireProbability(MatchingFocalAlpha, nameof(MatchingFocalAlpha));
    }

    private static void RequireNonnegative(double value, string name)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value < 0)
            throw new ArgumentOutOfRangeException(name, "Loss weights and exponents must be finite and nonnegative.");
    }

    private static void RequireProbability(double value, string name)
    {
        if (double.IsNaN(value) || value < 0 || value > 1)
            throw new ArgumentOutOfRangeException(name, "Focal alpha must be in [0, 1].");
    }
}
