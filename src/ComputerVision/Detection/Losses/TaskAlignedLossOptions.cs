namespace AiDotNet.ComputerVision.Detection.Losses;

/// <summary>Assignment and loss settings for task-aligned anchor-free YOLO training.</summary>
/// <remarks>
/// <para>
/// Defaults follow the published YOLO recipes. Task-aligned assignment ranks anchors inside each
/// object by t = s^alpha * IoU^beta (Feng et al. 2021, TOOD), with alpha 0.5 and beta 6 as in YOLOv8 and
/// both YOLOv10 heads (Wang et al. 2024, Sec. 3.1). Losses are BCE classification against the
/// normalized alignment target, CIoU box regression and distribution focal loss (Li et al. 2020),
/// weighted by the box/class/DFL gains 7.5/0.5/1.5 (YOLOv9 Table 1; YOLOv10 Table 14). YOLOv10's
/// one-to-one head uses top-1 selection.
/// </para>
/// <para><b>For Beginners:</b> A YOLO model predicts a box and class scores at every grid cell. These
/// settings decide which cells are responsible for each real object and how strongly each part of the
/// prediction is corrected.</para>
/// </remarks>
public sealed class TaskAlignedLossOptions
{
    /// <summary>Anchors selected per object by the one-to-many assignment.</summary>
    /// <remarks><para><b>For Beginners:</b> How many grid cells learn from each object. The YOLOv8 family
    /// uses 10; TOOD used 13.</para></remarks>
    public int TopK { get; set; } = 10;

    /// <summary>Anchors selected per object by YOLOv10's one-to-one head.</summary>
    /// <remarks><para><b>For Beginners:</b> YOLOv10 trains its inference head with exactly one cell per
    /// object, which is what lets it skip non-maximum suppression.</para></remarks>
    public int OneToOneTopK { get; set; } = 1;

    /// <summary>Exponent of the classification score in the alignment metric.</summary>
    /// <remarks><para><b>For Beginners:</b> Larger values favor cells that are already confident.</para></remarks>
    public double Alpha { get; set; } = 0.5;

    /// <summary>Exponent of the IoU in the alignment metric.</summary>
    /// <remarks><para><b>For Beginners:</b> Larger values favor cells whose box already overlaps well.</para></remarks>
    public double Beta { get; set; } = 6.0;

    /// <summary>Weight of the CIoU box loss.</summary>
    /// <remarks><para><b>For Beginners:</b> Larger values push box overlap harder.</para></remarks>
    public double BoxGain { get; set; } = 7.5;

    /// <summary>Weight of the BCE classification loss.</summary>
    /// <remarks><para><b>For Beginners:</b> Larger values push class scores harder.</para></remarks>
    public double ClassGain { get; set; } = 0.5;

    /// <summary>Weight of the distribution focal loss.</summary>
    /// <remarks><para><b>For Beginners:</b> Larger values sharpen the predicted box-edge distributions.</para></remarks>
    public double DflGain { get; set; } = 1.5;

    internal TaskAlignedLossOptions Snapshot()
    {
        var copy = (TaskAlignedLossOptions)MemberwiseClone();
        copy.Validate();
        return copy;
    }

    internal void Validate()
    {
        if (TopK < 1) throw new ArgumentOutOfRangeException(nameof(TopK), "At least one anchor must be selected per object.");
        if (OneToOneTopK < 1) throw new ArgumentOutOfRangeException(nameof(OneToOneTopK), "At least one anchor must be selected per object.");
        RequireNonnegative(Alpha, nameof(Alpha));
        RequireNonnegative(Beta, nameof(Beta));
        RequireNonnegative(BoxGain, nameof(BoxGain));
        RequireNonnegative(ClassGain, nameof(ClassGain));
        RequireNonnegative(DflGain, nameof(DflGain));
    }

    private static void RequireNonnegative(double value, string name)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value < 0)
            throw new ArgumentOutOfRangeException(name, "Exponents and gains must be finite and nonnegative.");
    }
}
