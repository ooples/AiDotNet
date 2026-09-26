namespace AiDotNet.Enums;

/// <summary>The classification objective used by a DETR-family set prediction loss.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> A DETR-style detector predicts a fixed set of candidate objects. After
/// each ground-truth object is matched to one candidate, this setting chooses how the class scores of
/// every candidate are trained.</para>
/// </remarks>
public enum SetPredictionClassificationLoss
{
    /// <summary>
    /// Softmax cross-entropy over the foreground classes plus a final no-object class, with the
    /// no-object class down-weighted (Carion et al. 2020, DETR).
    /// </summary>
    SoftmaxCrossEntropy,

    /// <summary>
    /// Per-class sigmoid focal loss with no no-object class (Lin et al. 2017), as used by Deformable
    /// DETR and DINO (Zhang et al. 2022).
    /// </summary>
    SigmoidFocal,

    /// <summary>
    /// IoU-aware varifocal loss (Zhang et al. 2021): the matched class is trained toward the IoU of
    /// its predicted box, as used by RT-DETR (Zhao et al. 2023).
    /// </summary>
    VariFocal
}
