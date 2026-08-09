using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the SAM2 segmentation model.
/// </summary>
public class SAM2Options : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets the focal-loss weight in the mask objective. The default, 20, is the focal:dice
    /// ratio of 20:1 that SAM 2 inherits from SAM (Ravi et al. 2024, §D Training details; Kirillov
    /// et al. 2023, §3).
    /// </summary>
    public double MaskFocalWeight { get; set; } = 20.0;

    /// <summary>
    /// Gets or sets the dice-loss weight in the mask objective. The default, 1, is the paper's
    /// focal:dice ratio of 20:1.
    /// </summary>
    public double MaskDiceWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets focal loss's focusing parameter gamma. The default, 2, is the RetinaNet value
    /// the SAM/SAM 2 mask objective uses.
    /// </summary>
    public double FocalGamma { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets focal loss's class-balance parameter alpha. The default, 0.25, is the RetinaNet
    /// value the SAM/SAM 2 mask objective uses.
    /// </summary>
    public double FocalAlpha { get; set; } = 0.25;
}
