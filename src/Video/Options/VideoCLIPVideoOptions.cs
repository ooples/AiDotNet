using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the VideoCLIP video understanding model.
/// </summary>
public class VideoCLIPVideoOptions : NeuralNetworkOptions
{
    /// <summary>Gets or sets the learning rate used by the default Adam optimizer.</summary>
    /// <value>
    /// Defaults to 5e-5, the initial learning rate VideoCLIP specifies (Xu et al. 2021,
    /// arXiv:2109.14084, Training Details). This was 1e-4 — twice the paper's rate.
    /// </value>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets Adam's first-moment decay.</summary>
    /// <value>Defaults to 0.9, per the paper's betas of (0.9, 0.98).</value>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets Adam's second-moment decay.</summary>
    /// <value>
    /// Defaults to 0.98, per the paper's betas of (0.9, 0.98). Adam's library default of 0.999 is
    /// NOT what VideoCLIP uses.
    /// </value>
    public double Beta2 { get; set; } = 0.98;

    /// <summary>Gets or sets the gradient-clipping norm.</summary>
    /// <value>Defaults to 2.0 — "Gradients are clipped at 2.0" in the paper's training details.</value>
    public double MaxGradientNorm { get; set; } = 2.0;

    /// <summary>Gets or sets the number of linear warm-up steps before decay begins.</summary>
    /// <value>Defaults to 1000 — "1000 steps of warm-up" in the paper's training details.</value>
    public int WarmupSteps { get; set; } = 1000;

    /// <summary>Gets or sets the total optimizer steps the decay schedule spans.</summary>
    /// <value>
    /// Defaults to 100000. This is the horizon of the paper's polynomial decay and is a property of
    /// the training run (epochs x steps-per-epoch), so set it to match yours.
    /// </value>
    public int TotalTrainingSteps { get; set; } = 100000;

    /// <summary>Gets or sets the exponent of the polynomial decay that follows warm-up.</summary>
    /// <value>
    /// Defaults to 1.0, matching the <c>polynomial_decay</c> schedule VideoCLIP is trained with in
    /// fairseq MMPT, whose own default power is 1.0.
    /// </value>
    public double DecayPower { get; set; } = 1.0;

    /// <summary>Gets or sets the hidden width of the video and text encoders.</summary>
    /// <value>Defaults to 768, matching the paper-scale configuration.</value>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>Gets or sets the number of spatial video encoder blocks.</summary>
    /// <value>Defaults to 12.</value>
    public int NumSpatialBlocks { get; set; } = 12;

    /// <summary>Gets or sets the number of temporal video encoder blocks.</summary>
    /// <value>Defaults to 4.</value>
    public int NumTemporalBlocks { get; set; } = 4;

    /// <summary>Gets or sets the number of text transformer blocks.</summary>
    /// <value>Defaults to 12.</value>
    public int NumTextBlocks { get; set; } = 12;
}
