using AiDotNet.Models.Options;

using AiDotNet.Video.Understanding;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the VideoCLIP video understanding model.
/// </summary>
public class VideoCLIPVideoOptions : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="VideoCLIPVideoOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They are the values this
    /// model has always used, moved here from its constructor so they can be seen and
    /// changed in one place.
    /// </para>
    /// <para>
    /// Carried over unchanged. Whether each matches the published paper is verified, and
    /// corrected where it does not, in a later phase of issue #2090.
    /// </para>
    /// </remarks>
    public VideoCLIPVideoOptions()
    {
        NumFrames = 32;
        EmbeddingDim = 512;
        TextMaxLength = 77;
        VocabSize = 49408;
        Temperature = 1.0;
        HiddenDim = 768;
        VocabPath = null;
        MergesPath = null;
    }

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

    /// <summary>
    /// Gets or sets the embedding dim.
    /// </summary>
    public int EmbeddingDim { get; set; }

    /// <summary>
    /// Gets or sets the text max length.
    /// </summary>
    public int TextMaxLength { get; set; }

    /// <summary>
    /// Gets or sets the vocab size.
    /// </summary>
    public int VocabSize { get; set; }

    /// <summary>
    /// Gets or sets the temperature.
    /// </summary>
    public double Temperature { get; set; }

    /// <summary>
    /// Gets or sets the hidden dim.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the vocab path.
    /// </summary>
    public string? VocabPath { get; set; }

    /// <summary>
    /// Gets or sets the merges path.
    /// </summary>
    public string? MergesPath { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
    }
}
