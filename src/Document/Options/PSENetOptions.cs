using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the PSENet document model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> PSENet finds text in a photograph, including text that is curved or at
/// an angle. It predicts several nested outlines of each text region at different sizes and then
/// grows the smallest one outwards, which is how it separates words that touch. The values here
/// are the ones it ships with.
/// </para>
/// </remarks>
public class PSENetOptions : DocumentNeuralNetworkOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    /// <remarks>
    /// <para>
    /// ImageSize and BackboneChannels are declared by DocumentNeuralNetworkOptions and were
    /// previously left unset, with the values living in constructor parameters instead. PSENet
    /// (Wang et al. 2019) detects on a 640-pixel image over a 256-channel ResNet/FPN backbone.
    /// </para>
    /// </remarks>
    public PSENetOptions()
    {
        ImageSize = 640;
        BackboneChannels = 256;
        FeatureChannels = 256;
        NumKernels = 7;
    }

    /// <summary>
    /// Gets or sets the width of the feature-pyramid outputs. Default: 256.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Distinct from <see cref="DocumentNeuralNetworkOptions.BackboneChannels"/>: that is the
    /// backbone's width, this is the width the pyramid levels are projected to before the
    /// kernel-prediction heads.
    /// </para>
    /// </remarks>
    public int FeatureChannels { get; set; }

    /// <summary>
    /// Gets or sets how many nested text kernels the model predicts. Default: 7.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model outlines each piece of text several times at
    /// shrinking sizes. More outlines separates crowded text better, at the cost of more work.
    /// </para>
    /// </remarks>
    public int NumKernels { get; set; }

    /// <summary>
    /// Gets or sets the learning rate used when the model creates its own optimizer.
    /// Default: 1e-4.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This was the literal 1e-4 in both constructors, with a comment explaining that the
    /// detector's multi-million-parameter ResNet/FPN stack needs a bounded fine-tuning step and
    /// that Adam's generic 1e-3 first step overshoots the BCE-with-logits objective. That
    /// reasoning is preserved; the value is now settable rather than baked in.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-4;
}
