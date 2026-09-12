using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the DBNet document model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DBNet finds text in a photograph. Instead of deciding "text or not" with
/// a fixed cut-off, it learns the cut-off for every pixel, which is what lets it separate lines of
/// text that sit close together. The values here are the ones it ships with.
/// </para>
/// </remarks>
public class DBNetOptions : DocumentNeuralNetworkOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    /// <remarks>
    /// <para>
    /// ImageSize and BackboneChannels are declared by DocumentNeuralNetworkOptions and were
    /// previously left unset, with the values living in constructor parameters instead. DBNet
    /// (Liao et al. 2020) detects on a 640-pixel image over a 256-channel backbone.
    /// </para>
    /// </remarks>
    public DBNetOptions()
    {
        ImageSize = 640;
        BackboneChannels = 256;
        InnerChannels = 256;
        ExpandRatio = 1.5;
        ThresholdK = 50;
        MinTextArea = 16;
    }

    /// <summary>
    /// Gets or sets the width of the feature-pyramid inner layers. Default: 256.
    /// </summary>
    public int InnerChannels { get; set; }

    /// <summary>
    /// Gets or sets how far a detected text kernel is grown back to its full outline.
    /// Default: 1.5.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model is trained to find a shrunken version of each text
    /// region, which keeps neighbouring lines apart. This says how much to expand that shrunken
    /// shape afterwards to recover the real one.</para>
    /// </remarks>
    public double ExpandRatio { get; set; }

    /// <summary>
    /// Gets or sets k, the steepness of the differentiable binarization. Default: 50.
    /// </summary>
    /// <remarks>
    /// <para>
    /// DBNet's contribution is that this step is differentiable, so the per-pixel threshold can be
    /// learned. k controls how sharply the approximation switches; the paper uses 50.
    /// </para>
    /// </remarks>
    public double ThresholdK { get; set; }

    /// <summary>
    /// Gets or sets the smallest region, in pixels, that is reported as text. Default: 16.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Anything smaller than this is treated as noise rather than
    /// text, which stops speckles in the image being reported as words.</para>
    /// </remarks>
    public int MinTextArea { get; set; }

    /// <summary>
    /// Gets or sets the learning rate used when the model creates its own optimizer.
    /// Default: 1e-3.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Both constructors previously built a BARE optimizer -- <c>new AdamOptimizer&lt;...&gt;(this)</c>
    /// -- so the model trained at the optimizer's own default and no configured rate could reach
    /// it. 1e-3 IS Adam's default, so behaviour is unchanged; the value is simply visible and
    /// overridable now.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;
}
