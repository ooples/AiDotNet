using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the LayoutLMv2 document model.
/// </summary>
public class LayoutLMv2Options : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LayoutLMv2Options"/> class carrying
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
    public LayoutLMv2Options()
    {
        NumClasses = 7;
        ImageSize = 224;
        MaxSequenceLength = 512;
        HiddenDim = 768;
        NumLayers = 12;
        NumHeads = 12;
        VocabSize = 30522;
        VisualBackboneChannels = 256;
    }


    /// <summary>
    /// Gets or sets the learning rate. Default 2e-5 — LayoutLMv2 Appendix B trains with Adam at this rate.
    /// It is a FINE-TUNING rate for an already-pretrained backbone, so training a randomly initialized model
    /// from scratch needs a larger one. Previously hardcoded in CreatePaperDefaultOptimizer, which left callers
    /// no way to change it without supplying an entire optimizer. (#1789)
    /// </summary>
    public double LearningRate { get; set; } = 2e-5;

    /// <summary>
    /// Gets or sets the decoupled weight decay. Default 1e-2, per LayoutLMv2 Appendix B. Previously hardcoded.
    /// </summary>
    public double WeightDecay { get; set; } = 1e-2;

    /// <summary>
    /// Gets or sets the visual backbone channels.
    /// </summary>
    public int VisualBackboneChannels { get; set; }

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
