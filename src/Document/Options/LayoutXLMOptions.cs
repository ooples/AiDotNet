using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the LayoutXLM document model.
/// </summary>
public class LayoutXLMOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LayoutXLMOptions"/> class carrying
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
    public LayoutXLMOptions()
    {
        NumClasses = 7;
        ImageSize = 224;
        MaxSequenceLength = 512;
        HiddenDim = 768;
        NumLayers = 12;
        NumHeads = 12;
        VocabSize = 250002;
        VisualBackboneChannels = 256;
        NumLanguages = 53;
    }


    /// <summary>
    /// Gets or sets the visual backbone channels.
    /// </summary>
    public int VisualBackboneChannels { get; set; }

    /// <summary>
    /// Gets or sets the num languages.
    /// </summary>
    public int NumLanguages { get; set; }

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
