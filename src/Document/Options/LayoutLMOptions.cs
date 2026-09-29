using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the LayoutLM document model.
/// </summary>
public class LayoutLMOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LayoutLMOptions"/> class carrying
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
    public LayoutLMOptions()
    {
        NumClasses = 7;
        MaxSequenceLength = 512;
        HiddenDim = 768;
        NumLayers = 12;
        NumHeads = 12;
        VocabSize = 30522;
        MaxPosition2D = 1024;
    }


    /// <summary>
    /// Gets or sets the max position2 d.
    /// </summary>
    public int MaxPosition2D { get; set; }

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
