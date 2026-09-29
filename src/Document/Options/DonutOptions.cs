using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the Donut document model.
/// </summary>
public class DonutOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DonutOptions"/> class carrying
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
    public DonutOptions()
    {
        ImageHeight = 1920;
        ImageWidth = 2560;
        MaxGenerationLength = 768;
        EmbedDim = 128;
        WindowSize = 10;
        PatchSize = 4;
        DecoderHiddenDim = 1024;
        NumDecoderLayers = 4;
        DecoderHeads = 16;
        VocabSize = 57522;
        MlpRatio = 4;
    }


    /// <summary>
    /// Gets or sets the max generation length.
    /// </summary>
    public int MaxGenerationLength { get; set; }

    /// <summary>
    /// Gets or sets the embed dim.
    /// </summary>
    public int EmbedDim { get; set; }

    /// <summary>
    /// Gets or sets the window size.
    /// </summary>
    public int WindowSize { get; set; }

    /// <summary>
    /// Gets or sets the decoder hidden dim.
    /// </summary>
    public int DecoderHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the decoder heads.
    /// </summary>
    public int DecoderHeads { get; set; }

    /// <summary>
    /// Gets or sets the mlp ratio.
    /// </summary>
    public int MlpRatio { get; set; }

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
