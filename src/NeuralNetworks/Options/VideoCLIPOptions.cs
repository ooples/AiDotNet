using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the VideoCLIPNeuralNetwork.
/// </summary>
public class VideoCLIPOptions : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="VideoCLIPOptions"/> class carrying
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
    public VideoCLIPOptions()
    {
        NumFrames = 8;
        FrameRate = 1.0;
        EmbeddingDimension = 512;
        MaxSequenceLength = 77;
        ImageSize = 224;
        Channels = 3;
        PatchSize = 16;
        VocabSize = 49408;
        VisionDim = 768;
        TextHiddenDim = 512;
        NumFrameEncoderLayers = 12;
        NumTemporalLayers = 4;
        NumTextLayers = 12;
        NumHeads = 12;
        TemporalAggregation = TemporalAggregationType.TemporalTransformer;
    }


    /// <summary>
    /// Gets or sets the num frames.
    /// </summary>
    public int NumFrames { get; set; }

    /// <summary>
    /// Gets or sets the frame rate.
    /// </summary>
    public double FrameRate { get; set; }

    /// <summary>
    /// Gets or sets the text hidden dim.
    /// </summary>
    public int TextHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the num frame encoder layers.
    /// </summary>
    public int NumFrameEncoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the num temporal layers.
    /// </summary>
    public int NumTemporalLayers { get; set; }

    /// <summary>
    /// Gets or sets the num text layers.
    /// </summary>
    public int NumTextLayers { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(ValidationRequirements.Text | ValidationRequirements.PatchGeometry);
    }

    /// <summary>
    /// Gets or sets the temporal aggregation.
    /// </summary>
    public TemporalAggregationType TemporalAggregation { get; set; }
}
