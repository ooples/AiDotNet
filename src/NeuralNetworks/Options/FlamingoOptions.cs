using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the FlamingoNeuralNetwork.
/// </summary>
public class FlamingoOptions : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FlamingoOptions"/> class carrying
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
    public FlamingoOptions()
    {
        EmbeddingDimension = 768;
        MaxSequenceLength = 2048;
        ImageSize = 224;
        NumPerceiverTokens = 64;
        MaxImagesInContext = 5;
        Channels = 3;
        VisionHiddenDim = 1024;
        LmHiddenDim = 2048;
        NumVisionLayers = 24;
        NumLmLayers = 32;
        NumHeads = 16;
        VocabSize = 32000;
        NumPerceiverLayers = 6;
        LearningRate = 1e-3;
        LanguageModelBackbone = LanguageModelBackbone.Chinchilla;
    }


    /// <summary>
    /// Gets or sets the num perceiver tokens.
    /// </summary>
    public int NumPerceiverTokens { get; set; }

    /// <summary>
    /// Gets or sets the max images in context.
    /// </summary>
    public int MaxImagesInContext { get; set; }

    /// <summary>
    /// Gets or sets the lm hidden dim.
    /// </summary>
    public int LmHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the num lm layers.
    /// </summary>
    public int NumLmLayers { get; set; }

    /// <summary>
    /// Gets or sets the num perceiver layers.
    /// </summary>
    public int NumPerceiverLayers { get; set; }

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    public double LearningRate { get; set; }

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

    /// <summary>
    /// Gets or sets the language model backbone.
    /// </summary>
    public LanguageModelBackbone LanguageModelBackbone { get; set; }
}
