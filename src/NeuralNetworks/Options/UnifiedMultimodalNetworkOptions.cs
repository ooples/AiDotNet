using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the UnifiedMultimodalNetwork.
/// </summary>
public class UnifiedMultimodalNetworkOptions : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="UnifiedMultimodalNetworkOptions"/> class carrying
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
    public UnifiedMultimodalNetworkOptions()
    {
        EmbeddingDimension = 768; // DEFAULT_EMBEDDING_DIM
        MaxSequenceLength = 2048; // DEFAULT_MAX_SEQ_LEN
        NumTransformerLayers = 12; // DEFAULT_NUM_LAYERS
        EmbeddingDimension = 768; // DEFAULT_EMBEDDING_DIM
        MaxSequenceLength = 2048; // DEFAULT_MAX_SEQ_LEN
        NumTransformerLayers = 12; // DEFAULT_NUM_LAYERS
        EmbeddingDimension = 768; // DEFAULT_EMBEDDING_DIM
        MaxSequenceLength = 2048; // DEFAULT_MAX_SEQ_LEN
        NumTransformerLayers = 12; // DEFAULT_NUM_LAYERS
    }


    /// <summary>
    /// Gets or sets the num transformer layers.
    /// </summary>
    public int NumTransformerLayers { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(ValidationRequirements.Text);
        if (NumTransformerLayers < 0)
            throw new ArgumentException($"{GetType().Name}.{nameof(NumTransformerLayers)} must be non-negative.", OptionsParameterName);
    }
}
