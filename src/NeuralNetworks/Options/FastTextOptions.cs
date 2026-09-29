using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the FastText model.
/// </summary>
public class FastTextOptions : EmbeddingModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FastTextOptions"/> class carrying
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
    public FastTextOptions()
    {
        VocabSize = 10000;
        BucketSize = 2000000;
        EmbeddingDimension = 100;
        MaxSequenceLength = 512;
        MaxGradNorm = 1.0;
    }


    /// <summary>
    /// Gets or sets the bucket size.
    /// </summary>
    public int BucketSize { get; set; }

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
