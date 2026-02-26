using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the QueryMeldNet (MQ-Former) model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> QueryMeldNet dynamically melds instance and stuff queries via
/// cross-attention to scale across diverse datasets. Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class QueryMeldNetOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public QueryMeldNetOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public QueryMeldNetOptions(QueryMeldNetOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
