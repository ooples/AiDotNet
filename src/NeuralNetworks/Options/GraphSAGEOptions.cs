using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for <see cref="GraphSAGENetwork{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> GraphSAGE learns by sampling a node's neighbours and aggregating them,
/// which lets it handle graphs too large to hold in memory at once. The values here are the ones
/// it ships with, so you can use the model without configuring anything.
/// </para>
/// </remarks>
public class GraphSAGEOptions : GraphEncoderOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public GraphSAGEOptions()
    {
        AggregatorType = SAGEAggregatorType.Mean;
        NumLayers = 2;
        Normalize = true;
        DropoutRate = 0.0;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public GraphSAGEOptions(GraphSAGEOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        AggregatorType = other.AggregatorType;
        NumLayers = other.NumLayers;
        Normalize = other.Normalize;
        DropoutRate = other.DropoutRate;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets how a node's sampled neighbours are combined.
    /// Default: <see cref="SAGEAggregatorType.Mean"/>.
    /// </summary>
    public SAGEAggregatorType AggregatorType { get; set; }

    /// <summary>
    /// Gets or sets whether each layer's output embeddings are L2-normalised. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Rescales every node's numbers to the same overall magnitude, so
    /// comparisons between nodes reflect direction rather than size.</para>
    /// </remarks>
    public bool Normalize { get; set; }

    /// <summary>
    /// Gets or sets the dropout rate. Default: 0.0, meaning no dropout.
    /// </summary>
    public double DropoutRate { get; set; }

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required dimension is zero or negative.</exception>
    public void Validate() => ValidateCore();
}
