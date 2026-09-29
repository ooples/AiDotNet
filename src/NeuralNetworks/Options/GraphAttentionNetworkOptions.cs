using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the GraphAttentionNetwork.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Graph Attention Network lets each node decide which of its
/// neighbours matter most, instead of treating them all equally. The values here are the ones the
/// paper publishes, so you can use the model without configuring anything.
/// </para>
/// </remarks>
public class GraphAttentionNetworkOptions : GraphEncoderOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public GraphAttentionNetworkOptions()
    {
        NumHeads = 8;
        NumLayers = 2;
        DropoutRate = 0.6;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public GraphAttentionNetworkOptions(GraphAttentionNetworkOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        NumHeads = other.NumHeads;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the number of attention heads. Default: 8.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each head forms its own opinion about which neighbours matter,
    /// and the results are combined — several viewpoints rather than one.</para>
    /// </remarks>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the dropout rate. Default: 0.6, which is unusually high and is the value the
    /// paper uses.
    /// </summary>
    public double DropoutRate { get; set; }

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required dimension is zero or negative.</exception>
    public void Validate()
    {
        ValidateCore();
        Require(NumHeads, nameof(NumHeads));
    }
}
