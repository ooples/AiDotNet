using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the GraphIsomorphismNetwork.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Graph Isomorphism Network is designed to tell apart graphs that
/// other designs confuse. The values here are the ones the paper publishes, so you can use the
/// model without configuring anything.
/// </para>
/// </remarks>
public class GraphIsomorphismNetworkOptions : GraphEncoderOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public GraphIsomorphismNetworkOptions()
    {
        MlpHiddenDim = 64;
        NumLayers = 5;
        LearnEpsilon = true;
        InitialEpsilon = 0.0;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public GraphIsomorphismNetworkOptions(GraphIsomorphismNetworkOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        MlpHiddenDim = other.MlpHiddenDim;
        NumLayers = other.NumLayers;
        LearnEpsilon = other.LearnEpsilon;
        InitialEpsilon = other.InitialEpsilon;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the width of the small network inside each layer. Default: 64.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Named MlpHiddenDim rather than HiddenDim because that is the name this model publishes,
    /// and because it describes the internal MLP rather than the layer output width.
    /// </para>
    /// </remarks>
    public int MlpHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets whether epsilon is learned rather than held fixed. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Epsilon controls how much weight a node gives its own features
    /// versus its neighbours'. Learning it lets the model decide per layer.</para>
    /// </remarks>
    public bool LearnEpsilon { get; set; }

    /// <summary>
    /// Gets or sets the starting value of epsilon. Default: 0.0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Zero is the paper's initialisation and a legitimate value, so this is deliberately NOT
    /// required by Validate.
    /// </para>
    /// </remarks>
    public double InitialEpsilon { get; set; }

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required dimension is zero or negative.</exception>
    public void Validate()
    {
        ValidateCore();
        Require(MlpHiddenDim, nameof(MlpHiddenDim));
    }
}
