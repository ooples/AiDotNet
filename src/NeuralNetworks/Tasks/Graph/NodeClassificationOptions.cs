using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Tasks.Graph;

/// <summary>
/// Configuration options for <see cref="NodeClassificationModel{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Node classification labels each item in a graph — which topic a paper
/// belongs to, say, given the papers it cites. The values here are the ones the model ships with,
/// so you can use it without configuring anything.
/// </para>
/// </remarks>
public class NodeClassificationOptions : GraphModelOptions
{
    /// <summary>
    /// Initializes a new instance carrying the model's published defaults.
    /// </summary>
    public NodeClassificationOptions()
    {
        HiddenDim = 64;
        NumLayers = 2;
        DropoutRate = 0.5;
    }

    /// <summary>
    /// Initializes a new instance by copying another instance.
    /// </summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public NodeClassificationOptions(NodeClassificationOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        HiddenDim = other.HiddenDim;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the number of message-passing layers. Default: 2.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many hops out from each node the model looks. Two layers
    /// means a node sees its neighbours and its neighbours' neighbours. Graph models usually stay
    /// shallow — going deeper tends to blur every node into the same representation.</para>
    /// </remarks>
    public int NumLayers { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
        Require(NumLayers, nameof(NumLayers));
    }
}
