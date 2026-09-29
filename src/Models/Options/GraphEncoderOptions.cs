namespace AiDotNet.Models.Options;

/// <summary>
/// Shared hyperparameters for the graph encoder networks — GraphSAGE, the Graph Isomorphism
/// Network and the Graph Attention Network.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These are the message-passing backbones a graph model is built from.
/// Each has its own way of combining a node's neighbours, but they all stack the same kind of
/// layer, so the number of layers is the one setting they share.
/// </para>
/// <para>
/// Deliberately separate from <see cref="GraphModelOptions"/>, which serves the graph TASK heads
/// (node classification, link prediction, graph classification). That base requires
/// <c>HiddenDim</c>, and these encoders do not all have one — GraphSAGE and the attention network
/// have none, and the isomorphism network calls its own <c>MlpHiddenDim</c>. Deriving them from a
/// base that requires a value they lack is precisely what made five earlier family bases throw at
/// their own members' published defaults.
/// </para>
/// </remarks>
public abstract class GraphEncoderOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the number of message-passing layers. Default: per model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many hops out from each node the model looks. Two layers
    /// means a node sees its neighbours and its neighbours' neighbours.</para>
    /// </remarks>
    public int NumLayers { get; set; }

    /// <summary>
    /// Throws if a dimension every graph encoder requires has been left unset.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="NumLayers"/> is zero or negative, which means the derived options
    /// class did not assign its published default.
    /// </exception>
    protected void ValidateCore()
    {
        Require(NumLayers, nameof(NumLayers));
    }
}
