using AiDotNet.Models.Options;

namespace AiDotNet.PhysicsInformed.Options;

/// <summary>
/// Configuration options for the GraphNeuralOperator.
/// </summary>
public class GraphNeuralOperatorOptions : PhysicsInformedOptions
{

    /// <summary>
    /// Gets or sets hidden dim. Default: <c>64</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How wide each hidden layer is.</para>
    /// </remarks>
    public int HiddenDim { get; set; } = 64;

    /// <summary>
    /// Gets or sets input dim. Default: <c>0</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many features each input node carries. Zero means infer it from the data.</para>
    /// </remarks>
    public int InputDim { get; set; } = 0;

    /// <summary>
    /// Gets or sets normalize adjacency. Default: <c>true</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Whether to normalise the graph connection matrix before use.</para>
    /// </remarks>
    public bool NormalizeAdjacency { get; set; } = true;

    /// <summary>
    /// Gets or sets num layers. Default: <c>4</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many stacked layers the network has.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(HiddenDim, nameof(HiddenDim));
        Require(NumLayers, nameof(NumLayers));
    }
}
