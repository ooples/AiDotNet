using AiDotNet.Models.Options;

namespace AiDotNet.PhysicsInformed.Options;

/// <summary>
/// Configuration options for the FourierNeuralOperator.
/// </summary>
public class FourierNeuralOperatorOptions : PhysicsInformedOptions
{

    /// <summary>
    /// Gets or sets modes. Default: <c>16</c>.
    /// </summary>
    public int Modes { get; set; } = 16;

    /// <summary>
    /// Gets or sets num layers. Default: <c>4</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many stacked layers the network has.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets width. Default: <c>64</c>.
    /// </summary>
    public int Width { get; set; } = 64;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(Modes, nameof(Modes));
        Require(NumLayers, nameof(NumLayers));
        Require(Width, nameof(Width));
    }
}
