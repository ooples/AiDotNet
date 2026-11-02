using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the RadialBasisFunctionNetwork.
/// </summary>
public class RadialBasisFunctionNetworkOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets how many radial basis units (centers) the hidden layer has; null derives it as
    /// max(10, (input size + output size) / 2).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Broomhead &amp; Lowe 1988 leave the number of centers to the problem: each center is one prototype
    /// the network measures inputs against. Used only when the architecture supplies no layers of its own.
    /// </para>
    /// <para><b>For Beginners:</b> More centers let the network tell apart more regions of the input
    /// space, at the cost of more parameters.</para>
    /// </remarks>
    public int? NumCenters { get; set; }
}
