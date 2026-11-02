using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the QuantumNeuralNetwork.
/// </summary>
public class QuantumNeuralNetworkOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets how many quantum layers (each followed by a measurement) the default stack has.
    /// Default: <c>2</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Used only when the architecture supplies no layers of its own. A dense layer projects each
    /// measurement back to the hidden width before the next quantum layer; the qubit count comes from
    /// the constructor's <c>numQubits</c>.
    /// </para>
    /// <para><b>For Beginners:</b> More quantum layers make the circuit deeper and more expressive, and
    /// slower to simulate.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;
}
