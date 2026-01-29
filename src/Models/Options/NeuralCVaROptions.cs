namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the NeuralCVaR model.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These options set the size of the neural network that
/// predicts CVaR (expected shortfall). More layers or larger layers increase
/// model capacity but can overfit if you don’t have enough data.
/// </para>
/// </remarks>
public class NeuralCVaROptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Number of hidden layers in the network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each hidden layer adds another step of processing.
    /// More layers allow more complex patterns but slow training.
    /// </para>
    /// </remarks>
    public int HiddenLayers { get; set; } = 2;

    /// <summary>
    /// Size of hidden layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the width of each hidden layer.
    /// Larger values make the model stronger but more expensive.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;
}
