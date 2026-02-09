namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the NeuralVaR model.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These options set the size and configuration of the neural
/// network that predicts Value-at-Risk (the worst-case loss estimate).
/// </para>
/// </remarks>
public class NeuralVaROptions<T> : RiskModelOptions<T>
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

    /// <summary>
    /// Validates the NeuralVaR options.
    /// </summary>
    public void Validate()
    {
        if (ConfidenceLevel <= 0 || ConfidenceLevel >= 1)
            throw new ArgumentException("ConfidenceLevel must be between 0 and 1.", nameof(ConfidenceLevel));
        if (TimeHorizon < 1)
            throw new ArgumentException("TimeHorizon must be at least 1.", nameof(TimeHorizon));
        if (NumFeatures < 1)
            throw new ArgumentException("NumFeatures must be at least 1.", nameof(NumFeatures));
        if (HiddenLayers < 1)
            throw new ArgumentException("HiddenLayers must be at least 1.", nameof(HiddenLayers));
        if (HiddenDimension < 1)
            throw new ArgumentException("HiddenDimension must be at least 1.", nameof(HiddenDimension));
    }
}
