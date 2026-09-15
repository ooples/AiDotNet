using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the RecurrentNeuralNetwork.
/// </summary>
public class RecurrentNeuralNetworkOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets learning rate. Default: <c>0.01</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big a step the model takes when it corrects itself.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(LearningRate, nameof(LearningRate));
    }
}
