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
    /// Gets or sets the width of each recurrent layer in the default stack; null derives it as
    /// max(64, input size, output size).
    /// </summary>
    /// <remarks>
    /// <para>Used only when the architecture supplies no layers of its own.</para>
    /// <para><b>For Beginners:</b> How many numbers the network keeps as its memory of the sequence so far.</para>
    /// </remarks>
    public int? HiddenSize { get; set; }

    /// <summary>
    /// Gets or sets how many Elman recurrent layers the default stack has. Default: <c>2</c>.
    /// </summary>
    /// <remarks>
    /// <para>Used only when the architecture supplies no layers of its own.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(LearningRate, nameof(LearningRate));
        if (HiddenSize is int hidden) Require(hidden, nameof(HiddenSize));
        Require(NumLayers, nameof(NumLayers));
    }
}
