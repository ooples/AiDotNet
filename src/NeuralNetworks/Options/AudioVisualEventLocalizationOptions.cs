using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the AudioVisualEventLocalizationNetwork.
/// </summary>
public class AudioVisualEventLocalizationOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets the learning rate for gradient descent parameter updates. Default: 0.001.
    /// </summary>
    public double LearningRate { get; set; } = 0.001;
}
