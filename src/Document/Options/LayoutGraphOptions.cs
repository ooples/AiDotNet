using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the LayoutGraph document model.
/// </summary>
public class LayoutGraphOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets the learning rate for gradient descent parameter updates. Default: 0.001.
    /// </summary>
    public double LearningRate { get; set; } = 0.001;
}
