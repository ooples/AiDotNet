using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the TrOCR document model.
/// </summary>
public class TrOCROptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets the learning rate for gradient descent parameter updates. Default: 0.0001.
    /// </summary>
    public double LearningRate { get; set; } = 0.0001;
}
