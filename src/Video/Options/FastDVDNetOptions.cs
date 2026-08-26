using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the FastDVDNet video model.
/// </summary>
public class FastDVDNetOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initial Adam learning rate. The released FastDVDnet training recipe uses 1e-3.
    /// </summary>
    public double LearningRate { get; set; } = 1e-3;
}
