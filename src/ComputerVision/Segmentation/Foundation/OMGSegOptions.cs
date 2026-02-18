using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the OMG-Seg model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> OMG-Seg handles 10+ segmentation tasks with one model using only 70M
/// trainable parameters. Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class OMGSegOptions : NeuralNetworkOptions
{
}
