using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the Segment Anything Model (SAM).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM is Meta AI's foundation model for image segmentation.
/// Options inherit from NeuralNetworkOptions and can be extended with SAM-specific settings.
/// </para>
/// </remarks>
public class SAMOptions : NeuralNetworkOptions
{
}
