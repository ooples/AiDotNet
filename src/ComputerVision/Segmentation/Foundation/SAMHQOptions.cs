using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the SAM-HQ (High-Quality Segment Anything) model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM-HQ extends SAM with a High-Quality output token for significantly
/// better mask boundaries. Options inherit from NeuralNetworkOptions and provide defaults
/// suitable for most use cases.
/// </para>
/// </remarks>
public class SAMHQOptions : NeuralNetworkOptions
{
}
