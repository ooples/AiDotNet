using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the SegFormer semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SegFormer options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. As the library evolves, additional segmentation-specific
/// settings can be added here.
/// </para>
/// </remarks>
public class SegFormerOptions : NeuralNetworkOptions
{
}
