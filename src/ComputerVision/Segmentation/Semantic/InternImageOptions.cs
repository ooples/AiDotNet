using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the InternImage semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> InternImage options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. InternImage is a large-scale CNN that uses Deformable
/// Convolution v3 (DCNv3) to compete with Vision Transformers on dense prediction tasks.
/// </para>
/// </remarks>
public class InternImageOptions : NeuralNetworkOptions
{
}
