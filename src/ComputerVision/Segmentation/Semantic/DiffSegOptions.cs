using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the DiffSeg semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DiffSeg options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. DiffSeg produces unsupervised segmentation by merging
/// self-attention maps from a diffusion model, requiring no training labels at all.
/// </para>
/// </remarks>
public class DiffSegOptions : NeuralNetworkOptions
{
}
