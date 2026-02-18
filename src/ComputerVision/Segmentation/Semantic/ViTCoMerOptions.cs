using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the ViT-CoMer semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ViT-CoMer options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. ViT-CoMer is a hybrid model that runs CNN and transformer
/// branches in parallel and fuses them to get excellent boundary quality in segmentation.
/// </para>
/// </remarks>
public class ViTCoMerOptions : NeuralNetworkOptions
{
}
