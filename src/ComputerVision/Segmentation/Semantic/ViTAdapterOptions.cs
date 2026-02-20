using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the ViT-Adapter semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ViT-Adapter options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. ViT-Adapter enables plain Vision Transformers to handle
/// dense prediction tasks by adding lightweight spatial prior modules, without requiring any
/// vision-specific architectural changes to the base ViT.
/// </para>
/// </remarks>
public class ViTAdapterOptions : NeuralNetworkOptions
{
}
