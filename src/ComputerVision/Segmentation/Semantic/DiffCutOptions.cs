using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the DiffCut semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DiffCut options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. DiffCut uses diffusion model features combined with
/// Normalized Cut graph partitioning for zero-shot semantic segmentation — no training labels needed.
/// </para>
/// </remarks>
public class DiffCutOptions : NeuralNetworkOptions
{
}
