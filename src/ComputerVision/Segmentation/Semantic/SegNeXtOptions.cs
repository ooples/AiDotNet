using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the SegNeXt semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SegNeXt options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. SegNeXt uses a purely convolutional architecture
/// with multi-scale attention — no transformers needed — making it one of the most efficient
/// semantic segmentation models available.
/// </para>
/// </remarks>
public class SegNeXtOptions : NeuralNetworkOptions
{
}
