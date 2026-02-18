using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the EoMT (Encoder-only Mask Transformer) model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> EoMT removes the pixel and transformer decoders used by Mask2Former,
/// placing mask queries directly inside a plain ViT (DINOv2). This yields 4.4x faster inference.
/// Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class EoMTOptions : NeuralNetworkOptions
{
}
