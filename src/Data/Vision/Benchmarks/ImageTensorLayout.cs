namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Specifies the axis ordering for image tensors returned by vision data loaders.
/// </summary>
public enum ImageTensorLayout
{
    /// <summary>
    /// Channel-last layout: <c>[B, H, W, C]</c>. TensorFlow/Keras convention.
    /// This is the default for AiDotNet vision data loaders.
    /// </summary>
    NHWC,

    /// <summary>
    /// Channel-first layout: <c>[B, C, H, W]</c>. PyTorch convention.
    /// Used by <c>ConvolutionalLayer&lt;T&gt;</c> and other channel-first models.
    /// </summary>
    NCHW,
}
