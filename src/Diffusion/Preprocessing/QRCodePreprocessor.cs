using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// QR code pattern preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Enhances QR code or grid-like binary patterns in images by applying adaptive
/// thresholding and contrast enhancement. The output preserves high-contrast
/// black/white patterns suitable for QR code-conditioned generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This preprocessor cleans up a QR code image so that
/// ControlNet can embed the QR code pattern into generated artwork. The result
/// is a high-contrast black-and-white image that clearly shows the QR pattern.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class QRCodePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly int _blockSize;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.QR;
    /// <inheritdoc />
    public override int OutputChannels => 1;

    /// <summary>
    /// Initializes a new QR code preprocessor.
    /// </summary>
    /// <param name="blockSize">Block size for adaptive thresholding. Default: 11.</param>
    public QRCodePreprocessor(int blockSize = 11)
    {
        _blockSize = blockSize;
    }

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var engine = AiDotNetEngine.Current;

        // Adaptive threshold: a pixel is dark (0) if below the local mean (neighbourhood clipped to the image) - 0.02.
        // The mean, threshold and comparison stay in double and only the binary result is narrowed: a Half mean and
        // threshold round enough to flip pixels within a Half ulp of the threshold.
        var pixel = TensorCopyHelper.CastLogical<T, double>(PreprocessorFilters.Channel(data, 0));
        var localMean = PreprocessorFilters.ClippedBoxMean(pixel, _blockSize / 2);
        var dark = engine.TensorLessThan(pixel, engine.TensorAddScalar(localMean, -0.02));
        return TensorCopyHelper.CastLogical<double, T>(engine.ScalarMinusTensor(1.0, dark));
    }
}
