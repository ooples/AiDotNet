using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

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
        var shape = data.Shape;
        int batch = shape[0];
        int height = shape[2];
        int width = shape[3];
        var result = new Tensor<T>(new[] { batch, 1, height, width });
        int halfBlock = _blockSize / 2;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double pixel = NumOps.ToDouble(data[b, 0, h, w]);

                    // Compute local mean for adaptive threshold
                    double sum = 0;
                    int count = 0;
                    int hMin = Math.Max(0, h - halfBlock);
                    int hMax = Math.Min(height - 1, h + halfBlock);
                    int wMin = Math.Max(0, w - halfBlock);
                    int wMax = Math.Min(width - 1, w + halfBlock);

                    for (int kh = hMin; kh <= hMax; kh++)
                    {
                        for (int kw = wMin; kw <= wMax; kw++)
                        {
                            sum += NumOps.ToDouble(data[b, 0, kh, kw]);
                            count++;
                        }
                    }

                    double localMean = sum / count;

                    // Adaptive threshold: pixel is dark if below local mean
                    result[b, 0, h, w] = pixel < localMean - 0.02
                        ? NumOps.Zero
                        : NumOps.One;
                }
            }
        }

        return result;
    }
}
