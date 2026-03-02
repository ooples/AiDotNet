using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Shuffle preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Shuffles image patches to create a permuted version of the input. This preserves
/// overall color distribution and texture while destroying spatial structure, enabling
/// color/style transfer without structural copying.
/// </para>
/// <para>
/// <b>For Beginners:</b> This cuts your image into small squares and randomly rearranges
/// them, like a jigsaw puzzle that's been mixed up. ControlNet uses this to transfer
/// the colors and textures of your image without copying its exact layout.
/// </para>
/// </remarks>
public class ShufflePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly int _patchSize;
    private readonly int _seed;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.ContentShuffle;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <summary>
    /// Initializes a new shuffle preprocessor.
    /// </summary>
    /// <param name="patchSize">Size of patches to shuffle. Default: 16.</param>
    /// <param name="seed">Random seed for reproducible shuffling. Default: 42.</param>
    public ShufflePreprocessor(int patchSize = 16, int seed = 42)
    {
        _patchSize = patchSize;
        _seed = seed;
    }

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var shape = data.Shape;
        int batch = shape[0];
        int channels = Math.Min(shape[1], 3);
        int height = shape[2];
        int width = shape[3];
        var result = new Tensor<T>(new[] { batch, 3, height, width });

        for (int b = 0; b < batch; b++)
        {
            int patchRows = height / _patchSize;
            int patchCols = width / _patchSize;
            int totalPatches = patchRows * patchCols;

            // Create shuffled indices
            var rng = new Random(_seed + b);
            var indices = new int[totalPatches];
            for (int i = 0; i < totalPatches; i++) indices[i] = i;
            for (int i = totalPatches - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            // Copy shuffled patches
            for (int p = 0; p < totalPatches; p++)
            {
                int srcRow = indices[p] / patchCols;
                int srcCol = indices[p] % patchCols;
                int dstRow = p / patchCols;
                int dstCol = p % patchCols;

                for (int c = 0; c < 3; c++)
                {
                    int srcC = Math.Min(c, channels - 1);
                    for (int ph = 0; ph < _patchSize && dstRow * _patchSize + ph < height; ph++)
                    {
                        for (int pw = 0; pw < _patchSize && dstCol * _patchSize + pw < width; pw++)
                        {
                            int sh = srcRow * _patchSize + ph;
                            int sw = srcCol * _patchSize + pw;
                            int dh = dstRow * _patchSize + ph;
                            int dw = dstCol * _patchSize + pw;

                            if (sh < height && sw < width)
                            {
                                result[b, c, dh, dw] = data[b, srcC, sh, sw];
                            }
                        }
                    }
                }
            }
        }

        return result;
    }
}
