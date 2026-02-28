using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Content shuffle preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Performs content-aware shuffling that rearranges image regions based on
/// similarity rather than random permutation. Groups similar pixels together
/// while destroying spatial layout, preserving texture and color distributions
/// more faithfully than random shuffling.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is like the Shuffle preprocessor but smarter —
/// it groups similar-looking parts of the image together before rearranging.
/// ControlNet uses this to capture the "feel" and colors of your image without
/// copying the exact arrangement.
/// </para>
/// </remarks>
public class ContentShufflePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly int _blockSize;
    private readonly int _seed;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.ContentShuffle;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <summary>
    /// Initializes a new content shuffle preprocessor.
    /// </summary>
    /// <param name="blockSize">Size of blocks to shuffle. Default: 8.</param>
    /// <param name="seed">Random seed for reproducible shuffling. Default: 42.</param>
    public ContentShufflePreprocessor(int blockSize = 8, int seed = 42)
    {
        _blockSize = blockSize;
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
            int blockRows = height / _blockSize;
            int blockCols = width / _blockSize;
            int totalBlocks = blockRows * blockCols;

            // Compute mean intensity per block for content-aware sorting
            var blockMeans = new (double mean, int index)[totalBlocks];
            for (int p = 0; p < totalBlocks; p++)
            {
                int row = p / blockCols;
                int col = p % blockCols;
                double sum = 0;
                int count = 0;

                for (int ph = 0; ph < _blockSize && row * _blockSize + ph < height; ph++)
                {
                    for (int pw = 0; pw < _blockSize && col * _blockSize + pw < width; pw++)
                    {
                        int h = row * _blockSize + ph;
                        int w = col * _blockSize + pw;
                        for (int c = 0; c < channels; c++)
                        {
                            sum += NumOps.ToDouble(data[b, c, h, w]);
                            count++;
                        }
                    }
                }

                blockMeans[p] = (sum / Math.Max(count, 1), p);
            }

            // Sort by intensity then apply local shuffling within similar groups
            Array.Sort(blockMeans, (a, bm) => a.mean.CompareTo(bm.mean));
            var rng = new Random(_seed + b);

            // Local shuffle within groups of similar blocks
            int groupSize = Math.Max(2, totalBlocks / 8);
            for (int g = 0; g < totalBlocks; g += groupSize)
            {
                int end = Math.Min(g + groupSize, totalBlocks);
                for (int i = end - 1; i > g; i--)
                {
                    int j = g + rng.Next(i - g + 1);
                    (blockMeans[i], blockMeans[j]) = (blockMeans[j], blockMeans[i]);
                }
            }

            // Copy blocks to result
            for (int p = 0; p < totalBlocks; p++)
            {
                int srcIdx = blockMeans[p].index;
                int srcRow = srcIdx / blockCols;
                int srcCol = srcIdx % blockCols;
                int dstRow = p / blockCols;
                int dstCol = p % blockCols;

                for (int c = 0; c < 3; c++)
                {
                    int srcC = Math.Min(c, channels - 1);
                    for (int ph = 0; ph < _blockSize && dstRow * _blockSize + ph < height; ph++)
                    {
                        for (int pw = 0; pw < _blockSize && dstCol * _blockSize + pw < width; pw++)
                        {
                            int sh = srcRow * _blockSize + ph;
                            int sw = srcCol * _blockSize + pw;
                            int dh = dstRow * _blockSize + ph;
                            int dw = dstCol * _blockSize + pw;

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
