using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
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
        var engine = AiDotNetEngine.Current;
        var shape = data._shape;
        int batch = shape[0];
        int channels = Math.Min(shape[1], 3);
        var rgb = PreprocessorFilters.ToRgb(data);
        var images = new Tensor<T>[batch];

        for (int b = 0; b < batch; b++)
        {
            var image = engine.TensorNarrow(data, 0, b, 1);
            int blockRows = shape[2] / _blockSize;
            int blockCols = shape[3] / _blockSize;
            int totalBlocks = blockRows * blockCols;

            // Sort blocks by mean intensity (stable for ties) then apply local shuffling within similar groups
            var order = totalBlocks > 0
                ? PreprocessorFilters.StableArgsort(PreprocessorFilters.PatchMeans(image, _blockSize, channels))
                : new int[0];
            var rng = RandomHelper.CreateSeededRandom(_seed + b);

            // Local shuffle within groups of similar blocks
            int groupSize = Math.Max(2, totalBlocks / 8);
            for (int g = 0; g < totalBlocks; g += groupSize)
            {
                int end = Math.Min(g + groupSize, totalBlocks);
                for (int i = end - 1; i > g; i--)
                {
                    int j = g + rng.Next(i - g + 1);
                    (order[i], order[j]) = (order[j], order[i]);
                }
            }

            // Copy blocks to result
            images[b] = PreprocessorFilters.PermutePatches(engine.TensorNarrow(rgb, 0, b, 1), _blockSize, order);
        }

        return engine.TensorConcatenate(images, 0);
    }
}
