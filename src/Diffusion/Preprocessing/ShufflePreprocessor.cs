using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
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
        var engine = AiDotNetEngine.Current;
        var shape = data._shape;
        int batch = shape[0];
        var rgb = PreprocessorFilters.ToRgb(data);
        var images = new Tensor<T>[batch];

        for (int b = 0; b < batch; b++)
        {
            int patchRows = shape[2] / _patchSize;
            int patchCols = shape[3] / _patchSize;
            int totalPatches = patchRows * patchCols;

            // Create shuffled indices
            var rng = RandomHelper.CreateSeededRandom(_seed + b);
            var indices = new int[totalPatches];
            for (int i = 0; i < totalPatches; i++) indices[i] = i;
            for (int i = totalPatches - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            // Copy shuffled patches
            images[b] = PreprocessorFilters.PermutePatches(engine.TensorNarrow(rgb, 0, b, 1), _patchSize, indices);
        }

        return engine.TensorConcatenate(images, 0);
    }
}
