using System;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// A test stand-in for a frozen CLIP image encoder: the embedding of an image is its per-channel mean colour,
/// placed in the first <c>ImageChannels</c> entries of an <c>EmbeddingDimension</c>-wide vector (the rest zero).
/// </summary>
/// <remarks>
/// Reference tests pair it with a <see cref="DeterministicTextEncoder{T}"/> whose class embeddings are colour axes,
/// so the expected CLIP-side classification of a masked image can be computed by hand: a region that is pure
/// channel <c>c</c> embeds (after L2 normalization) to the unit vector <c>e_c</c>.
/// </remarks>
public sealed class DeterministicVisualEncoder<T> : IVisualEncoder<T>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an encoder for <paramref name="channels"/>-channel images of side <paramref name="imageSize"/>.</summary>
    public DeterministicVisualEncoder(int embeddingDimension = 16, int channels = 3, int imageSize = 8)
    {
        if (embeddingDimension <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDimension));
        if (channels <= 0 || channels > embeddingDimension) throw new ArgumentOutOfRangeException(nameof(channels));
        if (imageSize <= 0) throw new ArgumentOutOfRangeException(nameof(imageSize));
        EmbeddingDimension = embeddingDimension;
        ImageChannels = channels;
        ImageSize = imageSize;
    }

    /// <inheritdoc/>
    public int EmbeddingDimension { get; }

    /// <inheritdoc/>
    public int ImageSize { get; }

    /// <inheritdoc/>
    public int ImageChannels { get; }

    /// <summary>Gets how many images have been encoded, so tests can check the encoder was consulted.</summary>
    public int EncodeCount { get; private set; }

    /// <inheritdoc/>
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (image.Rank != 3 || image.Shape[0] != ImageChannels)
            throw new ArgumentException($"Expected one [{ImageChannels}, H, W] image.", nameof(image));
        EncodeCount++;
        int height = image.Shape[1], width = image.Shape[2];
        var result = new Tensor<T>(new[] { EmbeddingDimension });
        for (int c = 0; c < ImageChannels; c++)
        {
            double sum = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++) sum += _numOps.ToDouble(image[c, y, x]);
            }

            result[c] = _numOps.FromDouble(sum / (height * width));
        }

        return result;
    }
}
