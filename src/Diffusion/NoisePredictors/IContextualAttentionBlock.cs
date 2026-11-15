namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// A U-Net attention block that takes the cross-attention context into its own forward, rather than being a
/// post-process of the block it wraps (<see cref="IAttentionBlockDecorator{T}"/>). The U-Net hands it the input and the
/// context (null without conditioning) and uses its output as is.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal interface IContextualAttentionBlock<T>
{
    /// <summary>The block's output for <paramref name="input"/> <c>[B, C, H, W]</c> and an optional context.</summary>
    Tensor<T> Forward(Tensor<T> input, Tensor<T>? context);
}
