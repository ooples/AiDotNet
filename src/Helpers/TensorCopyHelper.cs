namespace AiDotNet.Helpers;

/// <summary>
/// Helper class for tensor copy operations.
/// </summary>
/// <remarks>
/// <para>
/// TensorCopyHelper provides utility methods for copying data between tensors,
/// particularly for copying individual samples from one tensor to another.
/// </para>
/// <para><b>For Beginners:</b> When working with batched data in tensors,
/// you often need to copy individual samples between tensors. This helper
/// provides optimized methods for this common operation.
/// </para>
/// </remarks>
public static class TensorCopyHelper
{
    /// <summary>
    /// Copies a single sample from one tensor to another.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor elements.</typeparam>
    /// <param name="source">The source tensor to copy from.</param>
    /// <param name="dest">The destination tensor to copy to.</param>
    /// <param name="sourceIndex">The index of the sample in the source tensor (first dimension).</param>
    /// <param name="destIndex">The index where to place the sample in the destination tensor (first dimension).</param>
    /// <remarks>
    /// <para>
    /// This method copies all elements for a single sample from the source tensor to the
    /// destination tensor. For a tensor of shape [N, H, W, C], this copies all H*W*C elements
    /// for sample at sourceIndex to destIndex.
    /// </para>
    /// <para><b>For Beginners:</b> Think of a tensor as a multi-dimensional array where
    /// the first dimension represents individual samples. This method copies one complete
    /// sample (all its data across other dimensions) from one position to another.
    /// </para>
    /// </remarks>
    public static void CopySample<T>(Tensor<T> source, Tensor<T> dest, int sourceIndex, int destIndex)
    {
        // Validate tensor compatibility
        if (source.Shape.Length != dest.Shape.Length)
        {
            throw new ArgumentException(
                $"Source and destination tensors must have the same rank. " +
                $"Source has rank {source.Shape.Length}, destination has rank {dest.Shape.Length}.");
        }

        // Validate shapes match for all dimensions except the first (sample dimension)
        for (int d = 1; d < source.Shape.Length; d++)
        {
            if (source.Shape[d] != dest.Shape[d])
            {
                throw new ArgumentException(
                    $"Source and destination tensors must have matching shapes except for the first dimension. " +
                    $"Source shape: [{string.Join(", ", source.Shape)}], " +
                    $"Destination shape: [{string.Join(", ", dest.Shape)}].");
            }
        }

        // Validate indices are within bounds
        if (sourceIndex < 0 || sourceIndex >= source.Shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(sourceIndex),
                $"Source index {sourceIndex} is out of range. Valid range: 0 to {source.Shape[0] - 1}.");
        }

        if (destIndex < 0 || destIndex >= dest.Shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(destIndex),
                $"Destination index {destIndex} is out of range. Valid range: 0 to {dest.Shape[0] - 1}.");
        }

        // For 1D tensors, just copy the single element
        if (source.Shape.Length == 1)
        {
            dest[destIndex] = source[sourceIndex];
            return;
        }

        // Calculate elements per sample (product of dimensions after the first)
        int elementsPerSample = 1;
        for (int d = 1; d < source.Shape.Length; d++)
        {
            elementsPerSample *= source.Shape[d];
        }

        // Create index arrays for multi-dimensional access
        var sourceIndices = new int[source.Shape.Length];
        var destIndices = new int[dest.Shape.Length];
        sourceIndices[0] = sourceIndex;
        destIndices[0] = destIndex;

        // Copy all elements for this sample
        for (int i = 0; i < elementsPerSample; i++)
        {
            // Convert flat index to multi-dimensional indices
            int remaining = i;
            for (int d = source.Shape.Length - 1; d >= 1; d--)
            {
                sourceIndices[d] = remaining % source.Shape[d];
                destIndices[d] = remaining % dest.Shape[d];
                remaining /= source.Shape[d];
            }

            dest[destIndices] = source[sourceIndices];
        }
    }
}
