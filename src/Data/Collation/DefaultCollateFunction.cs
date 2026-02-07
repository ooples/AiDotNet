namespace AiDotNet.Data.Collation;

/// <summary>
/// Stacks equal-size tensors into a batch tensor along dimension 0.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// This is the default collation strategy, equivalent to PyTorch's default_collate.
/// All samples must have the same shape. The resulting batch tensor has shape
/// [N, ...sample_shape] where N is the number of samples.
/// </para>
/// </remarks>
public class DefaultCollateFunction<T> : ICollateFunction<Tensor<T>, Tensor<T>>
{
    /// <inheritdoc/>
    public Tensor<T> Collate(IReadOnlyList<Tensor<T>> samples)
    {
        if (samples.Count == 0)
            throw new ArgumentException("Cannot collate an empty sample list.", nameof(samples));

        int[] sampleShape = samples[0].Shape;

        // Verify all samples have the same shape
        for (int i = 1; i < samples.Count; i++)
        {
            int[] shape = samples[i].Shape;
            if (shape.Length != sampleShape.Length)
                throw new ArgumentException(
                    $"Sample {i} has rank {shape.Length} but expected {sampleShape.Length}.");

            for (int d = 0; d < sampleShape.Length; d++)
            {
                if (shape[d] != sampleShape[d])
                    throw new ArgumentException(
                        $"Sample {i} has shape mismatch at dimension {d}: {shape[d]} vs {sampleShape[d]}.");
            }
        }

        // Build batch shape: [N, ...sampleShape]
        int[] batchShape = new int[sampleShape.Length + 1];
        batchShape[0] = samples.Count;
        Array.Copy(sampleShape, 0, batchShape, 1, sampleShape.Length);

        var result = new Tensor<T>(batchShape);

        // Compute elements per sample
        int elementsPerSample = 1;
        for (int d = 0; d < sampleShape.Length; d++)
        {
            elementsPerSample *= sampleShape[d];
        }

        // Copy each sample into the batch
        var resultSpan = result.Data.Span;
        for (int i = 0; i < samples.Count; i++)
        {
            var sampleSpan = samples[i].Data.Span;
            int dstOffset = i * elementsPerSample;
            sampleSpan.Slice(0, elementsPerSample).CopyTo(resultSpan.Slice(dstOffset, elementsPerSample));
        }

        return result;
    }
}
