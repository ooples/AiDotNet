namespace AiDotNet.Data.Collation;

/// <summary>
/// Pads variable-length sequences to the maximum length in the batch, then stacks them.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// This collation strategy is essential for NLP and sequence models where inputs have
/// different lengths. Each sample is padded (or truncated) along the sequence dimension
/// to match the longest sample in the batch.
/// </para>
/// <para>
/// Assumes samples are 1D tensors (sequences of token IDs or values). The padding value
/// defaults to zero (common for PAD token in NLP).
/// </para>
/// </remarks>
public class PaddingCollateFunction<T> : ICollateFunction<Tensor<T>, Tensor<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _padValue;
    private readonly int? _maxLength;

    /// <summary>
    /// Creates a new padding collate function.
    /// </summary>
    /// <param name="padValue">The value to use for padding. Default is zero.</param>
    /// <param name="maxLength">Optional maximum sequence length. If null, uses the longest sample in the batch.</param>
    public PaddingCollateFunction(T? padValue = default, int? maxLength = null)
    {
        _padValue = padValue ?? NumOps.Zero;
        _maxLength = maxLength;
    }

    /// <inheritdoc/>
    public Tensor<T> Collate(IReadOnlyList<Tensor<T>> samples)
    {
        if (samples.Count == 0)
            throw new ArgumentException("Cannot collate an empty sample list.", nameof(samples));

        // Find maximum length across all samples
        int maxLen = 0;
        for (int i = 0; i < samples.Count; i++)
        {
            int sampleLen = samples[i].Data.Length;
            if (sampleLen > maxLen) maxLen = sampleLen;
        }

        if (_maxLength.HasValue)
        {
            maxLen = Math.Min(maxLen, _maxLength.Value);
        }

        // Create padded batch tensor [N, maxLen]
        var result = new Tensor<T>(new[] { samples.Count, maxLen });

        // Fill with pad value if non-zero
        var resultSpan = result.Data.Span;
        if (!NumOps.Equals(_padValue, NumOps.Zero))
        {
            for (int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i] = _padValue;
            }
        }

        // Copy each sample (possibly truncated)
        for (int i = 0; i < samples.Count; i++)
        {
            var sampleSpan = samples[i].Data.Span;
            int copyLen = Math.Min(sampleSpan.Length, maxLen);
            int dstOffset = i * maxLen;
            sampleSpan.Slice(0, copyLen).CopyTo(resultSpan.Slice(dstOffset, copyLen));
        }

        return result;
    }
}
