namespace AiDotNet.Pruning;

/// <summary>
/// Writes pruned weights back into the weights they were computed from, as one block copy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> pruning sets some weights to zero. A mask produces a pruned copy of the weights, and
/// that copy then has to replace the originals in place so the layer that owns them sees the change. Every pruning
/// strategy does this the same way, so it lives here once.</para>
/// </remarks>
internal static class PruningWriteBack
{
    /// <summary>Copies <paramref name="pruned"/> over <paramref name="weights"/> in place.</summary>
    /// <typeparam name="T">The numeric type of the weights.</typeparam>
    /// <param name="pruned">The pruned values.</param>
    /// <param name="weights">The weights to overwrite.</param>
    /// <exception cref="ArgumentException">The two vectors differ in length.</exception>
    internal static void Copy<T>(Vector<T> pruned, Vector<T> weights)
    {
        if (pruned is null) throw new ArgumentNullException(nameof(pruned));
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (pruned.Length != weights.Length)
            throw new ArgumentException(
                $"Pruned vector has {pruned.Length} elements but the weights have {weights.Length}.", nameof(pruned));

        pruned.AsSpan().CopyTo(weights.AsWritableSpan());
    }

    /// <summary>Copies <paramref name="pruned"/> over <paramref name="weights"/> in place.</summary>
    /// <typeparam name="T">The numeric type of the weights.</typeparam>
    /// <param name="pruned">The pruned values.</param>
    /// <param name="weights">The weights to overwrite.</param>
    /// <exception cref="ArgumentException">The two matrices differ in shape.</exception>
    internal static void Copy<T>(Matrix<T> pruned, Matrix<T> weights)
    {
        if (pruned is null) throw new ArgumentNullException(nameof(pruned));
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (pruned.Rows != weights.Rows || pruned.Columns != weights.Columns)
            throw new ArgumentException(
                $"Pruned matrix is {pruned.Rows}x{pruned.Columns} but the weights are {weights.Rows}x{weights.Columns}.",
                nameof(pruned));

        pruned.AsSpan().CopyTo(weights.AsWritableSpan());
    }

    /// <summary>
    /// Copies <paramref name="pruned"/> over <paramref name="weights"/> in place, in flat row-major order, and marks
    /// the weights as changed so a cached device copy is refreshed.
    /// </summary>
    /// <typeparam name="T">The numeric type of the weights.</typeparam>
    /// <param name="pruned">The pruned values.</param>
    /// <param name="weights">The weights to overwrite.</param>
    /// <exception cref="ArgumentException">The two tensors differ in element count.</exception>
    internal static void Copy<T>(Tensor<T> pruned, Tensor<T> weights)
    {
        if (pruned is null) throw new ArgumentNullException(nameof(pruned));
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (pruned.Length != weights.Length)
            throw new ArgumentException(
                $"Pruned tensor has {pruned.Length} elements but the weights have {weights.Length}.", nameof(pruned));
        if (weights.Length == 0)
            return;

        var source = pruned.IsContiguous ? pruned : pruned.Contiguous();
        if (weights.IsContiguous)
        {
            source.AsSpan().CopyTo(weights.AsWritableSpan());
            // A span write does not bump the version the way the indexer does
            weights.IncrementVersion();
            return;
        }

        CopyIntoView(source, weights);
    }

    [AiDotNet.Attributes.ScalarKernel("the destination is a strided view with no contiguous span; each element has "
        + "its own storage position")]
    private static void CopyIntoView<T>(Tensor<T> source, Tensor<T> weights)
    {
        var values = source.AsSpan();
        for (int i = 0; i < values.Length; i++)
        {
            weights.SetFlat(i, values[i]);
        }
    }
}
