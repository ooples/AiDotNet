namespace AiDotNet.LossFunctions;

/// <summary>
/// Builds the real DFT bases cos(2 pi n k / N) and sin(2 pi n k / N) shared by the spectral losses.
/// </summary>
/// <typeparam name="T">The numeric type of the bases.</typeparam>
internal static class DftBasis<T>
{
    /// <summary>
    /// Upper bound on the elements of one block of sample rows, so the integer and angle intermediates stay
    /// bounded and the two bases are the only full-size tensors.
    /// </summary>
    private const int BlockElements = 1 << 20;

    /// <summary>
    /// The [frameSize, bins] bases with angle[n, k] = 2 pi ((n k) mod N) / N.
    /// </summary>
    /// <remarks>
    /// The product n k and its remainder are computed in 64-bit integers, so they are exact for any frame size
    /// (a float product rounds once it passes 2^24); only the reduced remainder, which is below N, is converted
    /// to <typeparamref name="T"/>. The bases are filled block by block in place.
    /// </remarks>
    /// <param name="frameSize">The DFT length N (number of sample rows).</param>
    /// <param name="bins">The number of frequency bins (columns).</param>
    /// <param name="negateSine">Whether to return -sin, matching the e^(-i 2 pi k n / N) convention.</param>
    internal static (Tensor<T> Cos, Tensor<T> Sin) Build(int frameSize, int bins, bool negateSine)
    {
        // The bases are constants cached by the callers, so they are always built eagerly: under an ambient graph
        // trace the intermediates would otherwise be unrealized lazy buffers when they are cast and copied.
        using var suspension = AiDotNet.Tensors.Engines.Compilation.GraphMode.SuspendRecording();
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var cos = new Tensor<T>(new[] { frameSize, bins });
        var sin = new Tensor<T>(new[] { frameSize, bins });
        if (frameSize == 0 || bins == 0)
        {
            return (cos, sin);
        }

        T scale = ops.FromDouble(2.0 * Math.PI / frameSize);
        Tensor<long> binIndex = engine.TensorArange<long>(0, bins, 1);
        int block = Math.Max(1, Math.Min(frameSize, BlockElements / bins));
        for (int start = 0; start < frameSize; start += block)
        {
            int size = Math.Min(block, frameSize - start);
            Tensor<long> product = engine.TensorOuter(engine.TensorArange<long>(start, start + size, 1), binIndex);
            Tensor<long> wrapped = engine.TensorRemainder(product, engine.TensorFullLike(product, (long)frameSize));
            Tensor<T> angle = engine.TensorMultiplyScalar(wrapped.Cast<T>(), scale);
            Tensor<T> sinBlock = engine.TensorSin(angle);
            if (negateSine)
            {
                sinBlock = engine.TensorNegate(sinBlock);
            }

            engine.TensorCos(angle).Contiguous().CopyTo(cos.AsWritableSpan().Slice(start * bins, size * bins));
            sinBlock.Contiguous().CopyTo(sin.AsWritableSpan().Slice(start * bins, size * bins));
        }

        return (cos, sin);
    }
}
