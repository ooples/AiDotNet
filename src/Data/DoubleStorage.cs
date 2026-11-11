namespace AiDotNet.Data;

/// <summary>
/// Bulk conversion between <typeparamref name="T"/> buffers and the float64 storage that the data loaders,
/// file formats and snapshot writers use. Every conversion is a single
/// <c>Tensor&lt;TIn&gt;.Cast&lt;TOut&gt;()</c> over the whole buffer and every read/write moves the whole
/// buffer as one byte block, so callers never convert or serialize element by element.
/// </summary>
internal static class DoubleStorage
{
    /// <summary>Converts a double buffer to a new <typeparamref name="T"/> array in one bulk cast.</summary>
    internal static T[] ToArray<T>(double[] values)
    {
        if (values.Length == 0) return Array.Empty<T>();
        return new Tensor<double>(values, new[] { values.Length }).Cast<T>().AsSpan().ToArray();
    }

    /// <summary>Converts a double buffer to a <typeparamref name="T"/> tensor of the given shape in one bulk cast.</summary>
    internal static Tensor<T> ToTensor<T>(double[] values, int[] shape)
    {
        return new Tensor<double>(values, shape).Cast<T>();
    }

    /// <summary>Converts a <typeparamref name="T"/> array to doubles in one bulk cast.</summary>
    internal static double[] ToDoubles<T>(T[] values)
    {
        if (values.Length == 0) return Array.Empty<double>();
        return new Tensor<T>(values, new[] { values.Length }).Cast<double>().AsSpan().ToArray();
    }

    /// <summary>Converts a <typeparamref name="T"/> tensor's storage to doubles in one bulk cast.</summary>
    internal static double[] ToDoubles<T>(Tensor<T> values)
    {
        if (values.Length == 0) return Array.Empty<double>();
        // Cast reads the backing storage linearly, so materialize views (offset/strided) first.
        return values.Contiguous().Cast<double>().AsSpan().ToArray();
    }

    /// <summary>Returns <c>0, 1, ..., count - 1</c> as <typeparamref name="T"/> values (one bulk cast).</summary>
    internal static T[] Ramp<T>(int count)
    {
        var ramp = new int[Math.Max(count, 0)];
        for (int i = 0; i < ramp.Length; i++)
            ramp[i] = i;
        if (ramp.Length == 0) return Array.Empty<T>();
        return new Tensor<int>(ramp, new[] { ramp.Length }).Cast<T>().AsSpan().ToArray();
    }

    /// <summary>
    /// Rounds every element to the nearest integer with the engine and returns them as <see cref="int"/>,
    /// matching <c>INumericOperations&lt;T&gt;.ToInt32</c> element by element.
    /// </summary>
    internal static int[] RoundToInt32<T>(Tensor<T> values)
    {
        if (values.Length == 0) return Array.Empty<int>();
        return AiDotNetEngine.Current.TensorRound(values).Cast<int>().AsSpan().ToArray();
    }

    /// <inheritdoc cref="RoundToInt32{T}(Tensor{T})"/>
    internal static int[] RoundToInt32<T>(Vector<T> values)
    {
        if (values.Length == 0) return Array.Empty<int>();
        return RoundToInt32(Tensor<T>.FromVector(values));
    }

    /// <summary>
    /// Reinterprets <paramref name="count"/> doubles stored in machine byte order (as written by
    /// <see cref="BitConverter"/>) starting at <paramref name="byteOffset"/>.
    /// </summary>
    internal static double[] FromNativeBytes(byte[] bytes, int byteOffset, int count)
    {
        var result = new double[count];
        Buffer.BlockCopy(bytes, byteOffset, result, 0, count * sizeof(double));
        return result;
    }

    /// <summary>
    /// Reads <paramref name="count"/> little-endian doubles, matching <see cref="BinaryReader.ReadDouble"/>,
    /// and throws <see cref="EndOfStreamException"/> when the stream ends early.
    /// </summary>
    internal static double[] ReadDoubles(BinaryReader reader, int count)
    {
        // Same exception a negative-length array allocation raised in the per-element readers.
        if (count < 0) throw new OverflowException("Element count must be non-negative.");
        if (count == 0) return Array.Empty<double>();
        int byteCount = checked(count * sizeof(double));
        byte[] bytes = reader.ReadBytes(byteCount);
        if (bytes.Length < byteCount)
            throw new EndOfStreamException("Unable to read beyond the end of the stream.");
        if (!BitConverter.IsLittleEndian)
            ReverseEachDouble(bytes);
        return FromNativeBytes(bytes, 0, count);
    }

    /// <summary>Writes the doubles as little-endian float64, matching <see cref="BinaryWriter.Write(double)"/>.</summary>
    internal static void WriteDoubles(BinaryWriter writer, double[] values)
    {
        if (values.Length == 0) return;
        var bytes = new byte[values.Length * sizeof(double)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        if (!BitConverter.IsLittleEndian)
            ReverseEachDouble(bytes);
        writer.Write(bytes);
    }

    private static void ReverseEachDouble(byte[] bytes)
    {
        for (int i = 0; i + sizeof(double) <= bytes.Length; i += sizeof(double))
            Array.Reverse(bytes, i, sizeof(double));
    }
}
