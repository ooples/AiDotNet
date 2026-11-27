namespace AiDotNet.TextToSpeech;

/// <summary>
/// Token and code ids as tensors of the model's numeric type, for the networks and modules that feed ids to an
/// embedding table.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> text-to-speech models turn text and audio into lists of whole numbers (ids), one per
/// token. The layers that look those ids up expect a tensor, so the list is converted in one step here instead of
/// one number at a time in every model.</para>
/// </remarks>
internal static class TtsIdTensors
{
    /// <summary>
    /// A rank-1 tensor <c>[ids.Count]</c> holding each id as a value of <typeparamref name="T"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the model.</typeparam>
    /// <param name="ids">The ids, in order.</param>
    internal static Tensor<T> From<T>(IReadOnlyList<int> ids)
    {
        if (ids is null) throw new ArgumentNullException(nameof(ids));
        return From<T>(ids, new[] { ids.Count });
    }

    /// <summary>
    /// The ids, in row-major order, as a tensor of <paramref name="shape"/> (for example <c>[1, n]</c> for one batch
    /// row, or <c>[n, 1]</c> for a column mask).
    /// </summary>
    /// <typeparam name="T">The numeric type of the model.</typeparam>
    /// <param name="ids">The ids, in row-major order.</param>
    /// <param name="shape">The tensor's shape; its element count must equal <c>ids.Count</c>.</param>
    internal static Tensor<T> From<T>(IReadOnlyList<int> ids, int[] shape)
    {
        if (ids is null) throw new ArgumentNullException(nameof(ids));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        long elements = 1;
        foreach (int dimension in shape)
        {
            if (dimension < 0) throw new ArgumentOutOfRangeException(nameof(shape), "Dimensions must not be negative.");
            elements *= dimension;
        }
        if (elements != ids.Count)
            throw new ArgumentException($"A shape of {elements} elements cannot hold {ids.Count} ids.", nameof(shape));

        int count = ids.Count;
        var dimensions = (int[])shape.Clone();

        // Typed fast paths: the result tensor is the only allocation and is filled in one pass through its span,
        // with no Tensor<int> staging and no Cast copy, so a single id costs what the old element loop did. int ->
        // double is exact, and int -> float rounds once to nearest, the same as the int -> double -> float route of
        // FromDouble.
        if (typeof(T) == typeof(double))
        {
            var widened = new Tensor<double>(dimensions);
            var span = widened.AsWritableSpan();
            for (int i = 0; i < count; i++)
                span[i] = ids[i];
            return (Tensor<T>)(object)widened;
        }

        if (typeof(T) == typeof(float))
        {
            var narrowed = new Tensor<float>(dimensions);
            var span = narrowed.AsWritableSpan();
            for (int i = 0; i < count; i++)
                span[i] = ids[i];
            return (Tensor<T>)(object)narrowed;
        }

        if (count == 0) return new Tensor<T>(dimensions);
        var values = ids as int[] ?? ids.ToArray();
        return new Tensor<int>(values, dimensions).Cast<T>();
    }

    /// <summary>
    /// A grid of codes (for example a codec's <c>[codebooks, frames]</c> output) as a rank-2 tensor of the same layout,
    /// or of the transposed layout when <paramref name="transpose"/> is set (<c>[frames, codebooks]</c> from
    /// <c>[codebooks, frames]</c>).
    /// </summary>
    /// <typeparam name="T">The numeric type of the model.</typeparam>
    /// <param name="codes">The codes.</param>
    /// <param name="transpose">Whether the tensor's rows are the grid's columns.</param>
    internal static Tensor<T> FromGrid<T>(int[,] codes, bool transpose = false)
    {
        if (codes is null) throw new ArgumentNullException(nameof(codes));
        int rows = codes.GetLength(0), columns = codes.GetLength(1);
        var flat = new int[rows * columns];
        if (!transpose)
        {
            Buffer.BlockCopy(codes, 0, flat, 0, flat.Length * sizeof(int));
            return From<T>(flat, new[] { rows, columns });
        }
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++)
                flat[c * rows + r] = codes[r, c];
        return From<T>(flat, new[] { columns, rows });
    }

    /// <summary>
    /// One row of a code grid, <c>codes[row, :]</c>, as a rank-1 tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of the model.</typeparam>
    /// <param name="codes">The codes.</param>
    /// <param name="row">The row to take.</param>
    internal static Tensor<T> FromGridRow<T>(int[,] codes, int row)
    {
        if (codes is null) throw new ArgumentNullException(nameof(codes));
        if (row < 0 || row >= codes.GetLength(0)) throw new ArgumentOutOfRangeException(nameof(row));
        int columns = codes.GetLength(1);
        var flat = new int[columns];
        Buffer.BlockCopy(codes, row * columns * sizeof(int), flat, 0, columns * sizeof(int));
        return From<T>(flat);
    }
}
