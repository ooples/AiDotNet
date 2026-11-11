namespace AiDotNet.Data.Transforms.Numeric;

/// <summary>
/// Bulk buffer conversions shared by the numeric transforms so their statistics and scaling run as
/// whole-array engine operations in double precision.
/// </summary>
internal static class TransformTensors
{
    /// <summary>
    /// Stacks equally sized rows into a <c>[rows, featureCount]</c> double tensor (row copies, one bulk cast).
    /// </summary>
    internal static Tensor<double> RowsAsDouble<T>(T[][] rows, int featureCount)
    {
        var flat = new T[rows.Length * featureCount];
        for (int i = 0; i < rows.Length; i++)
        {
            Array.Copy(rows[i], 0, flat, i * featureCount, featureCount);
        }

        return new Tensor<double>(DoubleStorage.ToDoubles(flat), new[] { rows.Length, featureCount });
    }

    /// <summary>Returns the first <paramref name="length"/> entries converted to double (one bulk cast).</summary>
    internal static double[] PrefixAsDouble<T>(T[] values, int length)
    {
        return DoubleStorage.ToDoubles(length == values.Length ? values : Prefix(values, length));
    }

    /// <summary>Returns the first <paramref name="length"/> entries (the array itself when it already fits).</summary>
    internal static TValue[] Prefix<TValue>(TValue[] values, int length)
    {
        if (length == values.Length) return values;
        var prefix = new TValue[length];
        Array.Copy(values, prefix, length);
        return prefix;
    }
}
