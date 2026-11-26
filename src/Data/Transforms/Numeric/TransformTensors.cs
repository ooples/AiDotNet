using AiDotNet.Tensors;

namespace AiDotNet.Data.Transforms.Numeric;

/// <summary>
/// Bulk tensor builders shared by the numeric transforms so their statistics and scaling run as
/// whole-array engine operations in <typeparamref name="T"/>.
/// </summary>
internal static class TransformTensors
{
    /// <summary>Stacks equally sized rows into a <c>[rows, featureCount]</c> tensor (row block copies).</summary>
    internal static Tensor<T> Rows<T>(T[][] rows, int featureCount)
    {
        var flat = new T[rows.Length * featureCount];
        for (int i = 0; i < rows.Length; i++)
        {
            Array.Copy(rows[i], 0, flat, i * featureCount, featureCount);
        }

        return new Tensor<T>(flat, new[] { rows.Length, featureCount });
    }

    /// <summary>The first <paramref name="length"/> entries of <paramref name="values"/> as a 1-D tensor copy.</summary>
    internal static Tensor<T> Prefix<T>(T[] values, int length)
    {
        var prefix = new T[length];
        Array.Copy(values, prefix, length);
        return new Tensor<T>(prefix, new[] { length });
    }

    /// <summary>The first <paramref name="length"/> entries of a 1-D tensor.</summary>
    internal static Tensor<T> Prefix<T>(Tensor<T> values, int length)
    {
        return length == values.Length ? values : values.Slice(0, 0, length).Contiguous();
    }

    /// <summary>
    /// Copies caller-supplied per-feature statistics into a [length] tensor (a defensive copy, so later
    /// changes to the caller's array do not affect the transform).
    /// </summary>
    internal static Tensor<T> FromStatistics<T>(T[] values)
    {
        return new Tensor<T>((T[])values.Clone(), new[] { values.Length });
    }

    /// <summary>
    /// Mask of the entries of <paramref name="magnitudes"/> (non-negative values) below
    /// <paramref name="threshold"/>. When the threshold underflows <typeparamref name="T"/> to zero (1e-12 in
    /// half precision) the mask still catches exact zeros, which is what the double comparison caught there.
    /// </summary>
    internal static Tensor<Bit> NearZero<T>(Tensor<T> magnitudes, double threshold)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        T limit = numOps.FromDouble(threshold);
        return numOps.GreaterThan(limit, numOps.Zero)
            ? engine.TensorEqScalar(engine.TensorLessThan(magnitudes, limit), numOps.One)
            : engine.TensorEqScalar(magnitudes, numOps.Zero);
    }
}
