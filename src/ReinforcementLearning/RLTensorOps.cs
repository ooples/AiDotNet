using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning;

/// <summary>
/// Whole-vector engine helpers shared by the reinforcement-learning agents and policies.
/// </summary>
/// <remarks>
/// Every helper keeps the semantics of the scalar loops it replaces: arg-max returns the first
/// index holding the maximum (a strict greater-than scan starting at index 0, so a NaN in slot 0
/// stays selected exactly as before), and stacking/concatenation are pure data movement.
/// </remarks>
internal static class RLTensorOps
{
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>First index of the largest element (strict greater-than scan from index 0).</summary>
    internal static int ArgMax<T>(Vector<T> values)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Length == 0)
        {
            throw new ArgumentException("Cannot compute ArgMax of an empty vector.", nameof(values));
        }

        return Engine.TensorArgMax(values.AsTensor(), 0)[0];
    }

    /// <summary>
    /// First index whose element is strictly greater than zero, or 0 when there is none
    /// (the "decode a one-hot action" scan the tabular agents use).
    /// </summary>
    internal static int FirstPositiveIndex<T>(Vector<T> values, T zero)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Length == 0)
        {
            return 0;
        }

        // The comparison is 1 where values > 0 and 0 elsewhere (NaN compares false), so the first
        // arg-max of the indicator is the first positive slot, and 0 when no slot is positive.
        var positive = Engine.TensorGreaterThan(values.AsTensor(), zero);
        return Engine.TensorArgMax(positive, 0)[0];
    }

    /// <summary>Largest element (first-maximum scan semantics).</summary>
    internal static T Max<T>(Vector<T> values)
    {
        return values[ArgMax(values)];
    }

    /// <summary>Concatenates vectors end to end.</summary>
    internal static Vector<T> Concat<T>(params Vector<T>[] parts)
    {
        if (parts is null)
        {
            throw new ArgumentNullException(nameof(parts));
        }

        var tensors = new Tensor<T>[parts.Length];
        int total = 0;
        for (int p = 0; p < parts.Length; p++)
        {
            tensors[p] = parts[p].AsTensor();
            total += parts[p].Length;
        }

        if (total == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorConcatenate(tensors, 0).ToVector();
    }

    /// <summary>Stacks equal-length vectors as the rows of a [count, length] tensor.</summary>
    internal static Tensor<T> StackRows<T>(IReadOnlyList<Vector<T>> rows)
    {
        if (rows is null)
        {
            throw new ArgumentNullException(nameof(rows));
        }

        var tensors = new Tensor<T>[rows.Count];
        for (int r = 0; r < rows.Count; r++)
        {
            tensors[r] = rows[r].AsTensor();
        }

        return Engine.TensorStack(tensors, 0);
    }

    /// <summary>Packs a list of scalars into a rank-1 tensor of the given shape.</summary>
    internal static Tensor<T> FromList<T>(IReadOnlyList<T> values, params int[] shape)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        var vector = Vector<T>.FromEnumerable(values);
        return shape.Length == 0 ? vector.AsTensor() : new Tensor<T>(shape, vector);
    }
}
