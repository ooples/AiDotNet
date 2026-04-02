using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Shared utilities for second-order optimizers that operate on the full parameter vector
/// and need to flatten/unflatten between tensor and vector representations.
/// </summary>
internal static class SecondOrderHelper<T>
{
    /// <summary>
    /// Flattens parameter and gradient tensors into contiguous vectors for second-order update.
    /// </summary>
    public static (Vector<T> param, Vector<T> grad, int[] offsets) FlattenTensors(
        Tensor<T>[] parameters, Dictionary<Tensor<T>, Tensor<T>> gradients,
        INumericOperations<T> numOps)
    {
        int total = 0;
        var offsets = new int[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            offsets[i] = total;
            total += parameters[i].Length;
        }

        var pv = new Vector<T>(total);
        var gv = new Vector<T>(total);
        for (int i = 0; i < parameters.Length; i++)
        {
            var p = parameters[i];
            gradients.TryGetValue(p, out var g);
            for (int j = 0; j < p.Length; j++)
            {
                pv[offsets[i] + j] = p[j];
                gv[offsets[i] + j] = g is not null && j < g.Length ? g[j] : numOps.Zero;
            }
        }

        return (pv, gv, offsets);
    }

    /// <summary>
    /// Copies updated parameter values from a flat vector back into the original tensor references.
    /// </summary>
    public static void UnflattenIntoTensors(Vector<T> updated, Tensor<T>[] parameters, int[] offsets)
    {
        for (int i = 0; i < parameters.Length; i++)
        {
            var p = parameters[i];
            for (int j = 0; j < p.Length; j++)
                p[j] = updated[offsets[i] + j];
        }
    }
}
