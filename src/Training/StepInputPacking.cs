using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Packs a training step's several input tensors into the single <c>[1, N]</c> row a fused step takes, and reads
/// them back inside the forward with engine ops.
/// </summary>
/// <remarks>
/// A compiled plan rebinds only its input and target on each replay; any other host value a forward or loss reads
/// is baked in at trace time. A step whose forward reads two inputs (a branch and a trunk, a clip and its mask)
/// packs them into the input so every replay reads that step's own data.
/// </remarks>
internal static class StepInputPacking<T>
{
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Flattens <paramref name="parts"/>, in order, into one <c>[1, N]</c> row.</summary>
    public static Tensor<T> Pack(params Tensor<T>[] parts)
    {
        int total = 0;
        foreach (var part in parts) total += part.Length;
        var data = new T[total];
        int offset = 0;
        foreach (var part in parts)
        {
            // A view (a slice, a shifted window) is materialized first: only a contiguous tensor has one span.
            part.Contiguous().AsSpan().CopyTo(data.AsSpan(offset, part.Length));
            offset += part.Length;
        }
        return new Tensor<T>(new[] { 1, total }, new Vector<T>(data));
    }

    /// <summary>The part of <paramref name="packed"/> starting at <paramref name="offset"/>, as <paramref name="shape"/>.</summary>
    public static Tensor<T> Slice(Tensor<T> packed, int offset, int[] shape)
    {
        int length = 1;
        foreach (var dim in shape) length *= dim;
        return Engine.Reshape(Engine.TensorNarrow(packed, 1, offset, length), shape);
    }
}