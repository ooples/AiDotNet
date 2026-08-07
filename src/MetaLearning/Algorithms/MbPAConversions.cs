using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Per-example extraction from the generic batched <c>TInput</c>/<c>TOutput</c> the meta-learning
/// contract uses.
/// </summary>
/// <remarks>
/// MbPA's memory is keyed per EXAMPLE, so a batch has to be taken apart before it can be written.
/// The supported shapes match the rest of this library's meta-learning algorithms: a
/// <see cref="Matrix{T}"/> of rows, a <see cref="Tensor{T}"/> batched on axis 0, or a single
/// <see cref="Vector{T}"/>.
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class MbPAConversions<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Number of examples in a batch.</summary>
    /// <remarks>
    /// RANK 2 IS THE BATCHED CASE, NOT RANK 1. This returned <c>Shape[0]</c> for any rank &gt;= 1, so
    /// a rank-1 tensor of length N reported N examples -- while <see cref="SliceExample"/> has no
    /// rank-1 branch and returns the WHOLE tensor for every index. MbPAAlgorithm.EmbedBatch then
    /// embedded the same tensor N times and wrote N IDENTICAL keys into the episodic memory, so
    /// k-nearest-neighbour retrieval returned k copies of one neighbour and the local adaptation was
    /// computed from a single example presented k times.
    /// </remarks>
    /// <remarks>
    /// The two are aligned on rank 1 meaning ONE example, which is what SliceExample already assumed
    /// and what <see cref="Vector{T}"/> maps to. The alternative -- N scalar examples -- would feed
    /// the embedding network single scalars where it expects a feature vector.
    /// </remarks>
    internal static int GetBatchSize(object? input) => input switch
    {
        Matrix<T> matrix => matrix.Rows,
        Tensor<T> tensor when tensor.Rank >= 2 => tensor.Shape[0],
        Tensor<T> => 1,
        Vector<T> => 1,
        _ => 1,
    };

    /// <summary>
    /// Extracts example <paramref name="index"/> as a standalone input of the same static type, so
    /// it can be fed to the embedding network on its own.
    /// </summary>
    internal static TInput SliceExample<TInput>(TInput input, int index)
    {
        switch (input)
        {
            case Matrix<T> matrix when matrix.Rows > 0:
            {
                int row = index % matrix.Rows;
                var single = new Matrix<T>(1, matrix.Columns);
                for (int j = 0; j < matrix.Columns; j++) single[0, j] = matrix[row, j];
                return (TInput)(object)single;
            }

            case Tensor<T> tensor when tensor.Rank >= 2 && tensor.Shape[0] > 0:
            {
                int batch = tensor.Shape[0];
                int stride = tensor.Length / batch;
                int offset = (index % batch) * stride;

                // Keep the trailing dimensions and set the batch dimension to 1, so the embedding
                // network sees the shape it was built for rather than a flattened vector.
                var shape = tensor.Shape.ToArray();
                shape[0] = 1;
                var single = new Tensor<T>(shape);
                for (int i = 0; i < stride && offset + i < tensor.Length; i++) single[i] = tensor[offset + i];
                return (TInput)(object)single;
            }

            default:
                // A rank-1 tensor or a bare vector is already a single example.
                return input;
        }
    }

    /// <summary>
    /// Extracts row <paramref name="index"/> of a batched target as a vector.
    /// </summary>
    internal static Vector<T>? SliceTargetRow<TOutput>(TOutput output, int index)
    {
        switch (output)
        {
            case Vector<T> vector:
            {
                // A vector of targets is one scalar target per example, not one multi-valued target
                // — the batched case the meta-learning tasks in this library actually produce.
                if (vector.Length > 1 && index < vector.Length)
                {
                    var scalar = new Vector<T>(1);
                    scalar[0] = vector[index];
                    return scalar;
                }
                return vector;
            }

            case Matrix<T> matrix when matrix.Rows > 0:
            {
                int row = index % matrix.Rows;
                var result = new Vector<T>(matrix.Columns);
                for (int j = 0; j < matrix.Columns; j++) result[j] = matrix[row, j];
                return result;
            }

            case Tensor<T> tensor when tensor.Rank == 1:
            {
                if (tensor.Length > 1 && index < tensor.Length)
                {
                    var scalar = new Vector<T>(1);
                    scalar[0] = tensor[index];
                    return scalar;
                }
                var all = new Vector<T>(tensor.Length);
                for (int j = 0; j < tensor.Length; j++) all[j] = tensor[j];
                return all;
            }

            case Tensor<T> tensor when tensor.Rank >= 2 && tensor.Shape[0] > 0:
            {
                int batch = tensor.Shape[0];
                int stride = tensor.Length / batch;
                int offset = (index % batch) * stride;
                var result = new Vector<T>(stride);
                for (int j = 0; j < stride && offset + j < tensor.Length; j++) result[j] = tensor[offset + j];
                return result;
            }

            default:
                return null;
        }
    }

    /// <summary>
    /// Copies (truncating or zero-padding) into a vector of exactly <paramref name="length"/>
    /// entries, so keys and values always match the configured widths.
    /// </summary>
    internal static Vector<T> ResizeTo(Vector<T>? source, int length)
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++) result[i] = Ops.Zero;
        if (source is null) return result;
        int copy = Math.Min(length, source.Length);
        for (int i = 0; i < copy; i++) result[i] = source[i];
        return result;
    }
}
