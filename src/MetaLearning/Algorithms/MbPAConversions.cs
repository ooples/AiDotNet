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
                var single = AiDotNetEngine.Current.TensorSlice(
                    Tensor<T>.FromMatrix(matrix), new[] { row, 0 }, new[] { 1, matrix.Columns }).ToMatrix();
                return (TInput)(object)single;
            }

            case Tensor<T> tensor when tensor.Rank >= 2 && tensor.Shape[0] > 0:
            {
                // Keep the trailing dimensions and set the batch dimension to 1, so the embedding
                // network sees the shape it was built for rather than a flattened vector.
                var single = SliceBatchRow(tensor, index % tensor.Shape[0]);
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
                    return vector.GetSubVector(index, 1);
                }
                return vector;
            }

            case Matrix<T> matrix when matrix.Rows > 0:
            {
                return matrix.GetRow(index % matrix.Rows);
            }

            case Tensor<T> tensor when tensor.Rank == 1:
            {
                var all = tensor.ToVector();
                if (tensor.Length > 1 && index < tensor.Length)
                {
                    return all.GetSubVector(index, 1);
                }
                return all.Clone();
            }

            case Tensor<T> tensor when tensor.Rank >= 2 && tensor.Shape[0] > 0:
            {
                return SliceBatchRow(tensor, index % tensor.Shape[0]).ToVector();
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
        if (source is null || source.Length == 0 || length == 0) return new Vector<T>(length);
        return source.Length >= length
            ? source.GetSubVector(0, length)
            : Vector<T>.Concatenate(source.Clone(), new Vector<T>(length - source.Length));
    }

    /// <summary>Row <paramref name="row"/> of a batched tensor, keeping its rank with a batch size of 1.</summary>
    private static Tensor<T> SliceBatchRow(Tensor<T> tensor, int row)
    {
        var start = new int[tensor.Rank];
        var length = tensor.Shape.ToArray();
        start[0] = row;
        length[0] = 1;
        return AiDotNetEngine.Current.TensorSlice(tensor, start, length).Clone();
    }

    /// <summary>Converts a categorical class index or a full distribution to a head target.</summary>
    internal static Vector<T> CategoricalTarget(Vector<T> source, int classes)
    {
        if (source.Length == classes) return source;
        if (source.Length != 1)
            throw new ArgumentException("A categorical target must be one class index or one value per class.", nameof(source));
        double index = Ops.ToDouble(source[0]);
        if (double.IsNaN(index) || double.IsInfinity(index) || index != Math.Floor(index)
            || index < 0 || index >= classes)
            throw new ArgumentOutOfRangeException(nameof(source), "Class indices must be integers in the head's class range.");
        var target = new Vector<T>(classes);
        target[(int)index] = Ops.One;
        return target;
    }
}
