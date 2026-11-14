using AiDotNet.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Reductions for Concerto's objectives that accumulate in double when <typeparamref name="T"/> is
/// narrower than float (Half): the engine's ReduceMean and TensorScatterAdd accumulate in
/// <typeparamref name="T"/>, so once a Half running sum passes 2048 times the next term the term
/// rounds away (the sum stalls) and past 65504 it overflows.
/// </summary>
/// <remarks>
/// float and double keep the engine's own reduction, whose accumulator is already wide enough.
/// Other types are cast to double, reduced there and cast back, as the cross-modal projection already
/// does; the per-segment mean stays on the gradient tape through <see cref="WidenedSegmentMean{T}"/>.
/// </remarks>
internal static class ConcertoWideAccumulation<T>
{
    private static AiDotNet.Tensors.Engines.IEngine Engine => AiDotNet.Tensors.Engines.AiDotNetEngine.Current;

    /// <summary>True when <typeparamref name="T"/>'s own accumulator is too narrow for long sums.</summary>
    public static bool AccumulatesInDouble => typeof(T) != typeof(float) && typeof(T) != typeof(double);

    /// <summary>Mean over axis 0 (not differentiated), accumulated in double for narrow types.</summary>
    public static Tensor<T> MeanOverRows(Tensor<T> rows)
    {
        return AccumulatesInDouble
            ? Engine.ReduceMean(rows.Cast<double>(), new[] { 0 }, false).Cast<T>()
            : Engine.ReduceMean(rows, new[] { 0 }, false);
    }

    /// <summary>
    /// Mean of the rows of <paramref name="rows"/> ([N, D]) within each segment: result row s is the mean
    /// of the rows i with <c>segmentOfRow[i] == s</c>, and zero for an empty segment. Differentiable.
    /// </summary>
    /// <param name="rows">Values to pool, shape [N, D].</param>
    /// <param name="segmentOfRow">Segment of each row, shape [N], in [0, segmentCount).</param>
    /// <param name="rowsPerSegment">Rows in each segment, shape [segmentCount].</param>
    /// <param name="segmentCount">Number of segments.</param>
    public static Tensor<T> SegmentMean(
        Tensor<T> rows, Tensor<int> segmentOfRow, Tensor<int> rowsPerSegment, int segmentCount)
    {
        var engine = Engine;
        if (AccumulatesInDouble)
        {
            return new WidenedSegmentMean<T>(segmentOfRow, rowsPerSegment, segmentCount).Apply(rows);
        }

        // Each row carries 1 / (its segment's row count) and a scatter-add sums every segment's share,
        // which is exactly the per-segment mean and stays on the gradient tape.
        int numRows = rows.Shape[0];
        int width = rows.Shape[1];
        var weights = engine.TensorReciprocal(engine.TensorIndexSelect(rowsPerSegment, segmentOfRow, 0).Cast<T>());
        var weighted = engine.TensorMultiply(rows, engine.TensorBroadcastTo(
            engine.Reshape(weights, new[] { numRows, 1 }), new[] { numRows, width }));
        return engine.TensorScatterAdd(new Tensor<T>(new[] { segmentCount, width }), segmentOfRow, weighted, 0);
    }
}

/// <summary>
/// Per-segment mean computed in double and recorded on the <typeparamref name="T"/> gradient tape as one
/// operation. Forward: out[s] = sum_{i in s} x[i] / n_s. Backward: dx[i] = dout[segment(i)] / n_segment(i),
/// a gather that needs no accumulation, scaled in double.
/// </summary>
internal sealed class WidenedSegmentMean<T> : AutogradFunction<T>
{
    private static AiDotNet.Tensors.Engines.IEngine Engine => AiDotNet.Tensors.Engines.AiDotNetEngine.Current;

    private readonly Tensor<int> _segmentOfRow;
    private readonly int _segmentCount;

    // 1 / n_segment(i) per row, shape [N, 1], in double (a Half count overflows past 65504).
    private readonly Tensor<double> _inverseCount;

    public WidenedSegmentMean(Tensor<int> segmentOfRow, Tensor<int> rowsPerSegment, int segmentCount)
    {
        if (segmentOfRow is null) throw new ArgumentNullException(nameof(segmentOfRow));
        if (rowsPerSegment is null) throw new ArgumentNullException(nameof(rowsPerSegment));

        var engine = Engine;
        _segmentOfRow = segmentOfRow;
        _segmentCount = segmentCount;
        _inverseCount = engine.Reshape(
            engine.TensorReciprocal(engine.TensorIndexSelect(rowsPerSegment, segmentOfRow, 0).Cast<double>()),
            new[] { segmentOfRow.Shape[0], 1 });
    }

    public override Tensor<T> Forward(AutogradContext ctx, params Tensor<T>[] inputs)
    {
        var engine = Engine;
        var rows = inputs[0];
        var weighted = engine.TensorMultiply(rows.Cast<double>(), RowWeights(rows.Shape[1]));
        return engine.TensorScatterAdd(
            new Tensor<double>(new[] { _segmentCount, rows.Shape[1] }), _segmentOfRow, weighted, 0).Cast<T>();
    }

    public override Tensor<T>[] Backward(AutogradContext ctx, Tensor<T> gradOutput)
    {
        var engine = Engine;
        var perRow = engine.TensorIndexSelect(gradOutput, _segmentOfRow, 0).Cast<double>();
        return new[] { engine.TensorMultiply(perRow, RowWeights(gradOutput.Shape[1])).Cast<T>() };
    }

    private Tensor<double> RowWeights(int width)
        => Engine.TensorBroadcastTo(_inverseCount, new[] { _inverseCount.Shape[0], width });
}
