using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Shared detection post-processing for the grounding models: keep the eligible candidates, suppress
/// overlaps with greedy non-maximum suppression, and emit flat <c>(x1, y1, x2, y2, confidence)</c> rows.
/// </summary>
/// <typeparam name="T">The numeric type of the detections.</typeparam>
internal static class GroundingDetections<T>
{
    /// <summary>Values per emitted detection: x1, y1, x2, y2, confidence.</summary>
    internal const int FieldsPerDetection = 5;

    /// <summary>
    /// Takes the first <paramref name="maxDetections"/> eligible candidates in candidate order, then runs greedy
    /// non-maximum suppression in descending-confidence order (Neubeck and Van Gool, 2006; torchvision.ops.nms):
    /// a box is dropped when its IoU with a higher-confidence kept box exceeds <paramref name="iouThreshold"/>.
    /// </summary>
    /// <param name="engine">Engine to run on.</param>
    /// <param name="boxes">Candidate boxes <c>[n, 4]</c> as normalized (x1, y1, x2, y2).</param>
    /// <param name="confidences">Candidate confidences <c>[n]</c>.</param>
    /// <param name="eligible">A <c>[n]</c> mask, nonzero where the candidate may be kept.</param>
    /// <param name="maxDetections">How many eligible candidates to consider.</param>
    /// <param name="iouThreshold">IoU above which the lower-confidence box is suppressed.</param>
    /// <returns>The kept detections, highest confidence first, flattened to <c>[k * 5]</c>; <c>[5]</c> zeros when
    /// nothing is kept.</returns>
    internal static Tensor<T> Select(IEngine engine, Tensor<T> boxes, Tensor<T> confidences, Tensor<T> eligible,
        int maxDetections, double iouThreshold)
    {
        int count = confidences.Length;
        int[] isEligible = engine.TensorNotEquals(eligible.Reshape([count]),
            MathHelper.GetNumericOperations<T>().Zero).Cast<int>().ToArray();
        var candidates = new List<int>();
        for (int i = 0; i < count && candidates.Count < maxDetections; i++)
        {
            if (isEligible[i] != 0)
                candidates.Add(i);
        }

        if (candidates.Count == 0)
            return new Tensor<T>([FieldsPerDetection]);

        var candidateIndex = new Tensor<int>(new[] { candidates.Count }, Vector<int>.FromList(candidates));
        var candidateBoxes = engine.TensorIndexSelect(boxes.Reshape([count, 4]), candidateIndex, 0);
        var candidateConfidences = engine.TensorIndexSelect(confidences.Reshape([count]), candidateIndex, 0);

        // Suppression visits the candidates in descending confidence with ties in candidate order, as torchvision's
        // stable sort does: the engine's NMS ranks by strictly decreasing int-counted ranks, so its own sort cannot
        // reorder tied confidences.
        var orderable = engine.TensorWhere(engine.TensorEquals(candidateConfidences, candidateConfidences), candidateConfidences,
            Tensor<T>.CreateDefault([candidates.Count], MathHelper.GetNumericOperations<T>().MinValue));
        int[] order = AnomalyTensorOps.StableOrder(orderable, descending: true).ToArray();
        int candidateCount = order.Length;
        var rank = new int[candidateCount];
        for (int position = 0; position < candidateCount; position++)
            rank[order[position]] = candidateCount - position;
        var keep = engine.Nms(candidateBoxes, new Tensor<int>([candidateCount], Vector<int>.FromArray(rank)).Cast<T>(), iouThreshold);
        int kept = keep.Length;
        var rows = engine.TensorConcatenate(
            [
                engine.TensorIndexSelect(candidateBoxes, keep, 0),
                engine.TensorIndexSelect(candidateConfidences, keep, 0).Reshape([kept, 1])
            ], 1);
        return rows.Reshape([kept * FieldsPerDetection]);
    }

    /// <summary>
    /// The detections of scored boxes, highest score first: every box whose score is at least
    /// <paramref name="threshold"/> (NaN never is) and that has area, at most <paramref name="maxDetections"/> of the
    /// strongest, after greedy non-maximum suppression at <paramref name="iouThreshold"/>.
    /// </summary>
    /// <param name="engine">Engine to run on.</param>
    /// <param name="boxes">Boxes <c>[n, 4]</c> as normalized (x1, y1, x2, y2).</param>
    /// <param name="scores">Their scores <c>[n]</c>.</param>
    /// <param name="threshold">The least score a detection may have.</param>
    /// <param name="maxDetections">How many of the strongest eligible boxes to consider.</param>
    /// <param name="iouThreshold">IoU above which the weaker of two boxes is suppressed.</param>
    /// <returns>As <see cref="Select"/>: <c>[k * 5]</c> rows of (x1, y1, x2, y2, score), or <c>[5]</c> zeros.</returns>
    internal static Tensor<T> Ranked(IEngine engine, Tensor<T> boxes, Tensor<T> scores, double threshold, int maxDetections,
        double iouThreshold)
    {
        int count = scores.Length;
        if (count == 0 || maxDetections <= 0)
            return new Tensor<T>([FieldsPerDetection]);
        var numOps = MathHelper.GetNumericOperations<T>();
        var flatScores = scores.Reshape([count]);
        // Descending with tied scores in index order (torchvision's nms sorts with stable=true); a NaN score, never
        // eligible, sorts last.
        var orderable = engine.TensorWhere(engine.TensorEquals(flatScores, flatScores), flatScores,
            Tensor<T>.CreateDefault([count], numOps.MinValue));
        var order = AnomalyTensorOps.StableOrder(orderable, descending: true);
        var orderedBoxes = engine.TensorIndexSelect(boxes.Reshape([count, 4]), order, 0);
        var orderedScores = engine.TensorIndexSelect(flatScores, order, 0);
        var eligible = engine.TensorMultiply(
            AtLeast(engine, orderedScores, MathHelper.GetNumericOperations<T>().FromDouble(threshold)), HasArea(engine, orderedBoxes));
        return Select(engine, orderedBoxes, orderedScores, eligible, maxDetections, iouThreshold);
    }

    /// <summary>
    /// L2 norm of each row of a <c>[n, width]</c> matrix, as <c>[n]</c>. Rows are scaled by their largest magnitude
    /// before squaring, so narrow types such as Half cannot overflow.
    /// </summary>
    internal static Tensor<T> RowNorm(IEngine engine, Tensor<T> matrix)
    {
        int rows = matrix.Shape[0];
        int width = matrix.Shape[1];
        var scale = RowScale(engine, matrix);
        var scaled = engine.TensorDivide(matrix, engine.TensorBroadcastTo(scale, [rows, width]));
        var sumSquare = engine.ReduceSum(engine.TensorMultiply(scaled, scaled), [1], false);
        return engine.TensorMultiply(engine.TensorSqrt(sumSquare), scale.Reshape([rows]));
    }

    /// <summary>Each row's largest magnitude as <c>[n, 1]</c>, or 1 for an all-zero row.</summary>
    private static Tensor<T> RowScale(IEngine engine, Tensor<T> matrix)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int rows = matrix.Shape[0];
        var maxAbs = engine.ReduceMax(engine.TensorAbs(matrix), [1], true);
        return engine.TensorWhere(engine.TensorGreaterThan(maxAbs, numOps.Zero), maxAbs,
            Tensor<T>.CreateDefault([rows, 1], numOps.One));
    }

    /// <summary>
    /// A 0/1 mask of <paramref name="values"/> &gt;= <paramref name="threshold"/>; NaN is never at least the
    /// threshold.
    /// </summary>
    internal static Tensor<T> AtLeast(IEngine engine, Tensor<T> values, T threshold)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var notBelow = engine.TensorSubtract(
            Tensor<T>.CreateDefault(values.Shape.ToArray(), numOps.One), engine.TensorLessThan(values, threshold));
        return engine.TensorMultiply(notBelow, engine.TensorEquals(values, values));
    }

    /// <summary>
    /// The 0/1 mask of boxes with positive width and height: x2 &gt; x1 and y2 &gt; y1.
    /// </summary>
    /// <param name="engine">Engine to run on.</param>
    /// <param name="boxes">Boxes <c>[n, 4]</c> as (x1, y1, x2, y2).</param>
    /// <returns>The <c>[n]</c> mask.</returns>
    internal static Tensor<T> HasArea(IEngine engine, Tensor<T> boxes)
    {
        int count = boxes.Length / 4;
        var b = boxes.Reshape([count, 4]);
        var wide = engine.TensorGreaterThan(
            engine.TensorSlice(b, [0, 2], [count, 1]), engine.TensorSlice(b, [0, 0], [count, 1]));
        var tall = engine.TensorGreaterThan(
            engine.TensorSlice(b, [0, 3], [count, 1]), engine.TensorSlice(b, [0, 1], [count, 1]));
        return engine.TensorMultiply(wide, tall).Reshape([count]);
    }

    /// <summary>
    /// Boxes <c>[n, 4]</c> from centres and sizes, clipped to the unit square: (max(0, cx - w/2),
    /// max(0, cy - h/2), min(1, cx + w/2), min(1, cy + h/2)).
    /// </summary>
    internal static Tensor<T> ClippedCornerBoxes(IEngine engine, Tensor<T> cx, Tensor<T> cy, Tensor<T> w, Tensor<T> h)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int count = cx.Length;
        var half = numOps.FromDouble(0.5);
        var halfW = engine.TensorMultiplyScalar(w.Reshape([count, 1]), half);
        var halfH = engine.TensorMultiplyScalar(h.Reshape([count, 1]), half);
        var x = cx.Reshape([count, 1]);
        var y = cy.Reshape([count, 1]);
        return engine.TensorConcatenate(
            [
                engine.TensorClampMin(engine.TensorSubtract(x, halfW), numOps.Zero),
                engine.TensorClampMin(engine.TensorSubtract(y, halfH), numOps.Zero),
                engine.TensorClampMax(engine.TensorAdd(x, halfW), numOps.One),
                engine.TensorClampMax(engine.TensorAdd(y, halfH), numOps.One)
            ], 1);
    }
}
