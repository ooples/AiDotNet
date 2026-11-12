using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Solvers.Assignment;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Tracking;

/// <summary>
/// Whole-tensor helpers the trackers share: box and score tensors, IoU matrices and the
/// assignment variants used by SORT, DeepSORT and ByteTrack.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal static class TrackingAssociation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Packs boxes as an <c>[N, 4]</c> (x1, y1, x2, y2) tensor.</summary>
    internal static Tensor<T> BoxesToTensor(IReadOnlyCollection<BoundingBox<T>> boxes)
    {
        if (boxes.Count == 0)
            return new Tensor<T>(new[] { 0, 4 });

        var flat = boxes.SelectMany(b => new[] { b.X1, b.Y1, b.X2, b.Y2 }).ToArray();
        return new Tensor<T>(flat, new[] { boxes.Count, 4 });
    }

    /// <summary>
    /// (x1, y1, x2, y2) rows to (centre x, centre y, width / height, height) rows, <c>[N, 4]</c>.
    /// </summary>
    internal static Tensor<T> BoxesToXyah(Tensor<T> boxes)
    {
        var engine = AiDotNetEngine.Current;
        int n = boxes.Shape[0];
        var x1 = engine.TensorSlice(boxes, new[] { 0, 0 }, new[] { n, 1 });
        var y1 = engine.TensorSlice(boxes, new[] { 0, 1 }, new[] { n, 1 });
        var x2 = engine.TensorSlice(boxes, new[] { 0, 2 }, new[] { n, 1 });
        var y2 = engine.TensorSlice(boxes, new[] { 0, 3 }, new[] { n, 1 });
        T half = NumOps.FromDouble(0.5);
        var w = engine.TensorSubtract(x2, x1);
        var h = engine.TensorSubtract(y2, y1);
        var cx = engine.TensorAdd(x1, engine.TensorMultiplyScalar(w, half));
        var cy = engine.TensorAdd(y1, engine.TensorMultiplyScalar(h, half));
        return engine.TensorConcatenate(new[] { cx, cy, engine.TensorDivide(w, h), h }, 1);
    }

    /// <summary>Packs detection confidences as an <c>[N]</c> tensor.</summary>
    internal static Tensor<T> Scores(IReadOnlyCollection<Detection<T>> detections)
    {
        return new Tensor<T>(detections.Select(d => d.Confidence).ToArray(), new[] { detections.Count });
    }

    /// <summary>Pairwise IoU <c>[N, M]</c> between two (x1, y1, x2, y2) box tensors.</summary>
    internal static Tensor<T> IouMatrix(Tensor<T> boxesA, Tensor<T> boxesB)
    {
        int n = boxesA.Shape[0];
        int m = boxesB.Shape[0];
        if (n == 0 || m == 0)
            return new Tensor<T>(new[] { n, m });

        return AiDotNetEngine.Current.BoxIou(boxesA, boxesB);
    }

    /// <summary>1 - IoU <c>[N, M]</c>, the IoU distance of SORT, DeepSORT and ByteTrack.</summary>
    internal static Tensor<T> IouDistance(Tensor<T> boxesA, Tensor<T> boxesB)
    {
        return AiDotNetEngine.Current.ScalarMinusTensor(NumOps.One, IouMatrix(boxesA, boxesB));
    }

    /// <summary>Indices where a 0/1 mask tensor is non-zero, in ascending order.</summary>
    internal static int[] Where(Tensor<T> mask)
    {
        return mask.Length == 0 ? Array.Empty<int>() : AiDotNetEngine.Current.TensorNonzero(mask).ToArray();
    }

    /// <summary>
    /// Exact minimum-cost assignment of <paramref name="cost"/> (<c>[N, M]</c>); entry <c>i</c> is the
    /// column given to row <c>i</c>, or -1.
    /// </summary>
    internal static int[] SolveAssignment(Tensor<T> cost)
    {
        int rows = cost.Shape[0];
        int cols = cost.Shape[1];
        if (rows == 0 || cols == 0)
            return Enumerable.Repeat(-1, rows).ToArray();

        return new LinearAssignmentSolver<T>().Solve(cost.ToMatrix()).ToArray();
    }

    /// <summary>
    /// ByteTrack's <c>linear_assignment(cost, thresh)</c>: <c>lap.lapjv(cost, extend_cost=True,
    /// cost_limit=thresh)</c>. The cost is embedded in an <c>(N + M)</c> square problem in which leaving a
    /// row and a column unmatched costs <c>thresh / 2</c> each, so a pair is matched only when its cost is
    /// at most <paramref name="threshold"/> and matching it lowers the total.
    /// </summary>
    internal static (List<(int Row, int Col)> Matches, List<int> UnmatchedRows, List<int> UnmatchedCols)
        LinearAssignment(Tensor<T> cost, T threshold)
    {
        int rows = cost.Shape[0];
        int cols = cost.Shape[1];
        var matches = new List<(int Row, int Col)>();
        if (rows == 0 || cols == 0)
            return (matches, Enumerable.Range(0, rows).ToList(), Enumerable.Range(0, cols).ToList());

        var engine = AiDotNetEngine.Current;
        int size = rows + cols;
        var extended = new Tensor<T>(new[] { size, size });
        engine.TensorFill(extended, NumOps.Divide(threshold, NumOps.FromDouble(2.0)));
        extended = engine.TensorSetSlice(extended, new Tensor<T>(new[] { cols, rows }), new[] { rows, cols });
        extended = engine.TensorSetSlice(extended, cost, new[] { 0, 0 });

        int[] assignment = SolveAssignment(extended);
        var colTaken = new bool[cols];
        var unmatchedRows = new List<int>();
        for (int r = 0; r < rows; r++)
        {
            int c = assignment[r];
            if (c >= 0 && c < cols)
            {
                matches.Add((r, c));
                colTaken[c] = true;
            }
            else
            {
                unmatchedRows.Add(r);
            }
        }

        var unmatchedCols = Enumerable.Range(0, cols).Where(c => !colTaken[c]).ToList();
        return (matches, unmatchedRows, unmatchedCols);
    }

    /// <summary>
    /// deep_sort's <c>min_cost_matching</c>: costs above <paramref name="maxDistance"/> are raised to
    /// <c>maxDistance + 1e-5</c>, the assignment is solved exactly, and pairs whose cost still exceeds
    /// <paramref name="maxDistance"/> are returned unmatched.
    /// </summary>
    internal static (List<(int Row, int Col)> Matches, List<int> UnmatchedRows, List<int> UnmatchedCols)
        MinCostMatching(Tensor<T> cost, T maxDistance)
    {
        int rows = cost.Shape[0];
        int cols = cost.Shape[1];
        var matches = new List<(int Row, int Col)>();
        if (rows == 0 || cols == 0)
            return (matches, Enumerable.Range(0, rows).ToList(), Enumerable.Range(0, cols).ToList());

        var engine = AiDotNetEngine.Current;
        var capped = engine.TensorWhere(
            engine.TensorGreaterThan(cost, maxDistance),
            engine.TensorFullLike(cost, NumOps.Add(maxDistance, NumOps.FromDouble(1e-5))),
            cost);
        int[] assignment = SolveAssignment(capped);

        var assignedRows = Enumerable.Range(0, rows).Where(r => assignment[r] >= 0).ToArray();
        var flat = new Tensor<int>(assignedRows.Select(r => r * cols + assignment[r]).ToArray(), new[] { assignedRows.Length });
        var pairCost = engine.TensorTake(capped, flat);
        int[] tooFar = engine.TensorGreaterThan(pairCost, maxDistance).Cast<int>().ToArray();

        var colTaken = new bool[cols];
        var rowTaken = new bool[rows];
        for (int k = 0; k < assignedRows.Length; k++)
        {
            if (tooFar[k] != 0)
                continue;

            int r = assignedRows[k];
            matches.Add((r, assignment[r]));
            rowTaken[r] = true;
            colTaken[assignment[r]] = true;
        }

        return (matches,
            Enumerable.Range(0, rows).Where(r => !rowTaken[r]).ToList(),
            Enumerable.Range(0, cols).Where(c => !colTaken[c]).ToList());
    }
}
