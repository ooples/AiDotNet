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

        var keep = engine.Nms(candidateBoxes, candidateConfidences, iouThreshold);
        int kept = keep.Length;
        var rows = engine.TensorConcatenate(
            [
                engine.TensorIndexSelect(candidateBoxes, keep, 0),
                engine.TensorIndexSelect(candidateConfidences, keep, 0).Reshape([kept, 1])
            ], 1);
        return rows.Reshape([kept * FieldsPerDetection]);
    }

    /// <summary>
    /// DETR-style query decoding shared by DINO-X and Grounding DINO 1.5: the flattened decoder output is cut into
    /// <paramref name="queries"/> equal blocks of <c>max(1, outDim / queries)</c> values. A block's four quarters
    /// sum to cx, cy, w and h, each divided by <c>max(1, block / 4)</c> and squashed with a sigmoid, and its
    /// confidence is <c>sigmoid(1 - rms(block))</c>. A query past the end of a short output reads nothing.
    /// </summary>
    /// <param name="engine">Engine to run on.</param>
    /// <param name="decoderOut">Decoder output (read flattened).</param>
    /// <param name="queries">Number of queries (positive).</param>
    /// <param name="boxes">The <c>[queries, 4]</c> clipped (x1, y1, x2, y2) boxes.</param>
    /// <param name="confidences">The <c>[queries]</c> confidences.</param>
    internal static void QuarterBlockQueries(IEngine engine, Tensor<T> decoderOut, int queries,
        out Tensor<T> boxes, out Tensor<T> confidences)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int outDim = decoderOut.Length;
        int block = Math.Max(1, outDim / queries);
        int full = Math.Min(queries, outDim / block);
        int empty = queries - full;
        if (full == 0)
        {
            // Nothing to read: every sum is zero and every rms is zero.
            var middle = engine.TensorSigmoid(new Tensor<T>([queries]));
            confidences = engine.TensorSigmoid(Tensor<T>.CreateDefault([queries], numOps.One));
            boxes = ClippedCornerBoxes(engine, middle, middle, middle, middle);
            return;
        }

        var blocks = engine.TensorSlice(decoderOut.Reshape([outDim]), [0], [full * block]).Reshape([full, block]);
        var quarterDivisor = numOps.FromDouble(Math.Max(1, block / 4));
        Tensor<T> Part(int from, int to) =>
            WithEmptyRows(engine, engine.TensorDivideScalar(ColumnSum(engine, blocks, from, to), quarterDivisor), empty);

        var cx = engine.TensorSigmoid(Part(0, block / 4));
        var cy = engine.TensorSigmoid(Part(block / 4, block / 2));
        var w = engine.TensorSigmoid(Part(block / 2, 3 * block / 4));
        var h = engine.TensorSigmoid(Part(3 * block / 4, block));
        var rms = WithEmptyRows(engine, RowRms(engine, blocks), empty);
        confidences = engine.TensorSigmoid(engine.TensorAddScalar(engine.TensorNegate(rms), numOps.One));
        boxes = ClippedCornerBoxes(engine, cx, cy, w, h);
    }

    /// <summary>
    /// Per-patch box decoding shared by OWL-ViT and OWLv2: a patch's feature row splits into quarters of
    /// <c>q = max(1, dim / 4)</c> columns whose sums give the centre offset <c>tanh(s / q) * 0.5 / grid</c> from the
    /// patch centre and the size <c>sigmoid(s / q) * 3 / grid</c>.
    /// </summary>
    /// <param name="engine">Engine to run on.</param>
    /// <param name="features">Patch features <c>[patches, dim]</c>; row p is patch p of the grid in row-major order.</param>
    /// <param name="gridSize">Patches per grid side.</param>
    /// <returns>The <c>[patches, 4]</c> clipped (x1, y1, x2, y2) boxes.</returns>
    internal static Tensor<T> PatchOffsetBoxes(IEngine engine, Tensor<T> features, int gridSize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int patches = features.Shape[0];
        int dim = features.Shape[1];
        int quarter = Math.Max(1, dim / 4);
        var quarterDivisor = numOps.FromDouble(quarter);
        var offsetScale = numOps.FromDouble(0.5 / gridSize);
        var sizeScale = numOps.FromDouble(3.0 / gridSize);

        var dx = engine.TensorMultiplyScalar(engine.TensorTanh(
            engine.TensorDivideScalar(ColumnSum(engine, features, 0, quarter), quarterDivisor)), offsetScale);
        var dy = engine.TensorMultiplyScalar(engine.TensorTanh(
            engine.TensorDivideScalar(ColumnSum(engine, features, quarter, 2 * quarter), quarterDivisor)), offsetScale);
        var w = engine.TensorMultiplyScalar(engine.TensorSigmoid(
            engine.TensorDivideScalar(ColumnSum(engine, features, 2 * quarter, 3 * quarter), quarterDivisor)), sizeScale);
        var h = engine.TensorMultiplyScalar(engine.TensorSigmoid(
            engine.TensorDivideScalar(ColumnSum(engine, features, 3 * quarter, dim), quarterDivisor)), sizeScale);

        // Patch p sits at grid row p / grid and column p % grid; its centre is ((col + 0.5) / grid, (row + 0.5) / grid).
        var rows = new int[patches];
        var cols = new int[patches];
        for (int p = 0; p < patches; p++)
        {
            rows[p] = p / gridSize;
            cols[p] = p % gridSize;
        }

        var half = numOps.FromDouble(0.5);
        var invGrid = numOps.FromDouble(1.0 / gridSize);
        var centreX = engine.TensorMultiplyScalar(engine.TensorAddScalar(IntsAsT(cols), half), invGrid);
        var centreY = engine.TensorMultiplyScalar(engine.TensorAddScalar(IntsAsT(rows), half), invGrid);

        return ClippedCornerBoxes(engine, engine.TensorAdd(centreX, dx), engine.TensorAdd(centreY, dy), w, h);
    }

    /// <summary>
    /// Region proposals from a spatial attention map, shared by Ferret and Ferret-v2. The decoder output is read
    /// as a <c>G x G</c> grid (<c>G = max(2, floor(sqrt(outDim)))</c>); cell c sums the first
    /// <c>min(perCell, featureCap)</c> values from <c>c * perCell % outDim</c> (cyclically) and its attention is
    /// <c>exp(logitScale * sum)</c>. Every cell not below <c>seedFactor x mean</c> attention that no earlier region
    /// took seeds a region, which grows over 4-connected cells with at least <c>growFactor x seedFactor x mean</c>
    /// attention. A region's box is its cell bounding box and its confidence is
    /// <c>sigmoid(min(1, mean region attention / mean attention) - confidenceOffset)</c>.
    /// </summary>
    /// <remarks>Attention is computed relative to the largest cell logit, which leaves every ratio above unchanged
    /// and keeps the exponential finite in narrow types.</remarks>
    /// <param name="engine">Engine to run on.</param>
    /// <param name="decoderOut">Decoder output (read flattened, nonempty).</param>
    /// <param name="featureCap">Most values summed per cell.</param>
    /// <param name="logitScale">Scale applied to a cell sum before the exponential.</param>
    /// <param name="seedFactor">Seed threshold as a multiple of the mean attention.</param>
    /// <param name="growFactor">Growth threshold as a multiple of the seed threshold.</param>
    /// <param name="confidenceOffset">Offset subtracted before the confidence sigmoid.</param>
    /// <param name="boxes">The <c>[regions, 4]</c> (x1, y1, x2, y2) boxes, in discovery order.</param>
    /// <param name="confidences">The <c>[regions]</c> confidences.</param>
    internal static void AttentionRegions(IEngine engine, Tensor<T> decoderOut, int featureCap, double logitScale,
        double seedFactor, double growFactor, double confidenceOffset, out Tensor<T> boxes, out Tensor<T> confidences)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int outDim = decoderOut.Length;
        int gridSize = Math.Max(2, (int)Math.Sqrt(outDim));
        int cells = gridSize * gridSize;
        int perCell = Math.Max(1, outDim / cells);
        int read = Math.Min(perCell, featureCap);

        var positions = new int[cells * read];
        for (int cell = 0; cell < cells; cell++)
        {
            int cellStart = (cell * perCell) % outDim;
            for (int d = 0; d < read; d++)
                positions[cell * read + d] = (cellStart + d) % outDim;
        }

        var logits = engine.TensorMultiplyScalar(
            engine.ReduceSum(Take(engine, decoderOut, positions, [cells, read]), [1], false),
            numOps.FromDouble(logitScale));
        var attention = engine.TensorExp(engine.TensorSubtractScalar(logits, engine.TensorMaxValue(logits)));
        var meanAttention = numOps.Divide(engine.TensorSum(attention), numOps.FromDouble(cells));
        var seedThreshold = numOps.Multiply(meanAttention, numOps.FromDouble(seedFactor));
        var growThreshold = numOps.Multiply(seedThreshold, numOps.FromDouble(growFactor));

        // A cell seeds unless its attention is below the threshold; it joins a region when at least the growth one.
        var one = Tensor<T>.CreateDefault([cells], numOps.One);
        int[] seeds = engine.TensorSubtract(one, engine.TensorLessThan(attention, seedThreshold)).Cast<int>().ToArray();
        int[] grows = AtLeast(engine, attention, growThreshold).Cast<int>().ToArray();

        var visited = new bool[cells];
        var regionOf = new int[cells];
        var minRows = new List<int>();
        var minCols = new List<int>();
        var maxRows = new List<int>();
        var maxCols = new List<int>();
        var sizes = new List<int>();
        int[] dr = { -1, 1, 0, 0 };
        int[] dc = { 0, 0, -1, 1 };
        for (int cell = 0; cell < cells; cell++)
        {
            regionOf[cell] = -1;
        }

        for (int cell = 0; cell < cells; cell++)
        {
            if (visited[cell] || seeds[cell] == 0)
                continue;

            int region = sizes.Count;
            int minR = cell / gridSize, maxR = minR, minC = cell % gridSize, maxC = minC;
            int size = 0;
            var stack = new Stack<int>();
            stack.Push(cell);
            visited[cell] = true;
            while (stack.Count > 0)
            {
                int cur = stack.Pop();
                int cr = cur / gridSize;
                int cc = cur % gridSize;
                regionOf[cur] = region;
                size++;
                minR = Math.Min(minR, cr);
                maxR = Math.Max(maxR, cr);
                minC = Math.Min(minC, cc);
                maxC = Math.Max(maxC, cc);
                for (int n = 0; n < 4; n++)
                {
                    int nr = cr + dr[n];
                    int nc = cc + dc[n];
                    if (nr >= 0 && nr < gridSize && nc >= 0 && nc < gridSize)
                    {
                        int next = nr * gridSize + nc;
                        if (!visited[next] && grows[next] != 0)
                        {
                            visited[next] = true;
                            stack.Push(next);
                        }
                    }
                }
            }

            minRows.Add(minR);
            maxRows.Add(maxR + 1);
            minCols.Add(minC);
            maxCols.Add(maxC + 1);
            sizes.Add(size);
        }

        int regions = sizes.Count;
        if (regions == 0)
        {
            boxes = new Tensor<T>([0, 4]);
            confidences = new Tensor<T>([0]);
            return;
        }

        // Region attention sums: scatter-add each member cell's attention into its region.
        var memberCells = new List<int>();
        var memberRegions = new List<int>();
        for (int cell = 0; cell < cells; cell++)
        {
            if (regionOf[cell] >= 0)
            {
                memberCells.Add(cell);
                memberRegions.Add(regionOf[cell]);
            }
        }

        var regionAttention = engine.TensorScatterAdd(new Tensor<T>([regions]),
            new Tensor<int>(new[] { memberRegions.Count }, Vector<int>.FromList(memberRegions)),
            Take(engine, attention, memberCells.ToArray(), [memberCells.Count]), 0);
        var expected = engine.TensorMultiplyScalar(IntsAsT(sizes.ToArray()), meanAttention);
        var ratio = engine.TensorClampMax(engine.TensorDivide(regionAttention, expected), numOps.One);
        confidences = engine.TensorSigmoid(engine.TensorSubtractScalar(ratio, numOps.FromDouble(confidenceOffset)));

        var grid = numOps.FromDouble(gridSize);
        Tensor<T> Edge(List<int> values) =>
            engine.TensorDivideScalar(IntsAsT(values.ToArray()), grid).Reshape([regions, 1]);
        boxes = engine.TensorConcatenate([Edge(minCols), Edge(minRows), Edge(maxCols), Edge(maxRows)], 1);
    }

    /// <summary>Integer values as a <c>[n]</c> tensor of <typeparamref name="T"/>.</summary>
    internal static Tensor<T> IntsAsT(int[] values) =>
        new Tensor<int>(new[] { values.Length }, Vector<int>.FromArray(values)).Cast<T>();

    /// <summary>
    /// Sums columns <c>[from, min(to, width))</c> of a <c>[n, width]</c> matrix; zeros when the range is empty.
    /// </summary>
    internal static Tensor<T> ColumnSum(IEngine engine, Tensor<T> matrix, int from, int to)
    {
        int rows = matrix.Shape[0];
        int end = Math.Min(to, matrix.Shape[1]);
        if (end <= from)
            return new Tensor<T>([rows]);
        return engine.ReduceSum(engine.TensorSlice(matrix, [0, from], [rows, end - from]), [1], false);
    }

    /// <summary>
    /// Root mean square of each row of a <c>[n, width]</c> matrix. Rows are scaled by their largest magnitude
    /// before squaring, so narrow types such as Half cannot overflow.
    /// </summary>
    internal static Tensor<T> RowRms(IEngine engine, Tensor<T> matrix)
    {
        int rows = matrix.Shape[0];
        int width = matrix.Shape[1];
        var scale = RowScale(engine, matrix);
        var scaled = engine.TensorDivide(matrix, engine.TensorBroadcastTo(scale, [rows, width]));
        var meanSquare = engine.ReduceMean(engine.TensorMultiply(scaled, scaled), [1], false);
        return engine.TensorMultiply(engine.TensorSqrt(meanSquare), scale.Reshape([rows]));
    }

    /// <summary>
    /// L2 norm of each row of a <c>[n, width]</c> matrix, as <c>[n]</c>, with the same overflow-safe scaling as
    /// <see cref="RowRms"/>.
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

    /// <summary>Appends <paramref name="empty"/> zeros to a <c>[n]</c> tensor.</summary>
    private static Tensor<T> WithEmptyRows(IEngine engine, Tensor<T> values, int empty) => empty <= 0
        ? values
        : engine.TensorConcatenate([values.Reshape([values.Length]), new Tensor<T>([empty])], 0);

    /// <summary>
    /// Reads <paramref name="flat"/> at the given positions; a negative position reads zero.
    /// </summary>
    /// <param name="engine">Engine to run on.</param>
    /// <param name="flat">Values (read flattened).</param>
    /// <param name="positions">Positions into <paramref name="flat"/>, or -1 for zero.</param>
    /// <param name="shape">Shape of the result (its element count is <c>positions.Length</c>).</param>
    /// <returns>The gathered values.</returns>
    internal static Tensor<T> Take(IEngine engine, Tensor<T> flat, int[] positions, int[] shape)
    {
        int count = positions.Length;
        if (count == 0)
            return new Tensor<T>(shape);

        var valid = new int[count];
        var safe = new int[count];
        for (int i = 0; i < count; i++)
        {
            valid[i] = positions[i] >= 0 ? 1 : 0;
            safe[i] = positions[i] >= 0 ? positions[i] : 0;
        }

        var taken = engine.TensorTake(flat.Reshape([flat.Length]),
            new Tensor<int>(new[] { count }, Vector<int>.FromArray(safe)));
        return engine.TensorWhere(IntsAsT(valid), taken, new Tensor<T>([count])).Reshape(shape);
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
