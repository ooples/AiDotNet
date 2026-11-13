using AiDotNet.Audio;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Document;

/// <summary>
/// Whole-tensor helpers shared by the document models (input normalization, greedy decoding, score
/// maps), written with engine operations so callers do not loop over elements.
/// </summary>
internal static class DocumentTensorOps
{
    /// <summary>ImageNet channel means on the [0, 1] pixel scale.</summary>
    internal static readonly double[] ImageNetMean = { 0.485, 0.456, 0.406 };

    /// <summary>ImageNet channel standard deviations on the [0, 1] pixel scale.</summary>
    internal static readonly double[] ImageNetStd = { 0.229, 0.224, 0.225 };

    /// <summary>The 0.5 / 0.5 statistics of BEiT, DiT, LayoutLMv3, UDOP and Donut (maps [0, 1] to [-1, 1]).</summary>
    internal static readonly double[] HalfStatistics = { 0.5, 0.5, 0.5 };

    /// <summary>OpenAI CLIP channel means.</summary>
    internal static readonly double[] ClipMean = { 0.48145466, 0.4578275, 0.40821073 };

    /// <summary>OpenAI CLIP channel standard deviations.</summary>
    internal static readonly double[] ClipStd = { 0.26862954, 0.26130258, 0.27577711 };

    /// <summary>
    /// <c>(x - mean[c]) / std[c]</c> over a <c>[B, C, H, W]</c> image, where channels past the end of the
    /// tables use <paramref name="fallback"/> for both the mean and the standard deviation.
    /// </summary>
    internal static Tensor<T> NormalizeChannels<T>(
        IEngine engine, Tensor<T> image, double[] means, double[] stds, double fallback = 0.5)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int channels = image.Shape[1];
        var mean = Enumerable.Range(0, channels).Select(c => c < means.Length ? means[c] : fallback).ToArray();
        var inverse = Enumerable.Range(0, channels).Select(c => 1.0 / (c < stds.Length ? stds[c] : fallback)).ToArray();
        var shape = image.Shape.ToArray();
        var meanTensor = new Tensor<T>(new[] { 1, channels, 1, 1 }, new Vector<T>(mean.Select(numOps.FromDouble).ToArray()));
        var inverseTensor = new Tensor<T>(new[] { 1, channels, 1, 1 }, new Vector<T>(inverse.Select(numOps.FromDouble).ToArray()));
        return engine.TensorMultiply(
            engine.TensorSubtract(image, engine.TensorBroadcastTo(meanTensor, shape)),
            engine.TensorBroadcastTo(inverseTensor, shape));
    }

    /// <summary>
    /// The greedy token of every row of a <c>[rows, vocab]</c> score matrix: the first strict maximum of the
    /// row (later ties never replace it, NaN never wins, 0 when nothing beats -infinity).
    /// </summary>
    internal static int[] RowArgMax<T>(IEngine engine, Tensor<T> scores)
        => AudioEngineOps.RowFirstArgMax(engine, scores);

    /// <summary>
    /// The greedy token of each of the first <c>min(rows, maxLength)</c> rows of a decoder output whose first
    /// axis is the sequence (the remaining axes are the vocabulary).
    /// </summary>
    internal static int[] GreedyRowTokens<T>(IEngine engine, Tensor<T> output, int maxLength)
    {
        int rows = Math.Min(output.Shape[0], maxLength);
        if (rows <= 0 || output.Length == 0)
            return Array.Empty<int>();
        int vocab = output.Length / output.Shape[0];
        var matrix = engine.Reshape(output, new[] { output.Shape[0], vocab });
        var head = rows == output.Shape[0] ? matrix : engine.TensorSlice(matrix, new[] { 0, 0 }, new[] { rows, vocab });
        return RowArgMax(engine, head);
    }

    /// <summary>
    /// The rows <c>i &lt; min(rows, limit)</c> of an output viewed as <c>[rows, columns]</c> whose value in
    /// <paramref name="column"/> is at least <paramref name="threshold"/> (NaN never qualifies), in ascending
    /// row order, with those values.
    /// </summary>
    internal static (int[] Rows, T[] Scores) RowsAtLeast<T>(IEngine engine, Tensor<T> output, int limit, int column, T threshold)
    {
        var scores = ColumnHead(engine, output, limit, column);
        if (scores.Length == 0)
            return (Array.Empty<int>(), Array.Empty<T>());
        var hit = engine.TensorGreaterEqual(scores, Tensor<T>.CreateDefault(new[] { scores.Length }, threshold));
        int[] rows = engine.TensorNonzero(hit).ToArray();
        if (rows.Length == 0)
            return (rows, Array.Empty<T>());
        var picked = engine.TensorIndexSelect(scores, new Tensor<int>((int[])rows.Clone(), new[] { rows.Length }), 0);
        return (rows, picked.ToArray());
    }

    /// <summary>
    /// Column <paramref name="column"/> of the first <c>min(rows, limit)</c> rows of an output viewed as
    /// <c>[rows, columns]</c> (its first axis is the row axis).
    /// </summary>
    internal static Tensor<T> ColumnHead<T>(IEngine engine, Tensor<T> output, int limit, int column)
    {
        int rows = output.Rank == 0 ? 0 : Math.Min(output.Shape[0], limit);
        if (rows <= 0 || output.Length == 0)
            return new Tensor<T>(new[] { 0 });
        int columns = output.Length / output.Shape[0];
        if (column < 0 || column >= columns)
            throw new IndexOutOfRangeException($"Column {column} is outside the output's {columns} columns.");
        var matrix = engine.Reshape(output, new[] { output.Shape[0], columns });
        return engine.Reshape(engine.TensorSlice(matrix, new[] { 0, column }, new[] { rows, 1 }), new[] { rows });
    }

    /// <summary>
    /// The greedy token of every row of <c>[rows, vocab]</c> logits and its softmax probability
    /// <c>1 / sum_v exp(l_v - max)</c>; the probability is 0 where that sum is not positive (NaN rows).
    /// </summary>
    internal static (int[] Tokens, Tensor<T> Probabilities) GreedyWithProbabilities<T>(IEngine engine, Tensor<T> logits)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int rows = logits.Shape[0];
        var shape = logits.Shape.ToArray();
        int[] tokens = RowArgMax(engine, logits);
        var max = engine.TensorBroadcastTo(engine.ReduceMax(logits, new[] { 1 }, true), shape);
        var sumExp = engine.ReduceSum(engine.TensorExp(engine.TensorSubtract(logits, max)), new[] { 1 }, false);
        var zeros = Tensor<T>.CreateDefault(new[] { rows }, numOps.Zero);
        var ones = Tensor<T>.CreateDefault(new[] { rows }, numOps.One);
        var positive = engine.TensorGreaterThan(sumExp, numOps.Zero);
        var probabilities = engine.TensorWhere(positive, engine.TensorDivide(ones, engine.TensorWhere(positive, sumExp, ones)), zeros);
        return (tokens, probabilities);
    }

    /// <summary>
    /// Temperature sampling of one token per row of <c>[rows, vocab]</c> logits: with scores
    /// <c>s_v = exp(l_v / t - max)</c>, the token is the first v whose cumulative score reaches
    /// <c>u * sum(s)</c> for a uniform draw u (the last token, with probability 0, when rounding leaves
    /// none), and its probability is <c>s_v / sum(s)</c>. Rows whose sum is not positive give token 0 with
    /// probability 0. The draws come from one seeded engine call.
    /// </summary>
    internal static (int[] Tokens, Tensor<T> Probabilities) SampleWithProbabilities<T>(
        IEngine engine, Tensor<T> logits, T temperature, int seed)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int rows = logits.Shape[0];
        int vocab = logits.Shape[1];
        var shape = new[] { rows, vocab };
        var column = new[] { rows, 1 };
        var scaled = engine.TensorDivideScalar(logits, temperature);
        var scores = engine.TensorExp(engine.TensorSubtract(scaled,
            engine.TensorBroadcastTo(engine.ReduceMax(scaled, new[] { 1 }, true), shape)));
        var cumulative = engine.TensorCumSum(scores, 1);
        var sum = engine.TensorSlice(cumulative, new[] { 0, vocab - 1 }, column);
        var positive = engine.TensorGreaterThan(sum, numOps.Zero);
        var ones = Tensor<T>.CreateDefault(column, numOps.One);
        var zeros = Tensor<T>.CreateDefault(column, numOps.Zero);
        var safeSum = engine.TensorWhere(positive, sum, ones);
        var normalized = engine.TensorDivide(cumulative, engine.TensorBroadcastTo(safeSum, shape));
        var draw = engine.TensorUniform<T>(column, 0.0, 1.0, seed);
        var count = engine.ReduceSum(engine.TensorLessThan(normalized, engine.TensorBroadcastTo(draw, shape)), new[] { 1 }, true);
        var inRange = engine.TensorLessThan(count, numOps.FromDouble(vocab));
        var selected = engine.TensorWhere(inRange, count, Tensor<T>.CreateDefault(column, numOps.FromDouble(vocab - 1)));
        var index = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, vocab, 1), new[] { 1, vocab }), shape);
        var picked = engine.TensorEquals(index, engine.TensorBroadcastTo(selected, shape));
        var score = engine.ReduceSum(engine.TensorWhere(picked, scores, Tensor<T>.CreateDefault(shape, numOps.Zero)), new[] { 1 }, true);
        var probability = engine.TensorWhere(positive, engine.TensorWhere(inRange, engine.TensorDivide(score, safeSum), zeros), zeros);
        var tokens = engine.TensorWhere(positive, selected, zeros);
        return (engine.Reshape(tokens, new[] { rows }).Cast<int>().ToArray(), engine.Reshape(probability, new[] { rows }));
    }

    /// <summary>
    /// The mean of the positive values among the first <paramref name="count"/> of <paramref name="values"/>,
    /// or 0 when none is positive.
    /// </summary>
    internal static T MeanOfPositivePrefix<T>(IEngine engine, Tensor<T> values, int count)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (count <= 0)
            return numOps.Zero;
        var head = count == values.Length ? values : engine.TensorSlice(values, new[] { 0 }, new[] { count });
        var positive = engine.TensorGreaterThan(head, numOps.Zero);
        int hits = engine.TensorCountNonzero(positive);
        if (hits == 0)
            return numOps.Zero;
        var total = engine.TensorSum(engine.TensorWhere(positive, head, Tensor<T>.CreateDefault(new[] { count }, numOps.Zero)));
        return numOps.Divide(total, numOps.FromDouble(hits));
    }

    /// <summary>
    /// Category indices ordered by descending logit (ties in ascending index order, NaN last - the order a
    /// stable <c>OrderByDescending</c> gives) with their softmax probabilities in that order.
    /// </summary>
    internal static (int[] Order, T[] Probabilities) RankedSoftmax<T>(IEngine engine, Tensor<T> logits)
    {
        var flat = AudioEngineOps.Flatten(engine, logits);
        int n = flat.Length;
        if (n == 0)
            return (Array.Empty<int>(), Array.Empty<T>());
        int[] order = AudioEngineOps.StableDescendingOrder(engine, flat);
        var probabilities = engine.Softmax(flat, 0);
        var ranked = engine.TensorIndexSelect(probabilities, new Tensor<int>((int[])order.Clone(), new[] { n }), 0);
        return (order, ranked.ToArray());
    }

    /// <summary>
    /// The first <paramref name="count"/> values of row <paramref name="row"/> of an output viewed as
    /// <c>[rows, columns]</c>.
    /// </summary>
    internal static Tensor<T> RowHead<T>(IEngine engine, Tensor<T> output, int row, int count)
    {
        int columns = output.Length / output.Shape[0];
        var matrix = engine.Reshape(output, new[] { output.Shape[0], columns });
        return engine.Reshape(engine.TensorSlice(matrix, new[] { row, 0 }, new[] { 1, count }), new[] { count });
    }

    /// <summary>
    /// <c>sum_t scores[t, labels[t]]</c> over a <c>[steps, classes]</c> score matrix (for example the
    /// log-likelihood of a label sequence under per-step log-probabilities).
    /// </summary>
    internal static T PickedSum<T>(IEngine engine, Tensor<T> scores, int[] labels)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int steps = labels.Length;
        if (steps == 0)
            return numOps.Zero;
        int classes = scores.Shape[1];
        var shape = new[] { steps, classes };
        var head = scores.Shape[0] == steps ? scores : engine.TensorSlice(scores, new[] { 0, 0 }, shape);
        var picked = OneHot<T>(engine, labels, classes);
        return engine.TensorSum(engine.TensorWhere(picked, head, Tensor<T>.CreateDefault(shape, numOps.Zero)));
    }

    /// <summary>A <c>[labels, classes]</c> 0/1 matrix with a 1 at <c>(t, labels[t])</c>.</summary>
    internal static Tensor<T> OneHot<T>(IEngine engine, int[] labels, int classes)
    {
        int steps = labels.Length;
        var shape = new[] { steps, classes };
        var index = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, classes, 1), new[] { 1, classes }), shape);
        var label = engine.TensorBroadcastTo(new Tensor<int>((int[])labels.Clone(), new[] { steps, 1 }).Cast<T>(), shape);
        return engine.TensorEquals(index, label);
    }

    /// <summary>
    /// For each row of an output viewed as <c>[rows, columns]</c> (first <paramref name="columns"/> columns),
    /// the first strict maximum's column and the row maximum.
    /// </summary>
    internal static (int[] Arg, Tensor<T> Max) RowMax<T>(IEngine engine, Tensor<T> output, int columns)
    {
        int rows = output.Shape[0];
        int width = output.Length / rows;
        var matrix = engine.Reshape(output, new[] { rows, width });
        var head = columns == width ? matrix : engine.TensorSlice(matrix, new[] { 0, 0 }, new[] { rows, columns });
        return (RowArgMax(engine, head), engine.ReduceMax(head, new[] { 1 }, false));
    }

    /// <summary>The 1-D 0/1 mask of the <paramref name="values"/> that are at least <paramref name="threshold"/>.</summary>
    internal static Tensor<T> AtLeast<T>(IEngine engine, Tensor<T> values, T threshold)
        => engine.TensorGreaterEqual(values, Tensor<T>.CreateDefault(values.Shape.ToArray(), threshold));

    /// <summary>The positions of the nonzero entries of a 1-D mask, ascending.</summary>
    internal static int[] Hits<T>(IEngine engine, Tensor<T> mask)
        => mask.Length == 0 ? Array.Empty<int>() : engine.TensorNonzero(mask).ToArray();

    /// <summary>The entries of a 1-D tensor at the given positions.</summary>
    internal static T[] Pick<T>(IEngine engine, Tensor<T> values, int[] positions)
        => positions.Length == 0
            ? Array.Empty<T>()
            : engine.TensorIndexSelect(values, new Tensor<int>((int[])positions.Clone(), new[] { positions.Length }), 0).ToArray();

    /// <summary>Integer values (pixel coordinates, counts) converted to T in one bulk cast.</summary>
    internal static T[] ToT<T>(int[] values)
        => values.Length == 0 ? Array.Empty<T>() : new Tensor<int>((int[])values.Clone(), new[] { values.Length }).Cast<T>().ToArray();

    /// <summary>The integer <paramref name="values"/> as a 1-D T tensor.</summary>
    internal static Tensor<T> IntTensor<T>(int[] values)
        => new Tensor<int>((int[])values.Clone(), new[] { values.Length }).Cast<T>();

    /// <summary>The mean of <paramref name="values"/>, or 0 when there are none.</summary>
    internal static T Mean<T>(IEngine engine, T[] values)
        => values.Length == 0
            ? MathHelper.GetNumericOperations<T>().Zero
            : engine.TensorMean(new Tensor<T>((T[])values.Clone(), new[] { values.Length }));

    /// <summary>
    /// For each row of an output viewed as <c>[rows, columns]</c>: the rows whose best class (first strict
    /// maximum of the first <paramref name="classes"/> columns, above 0) is not the background class 0 and
    /// reaches <paramref name="threshold"/>, with that class and score.
    /// </summary>
    internal static (int[] Rows, int[] Classes, T[] Scores) ForegroundRows<T>(IEngine engine, Tensor<T> output, int classes, T threshold)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (output.Rank == 0 || output.Shape[0] == 0 || output.Length == 0 || classes <= 0)
            return (Array.Empty<int>(), Array.Empty<int>(), Array.Empty<T>());
        var (arg, max) = RowMax(engine, output, classes);
        var keep = engine.TensorMultiply(
            engine.TensorMultiply(engine.TensorGreaterThan(max, numOps.Zero), AtLeast(engine, max, threshold)),
            engine.TensorGreaterThan(IntTensor<T>(arg), numOps.Zero));
        int[] rows = Hits(engine, keep);
        return (rows, rows.Select(r => arg[r]).ToArray(), Pick(engine, max, rows));
    }

    /// <summary>
    /// Extractive span answering over a <c>[seq, hidden]</c> output whose column 0 holds start logits and
    /// column <c>min(1, hidden - 1)</c> end logits: the first best start, the first best end in
    /// <c>[start, start + maxAnswerLength)</c>, the greedy token (first maximum over the first
    /// <paramref name="vocab"/> columns, kept when above 0) of every position in the span, and the confidence
    /// <c>clamp((startLogit + endLogit) / 2, 0, 1)</c> (0 when the end window is empty).
    /// </summary>
    internal static (List<int> Tokens, T Confidence) SpanAnswer<T>(IEngine engine, Tensor<T> output, int hidden, int vocab, int maxAnswerLength)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int seqLen = output.Shape[0];
        var tokens = new List<int>();
        if (seqLen == 0 || output.Length == 0)
            return (tokens, numOps.Zero);
        int columns = output.Length / seqLen;
        var matrix = engine.Reshape(output, new[] { seqLen, columns });
        var start = engine.Reshape(engine.TensorSlice(matrix, new[] { 0, 0 }, new[] { seqLen, 1 }), new[] { 1, seqLen });
        int bestStart = RowArgMax(engine, start)[0];
        T startScore = engine.TensorMaxValue(start);
        int endLimit = Math.Min(seqLen, bestStart + maxAnswerLength);
        if (endLimit <= bestStart)
            return (tokens, numOps.Zero);
        int endColumn = Math.Min(1, hidden - 1);
        var end = engine.Reshape(engine.TensorSlice(matrix, new[] { bestStart, endColumn }, new[] { endLimit - bestStart, 1 }),
            new[] { 1, endLimit - bestStart });
        int bestEnd = bestStart + RowArgMax(engine, end)[0];
        T endScore = engine.TensorMaxValue(end);
        int width = Math.Min(Math.Min(hidden, vocab), columns);
        if (width > 0)
        {
            var span = engine.TensorSlice(matrix, new[] { bestStart, 0 }, new[] { bestEnd - bestStart + 1, width });
            tokens.AddRange(RowArgMax(engine, span).Where(t => t > 0));
        }
        T mean = numOps.Divide(numOps.Add(startScore, endScore), numOps.FromDouble(2.0));
        T confidence = numOps.LessThan(mean, numOps.Zero) ? numOps.Zero : numOps.GreaterThan(mean, numOps.One) ? numOps.One : mean;
        return (tokens, confidence);
    }

    /// <summary>
    /// Indices of a 1-D tensor by descending value, ties in ascending index order and NaN last (the order a
    /// stable <c>OrderByDescending</c> gives).
    /// </summary>
    internal static int[] DescendingOrder<T>(IEngine engine, Tensor<T> values)
        => values.Length == 0 ? Array.Empty<int>() : AudioEngineOps.StableDescendingOrder(engine, values);

    /// <summary>
    /// Node classification for the graph layout models (Doc-GCN, LayoutGraph): each of the first
    /// <paramref name="maxNodes"/> rows of a <c>[nodes, width]</c> head holds <paramref name="classes"/> class logits,
    /// optionally followed by a normalized <c>[x1, y1, x2, y2]</c> box in its last four columns. A node's class is
    /// its softmax argmax (every class is a real layout class, so none is skipped as background) and it is kept
    /// when that probability reaches <paramref name="threshold"/>. Without a box head the node's cell of a square
    /// grid over the page stands in for its box.
    /// </summary>
    internal static List<LayoutRegion<T>> GraphLayoutRegions<T>(
        IEngine engine, Tensor<T> output, int maxNodes, int classes, int imageSize, double threshold)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int numNodes = Math.Min(output.Shape[0], maxNodes);
        int width = output.Shape.Length > 1 ? output.Length / output.Shape[0] : classes;
        int numClasses = Math.Min(width, classes);
        if (numNodes <= 0 || numClasses <= 0)
            return new List<LayoutRegion<T>>();

        var rows = engine.TensorSlice(engine.Reshape(output, new[] { output.Shape[0], width }), new[] { 0, 0 }, new[] { numNodes, width });
        var probabilities = engine.Softmax(engine.TensorSlice(rows, new[] { 0, 0 }, new[] { numNodes, numClasses }), 1);
        var (labels, scores) = RowMax(engine, probabilities, numClasses);
        int[] kept = Hits(engine, AtLeast(engine, scores, numOps.FromDouble(threshold)));
        T[] confidences = Pick(engine, scores, kept);

        T[] boxes;
        if (width >= classes + 4)
        {
            T size = numOps.FromDouble(imageSize);
            var corners = engine.TensorMultiplyScalar(engine.TensorSlice(rows, new[] { 0, width - 4 }, new[] { numNodes, 4 }), size);
            var lower = engine.TensorMax(engine.TensorSlice(corners, new[] { 0, 0 }, new[] { numNodes, 2 }), numOps.Zero);
            var upper = engine.TensorMin(engine.TensorSlice(corners, new[] { 0, 2 }, new[] { numNodes, 2 }), size);
            boxes = engine.TensorConcatenate(new[] { lower, upper }, 1).ToArray();
        }
        else
        {
            int gridSize = Math.Max(1, (int)Math.Sqrt(numNodes));
            int cellSize = imageSize / gridSize;
            boxes = ToT<T>(Enumerable.Range(0, numNodes).SelectMany(i => new[]
            {
                (i % gridSize) * cellSize, (i / gridSize) * cellSize, ((i % gridSize) + 1) * cellSize, ((i / gridSize) + 1) * cellSize
            }).ToArray());
        }

        return kept.Select((node, k) => new LayoutRegion<T>
        {
            ElementType = (LayoutElementType)Math.Min(labels[node], (int)LayoutElementType.Other),
            Confidence = confidences[k],
            Index = node,
            BoundingBox = new Vector<T>(boxes.Skip(4 * node).Take(4).ToArray())
        }).ToList();
    }
}
