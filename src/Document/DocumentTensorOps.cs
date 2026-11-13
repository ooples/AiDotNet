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
    /// row (later ties never replace it, NaN never wins, 0 when nothing beats -infinity). The column index is
    /// produced as an int by the engine's arg-max, never compared in T: Half holds every integer only up to
    /// 2048, so an index ramp in T would merge neighbouring vocabulary ids.
    /// </summary>
    internal static int[] RowArgMax<T>(IEngine engine, Tensor<T> scores)
    {
        int rows = scores.Shape[0];
        if (rows == 0)
            return Array.Empty<int>();
        if (scores.Length == 0)
            return new int[rows];
        var ranked = engine.TensorWhere(engine.TensorIsNan(scores),
            Tensor<T>.CreateDefault(scores.Shape.ToArray(), AudioEngineOps.NegativeInfinityOrMin<T>()), scores);
        return engine.TensorArgMax(ranked, 1).ToArray();
    }

    /// <summary>
    /// The token of each of the first <c>min(rows, maxLength)</c> rows of a decoder output whose first axis is the
    /// sequence (the remaining axes are the vocabulary), with its softmax probability: the greedy token at
    /// <paramref name="temperature"/> 0, otherwise a temperature sample (one seeded engine draw per row, see
    /// <see cref="SampleWithProbabilities{T}"/>).
    /// </summary>
    internal static (int[] Tokens, T[] Probabilities) DecodeRows<T>(IEngine engine, Tensor<T> output, int maxLength, double temperature, int seed)
    {
        int rows = output.Rank == 0 ? 0 : Math.Min(output.Shape[0], maxLength);
        if (rows <= 0 || output.Length == 0)
            return (Array.Empty<int>(), Array.Empty<T>());
        int vocab = output.Length / output.Shape[0];
        var matrix = engine.Reshape(output, new[] { output.Shape[0], vocab });
        var head = rows == output.Shape[0] ? matrix : engine.TensorSlice(matrix, new[] { 0, 0 }, new[] { rows, vocab });
        var (tokens, probabilities) = temperature > 0.0
            ? SampleWithProbabilities(engine, head, MathHelper.GetNumericOperations<T>().FromDouble(temperature), seed)
            : GreedyWithProbabilities(engine, head);
        return (tokens, probabilities.ToArray());
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
        int rows = logits.Shape[0];
        var shape = logits.Shape.ToArray();
        int[] tokens = RowArgMax(engine, logits);
        var max = engine.TensorBroadcastTo(engine.ReduceMax(logits, new[] { 1 }, true), shape);
        // The partition function is summed in double: a Half running sum stops growing once its spacing exceeds
        // the addends (at 2048 for terms near 1), so a large vocabulary would report too high a probability.
        var sumExp = engine.ReduceSum(Wide(engine.TensorExp(engine.TensorSubtract(logits, max))), new[] { 1 }, false);
        var zeros = Tensor<double>.CreateDefault(new[] { rows }, 0.0);
        var ones = Tensor<double>.CreateDefault(new[] { rows }, 1.0);
        var positive = engine.TensorGreaterThan(sumExp, 0.0);
        var probabilities = engine.TensorWhere(positive, engine.TensorDivide(ones, engine.TensorWhere(positive, sumExp, ones)), zeros);
        return (tokens, probabilities.Cast<T>());
    }

    /// <summary>
    /// Temperature sampling of one token per row of <c>[rows, vocab]</c> logits: with scores
    /// <c>s_v = exp(l_v / t - max)</c>, the token is the first v whose cumulative score reaches
    /// <c>u * sum(s)</c> for a uniform draw u (the last token, with probability 0, when rounding leaves
    /// none), and its probability is <c>s_v / sum(s)</c>. Rows whose sum is not positive give token 0 with
    /// probability 0. The draws come from one seeded engine call.
    /// </summary>
    /// <remarks>
    /// The cumulative scores are accumulated in double (a Half running sum stalls at 2048 and loses every
    /// later term), and the chosen index is found and gathered as an int: it is the engine's first arg-max of
    /// the "cumulative share reached the draw" mask, never a count or an index ramp held in T, which Half
    /// cannot represent past 2048.
    /// </remarks>
    internal static (int[] Tokens, Tensor<T> Probabilities) SampleWithProbabilities<T>(
        IEngine engine, Tensor<T> logits, T temperature, int seed)
    {
        int rows = logits.Shape[0];
        int vocab = logits.Shape[1];
        var shape = new[] { rows, vocab };
        var column = new[] { rows, 1 };
        var scaled = engine.TensorDivideScalar(logits, temperature);
        var scores = engine.TensorExp(engine.TensorSubtract(scaled,
            engine.TensorBroadcastTo(engine.ReduceMax(scaled, new[] { 1 }, true), shape))).Cast<double>();
        var cumulative = engine.TensorCumSum(scores, 1);
        var sum = engine.TensorSlice(cumulative, new[] { 0, vocab - 1 }, column);
        var positive = engine.TensorGreaterThan(sum, 0.0);
        var safeSum = engine.TensorWhere(positive, sum, Tensor<double>.CreateDefault(column, 1.0));
        var normalized = engine.TensorDivide(cumulative, engine.TensorBroadcastTo(safeSum, shape));
        var draw = engine.TensorUniform<double>(column, 0.0, 1.0, seed);
        var reached = engine.TensorGreaterEqual(normalized, engine.TensorBroadcastTo(draw, shape));
        var (first, any) = RowMax(engine, reached, vocab);
        double[] anyReached = any.ToArray();
        double[] total = sum.ToArray();

        // Rows with no positive mass give token 0 (probability 0); rows where rounding left the draw unreached
        // give the last token (probability 0); every other row gives its first reaching token.
        int[] tokens = Enumerable.Range(0, rows)
            .Select(r => !(total[r] > 0.0) ? 0 : anyReached[r] > 0.0 ? first[r] : vocab - 1).ToArray();
        int[] flat = tokens.Select((token, r) => (r * vocab) + token).ToArray();
        var picked = engine.TensorIndexSelect(engine.Reshape(scores, new[] { rows * vocab }), new Tensor<int>(flat, new[] { rows }), 0);
        var valid = new Tensor<double>(Enumerable.Range(0, rows)
            .Select(r => total[r] > 0.0 && anyReached[r] > 0.0 ? 1.0 : 0.0).ToArray(), new[] { rows });
        var probability = engine.TensorWhere(valid,
            engine.TensorDivide(picked, engine.Reshape(safeSum, new[] { rows })), Tensor<double>.CreateDefault(new[] { rows }, 0.0));
        return (tokens, probability.Cast<T>());
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
        // Summed in double: a Half running sum of probabilities stalls at 2048 (a long decode has more steps).
        double total = engine.TensorSum(Wide(engine.TensorWhere(positive, head, Tensor<T>.CreateDefault(new[] { count }, numOps.Zero))));
        return numOps.FromDouble(total / hits);
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
        // Gathered at int positions t * classes + labels[t]: comparing an index ramp in T with the labels would
        // merge neighbouring ids once a vocabulary passes 2048 in Half.
        int rows = scores.Shape[0];
        int classes = scores.Length / rows;
        var flat = engine.Reshape(scores, new[] { rows * classes });
        int[] positions = labels.Select((label, t) => (t * classes) + label).ToArray();
        return engine.TensorSum(engine.TensorIndexSelect(flat, new Tensor<int>(positions, new[] { steps }), 0));
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
            : Mean(engine, new Tensor<T>((T[])values.Clone(), new[] { values.Length }));

    /// <summary>A contiguous double copy of <paramref name="values"/>, for reductions Half cannot carry.</summary>
    internal static Tensor<double> Wide<T>(Tensor<T> values) => values.Contiguous().Cast<double>();

    /// <summary>
    /// The mean of every value of <paramref name="values"/> (0 when it is empty), reduced in double and
    /// returned in T: a Half running sum stalls at 2048 for values near 1 (and overflows at 65504), so a mean
    /// over a large region would come out far too small.
    /// </summary>
    internal static T Mean<T>(IEngine engine, Tensor<T> values)
        => values.Length == 0
            ? MathHelper.GetNumericOperations<T>().Zero
            : MathHelper.GetNumericOperations<T>().FromDouble(engine.TensorMean(Wide(values)));

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
        var keep = engine.TensorMultiply(engine.TensorGreaterThan(max, numOps.Zero), AtLeast(engine, max, threshold));
        // The class test stays in int (class ids are indices).
        int[] rows = Hits(engine, keep).Where(r => arg[r] > 0).ToArray();
        return (rows, rows.Select(r => arg[r]).ToArray(), Pick(engine, max, rows));
    }

    /// <summary>
    /// Extractive span answering over a <c>[seq, hidden]</c> output whose column 0 holds start logits and
    /// column <c>min(1, hidden - 1)</c> end logits, decoded as BERT (Devlin et al. 2019, Sec. 4.2) and the
    /// LayoutLM-family DocVQA heads do: the span <c>(s, e)</c> with <c>s &lt;= e &lt; s + maxAnswerLength</c>
    /// maximizing <c>start[s] + end[e]</c> (first such pair in row-major order), its confidence the product of
    /// the start and end softmax probabilities <c>p_start(s) * p_end(e)</c> (the question-answering pipeline's
    /// score), and the greedy token (first maximum over the first <paramref name="vocab"/> columns, kept when
    /// above 0) of every position in the span. No span (an empty window) gives no tokens and confidence 0.
    /// </summary>
    internal static (List<int> Tokens, T Confidence) SpanAnswer<T>(IEngine engine, Tensor<T> output, int hidden, int vocab, int maxAnswerLength)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int seqLen = output.Shape[0];
        var tokens = new List<int>();
        int window = Math.Min(maxAnswerLength, seqLen);
        if (seqLen == 0 || output.Length == 0 || window <= 0)
            return (tokens, numOps.Zero);
        int columns = output.Length / seqLen;
        var matrix = engine.Reshape(output, new[] { seqLen, columns });
        int endColumn = Math.Min(1, Math.Min(hidden, columns) - 1);
        var start = engine.Reshape(engine.TensorSlice(matrix, new[] { 0, 0 }, new[] { seqLen, 1 }), new[] { seqLen });
        var end = engine.Reshape(engine.TensorSlice(matrix, new[] { 0, Math.Max(0, endColumn) }, new[] { seqLen, 1 }), new[] { seqLen });

        // Every (start, end) pair's score; pairs outside the band e - s in [0, window) are excluded. The band is
        // built from int offsets, so no index is ever compared in T.
        var pairShape = new[] { seqLen, seqLen };
        var pairs = engine.TensorAdd(
            engine.TensorBroadcastTo(engine.Reshape(start, new[] { seqLen, 1 }), pairShape),
            engine.TensorBroadcastTo(engine.Reshape(end, new[] { 1, seqLen }), pairShape));
        var band = new Tensor<int>(Enumerable.Range(0, seqLen * seqLen)
            .Select(i => (i % seqLen) - (i / seqLen))
            .Select(offset => offset >= 0 && offset < window ? 1 : 0).ToArray(), pairShape).Cast<T>();
        var candidates = engine.TensorWhere(band, pairs, Tensor<T>.CreateDefault(pairShape, AudioEngineOps.NegativeInfinityOrMin<T>()));
        int best = RowArgMax(engine, engine.Reshape(candidates, new[] { 1, seqLen * seqLen }))[0];
        int bestStart = best / seqLen, bestEnd = best % seqLen;

        int width = Math.Min(Math.Min(hidden, vocab), columns);
        if (width > 0)
        {
            var span = engine.TensorSlice(matrix, new[] { bestStart, 0 }, new[] { bestEnd - bestStart + 1, width });
            tokens.AddRange(RowArgMax(engine, span).Where(t => t > 0));
        }

        T startProbability = Pick(engine, engine.Softmax(start, 0), new[] { bestStart })[0];
        T endProbability = Pick(engine, engine.Softmax(end, 0), new[] { bestEnd })[0];
        return (tokens, numOps.Multiply(startProbability, endProbability));
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

    /// <summary>
    /// Per-image standardization (the Pix2Struct image processor, also used by MatCha): each image of the batch
    /// becomes <c>(x - mean) / max(std, 1 / sqrt(N))</c> with its own mean and population std over all its
    /// <c>N</c> values. The floor keeps a flat page finite (it maps to zeros).
    /// </summary>
    internal static Tensor<T> StandardizePerImage<T>(IEngine engine, Tensor<T> batch)
    {
        int images = batch.Shape[0];
        int n = batch.Length / Math.Max(1, images);
        if (images == 0 || n == 0)
            return batch;

        // The two moments are reduced in double and only they are brought back to T: a Half running sum over a
        // page (thousands of values) stalls at 2048 and overflows at 65504, which would wreck the mean and std.
        var shape = new[] { images, n };
        var flat = engine.Reshape(batch, shape);
        var wide = Wide(flat);
        var mean = engine.ReduceMean(wide, new[] { 1 }, keepDims: true);
        var deviation = engine.TensorSubtract(wide, engine.TensorBroadcastTo(mean, shape));
        var std = engine.TensorMax(
            engine.TensorSqrt(engine.ReduceMean(engine.TensorMultiply(deviation, deviation), new[] { 1 }, keepDims: true)),
            1.0 / Math.Sqrt(n));
        var centered = engine.TensorSubtract(flat, engine.TensorBroadcastTo(mean.Cast<T>(), shape));
        return engine.Reshape(engine.TensorDivide(centered, engine.TensorBroadcastTo(std.Cast<T>(), shape)), batch.Shape.ToArray());
    }
}
