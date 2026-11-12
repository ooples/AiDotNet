using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Solvers.Assignment;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// The set-prediction losses of the mask-classification segmentation models: every ground-truth segment is matched
/// to one predicted mask by bipartite (Hungarian) matching on the paper's matching cost, and the loss is taken over
/// that one-to-one assignment.
/// </summary>
/// <remarks>
/// <para>
/// <c>Mask2Former</c> is the loss ODISE trains with (Xu et al. 2023 build on Mask2Former, Cheng et al.
/// 2022): matching cost <c>2 * (-p(c)) + 5 * BCE + 5 * Dice</c>, then a class cross-entropy whose no-object class is
/// weighted 0.1, plus the mask binary cross-entropy and Dice losses of the matched pairs, weighted 2, 5 and 5.
/// </para>
/// <para>
/// <c>KMaX</c> is kMaX-DeepLab's loss (Yu et al. 2022, following MaX-DeepLab, Wang et al. 2021): masks are
/// a softmax over the cluster centers, segments are matched by maximizing <c>p(c) * Dice</c>, and the loss is the
/// PQ-style loss (weight 3.0; <c>alpha = 0.75</c> between matched and unmatched predictions) plus the mask-ID
/// cross-entropy (weight 0.3).
/// </para>
/// <para>
/// Ground-truth segments come from a semantic target: one segment per class present in the image, the class map
/// being the argmax over the class axis of a <c>[B, K, H, W]</c> target or a <c>[B, H, W]</c> class-index map.
/// The matching runs without a gradient; the assignment is a constant of the loss, as in the papers.
/// </para>
/// </remarks>
internal static class MaskSetPredictionLoss
{
    internal const double Mask2FormerClassWeight = 2.0;
    internal const double Mask2FormerMaskWeight = 5.0;
    internal const double Mask2FormerDiceWeight = 5.0;
    internal const double Mask2FormerNoObjectWeight = 0.1;
    internal const double KMaXPanopticQualityWeight = 3.0;
    internal const double KMaXMaskIdWeight = 0.3;
    internal const double KMaXPositiveWeight = 0.75;
    private const double DiceEpsilon = 1e-5;

    /// <summary>The ground-truth segments of one image: binary masks <c>[M, HW]</c> and their classes.</summary>
    internal readonly struct Segments<T>
    {
        internal Segments(Tensor<T> masks, int[] classes)
        {
            Masks = masks;
            Classes = classes;
        }

        internal Tensor<T> Masks { get; }
        internal int[] Classes { get; }
    }

    /// <summary>
    /// One segment per class present in each image of a semantic <paramref name="target"/>: <c>[B, K, H, W]</c>
    /// (scores or one-hot, argmaxed over K) or a <c>[B, H, W]</c> map of class indices.
    /// </summary>
    internal static Segments<T>[] SemanticSegments<T>(Tensor<T> target, int numClasses, int batch, int height, int width)
    {
        if (target is null) throw new ArgumentNullException(nameof(target));
        var engine = AiDotNetEngine.Current;
        int hw = height * width;
        Tensor<int> labels;
        if (target.Rank == 4 && target.Shape[0] == batch && target.Shape[1] == numClasses
            && target.Shape[2] == height && target.Shape[3] == width)
            labels = engine.Reshape(engine.TensorArgMax(target, 1), new[] { batch, hw });
        else if (target.Rank == 3 && target.Shape[0] == batch && target.Shape[1] == height && target.Shape[2] == width)
            labels = engine.Reshape(target, new[] { batch, hw }).Cast<int>();
        else
            throw new ArgumentException(
                $"The target must be [{batch}, {numClasses}, {height}, {width}] class scores or a [{batch}, {height}, {width}] " +
                $"class-index map; got [{string.Join(", ", target.Shape.ToArray())}].", nameof(target));

        var result = new Segments<T>[batch];
        for (int b = 0; b < batch; b++)
        {
            var row = engine.Reshape(engine.TensorSlice(labels, new[] { b, 0 }, new[] { 1, hw }), new[] { hw });
            int[] counts = engine.TensorBinCount(row, numClasses).ToArray();
            if (counts.Length > numClasses)
                throw new ArgumentException($"Target class indices must lie in [0, {numClasses}).", nameof(target));
            var present = new List<int>();
            for (int c = 0; c < numClasses; c++)
            {
                if (counts[c] > 0) present.Add(c);
            }

            var classMasks = engine.TensorTranspose(engine.TensorOneHot<T>(row, numClasses));                 // [K, HW]
            var masks = engine.TensorIndexSelect(classMasks, new Tensor<int>(present.ToArray(), new[] { present.Count }), 0);
            result[b] = new Segments<T>(masks, present.ToArray());
        }

        return result;
    }

    /// <summary>
    /// The minimum-cost one-to-one assignment of ground-truth rows to prediction columns of <paramref name="cost"/>
    /// (<c>[M, N]</c>), as (segment, prediction) pairs; segments left over when <c>M &gt; N</c> stay unmatched.
    /// </summary>
    internal static (int[] Segment, int[] Prediction) Match<T>(Tensor<T> cost)
    {
        if (cost is null) throw new ArgumentNullException(nameof(cost));
        if (cost.Rank != 2) throw new ArgumentException("The matching cost must be [segments, predictions].", nameof(cost));
        if (cost.Shape[0] == 0) return (Array.Empty<int>(), Array.Empty<int>());
        var assignment = new LinearAssignmentSolver<T>().Solve(cost.ToMatrix());
        var segments = new List<int>();
        var predictions = new List<int>();
        for (int i = 0; i < assignment.Length; i++)
        {
            if (assignment[i] < 0) continue;
            segments.Add(i);
            predictions.Add(assignment[i]);
        }

        return (segments.ToArray(), predictions.ToArray());
    }

    /// <summary>Mask2Former's matching cost <c>[M, N]</c> for one image.</summary>
    /// <param name="classLogits">Class logits <c>[N, K + 1]</c> (no-object last).</param>
    /// <param name="maskLogits">Mask logits <c>[N, HW]</c> (sigmoid masks).</param>
    /// <param name="segments">The image's ground-truth segments.</param>
    internal static Tensor<T> Mask2FormerCost<T>(Tensor<T> classLogits, Tensor<T> maskLogits, Segments<T> segments)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = maskLogits.Shape[0], hw = maskLogits.Shape[1], m = segments.Classes.Length, c = classLogits.Shape[1];
        var y = segments.Masks;                                                                           // [M, HW]
        var classOneHot = engine.TensorOneHot<T>(new Tensor<int>(segments.Classes, new[] { m }), c);       // [M, C]
        var probability = engine.TensorMatMul(classOneHot, engine.TensorTranspose(engine.TensorSoftmax(classLogits, 1))); // [M, N]

        var softplusSum = engine.ReduceSum(engine.Softplus(maskLogits), new[] { 1 }, false);              // [N]
        var crossEntropy = engine.TensorMultiplyScalar(
            engine.TensorSubtract(
                engine.TensorBroadcastTo(engine.Reshape(softplusSum, new[] { 1, n }), new[] { m, n }),
                engine.TensorMatMul(y, engine.TensorTranspose(maskLogits))),
            ops.FromDouble(1.0 / hw));

        var sigmoid = engine.TensorSigmoid(maskLogits);
        var overlap = engine.TensorMatMul(y, engine.TensorTranspose(sigmoid));                             // [M, N]
        var denominator = engine.TensorAdd(
            engine.TensorBroadcastTo(engine.ReduceSum(y, new[] { 1 }, true), new[] { m, n }),
            engine.TensorBroadcastTo(engine.Reshape(engine.ReduceSum(sigmoid, new[] { 1 }, false), new[] { 1, n }), new[] { m, n }));
        var dice = engine.ScalarMinusTensor(ops.One, engine.TensorDivide(
            engine.TensorAddScalar(engine.TensorMultiplyScalar(overlap, ops.FromDouble(2.0)), ops.One),
            engine.TensorAddScalar(denominator, ops.One)));

        return engine.TensorAdd(
            engine.TensorMultiplyScalar(probability, ops.FromDouble(-Mask2FormerClassWeight)),
            engine.TensorAdd(
                engine.TensorMultiplyScalar(crossEntropy, ops.FromDouble(Mask2FormerMaskWeight)),
                engine.TensorMultiplyScalar(dice, ops.FromDouble(Mask2FormerDiceWeight))));
    }

    /// <summary>
    /// Mask2Former's set-prediction loss over a batch (the loss ODISE trains with).
    /// </summary>
    /// <param name="classLogits">Class logits <c>[B, N, K + 1]</c> (no-object last).</param>
    /// <param name="maskLogits">Mask logits <c>[B, N, HW]</c>.</param>
    /// <param name="segments">Each image's ground-truth segments.</param>
    internal static Tensor<T> Mask2Former<T>(Tensor<T> classLogits, Tensor<T> maskLogits, IReadOnlyList<Segments<T>> segments)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int b = classLogits.Shape[0], n = classLogits.Shape[1], c = classLogits.Shape[2], hw = maskLogits.Shape[2];
        if (segments.Count != b) throw new ArgumentException("Give one segment set per image.", nameof(segments));

        Tensor<T>? weightedNll = null, weightSum = null, maskCrossEntropy = null, maskDice = null;
        int matchedTotal = 0;
        for (int i = 0; i < b; i++)
        {
            var cls = engine.Reshape(engine.TensorSlice(classLogits, new[] { i, 0, 0 }, new[] { 1, n, c }), new[] { n, c });
            var masks = engine.Reshape(engine.TensorSlice(maskLogits, new[] { i, 0, 0 }, new[] { 1, n, hw }), new[] { n, hw });
            var (segment, prediction) = MatchWithoutGradient(() => Mask2FormerCost(cls, masks, segments[i]));

            var targetClass = new int[n];
            var matched = new int[n];
            for (int q = 0; q < n; q++) targetClass[q] = c - 1;
            for (int p = 0; p < segment.Length; p++)
            {
                targetClass[prediction[p]] = segments[i].Classes[segment[p]];
                matched[prediction[p]] = 1;
            }

            // PyTorch's class-weighted cross-entropy: sum_q w_q * nll_q / sum_q w_q, w = 0.1 on no-object targets.
            var weights = engine.TensorAddScalar(
                engine.TensorMultiplyScalar(new Tensor<int>(matched, new[] { n }).Cast<T>(), ops.FromDouble(1.0 - Mask2FormerNoObjectWeight)),
                ops.FromDouble(Mask2FormerNoObjectWeight));
            var nll = engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(
                engine.TensorOneHot<T>(new Tensor<int>(targetClass, new[] { n }), c), engine.TensorLogSoftmax(cls, 1)), new[] { 1 }, false));
            weightedNll = Accumulate(engine, weightedNll, engine.ReduceSum(engine.TensorMultiply(weights, nll), null));
            weightSum = Accumulate(engine, weightSum, engine.ReduceSum(weights, null));

            if (segment.Length == 0) continue;
            matchedTotal += segment.Length;
            var (x, y) = MatchedPairs(masks, segments[i], segment, prediction, n);
            var bce = engine.TensorMultiplyScalar(
                engine.ReduceSum(engine.TensorSubtract(engine.Softplus(x), engine.TensorMultiply(x, y)), null),
                ops.FromDouble(1.0 / hw));
            var sigmoid = engine.TensorSigmoid(x);
            var numerator = engine.TensorAddScalar(
                engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(sigmoid, y), new[] { 1 }, false), ops.FromDouble(2.0)), ops.One);
            var denominator = engine.TensorAddScalar(
                engine.TensorAdd(engine.ReduceSum(sigmoid, new[] { 1 }, false), engine.ReduceSum(y, new[] { 1 }, false)), ops.One);
            var dice = engine.ReduceSum(engine.ScalarMinusTensor(ops.One, engine.TensorDivide(numerator, denominator)), null);
            maskCrossEntropy = Accumulate(engine, maskCrossEntropy, bce);
            maskDice = Accumulate(engine, maskDice, dice);
        }

        var classLoss = engine.TensorDivide(
            weightedNll ?? throw new InvalidOperationException("The batch is empty."),
            weightSum ?? throw new InvalidOperationException("The batch is empty."));
        var total = engine.TensorMultiplyScalar(classLoss, ops.FromDouble(Mask2FormerClassWeight));
        if (maskCrossEntropy is not null && maskDice is not null)
        {
            var perMask = ops.FromDouble(1.0 / Math.Max(1, matchedTotal));
            total = engine.TensorAdd(total, engine.TensorAdd(
                engine.TensorMultiplyScalar(maskCrossEntropy, ops.Multiply(perMask, ops.FromDouble(Mask2FormerMaskWeight))),
                engine.TensorMultiplyScalar(maskDice, ops.Multiply(perMask, ops.FromDouble(Mask2FormerDiceWeight)))));
        }

        return total;
    }

    /// <summary>
    /// kMaX-DeepLab's similarity <c>p(c_i) * Dice(m_i, m_hat_j)</c> <c>[M, N]</c> for one image, with the masks a softmax
    /// over the <c>N</c> clusters.
    /// </summary>
    internal static Tensor<T> KMaXSimilarity<T>(Tensor<T> classLogits, Tensor<T> maskLogits, Segments<T> segments)
    {
        var engine = AiDotNetEngine.Current;
        int m = segments.Classes.Length, c = classLogits.Shape[1];
        var classOneHot = engine.TensorOneHot<T>(new Tensor<int>(segments.Classes, new[] { m }), c);
        var probability = engine.TensorMatMul(classOneHot, engine.TensorTranspose(engine.TensorSoftmax(classLogits, 1))); // [M, N]
        return engine.TensorMultiply(probability, PairwiseDice(segments.Masks, engine.TensorSoftmax(maskLogits, 0)));
    }

    /// <summary>
    /// kMaX-DeepLab's loss over a batch: <c>3 * L_PQ + 0.3 * L_maskID</c>, where
    /// <c>L_PQ = 0.75 * L_pos + 0.25 * L_neg</c>,
    /// <c>L_pos = mean_i [ -sg(p_i) * Dice_i - sg(Dice_i) * log p_i ]</c> over the matched segments and
    /// <c>L_neg = mean_j [ -log p_j(no object) ]</c> over the unmatched clusters.
    /// </summary>
    /// <param name="classLogits">Class logits <c>[B, N, K + 1]</c> (no-object last).</param>
    /// <param name="maskLogits">Cluster logits <c>[B, N, HW]</c> (softmax over N gives the masks).</param>
    /// <param name="segments">Each image's ground-truth segments.</param>
    internal static Tensor<T> KMaX<T>(Tensor<T> classLogits, Tensor<T> maskLogits, IReadOnlyList<Segments<T>> segments)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int b = classLogits.Shape[0], n = classLogits.Shape[1], c = classLogits.Shape[2], hw = maskLogits.Shape[2];
        if (segments.Count != b) throw new ArgumentException("Give one segment set per image.", nameof(segments));

        Tensor<T>? total = null;
        for (int i = 0; i < b; i++)
        {
            var cls = engine.Reshape(engine.TensorSlice(classLogits, new[] { i, 0, 0 }, new[] { 1, n, c }), new[] { n, c });
            var logits = engine.Reshape(engine.TensorSlice(maskLogits, new[] { i, 0, 0 }, new[] { 1, n, hw }), new[] { n, hw });
            var (segment, prediction) = MatchWithoutGradient(
                () => engine.TensorNegate(KMaXSimilarity(cls, logits, segments[i])));
            var logProbability = engine.TensorLogSoftmax(cls, 1);
            Tensor<T> imageLoss;
            if (segment.Length == 0)
            {
                // Nothing to match: every cluster is a negative.
                var allVoid = engine.TensorNegate(engine.ReduceSum(engine.TensorNarrow(logProbability, 1, c - 1, 1), null));
                imageLoss = engine.TensorMultiplyScalar(allVoid,
                    ops.FromDouble(KMaXPanopticQualityWeight * (1 - KMaXPositiveWeight) / n));
                total = Accumulate(engine, total, imageLoss);
                continue;
            }

            int p = segment.Length;
            var querySelect = engine.TensorOneHot<T>(new Tensor<int>(prediction, new[] { p }), n);            // [P, N]
            var matchedClasses = new int[p];
            for (int k = 0; k < p; k++) matchedClasses[k] = segments[i].Classes[segment[k]];
            var classSelect = engine.TensorOneHot<T>(new Tensor<int>(matchedClasses, new[] { p }), c);       // [P, C]
            var segmentSelect = engine.TensorOneHot<T>(new Tensor<int>(segment, new[] { p }), segments[i].Classes.Length);
            var y = engine.TensorMatMul(segmentSelect, segments[i].Masks);                                    // [P, HW]

            var matchedLogProbability = engine.ReduceSum(
                engine.TensorMultiply(engine.TensorMatMul(querySelect, logProbability), classSelect), new[] { 1 }, false); // [P]
            var matchedProbability = engine.TensorExp(matchedLogProbability);
            var softMasks = engine.TensorSoftmax(logits, 0);
            var dice = Dice(engine.TensorMatMul(querySelect, softMasks), y);                                   // [P]
            var positive = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorNegate(engine.TensorAdd(
                engine.TensorMultiply(engine.StopGradient(matchedProbability), dice),
                engine.TensorMultiply(engine.StopGradient(dice), matchedLogProbability))), null), ops.FromDouble(1.0 / p));
            var panopticQuality = engine.TensorMultiplyScalar(positive, ops.FromDouble(KMaXPositiveWeight));
            if (n > p)
            {
                var unmatched = engine.ScalarMinusTensor(ops.One, engine.ReduceSum(querySelect, new[] { 0 }, false)); // [N]
                var voidLog = engine.Reshape(engine.TensorNarrow(logProbability, 1, c - 1, 1), new[] { n });
                var negative = engine.TensorMultiplyScalar(
                    engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(unmatched, voidLog), null)), ops.FromDouble(1.0 / (n - p)));
                panopticQuality = engine.TensorAdd(panopticQuality,
                    engine.TensorMultiplyScalar(negative, ops.FromDouble(1 - KMaXPositiveWeight)));
            }

            // Mask-ID cross-entropy: every pixel of a matched segment is classified (softmax over the clusters) as
            // the cluster matched to its segment.
            var pixelTarget = engine.TensorMatMul(engine.TensorTranspose(querySelect), y);                     // [N, HW]
            var coveredPixels = Math.Max(1, ops.ToInt32(engine.TensorSum(y)));
            var maskId = engine.TensorMultiplyScalar(engine.TensorNegate(engine.ReduceSum(
                engine.TensorMultiply(pixelTarget, engine.TensorLogSoftmax(logits, 0)), null)), ops.FromDouble(1.0 / coveredPixels));

            imageLoss = engine.TensorAdd(
                engine.TensorMultiplyScalar(panopticQuality, ops.FromDouble(KMaXPanopticQualityWeight)),
                engine.TensorMultiplyScalar(maskId, ops.FromDouble(KMaXMaskIdWeight)));
            total = Accumulate(engine, total, imageLoss);
        }

        return engine.TensorMultiplyScalar(
            total ?? throw new InvalidOperationException("The batch is empty."), ops.FromDouble(1.0 / b));
    }

    /// <summary>Dice coefficient <c>2 sum(a b) / (sum a + sum b)</c> of every row pair <c>[M, N]</c>.</summary>
    internal static Tensor<T> PairwiseDice<T>(Tensor<T> truth, Tensor<T> predicted)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = truth.Shape[0], n = predicted.Shape[0];
        var overlap = engine.TensorMatMul(truth, engine.TensorTranspose(predicted));
        var denominator = engine.TensorAdd(
            engine.TensorBroadcastTo(engine.ReduceSum(truth, new[] { 1 }, true), new[] { m, n }),
            engine.TensorBroadcastTo(engine.Reshape(engine.ReduceSum(predicted, new[] { 1 }, false), new[] { 1, n }), new[] { m, n }));
        return engine.TensorDivide(engine.TensorMultiplyScalar(overlap, ops.FromDouble(2.0)),
            engine.TensorClampMin(denominator, ops.FromDouble(DiceEpsilon)));
    }

    /// <summary>Dice coefficient of aligned rows <c>[P]</c>.</summary>
    private static Tensor<T> Dice<T>(Tensor<T> predicted, Tensor<T> truth)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var overlap = engine.ReduceSum(engine.TensorMultiply(predicted, truth), new[] { 1 }, false);
        var denominator = engine.TensorAdd(engine.ReduceSum(predicted, new[] { 1 }, false), engine.ReduceSum(truth, new[] { 1 }, false));
        return engine.TensorDivide(engine.TensorMultiplyScalar(overlap, ops.FromDouble(2.0)),
            engine.TensorClampMin(denominator, ops.FromDouble(DiceEpsilon)));
    }

    private static (int[] Segment, int[] Prediction) MatchWithoutGradient<T>(Func<Tensor<T>> cost)
    {
        using var noGrad = new NoGradScope<T>();
        return Match(cost());
    }

    /// <summary>The matched predictions' mask logits and their segments' masks, both <c>[P, HW]</c>.</summary>
    private static (Tensor<T> Logits, Tensor<T> Truth) MatchedPairs<T>(
        Tensor<T> maskLogits, Segments<T> segments, int[] segment, int[] prediction, int n)
    {
        var engine = AiDotNetEngine.Current;
        int p = segment.Length;
        var querySelect = engine.TensorOneHot<T>(new Tensor<int>(prediction, new[] { p }), n);
        var segmentSelect = engine.TensorOneHot<T>(new Tensor<int>(segment, new[] { p }), segments.Classes.Length);
        return (engine.TensorMatMul(querySelect, maskLogits), engine.TensorMatMul(segmentSelect, segments.Masks));
    }

    private static Tensor<T> Accumulate<T>(IEngine engine, Tensor<T>? sum, Tensor<T> term)
        => sum is null ? term : engine.TensorAdd(sum, term);
}
