using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Solvers.Assignment;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.NumericOperations;

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
/// cross-entropy (weight 0.3); the model adds the pixel-wise instance discrimination loss
/// (<c>InstanceDiscrimination</c>, weight 1.0) and the auxiliary semantic loss (<c>SemanticCrossEntropy</c>, weight 1.0).
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
        int[] assignment = new LinearAssignmentSolver<T>().Solve(cost.ToMatrix()).ToArray();
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

        // The matching cost carries no gradient, so the Dice sums run in double and are rounded to T once: summed in
        // Half, 0.5-valued sigmoids stop adding at 1024.
        var sigmoid = engine.TensorSigmoid(maskLogits).Cast<double>();
        var truth = y.Cast<double>();
        var overlap = engine.TensorMatMul(truth, engine.TensorTranspose(sigmoid));                         // [M, N]
        var denominator = engine.TensorAdd(
            engine.TensorBroadcastTo(engine.Reshape(engine.ReduceSum(truth, new[] { 1 }, false), new[] { m, 1 }), new[] { m, n }),
            engine.TensorBroadcastTo(engine.Reshape(engine.ReduceSum(sigmoid, new[] { 1 }, false), new[] { 1, n }), new[] { m, n }));
        var dice = engine.ScalarMinusTensor(1.0, engine.TensorDivide(
            engine.TensorAddScalar(engine.TensorMultiplyScalar(overlap, 2.0), 1.0),
            engine.TensorAddScalar(denominator, 1.0))).Cast<T>();

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
        var matchedWeightGap = ops.FromDouble(1.0 - Mask2FormerNoObjectWeight);
        var noObjectWeight = ops.FromDouble(Mask2FormerNoObjectWeight);
        var perPixel = ops.FromDouble(1.0 / hw);
        var two = ops.FromDouble(2.0);
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
                engine.TensorMultiplyScalar(new Tensor<int>(matched, new[] { n }).Cast<T>(), matchedWeightGap),
                noObjectWeight);
            var nll = engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(
                engine.TensorOneHot<T>(new Tensor<int>(targetClass, new[] { n }), c), engine.TensorLogSoftmax(cls, 1)), new[] { 1 }, false));
            weightedNll = Accumulate(engine, weightedNll, engine.ReduceSum(engine.TensorMultiply(weights, nll), null));
            weightSum = Accumulate(engine, weightSum, engine.ReduceSum(weights, null));

            if (segment.Length == 0) continue;
            matchedTotal += segment.Length;
            var (x, y) = MatchedPairs(masks, segments[i], segment, prediction, n);
            // The per-pixel BCE and the Dice numerator and denominator are summed in float for Half and BFloat16
            // (WideRowSums): summed in Half, 0.5-valued sigmoids stop adding at 1024.
            var bce = engine.TensorMultiplyScalar(
                WideSum(engine.TensorSubtract(engine.Softplus(x), engine.TensorMultiply(x, y))), perPixel);
            var sigmoid = engine.TensorSigmoid(x);
            var numerator = engine.TensorAddScalar(
                engine.TensorMultiplyScalar(WideRowSums(engine.TensorMultiply(sigmoid, y)), two), ops.One);
            var denominator = engine.TensorAddScalar(engine.TensorAdd(WideRowSums(sigmoid), TruthAreas(y)), ops.One);
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
            // weight / num_masks is formed in double and rounded to T once; rounding 1 / num_masks first and then
            // multiplying in T rounds twice (float: 5 * (1/3) is 1.6666667461 where 5/3 is 1.6666666269).
            double masks = Math.Max(1, matchedTotal);
            total = engine.TensorAdd(total, engine.TensorAdd(
                engine.TensorMultiplyScalar(maskCrossEntropy, ops.FromDouble(Mask2FormerMaskWeight / masks)),
                engine.TensorMultiplyScalar(maskDice, ops.FromDouble(Mask2FormerDiceWeight / masks))));
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

    /// <summary>kMaX-DeepLab's PQ-style and mask-ID terms over a batch with the paper's weights.</summary>
    internal static Tensor<T> KMaX<T>(Tensor<T> classLogits, Tensor<T> maskLogits, IReadOnlyList<Segments<T>> segments)
        => KMaX(classLogits, maskLogits, segments, KMaXLossSettings.Default);

    /// <summary>
    /// kMaX-DeepLab's Hungarian matching of every image, maximizing <c>p(c) * Dice</c> (no gradient). Yu et al. 2022,
    /// Sec. 4.1: "Hungarian matching result based on the final prediction is used to assign supervisions for all
    /// auxiliary outputs"; deeplab2's <c>MaXDeepLabLoss.call</c> (model/loss/max_deeplab_loss.py L930-L1020 at
    /// 64bc1c1a) matches on the main output once and permutes every auxiliary output with that permutation.
    /// </summary>
    /// <param name="classLogits">Class logits <c>[B, N, K + 1]</c> of the final prediction.</param>
    /// <param name="maskLogits">Cluster logits <c>[B, N, HW]</c> of the final prediction.</param>
    /// <param name="segments">Each image's ground-truth segments.</param>
    internal static (int[] Segment, int[] Prediction)[] KMaXMatching<T>(
        Tensor<T> classLogits, Tensor<T> maskLogits, IReadOnlyList<Segments<T>> segments)
    {
        if (classLogits is null) throw new ArgumentNullException(nameof(classLogits));
        if (maskLogits is null) throw new ArgumentNullException(nameof(maskLogits));
        if (segments is null) throw new ArgumentNullException(nameof(segments));
        var engine = AiDotNetEngine.Current;
        int b = classLogits.Shape[0], n = classLogits.Shape[1], c = classLogits.Shape[2], hw = maskLogits.Shape[2];
        if (segments.Count != b) throw new ArgumentException("Give one segment set per image.", nameof(segments));
        var matchings = new (int[] Segment, int[] Prediction)[b];
        for (int i = 0; i < b; i++)
        {
            var cls = engine.Reshape(engine.TensorSlice(classLogits, new[] { i, 0, 0 }, new[] { 1, n, c }), new[] { n, c });
            var logits = engine.Reshape(engine.TensorSlice(maskLogits, new[] { i, 0, 0 }, new[] { 1, n, hw }), new[] { n, hw });
            var segment = segments[i];
            matchings[i] = MatchWithoutGradient(() => engine.TensorNegate(KMaXSimilarity(cls, logits, segment)));
        }

        return matchings;
    }

    /// <summary>
    /// kMaX-DeepLab's loss over a batch: <c>w_PQ * L_PQ + w_ID * L_maskID</c>, where
    /// <c>L_PQ = alpha * L_pos + (1 - alpha) * L_neg</c>,
    /// <c>L_pos = mean_i [ -sg(p_i) * Dice_i - sg(Dice_i) * log p_i ]</c> over the matched segments and
    /// <c>L_neg = mean_j [ -log p_j(no object) ]</c> over the unmatched clusters.
    /// </summary>
    /// <param name="classLogits">Class logits <c>[B, N, K + 1]</c> (no-object last).</param>
    /// <param name="maskLogits">Cluster logits <c>[B, N, HW]</c> (softmax over N gives the masks).</param>
    /// <param name="segments">Each image's ground-truth segments.</param>
    /// <param name="settings">The loss weights (paper: 3.0, 0.3, alpha = 0.75).</param>
    /// <param name="matchings">Each image's (segment, cluster) assignment, from <see cref="KMaXMatching"/> on the final
    /// prediction; when null, this prediction is matched on its own.</param>
    internal static Tensor<T> KMaX<T>(
        Tensor<T> classLogits, Tensor<T> maskLogits, IReadOnlyList<Segments<T>> segments, KMaXLossSettings settings,
        IReadOnlyList<(int[] Segment, int[] Prediction)>? matchings = null)
    {
        if (settings is null) throw new ArgumentNullException(nameof(settings));
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int b = classLogits.Shape[0], n = classLogits.Shape[1], c = classLogits.Shape[2], hw = maskLogits.Shape[2];
        if (segments.Count != b) throw new ArgumentException("Give one segment set per image.", nameof(segments));
        var assignments = matchings ?? KMaXMatching(classLogits, maskLogits, segments);
        if (assignments.Count != b) throw new ArgumentException("Give one matching per image.", nameof(matchings));

        Tensor<T>? total = null;
        var allVoidWeight = ops.FromDouble(settings.PanopticQualityWeight * (1 - settings.PositiveWeight) / n);
        var positiveWeight = ops.FromDouble(settings.PositiveWeight);
        var negativeWeight = ops.FromDouble(1 - settings.PositiveWeight);
        var panopticQualityWeight = ops.FromDouble(settings.PanopticQualityWeight);
        var maskIdWeight = ops.FromDouble(settings.MaskIdWeight);
        for (int i = 0; i < b; i++)
        {
            var cls = engine.Reshape(engine.TensorSlice(classLogits, new[] { i, 0, 0 }, new[] { 1, n, c }), new[] { n, c });
            var logits = engine.Reshape(engine.TensorSlice(maskLogits, new[] { i, 0, 0 }, new[] { 1, n, hw }), new[] { n, hw });
            var (segment, prediction) = assignments[i];
            if (segment.Length != prediction.Length || prediction.Any(q => q < 0 || q >= n))
                throw new ArgumentException($"The matching of image {i} does not fit {n} clusters.", nameof(matchings));
            var logProbability = engine.TensorLogSoftmax(cls, 1);
            Tensor<T> imageLoss;
            if (segment.Length == 0)
            {
                // Nothing to match: every cluster is a negative.
                var allVoid = engine.TensorNegate(engine.ReduceSum(engine.TensorNarrow(logProbability, 1, c - 1, 1), null));
                imageLoss = engine.TensorMultiplyScalar(allVoid, allVoidWeight);
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
            var positive = engine.TensorDivide(engine.ReduceSum(engine.TensorNegate(engine.TensorAdd(
                engine.TensorMultiply(engine.StopGradient(matchedProbability), dice),
                engine.TensorMultiply(engine.StopGradient(dice), matchedLogProbability))), null), Count<T>(p));
            var panopticQuality = engine.TensorMultiplyScalar(positive, positiveWeight);
            if (n > p)
            {
                var unmatched = engine.ScalarMinusTensor(ops.One, engine.ReduceSum(querySelect, new[] { 0 }, false)); // [N]
                var voidLog = engine.Reshape(engine.TensorNarrow(logProbability, 1, c - 1, 1), new[] { n });
                var negative = engine.TensorDivide(
                    engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(unmatched, voidLog), null)), Count<T>(n - p));
                panopticQuality = engine.TensorAdd(panopticQuality, engine.TensorMultiplyScalar(negative, negativeWeight));
            }

            // Mask-ID cross-entropy: every pixel of a matched segment is classified (softmax over the clusters) as
            // the cluster matched to its segment. The pixel sum runs in float for Half and BFloat16.
            var pixelTarget = engine.TensorMatMul(engine.TensorTranspose(querySelect), y);                     // [N, HW]
            // Counted in int: the 0/1 sum in T stops at 2048 in Half.
            int coveredPixels = Math.Max(1, engine.TensorSum(y.Cast<int>()));
            var maskId = engine.TensorDivide(engine.TensorNegate(
                WideSum(engine.TensorMultiply(pixelTarget, engine.TensorLogSoftmax(logits, 0)))), Count<T>(coveredPixels));

            imageLoss = engine.TensorAdd(
                engine.TensorMultiplyScalar(panopticQuality, panopticQualityWeight),
                engine.TensorMultiplyScalar(maskId, maskIdWeight));
            total = Accumulate(engine, total, imageLoss);
        }

        return engine.TensorMultiplyScalar(
            total ?? throw new InvalidOperationException("The batch is empty."), ops.FromDouble(1.0 / b));
    }

    /// <summary>
    /// The pixels a kMaX auxiliary loss reads in one image (MaX-DeepLab / kMaX-DeepLab): every pixel covered by a
    /// segment when there are at most <paramref name="count"/>, otherwise <paramref name="count"/> of them drawn without
    /// replacement with probability proportional to <c>(HW / area of the pixel's segment)^temperature</c> (Gumbel top-k
    /// on <c>temperature * log(inverse area)</c>), so small segments are not drowned out by large ones.
    /// </summary>
    internal static int[] SamplePixels<T>(Segments<T> segments, int count, double temperature, Random random)
    {
        if (random is null) throw new ArgumentNullException(nameof(random));
        if (count < 1) throw new ArgumentOutOfRangeException(nameof(count), "At least one pixel must be sampled.");
        if (segments.Classes.Length == 0) return Array.Empty<int>();
        var engine = AiDotNetEngine.Current;
        var masks = segments.Masks.Cast<int>();                                                               // [M, HW]
        int hw = masks.Shape[1];
        int[] areas = engine.ReduceSum(masks, new[] { 1 }, false).ToArray();
        int[] owner = engine.TensorArgMax(masks, 0).ToArray();
        int[] covered = engine.ReduceMax(masks, new[] { 0 }, false).ToArray();
        var candidates = new List<int>(hw);
        for (int p = 0; p < hw; p++)
        {
            if (covered[p] > 0) candidates.Add(p);
        }

        if (candidates.Count <= count) return candidates.ToArray();
        var keys = new double[candidates.Count];
        var order = candidates.ToArray();
        for (int i = 0; i < order.Length; i++)
        {
            double logit = temperature * Math.Log((double)hw / areas[owner[order[i]]]);
            double gumbel = -Math.Log(-Math.Log(random.NextDouble()));
            keys[i] = -(logit + gumbel);
        }

        Array.Sort(keys, order);
        var sampled = order.Take(count).ToArray();
        Array.Sort(sampled);
        return sampled;
    }

    /// <summary>
    /// MaX-DeepLab's pixel-wise instance discrimination loss for one image (kept by kMaX-DeepLab): over the sampled
    /// pixels, the target of pixel <c>j</c> spreads evenly over the sampled pixels of its own segment and the prediction
    /// is <c>softmax_k(f_k . f_j / tau)</c> of the L2-normalized pixel features; the loss is the cross-entropy, averaged
    /// over the pixels whose loss is not zero.
    /// </summary>
    /// <param name="features">L2-normalized pixel features <c>[D, HW]</c>.</param>
    /// <param name="segments">The image's ground-truth segments.</param>
    /// <param name="sampled">The sampled pixels (<see cref="SamplePixels"/>).</param>
    /// <param name="temperature">The similarity temperature tau (0.3).</param>
    internal static Tensor<T> InstanceDiscrimination<T>(Tensor<T> features, Segments<T> segments, int[] sampled, double temperature)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (sampled is null) throw new ArgumentNullException(nameof(sampled));
        if (!(temperature > 0)) throw new ArgumentOutOfRangeException(nameof(temperature), "The temperature must be positive.");
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        if (sampled.Length == 0) return Tensor<T>.CreateDefault(new[] { 1 }, ops.Zero);
        var index = new Tensor<int>(sampled, new[] { sampled.Length });
        var f = engine.TensorIndexSelect(features, index, 1);                                                  // [D, S]
        // The target is a constant of the loss, built in double and rounded once: column totals count up to S.
        var y = engine.TensorIndexSelect(segments.Masks, index, 1).Cast<double>();                            // [M, S]
        var same = engine.TensorMatMul(engine.TensorTranspose(y), y);                                          // [S, S]
        var columnTotals = engine.TensorClampMin(engine.ReduceSum(same, new[] { 0 }, true), 1.0);
        var target = engine.TensorDivide(same, engine.TensorBroadcastTo(columnTotals, same.Shape.ToArray())).Cast<T>();
        var similarity = engine.TensorMultiplyScalar(
            engine.TensorMatMul(engine.TensorTranspose(f), f), ops.FromDouble(1.0 / temperature));              // [S, S]
        // Cross-entropy of column j over the rows k (PyTorch's class axis 1 of [B, S, S]).
        var losses = engine.TensorNegate(WideRowSums(engine.TensorTranspose(
            engine.TensorMultiply(target, engine.TensorLogSoftmax(similarity, 0)))));                          // [S]
        int nonZero = engine.TensorSum(engine.TensorGreaterThan(engine.StopGradient(losses), ops.Zero).Cast<int>());
        return engine.TensorDivide(WideSum(losses), Count<T>(Math.Max(1, nonZero)));
    }

    /// <summary>
    /// kMaX-DeepLab's auxiliary semantic segmentation loss for one image: the cross-entropy of the semantic head's
    /// logits at the sampled pixels against each pixel's class, averaged over them.
    /// </summary>
    /// <param name="logits">Semantic logits <c>[K, HW]</c>.</param>
    /// <param name="segments">The image's ground-truth segments.</param>
    /// <param name="sampled">The sampled pixels (<see cref="SamplePixels"/>; void pixels are never sampled).</param>
    internal static Tensor<T> SemanticCrossEntropy<T>(Tensor<T> logits, Segments<T> segments, int[] sampled)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (sampled is null) throw new ArgumentNullException(nameof(sampled));
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        if (sampled.Length == 0) return Tensor<T>.CreateDefault(new[] { 1 }, ops.Zero);
        int k = logits.Shape[0];
        int[] owner = engine.TensorArgMax(segments.Masks.Cast<int>(), 0).ToArray();
        var labels = new int[sampled.Length];
        for (int i = 0; i < labels.Length; i++) labels[i] = segments.Classes[owner[sampled[i]]];
        var picked = engine.TensorIndexSelect(logits, new Tensor<int>(sampled, new[] { sampled.Length }), 1);  // [K, S]
        var oneHot = engine.TensorTranspose(engine.TensorOneHot<T>(new Tensor<int>(labels, new[] { labels.Length }), k));
        var losses = engine.TensorNegate(WideRowSums(engine.TensorTranspose(
            engine.TensorMultiply(oneHot, engine.TensorLogSoftmax(picked, 0)))));                              // [S]
        return engine.TensorDivide(WideSum(losses), Count<T>(sampled.Length));
    }

    /// <summary>
    /// Dice coefficient <c>2 sum(a b) / (sum a + sum b)</c> of every row pair <c>[M, N]</c>. Only the matching reads it
    /// (no gradient), so it is computed in double and rounded to T once.
    /// </summary>
    internal static Tensor<T> PairwiseDice<T>(Tensor<T> truth, Tensor<T> predicted)
    {
        var engine = AiDotNetEngine.Current;
        int m = truth.Shape[0], n = predicted.Shape[0];
        var a = truth.Cast<double>();
        var p = predicted.Cast<double>();
        var overlap = engine.TensorMatMul(a, engine.TensorTranspose(p));
        var denominator = engine.TensorAdd(
            engine.TensorBroadcastTo(engine.Reshape(engine.ReduceSum(a, new[] { 1 }, false), new[] { m, 1 }), new[] { m, n }),
            engine.TensorBroadcastTo(engine.Reshape(engine.ReduceSum(p, new[] { 1 }, false), new[] { 1, n }), new[] { m, n }));
        return engine.TensorDivide(engine.TensorMultiplyScalar(overlap, 2.0), engine.TensorClampMin(denominator, DiceEpsilon)).Cast<T>();
    }

    /// <summary>Dice coefficient of aligned rows <c>[P]</c>; the sums run in float for Half and BFloat16.</summary>
    private static Tensor<T> Dice<T>(Tensor<T> predicted, Tensor<T> truth)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var overlap = WideRowSums(engine.TensorMultiply(predicted, truth));
        var denominator = engine.TensorAdd(WideRowSums(predicted), TruthAreas(truth));
        return engine.TensorDivide(engine.TensorMultiplyScalar(overlap, ops.FromDouble(2.0)),
            engine.TensorClampMin(denominator, ops.FromDouble(DiceEpsilon)));
    }

    /// <summary>True for the 16-bit floating types, whose sums stall: Half stops adding 0.5 at 1024.</summary>
    private static bool IsNarrowFloat<T>() => typeof(T) == typeof(Half) || typeof(T) == typeof(BFloat16);

    /// <summary>
    /// The row sums <c>[P]</c> of <paramref name="x"/> <c>[P, L]</c>. For Half and BFloat16 the sums accumulate in float
    /// and are rounded to T once (the gradient, a broadcast of the upstream gradient, is recorded on the tape); other
    /// types sum in T as before.
    /// </summary>
    internal static Tensor<T> WideRowSums<T>(Tensor<T> x)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (x.Rank != 2) throw new ArgumentException("Row sums need a [rows, columns] tensor.", nameof(x));
        var engine = AiDotNetEngine.Current;
        if (!IsNarrowFloat<T>()) return engine.ReduceSum(x, new[] { 1 }, false);
        int rows = x.Shape[0], columns = x.Shape[1];
        var sums = engine.ReduceSum(x.Cast<float>(), new[] { 1 }, false).Cast<T>();
        DifferentiableOps.RecordUnary("WideRowSums", sums, x, (gradOutput, inputs, _, _, eng, grads) =>
            DifferentiableOps.AccumulateGrad(grads, inputs[0],
                eng.TensorBroadcastTo(eng.Reshape(gradOutput, new[] { rows, 1 }), new[] { rows, columns }), eng));
        return sums;
    }

    /// <summary>The sum of every element, accumulated in float for Half and BFloat16.</summary>
    internal static Tensor<T> WideSum<T>(Tensor<T> x)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        var engine = AiDotNetEngine.Current;
        if (!IsNarrowFloat<T>()) return engine.ReduceSum(x, null);
        return WideRowSums(engine.Reshape(x, new[] { 1, x.Length }));
    }

    /// <summary>A count as a one-element tensor <c>[1]</c> of T, the divisor of a summed loss term.</summary>
    private static Tensor<T> Count<T>(int count) => new Tensor<int>(new[] { count }, new[] { 1 }).Cast<T>();

    /// <summary>
    /// The pixel count <c>[M]</c> of each binary ground-truth mask row, counted in int and converted to T once: summed
    /// in T, Half stops counting at 2048, so a 4096-pixel segment would count as 2048. The masks are constants of the
    /// loss, so no gradient flows through the count.
    /// </summary>
    private static Tensor<T> TruthAreas<T>(Tensor<T> truth)
        => AiDotNetEngine.Current.ReduceSum(truth.Cast<int>(), new[] { 1 }, false).Cast<T>();

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

/// <summary>
/// The weights and sampling of kMaX-DeepLab's training loss (Yu et al., "k-means Mask Transformer", ECCV 2022,
/// following MaX-DeepLab, Wang et al. 2021). The defaults are the reference's: PQ-style loss 3.0 with alpha 0.75,
/// mask-ID cross-entropy 0.3, pixel-wise instance discrimination 1.0 (temperature 0.3; 4096 pixels sampled at sampling
/// temperature 1.5) and the auxiliary semantic loss 1.0 (4096 pixels at sampling temperature 2.0).
/// </summary>
/// <remarks>
/// <para>Sources, checked against google-research/deeplab2 at commit 64bc1c1a, bytedance/kmax-deeplab (the authors'
/// PyTorch port) at f8e1d046 and arXiv 2207.04044v5:</para>
/// <list type="bullet">
/// <item>Loss weights 3.0 / 1.0 / 0.3 / 1.0 (PQ-style, semantic, mask-ID, instance discrimination): the paper, Sec. 4.1
/// "COCO dataset"; deeplab2 configs/coco/kmax_deeplab/kmax_meta_r50_os32.textproto L70 (semantic_loss weight 1.0),
/// L73 (pq_style_loss 3.0), L76 (mask_id_cross_entropy_loss 0.3), L79 (instance_discrimination_loss 1.0); PyTorch
/// port kmax_deeplab/config.py L39-L43.</item>
/// <item>alpha 0.75: deeplab2 model/loss/max_deeplab_loss.py L211 (focal_loss_alpha=0.75).</item>
/// <item>Instance discrimination temperature 0.3: deeplab2 model/loss/max_deeplab_loss.py L214
/// (instance_discrimination_temperature=0.3, not overridden by model/loss/loss_builder.py L171-L179); PyTorch port
/// kmax_deeplab/modeling/criterion.py L294.</item>
/// <item>Instance discrimination sampling 4096 pixels at temperature 1.5: kmax_meta_r50_os32.textproto L57-L58
/// (instance_discrimination_sample_k, instance_discrimination_sample_temperature); PyTorch port config.py L46-L47.</item>
/// <item>Semantic sampling 4096 pixels at temperature 2.0: kmax_meta_r50_os32.textproto L59-L60 (semantic_sample_k,
/// semantic_sample_temperature); PyTorch port config.py L48-L49 and criterion.py L230. (deeplab2's loss_builder.py does
/// not forward these two fields to MaXDeepLabLoss, which then falls back to the full-image semantic loss; the config
/// and the authors' port both sample.)</item>
/// </list>
/// </remarks>
internal sealed class KMaXLossSettings
{
    /// <summary>The paper's weights.</summary>
    internal static readonly KMaXLossSettings Default = new KMaXLossSettings();

    /// <summary>3.0: kmax_meta_r50_os32.textproto L73.</summary>
    internal double PanopticQualityWeight { get; init; } = MaskSetPredictionLoss.KMaXPanopticQualityWeight;
    /// <summary>0.75: max_deeplab_loss.py L211.</summary>
    internal double PositiveWeight { get; init; } = MaskSetPredictionLoss.KMaXPositiveWeight;
    /// <summary>0.3: kmax_meta_r50_os32.textproto L76.</summary>
    internal double MaskIdWeight { get; init; } = MaskSetPredictionLoss.KMaXMaskIdWeight;
    /// <summary>1.0: kmax_meta_r50_os32.textproto L79.</summary>
    internal double InstanceDiscriminationWeight { get; init; } = 1.0;
    /// <summary>0.3: max_deeplab_loss.py L214.</summary>
    internal double InstanceDiscriminationTemperature { get; init; } = 0.3;
    /// <summary>4096: kmax_meta_r50_os32.textproto L57.</summary>
    internal int InstanceDiscriminationSampleCount { get; init; } = 4096;
    /// <summary>1.5: kmax_meta_r50_os32.textproto L58.</summary>
    internal double InstanceDiscriminationSampleTemperature { get; init; } = 1.5;
    /// <summary>1.0: kmax_meta_r50_os32.textproto L70.</summary>
    internal double SemanticWeight { get; init; } = 1.0;
    /// <summary>4096: kmax_meta_r50_os32.textproto L59.</summary>
    internal int SemanticSampleCount { get; init; } = 4096;
    /// <summary>2.0: kmax_meta_r50_os32.textproto L60.</summary>
    internal double SemanticSampleTemperature { get; init; } = 2.0;
}
