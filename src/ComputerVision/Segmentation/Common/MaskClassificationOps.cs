using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Mask-classification heads and panoptic merges shared by the query-based panoptic models
/// (kMaX-DeepLab, ODISE) and the semantic-plus-instance merge used by CUPS, built from whole-tensor
/// engine operations.
/// </summary>
/// <remarks>
/// <para>
/// A mask-classification model predicts <c>N</c> masks and one class distribution per mask instead of a
/// class per pixel (MaskFormer, Cheng et al. 2021). The query decoders that produce them (kMaX-DeepLab's k-means
/// decoder and Mask2Former's masked-attention decoder, which ODISE uses) are in <see cref="KMaXTransformerDecoder"/>
/// and <see cref="Mask2FormerTransformerDecoder"/>.
/// </para>
/// </remarks>
internal static class MaskClassificationOps
{
    /// <summary>Added before taking the logarithm of a probability, as MaskFormer-family code does.</summary>
    internal const double LogEpsilon = 1e-8;

    /// <summary>
    /// MaskFormer's semantic inference: <c>semseg[k, p] = sum_n classProb[n, k] * maskProb[n, p]</c>, returned
    /// as <c>log(semseg + 1e-8)</c> so it can be trained with a logits loss and argmaxed like class logits.
    /// </summary>
    /// <param name="classProbabilities">Per-mask class probabilities <c>[B, N, K]</c> (void excluded).</param>
    /// <param name="maskProbabilities">Mask probabilities <c>[B, N, HW]</c>.</param>
    /// <param name="height">Output height.</param>
    /// <param name="width">Output width.</param>
    /// <returns>Log semantic scores <c>[B, K, H, W]</c>.</returns>
    internal static Tensor<T> SemanticLogScores<T>(Tensor<T> classProbabilities, Tensor<T> maskProbabilities, int height, int width)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int b = classProbabilities.Shape[0], k = classProbabilities.Shape[2];
        var semantic = engine.TensorMatMul(engine.TensorPermute(classProbabilities, new[] { 0, 2, 1 }), maskProbabilities);
        var logScores = engine.TensorLog(engine.TensorAddScalar(semantic, ops.FromDouble(LogEpsilon)));
        return engine.Reshape(logScores, new[] { b, k, height, width });
    }

    /// <summary>
    /// L2-normalizes <paramref name="x"/> along <paramref name="axis"/>: <c>x / sqrt(sum(x^2) + eps)</c>.
    /// </summary>
    /// <remarks>
    /// Each slice is first divided by its largest magnitude (the direction is unchanged), so the squares stay in
    /// <c>[0, 1]</c>: squaring the raw values overflows Half (65504) once a component exceeds 256, which turned
    /// (300, 400) into (0, 0). <c>eps</c> is 1e-24 or, where <typeparamref name="T"/> flushes that to zero, the
    /// smallest power-of-two multiple of it that <typeparamref name="T"/> holds, so an all-zero slice still divides
    /// by a positive number.
    /// </remarks>
    internal static Tensor<T> NormalizeAxis<T>(Tensor<T> x, int axis)
    {
        var engine = AiDotNetEngine.Current;
        var shape = x.Shape.ToArray();
        var epsilon = PositiveTolerance<T>(1e-24);
        var scale = engine.TensorClampMin(engine.ReduceMax(engine.TensorAbs(x), new[] { axis }, true), epsilon);
        var scaled = engine.TensorDivide(x, engine.TensorBroadcastTo(scale, shape));
        var sumSquares = engine.ReduceSum(engine.TensorSquare(scaled), new[] { axis }, true);
        var norm = engine.TensorSqrt(engine.TensorAddScalar(sumSquares, epsilon));
        return engine.TensorDivide(scaled, engine.TensorBroadcastTo(norm, shape));
    }

    /// <summary>Half's smallest normal value, 2^-14.</summary>
    private const double HalfSmallestNormal = 6.103515625e-05;

    /// <summary>Half's largest value below one, 1 - 2^-11.</summary>
    private const double HalfLargestBelowOne = 0.99951171875;

    /// <summary>
    /// <paramref name="value"/> in <typeparamref name="T"/>, or Half's smallest normal value (2^-14) when
    /// <typeparamref name="T"/> rounds <paramref name="value"/> to zero (Half flushes 1e-8 and smaller to 0).
    /// </summary>
    internal static T PositiveTolerance<T>(double value)
    {
        if (!(value > 0)) throw new ArgumentOutOfRangeException(nameof(value), "A tolerance must be positive.");
        var ops = MathHelper.GetNumericOperations<T>();
        var rounded = ops.FromDouble(value);
        return ops.ToDouble(rounded) > 0 ? rounded : ops.FromDouble(Math.Max(value, HalfSmallestNormal));
    }

    /// <summary>
    /// <c>1 - value</c> in <typeparamref name="T"/>, or Half's largest value below one (1 - 2^-11) when
    /// <typeparamref name="T"/> rounds <c>1 - value</c> to one (Half rounds 1 - 1e-7 to 1).
    /// </summary>
    internal static T BelowOne<T>(double value)
    {
        if (!(value > 0 && value < 1)) throw new ArgumentOutOfRangeException(nameof(value), "The gap below one must lie in (0, 1).");
        var ops = MathHelper.GetNumericOperations<T>();
        var rounded = ops.FromDouble(1.0 - value);
        return ops.ToDouble(rounded) < 1.0 ? rounded : ops.FromDouble(Math.Min(1.0 - value, HalfLargestBelowOne));
    }

    /// <summary>
    /// The mean of each row of <paramref name="values"/> <c>[R, P]</c> over the entries where <paramref name="mask"/>
    /// is 1 (0 for an empty row), given the per-row counts <paramref name="counts"/> <c>[R]</c>, reduced in double
    /// and rounded to T once.
    /// </summary>
    /// <remarks>
    /// A row sum in Half accumulates in Half: it overflows past 65504 and, long before that, stops growing once its
    /// spacing exceeds twice the addend (3000 probabilities of 0.75 sum to about 2048, not 2250). The engine has no
    /// Half reduction with a wider accumulator, so the sum is taken in double. These means are post-processing
    /// statistics and carry no gradient.
    /// </remarks>
    internal static Tensor<T> MeanOverMask<T>(Tensor<T> values, Tensor<T> mask, Tensor<int> counts)
    {
        var engine = AiDotNetEngine.Current;
        var sums = engine.ReduceSum(engine.TensorMultiply(values.Cast<double>(), mask.Cast<double>()), new[] { 1 }, false);
        return engine.TensorDivide(sums, engine.TensorClampMin(counts, 1).Cast<double>()).Cast<T>();
    }

    /// <summary>
    /// Returns the indices of <paramref name="scores"/> in descending order, ties broken by the lower index
    /// (a composite (score, index) key, so the order does not depend on a sort's stability).
    /// </summary>
    /// <exception cref="ArgumentException">A score is NaN.</exception>
    internal static int[] DescendingOrder<T>(Tensor<T> scores)
    {
        var engine = AiDotNetEngine.Current;
        int n = scores.Length;
        if (n == 0) return Array.Empty<int>();
        var column = engine.TensorBroadcastTo(engine.Reshape(scores, new[] { n, 1 }), new[] { n, n }); // [i, j] = s_i
        var row = engine.TensorBroadcastTo(engine.Reshape(scores, new[] { 1, n }), new[] { n, n });    // [i, j] = s_j
        // Indices and counts stay in int: Half holds integers exactly only up to 2048, so indices compared in T
        // tie above it and a rank summed in T stops counting there.
        var index = engine.TensorArange<int>(0, n, 1);
        var indexColumn = engine.TensorBroadcastTo(engine.Reshape(index, new[] { n, 1 }), new[] { n, n });
        var indexRow = engine.TensorBroadcastTo(engine.Reshape(index, new[] { 1, n }), new[] { n, n });
        // rank_i = #{j : s_j > s_i} + #{j < i : s_j == s_i}
        var ahead = engine.TensorAdd(
            engine.TensorGreaterThan(row, column).Cast<int>(),
            engine.TensorMultiply(engine.TensorEquals(row, column).Cast<int>(), engine.TensorLessThan(indexRow, indexColumn)));
        int[] rank = engine.ReduceSum(ahead, new[] { 1 }, false).ToArray();

        var order = new int[n];
        var filled = new bool[n];
        for (int i = 0; i < n; i++)
        {
            int r = rank[i];
            if (r < 0 || r >= n || filled[r])
                throw new ArgumentException("Scores must not be NaN.", nameof(scores));
            filled[r] = true;
            order[r] = i;
        }

        return order;
    }

    /// <summary>
    /// kMaX-DeepLab's panoptic post-processing (bytedance/kmax-deeplab <c>panoptic_inference</c>): masks are
    /// the pixels whose softmax-over-clusters probability exceeds the object-mask threshold; clusters are
    /// visited by <c>classScore^a * maskScore^b</c> (descending); a cluster becomes a segment when its class
    /// score clears the thing/stuff threshold and more than <c>overlap</c> of its mask is still unclaimed;
    /// stuff clusters of the same class merge into one segment.
    /// </summary>
    /// <param name="classProbabilities">Class probabilities <c>[N, K + 1]</c> (softmax, void last).</param>
    /// <param name="maskProbabilities">Mask probabilities <c>[N, H, W]</c> (softmax over the N clusters).</param>
    /// <param name="isThingClass">Whether each of the K classes is a thing class.</param>
    /// <param name="settings">Thresholds.</param>
    internal static PanopticSegmentationResult<T> MergeKMaX<T>(
        Tensor<T> classProbabilities, Tensor<T> maskProbabilities, bool[] isThingClass, KMaXPanopticSettings settings)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = maskProbabilities.Shape[0], h = maskProbabilities.Shape[1], w = maskProbabilities.Shape[2], hw = h * w;
        int k = classProbabilities.Shape[1] - 1;
        if (isThingClass.Length != k)
            throw new ArgumentException($"Expected {k} thing/stuff flags, got {isThingClass.Length}.", nameof(isThingClass));

        var classOnly = engine.TensorSlice(classProbabilities, new[] { 0, 0 }, new[] { n, k });
        var classScores = engine.ReduceMax(classOnly, new[] { 1 }, false);             // [N]
        var classLabels = engine.TensorArgMax(classOnly, 1);                           // [N] int
        var masks = engine.Reshape(maskProbabilities, new[] { n, hw });
        var binary = engine.TensorGreaterThan(masks, ops.FromDouble(settings.ObjectMaskThreshold));
        // Pixel counts in int: a 0/1 sum in Half stops at 2048.
        var pixelCount = engine.ReduceSum(binary.Cast<int>(), new[] { 1 }, false);
        var maskScores = MeanOverMask(masks, binary, pixelCount);
        var reorder = engine.TensorMultiply(
            engine.TensorPower(classScores, ops.FromDouble(settings.ReorderClassWeight)),
            engine.TensorPower(maskScores, ops.FromDouble(settings.ReorderMaskWeight)));
        int[] order = DescendingOrder(reorder);

        var thingTable = BoolTable<T>(isThingClass);
        var queryIsThing = engine.TensorIndexSelect(thingTable, classLabels, 0);
        var threshold = engine.TensorAdd(
            engine.TensorMultiplyScalar(queryIsThing, ops.FromDouble(settings.ClassThresholdThing)),
            engine.TensorMultiplyScalar(engine.ScalarMinusTensor(ops.One, queryIsThing), ops.FromDouble(settings.ClassThresholdStuff)));
        int[] confident = engine.TensorGreaterThan(classScores, threshold).Cast<int>().ToArray();
        int[] labels = classLabels.ToArray();
        int[] originalCount = pixelCount.ToArray();
        double[] confidence = classScores.Cast<double>().ToArray();
        var binaryInt = binary.Cast<int>();

        var segmentMap = new Tensor<int>(new[] { hw });
        var segments = new List<SegmentInfo>();
        var stuffSegment = new Dictionary<int, int>();
        foreach (int q in order)
        {
            if (confident[q] == 0) continue;
            var queryMask = engine.Reshape(engine.TensorSlice(binaryInt, new[] { q, 0 }, new[] { 1, hw }), new[] { hw });
            var newMask = engine.TensorMultiply(queryMask, engine.TensorEquals(segmentMap, 0));
            int newCount = engine.TensorSum(newMask);
            if (!(newCount > originalCount[q] * settings.OverlapThreshold)) continue;
            int label = labels[q];
            bool thing = isThingClass[label];
            if (!thing && stuffSegment.TryGetValue(label, out int existing))
            {
                segmentMap = engine.TensorAdd(segmentMap, engine.TensorMultiplyScalar(newMask, existing));
                continue;
            }

            int id = segments.Count + 1;
            if (!thing) stuffSegment[label] = id;
            segmentMap = engine.TensorAdd(segmentMap, engine.TensorMultiplyScalar(newMask, id));
            segments.Add(new SegmentInfo(id, label, thing, confidence[q]));
        }

        return Assemble<T>(segmentMap, h, w, segments);
    }

    /// <summary>
    /// Mask2Former's panoptic inference (<c>panoptic_inference</c>), which ODISE uses: queries whose best class
    /// is not void and scores above the object-mask threshold are kept; each pixel goes to the kept query
    /// with the largest <c>score * maskProb</c>; a query becomes a segment when its share of its own mask
    /// (<c>maskProb &gt;= 0.5</c>) is at least the overlap threshold; stuff queries of one class merge.
    /// </summary>
    /// <param name="classProbabilities">Class probabilities <c>[N, K + 1]</c> (void last).</param>
    /// <param name="maskProbabilities">Sigmoid mask probabilities <c>[N, H, W]</c>.</param>
    /// <param name="isThingClass">Whether each of the K classes is a thing class.</param>
    /// <param name="objectMaskThreshold">Minimum class score of a kept query.</param>
    /// <param name="overlapThreshold">Minimum share of a query's own mask it must win.</param>
    internal static PanopticSegmentationResult<T> MergeMask2Former<T>(
        Tensor<T> classProbabilities, Tensor<T> maskProbabilities, bool[] isThingClass,
        double objectMaskThreshold, double overlapThreshold)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = maskProbabilities.Shape[0], h = maskProbabilities.Shape[1], w = maskProbabilities.Shape[2], hw = h * w;
        int k = classProbabilities.Shape[1] - 1;
        if (isThingClass.Length != k)
            throw new ArgumentException($"Expected {k} thing/stuff flags, got {isThingClass.Length}.", nameof(isThingClass));

        var scores = engine.ReduceMax(classProbabilities, new[] { 1 }, false);   // [N], over K + 1
        var labelTensor = engine.TensorArgMax(classProbabilities, 1);            // [N]
        var notVoid = engine.TensorNotEquals(labelTensor, k).Cast<T>();
        var keep = engine.TensorMultiply(notVoid, engine.TensorGreaterThan(scores, ops.FromDouble(objectMaskThreshold)));
        var masks = engine.Reshape(maskProbabilities, new[] { n, hw });
        // Dropped queries get -1 so they never win the per-pixel argmax over kept queries.
        var weighted = engine.TensorAdd(
            engine.TensorMultiply(engine.TensorBroadcastTo(engine.Reshape(engine.TensorMultiply(scores, keep), new[] { n, 1 }), new[] { n, hw }), masks),
            engine.TensorBroadcastTo(engine.Reshape(engine.TensorSubtractScalar(keep, ops.One), new[] { n, 1 }), new[] { n, hw }));
        var owner = engine.TensorArgMax(weighted, 0);                              // [HW]
        var ownedOneHot = engine.TensorPermute(engine.TensorOneHot<int>(owner, n), new[] { 1, 0 }); // [N, HW]
        var ownMask = engine.TensorGreaterEqual(masks, Tensor<T>.CreateDefault(new[] { n, hw }, ops.FromDouble(0.5))).Cast<int>();
        var finalMask = engine.TensorMultiply(ownedOneHot, ownMask);
        int[] ownedArea = engine.ReduceSum(ownedOneHot, new[] { 1 }, false).ToArray();
        int[] originalArea = engine.ReduceSum(ownMask, new[] { 1 }, false).ToArray();
        int[] finalArea = engine.ReduceSum(finalMask, new[] { 1 }, false).ToArray();
        int[] kept = keep.Cast<int>().ToArray();
        int[] labels = labelTensor.ToArray();
        double[] confidence = scores.Cast<double>().ToArray();

        var segmentMap = new Tensor<int>(new[] { hw });
        var segments = new List<SegmentInfo>();
        var stuffSegment = new Dictionary<int, int>();
        for (int q = 0; q < n; q++)
        {
            if (kept[q] == 0) continue;
            if (ownedArea[q] <= 0 || originalArea[q] <= 0 || finalArea[q] <= 0) continue;
            if ((double)ownedArea[q] / originalArea[q] < overlapThreshold) continue;
            var queryMask = engine.Reshape(engine.TensorSlice(finalMask, new[] { q, 0 }, new[] { 1, hw }), new[] { hw });
            int label = labels[q];
            bool thing = isThingClass[label];
            if (!thing && stuffSegment.TryGetValue(label, out int existing))
            {
                segmentMap = engine.TensorAdd(segmentMap, engine.TensorMultiplyScalar(queryMask, existing));
                continue;
            }

            int id = segments.Count + 1;
            if (!thing) stuffSegment[label] = id;
            segmentMap = engine.TensorAdd(segmentMap, engine.TensorMultiplyScalar(queryMask, id));
            segments.Add(new SegmentInfo(id, label, thing, confidence[q]));
        }

        return Assemble<T>(segmentMap, h, w, segments);
    }

    /// <summary>
    /// Panoptic FPN's combine step (Kirillov et al. 2019; detectron2 <c>combine_semantic_and_instance_outputs</c>),
    /// the inference merge of the Panoptic Cascade Mask R-CNN that CUPS trains: instances are pasted by
    /// descending score while their score reaches the instance threshold and at most <c>overlap</c> of each
    /// instance is already claimed; then every stuff class of the semantic map whose unclaimed area reaches
    /// the stuff-area threshold becomes one segment.
    /// </summary>
    /// <param name="semanticLabels">Semantic class map <c>[HW]</c> (int).</param>
    /// <param name="semanticProbabilities">Class probabilities <c>[K, HW]</c>; a stuff segment's confidence is
    /// the mean probability of its class over the segment.</param>
    /// <param name="instanceMasks">Binary instance masks <c>[M, HW]</c> (int).</param>
    /// <param name="instanceClasses">Class of each instance.</param>
    /// <param name="instanceScores">Score of each instance <c>[M]</c>.</param>
    /// <param name="isThingClass">Whether each class is a thing class.</param>
    /// <param name="height">Map height.</param>
    /// <param name="width">Map width.</param>
    /// <param name="settings">Thresholds; <see cref="PanopticFpnSettings.StuffAreaThreshold"/> is counted
    /// in input-resolution pixels, so each map pixel counts as <paramref name="pixelArea"/> of them.</param>
    /// <param name="pixelArea">Input pixels covered by one map pixel.</param>
    internal static PanopticSegmentationResult<T> MergePanopticFpn<T>(
        Tensor<int> semanticLabels, Tensor<T> semanticProbabilities, Tensor<int> instanceMasks, int[] instanceClasses,
        Tensor<T> instanceScores, bool[] isThingClass, int height, int width, PanopticFpnSettings settings, double pixelArea)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = instanceClasses.Length, hw = height * width;
        var segmentMap = new Tensor<int>(new[] { hw });
        var segments = new List<SegmentInfo>();

        if (m > 0)
        {
            int[] order = DescendingOrder(instanceScores);
            int[] passes = engine.TensorGreaterEqual(
                instanceScores, Tensor<T>.CreateDefault(new[] { m }, ops.FromDouble(settings.InstanceScoreThreshold))).Cast<int>().ToArray();
            double[] confidence = instanceScores.Cast<double>().ToArray();
            int[] area = engine.ReduceSum(instanceMasks, new[] { 1 }, false).ToArray();
            foreach (int i in order)
            {
                if (passes[i] == 0) break;
                if (area[i] == 0) continue;
                var mask = engine.Reshape(engine.TensorSlice(instanceMasks, new[] { i, 0 }, new[] { 1, hw }), new[] { hw });
                var free = engine.TensorEquals(segmentMap, 0);
                int intersect = area[i] - engine.TensorSum(engine.TensorMultiply(mask, free));
                if ((double)intersect / area[i] > settings.OverlapThreshold) continue;
                var newMask = engine.TensorMultiply(mask, free);
                int id = segments.Count + 1;
                segmentMap = engine.TensorAdd(segmentMap, engine.TensorMultiplyScalar(newMask, id));
                segments.Add(new SegmentInfo(id, instanceClasses[i], true, confidence[i]));
            }
        }

        int numClasses = isThingClass.Length;
        var unclaimed = engine.TensorEquals(segmentMap, 0);
        // Unclaimed pixels keep their class; claimed pixels move to an extra bin that is never read.
        var binned = engine.TensorAdd(
            engine.TensorMultiply(semanticLabels, unclaimed),
            engine.TensorMultiplyScalar(engine.ScalarMinusTensor(1, unclaimed), numClasses));
        var areaTensor = engine.TensorBinCount(binned, numClasses + 1);
        int[] stuffArea = areaTensor.ToArray();
        // Mean probability of each class over its unclaimed pixels, summed in double: a per-class sum in Half stops
        // growing near 2048 and overflows past 65504 (see MeanOverMask). The confidence is reported as a double.
        var ownProbability = engine.TensorGatherClassValues(
            engine.TensorPermute(semanticProbabilities, new[] { 1, 0 }), semanticLabels.Cast<T>()).Cast<double>();
        var probabilitySums = engine.TensorScatterAdd(
            new Tensor<double>(new[] { numClasses + 1 }), binned, engine.TensorMultiply(ownProbability, unclaimed.Cast<double>()), 0);
        double[] stuffConfidence = engine.TensorDivide(
            probabilitySums, engine.TensorClampMin(areaTensor, 1).Cast<double>()).ToArray();
        for (int cls = 0; cls < numClasses; cls++)
        {
            if (isThingClass[cls] || stuffArea[cls] == 0) continue;
            if (stuffArea[cls] * pixelArea < settings.StuffAreaThreshold) continue;
            var mask = engine.TensorMultiply(engine.TensorEquals(semanticLabels, cls), unclaimed);
            int id = segments.Count + 1;
            segmentMap = engine.TensorAdd(segmentMap, engine.TensorMultiplyScalar(mask, id));
            segments.Add(new SegmentInfo(id, cls, false, stuffConfidence[cls]));
        }

        return Assemble<T>(segmentMap, height, width, segments);
    }

    /// <summary>
    /// Builds the result maps from a segment-id map (<c>0</c> = void): <c>SemanticMap</c> holds the class (or
    /// <c>-1</c> on void pixels), <c>InstanceMap</c> the segment id of thing segments (<c>0</c> for stuff and
    /// void), and <c>PanopticMap</c> <c>classId * 1000 + instanceId</c> (<c>-1</c> on void).
    /// </summary>
    internal static PanopticSegmentationResult<T> Assemble<T>(Tensor<int> segmentMap, int height, int width, List<SegmentInfo> segments)
    {
        var engine = AiDotNetEngine.Current;
        int s = segments.Count;
        var semantic = new int[s + 1];
        var instance = new int[s + 1];
        var panoptic = new int[s + 1];
        semantic[0] = -1;
        panoptic[0] = -1;
        foreach (var segment in segments)
        {
            int instanceId = segment.IsThing ? segment.SegmentId : 0;
            semantic[segment.SegmentId] = segment.ClassId;
            instance[segment.SegmentId] = instanceId;
            panoptic[segment.SegmentId] = segment.ClassId * 1000 + instanceId;
        }

        Tensor<T> Lookup(int[] table) => engine.Reshape(
            engine.TensorIndexSelect(new Tensor<int>(table, new[] { s + 1 }), segmentMap, 0).Cast<T>(),
            new[] { height, width });

        int[] areas = engine.TensorBinCount(segmentMap, s + 1).ToArray();
        var result = new List<PanopticSegment<T>>(s);
        foreach (var segment in segments)
        {
            result.Add(new PanopticSegment<T>
            {
                SegmentId = segment.SegmentId,
                ClassId = segment.ClassId,
                IsThing = segment.IsThing,
                Confidence = segment.Confidence,
                Area = areas[segment.SegmentId],
                Mask = engine.Reshape(engine.TensorEquals(segmentMap, segment.SegmentId).Cast<T>(), new[] { height, width }),
            });
        }

        return new PanopticSegmentationResult<T>
        {
            SemanticMap = Lookup(semantic),
            InstanceMap = Lookup(instance),
            PanopticMap = Lookup(panoptic),
            Segments = result,
        };
    }

    /// <summary>A 0/1 lookup table <c>[K]</c> of <paramref name="flags"/>.</summary>
    internal static Tensor<T> BoolTable<T>(bool[] flags)
    {
        var values = new int[flags.Length];
        for (int i = 0; i < flags.Length; i++) values[i] = flags[i] ? 1 : 0;
        return new Tensor<int>(values, new[] { flags.Length }).Cast<T>();
    }

    /// <summary>One panoptic segment before the result maps are built.</summary>
    internal readonly struct SegmentInfo
    {
        internal SegmentInfo(int segmentId, int classId, bool isThing, double confidence)
        {
            SegmentId = segmentId;
            ClassId = classId;
            IsThing = isThing;
            Confidence = confidence;
        }

        internal int SegmentId { get; }
        internal int ClassId { get; }
        internal bool IsThing { get; }
        internal double Confidence { get; }
    }
}

/// <summary>Thresholds of kMaX-DeepLab's panoptic post-processing.</summary>
internal readonly struct KMaXPanopticSettings
{
    internal KMaXPanopticSettings(double classThresholdThing, double classThresholdStuff, double objectMaskThreshold,
        double overlapThreshold, double reorderClassWeight, double reorderMaskWeight)
    {
        ClassThresholdThing = classThresholdThing;
        ClassThresholdStuff = classThresholdStuff;
        ObjectMaskThreshold = objectMaskThreshold;
        OverlapThreshold = overlapThreshold;
        ReorderClassWeight = reorderClassWeight;
        ReorderMaskWeight = reorderMaskWeight;
    }

    internal double ClassThresholdThing { get; }
    internal double ClassThresholdStuff { get; }
    internal double ObjectMaskThreshold { get; }
    internal double OverlapThreshold { get; }
    internal double ReorderClassWeight { get; }
    internal double ReorderMaskWeight { get; }
}

/// <summary>Thresholds of Panoptic FPN's semantic-plus-instance combine step.</summary>
internal readonly struct PanopticFpnSettings
{
    internal PanopticFpnSettings(double instanceScoreThreshold, double overlapThreshold, double stuffAreaThreshold)
    {
        InstanceScoreThreshold = instanceScoreThreshold;
        OverlapThreshold = overlapThreshold;
        StuffAreaThreshold = stuffAreaThreshold;
    }

    internal double InstanceScoreThreshold { get; }
    internal double OverlapThreshold { get; }
    internal double StuffAreaThreshold { get; }
}
