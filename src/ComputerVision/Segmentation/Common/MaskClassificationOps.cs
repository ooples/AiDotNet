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
/// class per pixel (MaskFormer, Cheng et al. 2021). The heads here work in "assignment space": the decoder's
/// last convolution emits the pixel embedding <c>P</c> (<c>D</c> channels), the initial mask logits
/// <c>L0 = C0 P</c> of the <c>N</c> learnable cluster centers / queries <c>C0</c> (the convolution's weight
/// rows), and per-pixel class projections. A center update <c>C &lt;- C + A P^T</c> then changes the mask
/// logits by <c>(A P^T) P</c>, so the cross-attention runs without materializing the centers.
/// </para>
/// </remarks>
internal static class MaskClassificationOps
{
    /// <summary>Added before taking the logarithm of a probability, as MaskFormer-family code does.</summary>
    internal const double LogEpsilon = 1e-8;

    /// <summary>
    /// kMaX-DeepLab's k-means cross-attention (Yu et al. 2022, Eq. 4): each pixel is assigned to the cluster
    /// center with the largest affinity (argmax over the <c>N</c> centers, a hard assignment), and every
    /// center adds the sum of the pixel embeddings assigned to it.
    /// </summary>
    /// <param name="raw">Decoder output <c>[B, D + N + K + 1, H, W]</c>: pixel embedding, initial cluster
    /// logits and per-pixel class projections (K classes plus void).</param>
    /// <param name="pixelDim">Pixel embedding width <c>D</c>.</param>
    /// <param name="numClusters">Number of cluster centers <c>N</c>.</param>
    /// <param name="iterations">Number of k-means cross-attention updates (the kMaX decoder layers).</param>
    /// <returns>Mask logits <c>[B, N, H*W]</c> and class logits <c>[B, N, K + 1]</c>.</returns>
    /// <remarks>
    /// Pixel embeddings are L2-normalized along <c>D</c> so the affinities stay bounded. Keys and values share
    /// the pixel embedding on the mask path; the class head acts on the per-pixel class projections, so a
    /// center's class logits are the sum of the projections of the pixels assigned to it over all updates.
    /// The argmax is not differentiable (as in the paper); gradients reach the network through the summed
    /// pixel embeddings and projections.
    /// </remarks>
    internal static (Tensor<T> MaskLogits, Tensor<T> ClassLogits) KMeansCrossAttention<T>(
        Tensor<T> raw, int pixelDim, int numClusters, int iterations)
    {
        if (raw is null) throw new ArgumentNullException(nameof(raw));
        if (raw.Rank != 4) throw new ArgumentException("The head input must be [B, C, H, W].", nameof(raw));
        if (iterations < 1) throw new ArgumentOutOfRangeException(nameof(iterations), "At least one k-means update is required.");
        int b = raw.Shape[0], c = raw.Shape[1], hw = raw.Shape[2] * raw.Shape[3];
        int classWidth = c - pixelDim - numClusters;
        if (pixelDim < 1 || numClusters < 1 || classWidth < 2)
            throw new ArgumentException(
                $"The head input has {c} channels; it needs {pixelDim} pixel, {numClusters} cluster and at least 2 class channels.",
                nameof(raw));

        var engine = AiDotNetEngine.Current;
        var flat = engine.Reshape(raw, new[] { b, c, hw });
        var pixel = NormalizeAxis(engine.TensorSlice(flat, new[] { 0, 0, 0 }, new[] { b, pixelDim, hw }), 1);
        var logits = engine.TensorSlice(flat, new[] { 0, pixelDim, 0 }, new[] { b, numClusters, hw });
        var classProjection = engine.TensorSlice(flat, new[] { 0, pixelDim + numClusters, 0 }, new[] { b, classWidth, hw });
        var pixelT = engine.TensorPermute(pixel, new[] { 0, 2, 1 });
        var classProjectionT = engine.TensorPermute(classProjection, new[] { 0, 2, 1 });

        Tensor<T>? classLogits = null;
        for (int t = 0; t < iterations; t++)
        {
            // Hard cluster assignment: one-hot of the argmax over the N centers, [B, N, HW].
            var assignment = engine.TensorPermute(
                engine.TensorOneHot<T>(engine.TensorArgMax(logits, 1), numClusters), new[] { 0, 2, 1 });
            var centerUpdate = engine.TensorMatMul(assignment, pixelT);                 // [B, N, D]
            logits = engine.TensorAdd(logits, engine.TensorMatMul(centerUpdate, pixel)); // [B, N, HW]
            var classUpdate = engine.TensorMatMul(assignment, classProjectionT);         // [B, N, K + 1]
            classLogits = classLogits is null ? classUpdate : engine.TensorAdd(classLogits, classUpdate);
        }

        return (logits, classLogits ?? throw new InvalidOperationException("No k-means update ran."));
    }

    /// <summary>
    /// Mask2Former's masked cross-attention (Cheng et al. 2022), the transformer decoder ODISE uses: each query
    /// attends (softmax over pixels) only to the pixels its previous mask marks as foreground
    /// (<c>sigmoid(logit) &gt;= 0.5</c>), and attends to every pixel when that mask is empty.
    /// </summary>
    /// <param name="raw">Decoder output <c>[B, D + N + E + 1, H, W]</c>: pixel embedding, initial query
    /// mask logits, per-pixel text-space embeddings (<c>E</c>) and a per-pixel void score.</param>
    /// <param name="pixelDim">Pixel embedding width <c>D</c>.</param>
    /// <param name="numQueries">Number of queries <c>N</c>.</param>
    /// <param name="embeddingDim">Text-space embedding width <c>E</c>.</param>
    /// <param name="iterations">Number of decoder layers.</param>
    /// <returns>Mask logits <c>[B, N, HW]</c>, query embeddings <c>[B, N, E]</c> (the attention-weighted
    /// text-space features summed over the layers) and void logits <c>[B, N, 1]</c>.</returns>
    internal static (Tensor<T> MaskLogits, Tensor<T> QueryEmbeddings, Tensor<T> VoidLogits) MaskedAttention<T>(
        Tensor<T> raw, int pixelDim, int numQueries, int embeddingDim, int iterations)
    {
        if (raw is null) throw new ArgumentNullException(nameof(raw));
        if (raw.Rank != 4) throw new ArgumentException("The head input must be [B, C, H, W].", nameof(raw));
        if (iterations < 1) throw new ArgumentOutOfRangeException(nameof(iterations), "At least one decoder layer is required.");
        int b = raw.Shape[0], c = raw.Shape[1], hw = raw.Shape[2] * raw.Shape[3];
        if (pixelDim < 1 || numQueries < 1 || embeddingDim < 1 || c != pixelDim + numQueries + embeddingDim + 1)
            throw new ArgumentException(
                $"The head input has {c} channels; it needs {pixelDim} pixel, {numQueries} query, {embeddingDim} " +
                "embedding channels and one void channel.", nameof(raw));

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var flat = engine.Reshape(raw, new[] { b, c, hw });
        var pixel = engine.TensorSlice(flat, new[] { 0, 0, 0 }, new[] { b, pixelDim, hw });
        var logits = engine.TensorSlice(flat, new[] { 0, pixelDim, 0 }, new[] { b, numQueries, hw });
        var embedding = engine.TensorSlice(flat, new[] { 0, pixelDim + numQueries, 0 }, new[] { b, embeddingDim, hw });
        var voidScore = engine.TensorSlice(flat, new[] { 0, pixelDim + numQueries + embeddingDim, 0 }, new[] { b, 1, hw });
        var pixelT = engine.TensorPermute(pixel, new[] { 0, 2, 1 });
        var embeddingT = engine.TensorPermute(embedding, new[] { 0, 2, 1 });
        var voidT = engine.TensorPermute(voidScore, new[] { 0, 2, 1 });
        var blocked = Tensor<T>.CreateDefault(new[] { b, numQueries, hw }, ops.FromDouble(-1e9));
        var zeros = Tensor<T>.CreateDefault(new[] { b, numQueries, hw }, ops.Zero);
        var one = ops.One;

        Tensor<T>? queryEmbeddings = null;
        Tensor<T>? voidLogits = null;
        for (int t = 0; t < iterations; t++)
        {
            // 1 where the query may not attend: background of a non-empty previous mask.
            var foreground = engine.TensorGreaterEqual(logits, zeros);
            var anyForeground = engine.ReduceMax(foreground, new[] { 2 }, true);
            var masked = engine.TensorMultiply(
                engine.ScalarMinusTensor(one, foreground),
                engine.TensorBroadcastTo(anyForeground, foreground.Shape.ToArray()));

            var weights = engine.TensorSoftmax(engine.TensorWhere(masked, blocked, logits), 2); // [B, N, HW]
            var queryUpdate = engine.TensorMatMul(weights, pixelT);                                 // [B, N, D]
            logits = engine.TensorAdd(logits, engine.TensorMatMul(queryUpdate, pixel));
            var embeddingUpdate = engine.TensorMatMul(weights, embeddingT);                         // [B, N, E]
            var voidUpdate = engine.TensorMatMul(weights, voidT);                                   // [B, N, 1]
            queryEmbeddings = queryEmbeddings is null ? embeddingUpdate : engine.TensorAdd(queryEmbeddings, embeddingUpdate);
            voidLogits = voidLogits is null ? voidUpdate : engine.TensorAdd(voidLogits, voidUpdate);
        }

        return (logits,
            queryEmbeddings ?? throw new InvalidOperationException("No decoder layer ran."),
            voidLogits ?? throw new InvalidOperationException("No decoder layer ran."));
    }

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

    /// <summary>L2-normalizes <paramref name="x"/> along <paramref name="axis"/> with <c>sqrt(sum(x^2) + 1e-24)</c>.</summary>
    internal static Tensor<T> NormalizeAxis<T>(Tensor<T> x, int axis)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var sumSquares = engine.ReduceSum(engine.TensorSquare(x), new[] { axis }, true);
        var norm = engine.TensorSqrt(engine.TensorAddScalar(sumSquares, ops.FromDouble(1e-24)));
        return engine.TensorDivide(x, engine.TensorBroadcastTo(norm, x.Shape.ToArray()));
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
        var index = engine.TensorArange<T>(0, n, 1);
        var indexColumn = engine.TensorBroadcastTo(engine.Reshape(index, new[] { n, 1 }), new[] { n, n });
        var indexRow = engine.TensorBroadcastTo(engine.Reshape(index, new[] { 1, n }), new[] { n, n });
        // rank_i = #{j : s_j > s_i} + #{j < i : s_j == s_i}
        var ahead = engine.TensorAdd(
            engine.TensorGreaterThan(row, column),
            engine.TensorMultiply(engine.TensorEquals(row, column), engine.TensorLessThan(indexRow, indexColumn)));
        int[] rank = engine.ReduceSum(ahead, new[] { 1 }, false).Cast<int>().ToArray();

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
        var pixelCount = engine.ReduceSum(binary, new[] { 1 }, false);
        var maskScores = engine.TensorDivide(
            engine.ReduceSum(engine.TensorMultiply(masks, binary), new[] { 1 }, false),
            engine.TensorClampMin(pixelCount, ops.One));
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
        int[] originalCount = pixelCount.Cast<int>().ToArray();
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
        // Mean probability of each class over its unclaimed pixels.
        var ownProbability = engine.TensorGatherClassValues(
            engine.TensorPermute(semanticProbabilities, new[] { 1, 0 }), semanticLabels.Cast<T>());
        var probabilitySums = engine.TensorScatterAdd(
            new Tensor<T>(new[] { numClasses + 1 }), binned, engine.TensorMultiply(ownProbability, unclaimed.Cast<T>()), 0);
        double[] stuffConfidence = engine.TensorDivide(
            probabilitySums, engine.TensorClampMin(areaTensor.Cast<T>(), ops.One)).Cast<double>().ToArray();
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
