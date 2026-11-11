using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Static helper methods for common segmentation tensor operations (argmax, softmax, etc.)
/// used by models implementing segmentation interfaces.
/// </summary>
public static class SegmentationTensorOps
{
    /// <summary>
    /// Removes the batch dimension from a tensor if present, keeping only batch index 0.
    /// Converts [B, ...] to [...]. If already unbatched, returns as-is.
    /// </summary>
    public static Tensor<T> EnsureUnbatched<T>(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        if (tensor.Rank == 4)
        {
            // [B, C, H, W] -> [C, H, W]: batch 0 as a new tensor.
            int c = tensor.Shape[1], h = tensor.Shape[2], w = tensor.Shape[3];
            var engine = AiDotNetEngine.Current;
            return engine.Reshape(engine.TensorSlice(tensor, [0, 0, 0, 0], [1, c, h, w]), [c, h, w]);
        }
        return tensor;
    }

    /// <summary>
    /// Removes the leading batch dimension from a class-index map, returning [H, W].
    /// </summary>
    /// <typeparam name="T">The tensor element type.</typeparam>
    /// <param name="classMap">A class map shaped [H,W] or [B,H,W].</param>
    /// <returns>The unbatched [H,W] class map. For a batched map, batch index 0 is selected.</returns>
    public static Tensor<T> EnsureUnbatchedClassMap<T>(Tensor<T> classMap)
    {
        if (classMap is null) throw new ArgumentNullException(nameof(classMap));
        if (classMap.Rank == 2) return classMap;
        if (classMap.Rank != 3)
            throw new ArgumentException(
                $"Class map must be [H,W] or [B,H,W]; got rank {classMap.Rank}.",
                nameof(classMap));
        if (classMap.Shape[0] <= 0)
            throw new ArgumentException("A batched class map must contain at least one batch.", nameof(classMap));

        int height = classMap.Shape[1];
        int width = classMap.Shape[2];
        var engine = AiDotNetEngine.Current;
        return engine.Reshape(engine.TensorSlice(classMap, [0, 0, 0], [1, height, width]), [height, width]);
    }

    /// <summary>
    /// Computes argmax along the class dimension, producing a per-pixel class index map.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="logits">Logits tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Class index map [H, W] or [B, H, W].</returns>
    public static Tensor<T> ArgmaxAlongClassDim<T>(Tensor<T> logits)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (logits.Rank < 3 || logits.Rank > 4)
            throw new ArgumentException("Logits must be rank 3 [C, H, W] or rank 4 [B, C, H, W].", nameof(logits));

        // First maximum wins (a strictly greater value replaces the best, NaN never does), and the
        // index is returned as a value of T, exactly as the per-pixel scan did.
        int classAxis = logits.Rank - 3;
        int numClasses = logits.Shape[classAxis];
        var classIndices = AiDotNetEngine.Current.TensorArgMax(logits, classAxis);
        return CvTensorOps<T>.IndicesAsValues(classIndices, numClasses);
    }

    /// <summary>
    /// Computes softmax along the class dimension, producing per-pixel probabilities.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="logits">Logits tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Probability map [C, H, W] or [B, C, H, W] with values in [0, 1].</returns>
    public static Tensor<T> SoftmaxAlongClassDim<T>(Tensor<T> logits)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (logits.Rank < 3 || logits.Rank > 4)
            throw new ArgumentException("Logits must be rank 3 [C, H, W] or rank 4 [B, C, H, W].", nameof(logits));

        // Max-subtracted softmax over the class axis.
        return AiDotNetEngine.Current.Softmax(logits, logits.Rank - 3);
    }

    /// <summary>
    /// Labels connected components of pixels matching a target class in the class map.
    /// Uses BFS flood fill with 4-connectivity.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="classMap">Per-pixel class indices [H, W].</param>
    /// <param name="targetClass">The class value to find components of.</param>
    /// <returns>A label map [H, W] where each pixel's value is its component ID (1-based), and the total component count.</returns>
    public static (Tensor<T> LabelMap, int ComponentCount) LabelConnectedComponents<T>(
        Tensor<T> classMap, int targetClass)
    {
        if (classMap is null) throw new ArgumentNullException(nameof(classMap));
        if (classMap.Rank != 2)
            throw new ArgumentException("classMap must be rank 2 [H, W].", nameof(classMap));

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int h = classMap.Shape[0], w = classMap.Shape[1];

        // Distance of every pixel's class value from the target, computed once. A component starts at a
        // pixel within 0.5 of the target (inclusive) and grows into neighbours strictly within 0.5, as before.
        var distance = engine.TensorAbs(engine.TensorSubtractScalar(classMap, numOps.FromDouble(targetClass)));
        var halfMap = CvTensorOps<T>.Fill([h, w], 0.5);
        var canStart = CvTensorOps<T>.ToDoubleArray(engine.TensorLessEqual(distance, halfMap));
        var canGrow = CvTensorOps<T>.ToDoubleArray(engine.TensorLessThan(distance, halfMap));

        // The flood fill itself is a graph traversal over those host-side flags, labelling in raster order.
        var labels = new int[h * w];
        var visited = new bool[h * w];
        int componentCount = 0;
        var queue = new Queue<int>();
        for (int idx = 0; idx < h * w; idx++)
        {
            if (visited[idx] || canStart[idx] == 0) continue;

            componentCount++;
            queue.Enqueue(idx);
            visited[idx] = true;
            while (queue.Count > 0)
            {
                int cur = queue.Dequeue();
                int r = cur / w, c = cur % w;
                labels[cur] = componentCount;

                // 4-connected neighbors
                if (r > 0) Visit(cur - w);
                if (r < h - 1) Visit(cur + w);
                if (c > 0) Visit(cur - 1);
                if (c < w - 1) Visit(cur + 1);
            }
        }

        var labelMap = engine.Reshape(CvTensorOps<T>.IndicesAsValues(labels, componentCount + 1), [h, w]);
        return (labelMap, componentCount);

        void Visit(int neighbor)
        {
            if (visited[neighbor] || canGrow[neighbor] == 0) return;
            visited[neighbor] = true;
            queue.Enqueue(neighbor);
        }
    }

    /// <summary>
    /// Thresholds a tensor to produce a binary {0, 1} mask.
    /// </summary>
    public static Tensor<T> ThresholdMask<T>(Tensor<T> values, double threshold = 0.5)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        var numOps = MathHelper.GetNumericOperations<T>();
        return AiDotNetEngine.Current.TensorGreaterThan(values, numOps.FromDouble(threshold));
    }

    /// <summary>
    /// Applies element-wise sigmoid to a tensor: 1 / (1 + exp(-x)).
    /// </summary>
    public static Tensor<T> Sigmoid<T>(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        return AiDotNetEngine.Current.Sigmoid(input);
    }

    /// <summary>
    /// Generates a 2D Gaussian attention mask centered at (cx, cy) with given sigma.
    /// Output shape is [H, W] with values in (0, 1].
    /// </summary>
    public static Tensor<T> GaussianMask<T>(int height, int width, double cx, double cy, double sigma = 10.0)
    {
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "Must be positive.");
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "Must be positive.");
        if (sigma <= 0) throw new ArgumentOutOfRangeException(nameof(sigma), "Must be positive.");

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // exp(-(dx^2 + dy^2) / (2 sigma^2)) with dx a column ramp and dy a row ramp, broadcast to [H, W].
        var dx = engine.TensorSubtractScalar(engine.TensorArange<T>(0, width, 1), numOps.FromDouble(cx));
        var dy = engine.TensorSubtractScalar(engine.TensorArange<T>(0, height, 1), numOps.FromDouble(cy));
        var dx2 = engine.TensorBroadcastTo(engine.Reshape(engine.TensorSquare(dx), [1, width]), [height, width]);
        var dy2 = engine.TensorBroadcastTo(engine.Reshape(engine.TensorSquare(dy), [height, 1]), [height, width]);
        var scaled = engine.TensorMultiplyScalar(engine.TensorAdd(dx2, dy2), numOps.FromDouble(-1.0 / (2.0 * sigma * sigma)));
        return engine.TensorExp(scaled);
    }

    /// <summary>
    /// Creates a spatial mask that is 1.0 inside the box [x1, y1, x2, y2) and 0.0 outside.
    /// </summary>
    public static Tensor<T> BoxMask<T>(int height, int width, int x1, int y1, int x2, int y2)
    {
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "height must be positive.");
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "width must be positive.");

        var result = new Tensor<T>([height, width]);
        int clampX1 = Math.Max(0, x1), clampY1 = Math.Max(0, y1);
        int clampX2 = Math.Min(width, x2), clampY2 = Math.Min(height, y2);
        if (clampX2 <= clampX1 || clampY2 <= clampY1) return result;

        var inside = CvTensorOps<T>.Fill([clampY2 - clampY1, clampX2 - clampX1], 1.0);
        return AiDotNetEngine.Current.TensorSetSlice(result, inside, [clampY1, clampX1]);
    }

    /// <summary>
    /// Computes weighted sum across the channel dimension.
    /// Features shape: [C, H, W] or [B, C, H, W], weights length: C. Output: [H, W].
    /// For 4D input, uses batch index 0.
    /// </summary>
    public static Tensor<T> WeightedChannelSum<T>(Tensor<T> features, double[] weights)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (features.Rank < 3 || features.Rank > 4)
            throw new ArgumentException("features must be rank 3 [C, H, W] or rank 4 [B, C, H, W].", nameof(features));

        var engine = AiDotNetEngine.Current;
        var chw = EnsureUnbatched(features);
        int c = chw.Shape[0], h = chw.Shape[1], w = chw.Shape[2];
        int numWeights = Math.Min(c, weights.Length);
        if (numWeights == 0) return new Tensor<T>([h, w]);

        // The first numWeights channels, each scaled by its weight, summed over the channel axis.
        var used = engine.TensorSlice(chw, [0, 0, 0], [numWeights, h, w]);
        var channelWeights = CvTensorOps<T>.FromDoubleArray(weights.Take(numWeights).ToArray(), [numWeights, 1, 1]);
        var weighted = engine.TensorMultiply(used, engine.TensorBroadcastTo(channelWeights, [numWeights, h, w]));
        return engine.ReduceSum(weighted, [0], keepDims: false);
    }

    /// <summary>
    /// Generates deterministic channel weights from text using character-level hashing.
    /// The weights are normalized to unit length for use as cosine-similarity queries.
    /// </summary>
    public static double[] TextToWeights(string text, int numChannels)
    {
        if (numChannels <= 0) throw new ArgumentOutOfRangeException(nameof(numChannels), "numChannels must be positive.");

        var weights = new double[numChannels];
        if (string.IsNullOrEmpty(text)) return weights;

        // Deterministic hash-based embedding: each character contributes to multiple channels
        for (int i = 0; i < text.Length; i++)
        {
            int charVal = text[i];
            // Mix character value across channels using prime-based hashing
            for (int ch = 0; ch < numChannels; ch++)
            {
                int hash = (charVal * 31 + i * 97 + ch * 53) % 1000;
                weights[ch] += Math.Sin(hash * 0.00628318530718); // sin(hash * 2*pi/1000)
            }
        }

        // Normalize to unit length
        double norm = 0;
        for (int i = 0; i < numChannels; i++) norm += weights[i] * weights[i];
        norm = Math.Sqrt(norm);
        if (norm > 1e-8)
            for (int i = 0; i < numChannels; i++) weights[i] /= norm;

        return weights;
    }

    /// <summary>
    /// Computes per-pixel cosine similarity between two feature maps [C, H, W] or [B, C, H, W].
    /// Output: [H, W] with values in [-1, 1]. For 4D input, uses batch index 0.
    /// </summary>
    public static Tensor<T> PixelAffinity<T>(Tensor<T> features1, Tensor<T> features2)
    {
        if (features1 is null) throw new ArgumentNullException(nameof(features1));
        if (features2 is null) throw new ArgumentNullException(nameof(features2));
        if (features1.Rank < 3 || features1.Rank > 4)
            throw new ArgumentException("features1 must be rank 3 [C, H, W] or rank 4 [B, C, H, W].", nameof(features1));
        if (features2.Rank != features1.Rank)
            throw new ArgumentException("features2 must have the same rank as features1.", nameof(features2));

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var a = EnsureUnbatched(features1);
        var b = EnsureUnbatched(features2);
        int c = a.Shape[0], h = a.Shape[1], w = a.Shape[2];
        if (b.Shape[0] != c)
            b = engine.TensorSlice(b, [0, 0, 0], [c, h, w]);

        // dot / (|a| |b|) over the channel axis, 0 where the norm product is at most 1e-8.
        var dot = engine.ReduceSum(engine.TensorMultiply(a, b), [0], keepDims: false);
        var norm1 = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(a), [0], keepDims: false));
        var norm2 = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(b), [0], keepDims: false));
        var denom = engine.TensorMultiply(norm1, norm2);
        var valid = engine.TensorGreaterThan(denom, numOps.FromDouble(1e-8));
        var safeDenom = engine.TensorWhere(valid, denom, CvTensorOps<T>.Fill([h, w], 1.0));
        return engine.TensorWhere(valid, engine.TensorDivide(dot, safeDenom), new Tensor<T>([h, w]));
    }

    /// <summary>
    /// Warps masks from a reference frame to a target frame using pixel-wise affinity scores.
    /// Reference masks: [N, H, W], affinity: [H_a, W_a] with values in [0, 1].
    /// Output: [N, H, W] warped masks weighted by affinity.
    /// When affinity spatial dims differ from mask dims, uses nearest-neighbor sampling.
    /// </summary>
    public static Tensor<T> WarpMasksByAffinity<T>(Tensor<T> refMasks, Tensor<T> affinity)
    {
        if (refMasks is null) throw new ArgumentNullException(nameof(refMasks));
        if (affinity is null) throw new ArgumentNullException(nameof(affinity));
        if (refMasks.Rank != 3)
            throw new ArgumentException("refMasks must be rank 3 [N, H, W].", nameof(refMasks));
        if (affinity.Rank != 2)
            throw new ArgumentException("affinity must be rank 2 [H, W].", nameof(affinity));

        var engine = AiDotNetEngine.Current;
        int n = refMasks.Shape[0], h = refMasks.Shape[1], w = refMasks.Shape[2];
        int aH = affinity.Shape[0], aW = affinity.Shape[1];

        // Nearest-neighbour source row/column for every target row/column (integer index maps).
        var rowIndex = new int[h];
        for (int row = 0; row < h; row++)
            rowIndex[row] = aH == h ? row : (int)((long)row * aH / h);
        var colIndex = new int[w];
        for (int col = 0; col < w; col++)
            colIndex[col] = aW == w ? col : (int)((long)col * aW / w);

        var sampled = CvTensorOps<T>.Select(CvTensorOps<T>.Select(affinity, rowIndex, 0), colIndex, 1);
        return engine.TensorMultiply(refMasks, engine.TensorBroadcastTo(engine.Reshape(sampled, [1, h, w]), [n, h, w]));
    }

    /// <summary>
    /// The per-pixel channel mean of a <c>[C, H, W]</c> embedding, <c>[H, W]</c>: the base score map the
    /// promptable models gate with their prompt.
    /// </summary>
    internal static Tensor<T> ChannelMean<T>(Tensor<T> features)
        => AiDotNetEngine.Current.ReduceMean(features, [0], keepDims: false);

    /// <summary>
    /// Point-prompt attention <c>[H, W]</c>: the signed sum of one Gaussian per point (sigma = max(H, W) / 10),
    /// added where <paramref name="positive"/> is non-zero and subtracted elsewhere.
    /// </summary>
    /// <param name="points">Point coordinates <c>[N, 2+]</c> as (x, y).</param>
    /// <param name="positive">Per-point indicator; only its first N elements are read.</param>
    /// <param name="h">Map height.</param>
    /// <param name="w">Map width.</param>
    internal static Tensor<T> PointPromptAttention<T>(Tensor<T> points, Tensor<T> positive, int h, int w)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int numPts = points.Shape[0];
        if (numPts == 0) return new Tensor<T>([h, w]);

        double sigma = Math.Max(h, w) / 10.0;
        var px = engine.Reshape(engine.TensorSlice(points, [0, 0], [numPts, 1]), [numPts, 1, 1]);
        var py = engine.Reshape(engine.TensorSlice(points, [0, 1], [numPts, 1]), [numPts, 1, 1]);
        var cols = engine.Reshape(engine.TensorArange<T>(0, w, 1), [1, 1, w]);
        var rows = engine.Reshape(engine.TensorArange<T>(0, h, 1), [1, h, 1]);
        var dx2 = engine.TensorBroadcastTo(engine.TensorSquare(engine.TensorSubtract(
            engine.TensorBroadcastTo(cols, [numPts, 1, w]), engine.TensorBroadcastTo(px, [numPts, 1, w]))), [numPts, h, w]);
        var dy2 = engine.TensorBroadcastTo(engine.TensorSquare(engine.TensorSubtract(
            engine.TensorBroadcastTo(rows, [numPts, h, 1]), engine.TensorBroadcastTo(py, [numPts, h, 1]))), [numPts, h, w]);
        var gaussians = engine.TensorExp(engine.TensorMultiplyScalar(
            engine.TensorAdd(dx2, dy2), numOps.FromDouble(-1.0 / (2.0 * sigma * sigma))));

        // sign = +1 for a positive point, -1 otherwise: 2 * [positive != 0] - 1.
        var indicator = engine.TensorNotEquals(engine.TensorSlice(engine.Reshape(positive, [positive.Length]), [0], [numPts]), numOps.Zero);
        var sign = engine.TensorSubtractScalar(engine.TensorMultiplyScalar(indicator, numOps.FromDouble(2.0)), numOps.One);
        var signed = engine.TensorMultiply(gaussians, engine.TensorBroadcastTo(engine.Reshape(sign, [numPts, 1, 1]), [numPts, h, w]));
        return engine.ReduceSum(signed, [0], keepDims: false);
    }

    /// <summary>
    /// Point-prompt score map: the channel mean of <paramref name="features"/> gated by the sigmoid of
    /// <see cref="PointPromptAttention"/>, where a point is positive when its label equals one.
    /// </summary>
    internal static Tensor<T> PointPromptScoreMap<T>(Tensor<T> features, Tensor<T> points, Tensor<T> labels)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int h = features.Shape[1], w = features.Shape[2];
        var positive = AiDotNetEngine.Current.TensorEquals(labels, numOps.One);
        var attention = PointPromptAttention(points, positive, h, w);
        return AiDotNetEngine.Current.TensorMultiply(ChannelMean(features), Sigmoid(attention));
    }

    /// <summary>
    /// Scales every channel of a <c>[C, H, W]</c> feature map by the top-left <c>[H, W]</c> of
    /// <paramref name="spatialMask"/>.
    /// </summary>
    internal static Tensor<T> ModulateBySpatialMask<T>(Tensor<T> features, Tensor<T> spatialMask, int h, int w)
    {
        var engine = AiDotNetEngine.Current;
        var gate = spatialMask.Shape[0] == h && spatialMask.Shape[1] == w
            ? spatialMask
            : engine.TensorSlice(spatialMask, [0, 0], [h, w]);
        int numC = features.Shape[0];
        return engine.TensorMultiply(features, engine.TensorBroadcastTo(engine.Reshape(gate, [1, h, w]), [numC, h, w]));
    }

    /// <summary>
    /// Box-prompt score map: the channel mean of <paramref name="features"/> inside the box
    /// <c>[x1, y1, x2, y2)</c> (coordinates truncated to integers), zero outside.
    /// </summary>
    internal static Tensor<T> BoxPromptScoreMap<T>(Tensor<T> features, Tensor<T> box)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int h = features.Shape[1], w = features.Shape[2];
        int bx1 = (int)numOps.ToDouble(box[0]), by1 = (int)numOps.ToDouble(box[1]);
        int bx2 = (int)numOps.ToDouble(box[2]), by2 = (int)numOps.ToDouble(box[3]);
        var boxMask = BoxMask<T>(h, w, bx1, by1, bx2, by2);
        return AiDotNetEngine.Current.TensorMultiply(ChannelMean(features), boxMask);
    }

    /// <summary>
    /// Mask-prompt score map: the channel mean of <paramref name="features"/> where the <c>[Hm, Wm]</c> prompt
    /// mask is positive. Pixels outside the prompt mask's extent count as not selected.
    /// </summary>
    internal static Tensor<T> MaskPromptScoreMap<T>(Tensor<T> features, Tensor<T> mask)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int h = features.Shape[1], w = features.Shape[2];
        int coverH = Math.Min(h, mask.Shape[0]), coverW = Math.Min(w, mask.Shape[1]);
        var gate = new Tensor<T>([h, w]);
        if (coverH > 0 && coverW > 0)
        {
            var covered = engine.TensorSlice(mask, [0, 0], [coverH, coverW]);
            gate = engine.TensorSetSlice(gate, engine.TensorGreaterThan(covered, numOps.Zero), [0, 0]);
        }

        return engine.TensorMultiply(ChannelMean(features), gate);
    }

    /// <summary>
    /// Turns a prompt score map into the three SAM-style candidate masks (sigmoid probability at least 0.3,
    /// 0.5 and 0.7), each scored by its mean probability and by its stability - the ratio of the areas at the
    /// threshold plus and minus 0.05 - with a 4x-subsampled copy of the score map as the low-resolution logits.
    /// </summary>
    internal static PromptedSegmentationResult<T> BuildPromptMaskResult<T>(Tensor<T> scoreMap, int h, int w)
    {
        var engine = AiDotNetEngine.Current;
        var probs = engine.TensorBroadcastTo(engine.Reshape(Sigmoid(scoreMap), [1, h, w]), [3, h, w]);

        Tensor<T> AtLeast(double[] thresholds)
            => engine.TensorGreaterEqual(probs, engine.TensorBroadcastTo(CvTensorOps<T>.FromDoubleArray(thresholds, [3, 1, 1]), [3, h, w]));

        var masks = AtLeast([0.3, 0.5, 0.7]);
        var areaLo = engine.ReduceSum(AtLeast([0.3 - 0.05, 0.5 - 0.05, 0.7 - 0.05]), [1, 2], keepDims: false);
        var areaHi = engine.ReduceSum(AtLeast([0.3 + 0.05, 0.5 + 0.05, 0.7 + 0.05]), [1, 2], keepDims: false);
        var area = engine.ReduceSum(masks, [1, 2], keepDims: false);
        var confSum = engine.ReduceSum(engine.TensorMultiply(masks, probs), [1, 2], keepDims: false);

        var lowRes = SubsampleNearest(scoreMap, h, w, 4);
        return new PromptedSegmentationResult<T>
        {
            Masks = masks,
            Scores = CvTensorOps<T>.ToDoubleArray(SafeRatio(confSum, area)),
            LowResLogits = engine.Reshape(lowRes, [1, lowRes.Shape[0], lowRes.Shape[1]]),
            StabilityScores = CvTensorOps<T>.ToDoubleArray(SafeRatio(areaHi, areaLo)),
        };
    }

    /// <summary>
    /// One candidate per class of the argmax map (up to 100, in class order) covering at least four pixels,
    /// scored by the mean class probability over its pixels; a single empty candidate when none qualifies.
    /// </summary>
    internal static List<PromptedSegmentationResult<T>> ClassMaskCandidates<T>(Tensor<T> features, Tensor<T> probs)
    {
        var engine = AiDotNetEngine.Current;
        var classMap = ArgmaxAlongClassDim(features);
        int numC = features.Shape[0], h = classMap.Shape[0], w = classMap.Shape[1];

        // All class masks at once: [C, H, W] with 1 where the argmax is that class.
        var classIds = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, numC, 1), [numC, 1, 1]), [numC, h, w]);
        var classMasks = engine.TensorEquals(engine.TensorBroadcastTo(engine.Reshape(classMap, [1, h, w]), [numC, h, w]), classIds);
        var areas = CvTensorOps<T>.ToDoubleArray(engine.ReduceSum(classMasks, [1, 2], keepDims: false));
        var classProbs = engine.TensorSlice(probs, [0, 0, 0], [numC, h, w]);
        var confSums = CvTensorOps<T>.ToDoubleArray(engine.ReduceSum(engine.TensorMultiply(classMasks, classProbs), [1, 2], keepDims: false));

        var results = new List<PromptedSegmentationResult<T>>();
        for (int cls = 0; cls < numC && results.Count < 100; cls++)
        {
            double area = areas[cls];
            if (area < 4) continue;
            double conf = confSums[cls] / area;
            var mask = engine.TensorSlice(classMasks, [cls, 0, 0], [1, h, w]);
            results.Add(new PromptedSegmentationResult<T> { Masks = mask, Scores = [conf], StabilityScores = [conf > 0.7 ? 0.95 : conf] });
        }

        if (results.Count == 0)
            results.Add(new PromptedSegmentationResult<T> { Masks = new Tensor<T>([1, features.Shape[1], features.Shape[2]]), Scores = [0.0], StabilityScores = [0.0] });
        return results;
    }

    /// <summary><c>numerator / denominator</c> where the denominator is positive, else zero.</summary>
    internal static Tensor<T> SafeRatio<T>(Tensor<T> numerator, Tensor<T> denominator)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var positive = engine.TensorGreaterThan(denominator, numOps.Zero);
        var ones = engine.TensorOnesLike(denominator);
        return engine.TensorWhere(positive, engine.TensorDivide(numerator, engine.TensorWhere(positive, denominator, ones)), engine.TensorZerosLike(denominator));
    }

    /// <summary>
    /// Every <paramref name="factor"/>-th row and column of an <c>[H, W]</c> map (clamped to the last one),
    /// <c>[max(1, H / factor), max(1, W / factor)]</c>.
    /// </summary>
    internal static Tensor<T> SubsampleNearest<T>(Tensor<T> map, int h, int w, int factor)
    {
        int lrH = Math.Max(1, h / factor), lrW = Math.Max(1, w / factor);
        var rowIndex = new int[lrH];
        for (int y = 0; y < lrH; y++) rowIndex[y] = Math.Min(y * factor, h - 1);
        var colIndex = new int[lrW];
        for (int x = 0; x < lrW; x++) colIndex[x] = Math.Min(x * factor, w - 1);
        return CvTensorOps<T>.Select(CvTensorOps<T>.Select(map, rowIndex, 0), colIndex, 1);
    }

    /// <summary>
    /// Panoptic labelling of per-pixel class logits <c>[C, H, W]</c>: classes below
    /// <paramref name="numStuff"/> become one stuff segment each, and every 4-connected component of a thing
    /// class becomes an instance with a running id starting at 1. Panoptic labels are <c>class * 1000</c> for
    /// stuff and <c>class * 1000 + instance id</c> for things; each segment's confidence is its mean class
    /// probability.
    /// </summary>
    /// <param name="logits">Class logits <c>[C, H, W]</c>.</param>
    /// <param name="numStuff">Number of stuff classes (ids <c>0 .. numStuff-1</c>).</param>
    /// <param name="numClasses">Total number of classes considered.</param>
    /// <param name="segmentIdIsPanopticLabel">When true a segment's id is its panoptic label; otherwise
    /// it is the class id for stuff and the instance id for things.</param>
    internal static PanopticSegmentationResult<T> PanopticFromLogits<T>(
        Tensor<T> logits, int numStuff, int numClasses, bool segmentIdIsPanopticLabel)
    {
        const int PanopticLabelDivisor = 1000;
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var probMap = SoftmaxAlongClassDim(logits);
        var semanticMap = ArgmaxAlongClassDim(logits);
        int numC = logits.Shape[0], h = semanticMap.Shape[0], w = semanticMap.Shape[1];
        var instanceMap = new Tensor<T>([h, w]);
        var panopticMap = new Tensor<T>([h, w]);
        var segments = new List<PanopticSegment<T>>();

        // Stuff classes: one mask per class id (classes past the logits' channels never occur).
        int stuffCount = Math.Min(numStuff, numC);
        if (stuffCount > 0)
        {
            var stuffIds = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, stuffCount, 1), [stuffCount, 1, 1]), [stuffCount, h, w]);
            var stuffMasks = engine.TensorEquals(engine.TensorBroadcastTo(engine.Reshape(semanticMap, [1, h, w]), [stuffCount, h, w]), stuffIds);
            var stuffLabels = engine.TensorMultiplyScalar(stuffIds, numOps.FromDouble(PanopticLabelDivisor));
            panopticMap = engine.ReduceSum(engine.TensorMultiply(stuffMasks, stuffLabels), [0], keepDims: false);
            var stuffProbs = engine.TensorSlice(probMap, [0, 0, 0], [stuffCount, h, w]);
            var areas = ToDoubleArray(engine.ReduceSum(stuffMasks, [1, 2], keepDims: false));
            var confSums = ToDoubleArray(engine.ReduceSum(engine.TensorMultiply(stuffMasks, stuffProbs), [1, 2], keepDims: false));
            for (int cls = 0; cls < stuffCount; cls++)
            {
                int area = (int)areas[cls];
                if (area > 0)
                {
                    segments.Add(new PanopticSegment<T>
                    {
                        SegmentId = segmentIdIsPanopticLabel ? cls * PanopticLabelDivisor : cls,
                        ClassId = cls,
                        IsThing = false,
                        Confidence = confSums[cls] / area,
                        Area = area
                    });
                }
            }
        }

        // Thing classes: one instance per connected component, numbered in raster order of discovery.
        int nextInstId = 1;
        for (int cls = numStuff; cls < numClasses && cls < numC; cls++)
        {
            var (labelMap, count) = LabelConnectedComponents(semanticMap, cls);
            if (count == 0) continue;

            // Component k (1-based) becomes instance nextInstId + k - 1.
            var inComponent = engine.TensorGreaterThan(labelMap, numOps.Zero);
            var instanceIds = engine.TensorMultiply(inComponent, engine.TensorAddScalar(labelMap, numOps.FromDouble(nextInstId - 1)));
            instanceMap = engine.TensorAdd(instanceMap, instanceIds);
            panopticMap = engine.TensorAdd(panopticMap, engine.TensorAdd(
                instanceIds, engine.TensorMultiplyScalar(inComponent, numOps.FromDouble(cls * PanopticLabelDivisor))));

            var componentIds = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(1, count + 1, 1), [count, 1, 1]), [count, h, w]);
            var componentMasks = engine.TensorEquals(engine.TensorBroadcastTo(engine.Reshape(labelMap, [1, h, w]), [count, h, w]), componentIds);
            var classProb = engine.TensorBroadcastTo(engine.TensorSlice(probMap, [cls, 0, 0], [1, h, w]), [count, h, w]);
            var areas = ToDoubleArray(engine.ReduceSum(componentMasks, [1, 2], keepDims: false));
            var confSums = ToDoubleArray(engine.ReduceSum(engine.TensorMultiply(componentMasks, classProb), [1, 2], keepDims: false));
            for (int comp = 0; comp < count; comp++)
            {
                int instId = nextInstId++;
                int area = (int)areas[comp];
                if (area > 0)
                {
                    segments.Add(new PanopticSegment<T>
                    {
                        SegmentId = segmentIdIsPanopticLabel ? cls * PanopticLabelDivisor + instId : instId,
                        ClassId = cls,
                        IsThing = true,
                        Confidence = confSums[comp] / area,
                        Area = area,
                        Mask = engine.Reshape(engine.TensorSlice(componentMasks, [comp, 0, 0], [1, h, w]), [h, w])
                    });
                }
            }
        }

        return new PanopticSegmentationResult<T> { SemanticMap = semanticMap, InstanceMap = instanceMap, PanopticMap = panopticMap, Segments = segments };
    }

    private static double[] ToDoubleArray<T>(Tensor<T> values) => CvTensorOps<T>.ToDoubleArray(values);

    /// <summary>
    /// Instance candidates from a per-pixel class map: every 4-connected component of each non-background
    /// class (1 .. <paramref name="numClasses"/>-1), in class then raster order, that covers at least four
    /// pixels and whose mean class probability reaches <paramref name="confidenceThreshold"/>; at most
    /// <paramref name="maxInstances"/>. Each carries its binary mask, its tight XYXY pixel box (exclusive
    /// right/bottom) and its mean probability as confidence.
    /// </summary>
    internal static List<InstanceSegmentation.InstanceMask<T>> ComponentInstances<T>(
        Tensor<T> classMap, Tensor<T> probMap, int numClasses, double confidenceThreshold, int maxInstances = 100)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int h = classMap.Shape[0], w = classMap.Shape[1];
        var instances = new List<InstanceSegmentation.InstanceMask<T>>();
        for (int cls = 1; cls < numClasses && instances.Count < maxInstances; cls++)
        {
            var (labelMap, count) = LabelConnectedComponents(classMap, cls);
            if (count == 0) continue;

            var componentIds = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(1, count + 1, 1), [count, 1, 1]), [count, h, w]);
            var masks = engine.TensorEquals(engine.TensorBroadcastTo(engine.Reshape(labelMap, [1, h, w]), [count, h, w]), componentIds);
            var classProb = engine.TensorBroadcastTo(engine.TensorSlice(probMap, [cls, 0, 0], [1, h, w]), [count, h, w]);
            var area = engine.ReduceSum(masks, [1, 2], keepDims: false);
            var confidence = SafeRatio(engine.ReduceSum(engine.TensorMultiply(masks, classProb), [1, 2], keepDims: false), area);
            var (minX, minY, maxX, maxY) = MaskExtents(masks, h, w);

            var areas = CvTensorOps<T>.ToDoubleArray(area);
            var confidences = confidence.ToArray();
            var confidenceValues = CvTensorOps<T>.ToDoubleArray(confidence);
            var x1 = minX.ToArray();
            var y1 = minY.ToArray();
            var x2 = engine.TensorAddScalar(maxX, numOps.One).ToArray();
            var y2 = engine.TensorAddScalar(maxY, numOps.One).ToArray();
            for (int comp = 0; comp < count && instances.Count < maxInstances; comp++)
            {
                if (areas[comp] < 4) continue;
                if (confidenceValues[comp] < confidenceThreshold) continue;
                var box = new BoundingBox<T>(x1[comp], y1[comp], x2[comp], y2[comp], BoundingBoxFormat.XYXY, cls);
                var mask = engine.Reshape(engine.TensorSlice(masks, [comp, 0, 0], [1, h, w]), [h, w]);
                instances.Add(new InstanceSegmentation.InstanceMask<T>(box, mask, cls, confidences[comp]));
            }
        }

        return instances;
    }

    /// <summary>
    /// Per mask of a <c>[N, H, W]</c> binary stack, the smallest and largest column and row it covers
    /// (each <c>[N]</c>). An empty mask yields <c>(W, H, 0, 0)</c>, the scan's starting values.
    /// </summary>
    internal static (Tensor<T> MinX, Tensor<T> MinY, Tensor<T> MaxX, Tensor<T> MaxY) MaskExtents<T>(Tensor<T> masks, int h, int w)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = masks.Shape[0];
        var inside = engine.TensorGreaterThan(masks, numOps.Zero);
        var cols = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, w, 1), [1, 1, w]), [n, h, w]);
        var rows = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, h, 1), [1, h, 1]), [n, h, w]);
        var widthFill = CvTensorOps<T>.Fill([n, h, w], numOps.FromDouble(w));
        var heightFill = CvTensorOps<T>.Fill([n, h, w], numOps.FromDouble(h));
        var zeros = new Tensor<T>([n, h, w]);
        var minX = engine.Reshape(engine.TensorAmin(engine.TensorWhere(inside, cols, widthFill), [1, 2], false), [n]);
        var minY = engine.Reshape(engine.TensorAmin(engine.TensorWhere(inside, rows, heightFill), [1, 2], false), [n]);
        var maxX = engine.Reshape(engine.ReduceMax(engine.TensorWhere(inside, cols, zeros), [1, 2], false), [n]);
        var maxY = engine.Reshape(engine.ReduceMax(engine.TensorWhere(inside, rows, zeros), [1, 2], false), [n]);
        return (minX, minY, maxX, maxY);
    }

    /// <summary>
    /// Greedy mask NMS: visits the instances by descending confidence (stable for ties) and keeps each one
    /// whose mask IoU with every instance already kept is below <paramref name="iouThreshold"/>.
    /// </summary>
    internal static List<InstanceSegmentation.InstanceMask<T>> MaskNms<T>(
        List<InstanceSegmentation.InstanceMask<T>> instances, double iouThreshold)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var sorted = SortByConfidenceDescending(instances);
        var kept = new List<InstanceSegmentation.InstanceMask<T>>();
        while (sorted.Count > 0)
        {
            var best = sorted[0];
            kept.Add(best);
            sorted.RemoveAt(0);
            sorted = sorted.Where(inst => best.ComputeMaskIoU(inst, numOps) < iouThreshold).ToList();
        }

        return kept;
    }

    /// <summary>The instances ordered by descending confidence; ties keep their input order.</summary>
    internal static List<InstanceSegmentation.InstanceMask<T>> SortByConfidenceDescending<T>(
        List<InstanceSegmentation.InstanceMask<T>> instances)
    {
        if (instances.Count == 0) return new List<InstanceSegmentation.InstanceMask<T>>();
        var confidences = new Tensor<T>([instances.Count], new Vector<T>(instances.Select(i => i.Confidence).ToArray()));
        var keys = CvTensorOps<T>.ToDoubleArray(confidences);
        return Enumerable.Range(0, instances.Count).OrderByDescending(i => keys[i]).Select(i => instances[i]).ToList();
    }
}
