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

        var numOps = MathHelper.GetNumericOperations<T>();
        int h = classMap.Shape[0], w = classMap.Shape[1];
        var labelMap = new Tensor<T>([h, w]);
        var visited = new bool[h * w];
        int componentCount = 0;
        double targetVal = targetClass;

        for (int row = 0; row < h; row++)
        {
            for (int col = 0; col < w; col++)
            {
                int idx = row * w + col;
                if (visited[idx]) continue;
                if (Math.Abs(numOps.ToDouble(classMap[row, col]) - targetVal) > 0.5) continue;

                componentCount++;
                var label = numOps.FromDouble(componentCount);
                var queue = new Queue<int>();
                queue.Enqueue(idx);
                visited[idx] = true;

                while (queue.Count > 0)
                {
                    int cur = queue.Dequeue();
                    int r = cur / w, c = cur % w;
                    labelMap[r, c] = label;

                    // 4-connected neighbors
                    if (r > 0 && !visited[(r - 1) * w + c] &&
                        Math.Abs(numOps.ToDouble(classMap[r - 1, c]) - targetVal) < 0.5)
                    { visited[(r - 1) * w + c] = true; queue.Enqueue((r - 1) * w + c); }
                    if (r < h - 1 && !visited[(r + 1) * w + c] &&
                        Math.Abs(numOps.ToDouble(classMap[r + 1, c]) - targetVal) < 0.5)
                    { visited[(r + 1) * w + c] = true; queue.Enqueue((r + 1) * w + c); }
                    if (c > 0 && !visited[r * w + c - 1] &&
                        Math.Abs(numOps.ToDouble(classMap[r, c - 1]) - targetVal) < 0.5)
                    { visited[r * w + c - 1] = true; queue.Enqueue(r * w + c - 1); }
                    if (c < w - 1 && !visited[r * w + c + 1] &&
                        Math.Abs(numOps.ToDouble(classMap[r, c + 1]) - targetVal) < 0.5)
                    { visited[r * w + c + 1] = true; queue.Enqueue(r * w + c + 1); }
                }
            }
        }

        return (labelMap, componentCount);
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

        var numOps = MathHelper.GetNumericOperations<T>();

        if (features.Rank == 4)
        {
            int c = features.Shape[1], h = features.Shape[2], w = features.Shape[3];
            int numWeights = Math.Min(c, weights.Length);
            var result = new Tensor<T>([h, w]);
            for (int ch = 0; ch < numWeights; ch++)
            {
                double wt = weights[ch];
                for (int row = 0; row < h; row++)
                    for (int col = 0; col < w; col++)
                        result[row, col] = numOps.Add(result[row, col],
                            numOps.FromDouble(numOps.ToDouble(features[0, ch, row, col]) * wt));
            }
            return result;
        }
        else
        {
            int c = features.Shape[0], h = features.Shape[1], w = features.Shape[2];
            int numWeights = Math.Min(c, weights.Length);
            var result = new Tensor<T>([h, w]);
            for (int ch = 0; ch < numWeights; ch++)
            {
                double wt = weights[ch];
                for (int row = 0; row < h; row++)
                    for (int col = 0; col < w; col++)
                        result[row, col] = numOps.Add(result[row, col],
                            numOps.FromDouble(numOps.ToDouble(features[ch, row, col]) * wt));
            }
            return result;
        }
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
        // Only the first C channels and the top-left H x W of features2 pair with features1.
        if (b.Shape[0] != c || b.Shape[1] != h || b.Shape[2] != w)
            b = engine.TensorSlice(b, [0, 0, 0], [c, h, w]);

        // Cosine similarity is scale-invariant, so each pixel's channel vector is first divided by its largest
        // magnitude: the squared norms then stay at most C, where raw squares overflow half precision (65504) for
        // features as small as 64 over 16 channels and the ratio became NaN.
        var (unitA, scaleA) = ScaleByChannelMax(a, h, w);
        var (unitB, scaleB) = ScaleByChannelMax(b, h, w);

        // dot / (|a| |b|) over the channel axis, 0 where the norm product |a| |b| is at most 1e-8.
        var dot = engine.ReduceSum(engine.TensorMultiply(unitA, unitB), [0], keepDims: false);
        var norm1 = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(unitA), [0], keepDims: false));
        var norm2 = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(unitB), [0], keepDims: false));
        var denom = engine.TensorMultiply(norm1, norm2);
        var originalDenom = engine.TensorMultiply(engine.TensorMultiply(denom, scaleA), scaleB);
        var valid = engine.TensorGreaterThan(originalDenom, numOps.FromDouble(1e-8));
        var safeDenom = engine.TensorWhere(valid, denom, CvTensorOps<T>.Fill([h, w], 1.0));
        return engine.TensorWhere(valid, engine.TensorDivide(dot, safeDenom), new Tensor<T>([h, w]));
    }

    /// <summary>
    /// <paramref name="features"/> <c>[C, H, W]</c> divided per pixel by its largest channel magnitude (1 where that
    /// is 0), and that magnitude <c>[H, W]</c> (0 for an all-zero pixel).
    /// </summary>
    private static (Tensor<T> Scaled, Tensor<T> Scale) ScaleByChannelMax<T>(Tensor<T> features, int h, int w)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var largest = engine.ReduceMax(engine.TensorAbs(features), [0], keepDims: true);
        var safe = engine.TensorWhere(engine.TensorGreaterThan(largest, numOps.Zero), largest, CvTensorOps<T>.Fill([1, h, w], 1.0));
        var scaled = engine.TensorDivide(features, engine.TensorBroadcastTo(safe, features.Shape.ToArray()));
        return (scaled, engine.Reshape(largest, [h, w]));
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
    /// <remarks>
    /// Each channel is divided by C before the sum: the engine's mean accumulates in T, and a half-precision running
    /// sum passes 65504 (and becomes infinite) for 256 channels near 300, where the mean itself is in range.
    /// </remarks>
    internal static Tensor<T> ChannelMean<T>(Tensor<T> features)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var share = engine.TensorDivideScalar(features, numOps.FromDouble(features.Shape[0]));
        return engine.ReduceSum(share, [0], keepDims: false);
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

}
