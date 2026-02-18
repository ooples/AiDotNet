using AiDotNet.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Static helper methods for common segmentation tensor operations (argmax, softmax, etc.)
/// used by models implementing segmentation interfaces.
/// </summary>
public static class SegmentationTensorOps
{
    /// <summary>
    /// Computes argmax along the class dimension, producing a per-pixel class index map.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="logits">Logits tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Class index map [H, W] or [B, H, W].</returns>
    public static Tensor<T> ArgmaxAlongClassDim<T>(Tensor<T> logits)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (logits.Rank == 3)
        {
            int c = logits.Shape[0], h = logits.Shape[1], w = logits.Shape[2];
            var result = new Tensor<T>([h, w]);
            for (int row = 0; row < h; row++)
            {
                for (int col = 0; col < w; col++)
                {
                    int bestClass = 0;
                    T bestVal = logits[0, row, col];
                    for (int cls = 1; cls < c; cls++)
                    {
                        T val = logits[cls, row, col];
                        if (numOps.GreaterThan(val, bestVal))
                        {
                            bestVal = val;
                            bestClass = cls;
                        }
                    }
                    result[row, col] = numOps.FromDouble(bestClass);
                }
            }
            return result;
        }
        else
        {
            int b = logits.Shape[0], c = logits.Shape[1], h = logits.Shape[2], w = logits.Shape[3];
            var result = new Tensor<T>([b, h, w]);
            for (int batch = 0; batch < b; batch++)
            {
                for (int row = 0; row < h; row++)
                {
                    for (int col = 0; col < w; col++)
                    {
                        int bestClass = 0;
                        T bestVal = logits[batch, 0, row, col];
                        for (int cls = 1; cls < c; cls++)
                        {
                            T val = logits[batch, cls, row, col];
                            if (numOps.GreaterThan(val, bestVal))
                            {
                                bestVal = val;
                                bestClass = cls;
                            }
                        }
                        result[batch, row, col] = numOps.FromDouble(bestClass);
                    }
                }
            }
            return result;
        }
    }

    /// <summary>
    /// Computes softmax along the class dimension, producing per-pixel probabilities.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="logits">Logits tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Probability map [C, H, W] or [B, C, H, W] with values in [0, 1].</returns>
    public static Tensor<T> SoftmaxAlongClassDim<T>(Tensor<T> logits)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(logits.Shape);

        if (logits.Rank == 3)
        {
            int c = logits.Shape[0], h = logits.Shape[1], w = logits.Shape[2];
            for (int row = 0; row < h; row++)
            {
                for (int col = 0; col < w; col++)
                {
                    T maxVal = logits[0, row, col];
                    for (int cls = 1; cls < c; cls++)
                    {
                        T val = logits[cls, row, col];
                        if (numOps.GreaterThan(val, maxVal)) maxVal = val;
                    }

                    T sumExp = numOps.Zero;
                    for (int cls = 0; cls < c; cls++)
                    {
                        T expVal = numOps.Exp(numOps.Subtract(logits[cls, row, col], maxVal));
                        result[cls, row, col] = expVal;
                        sumExp = numOps.Add(sumExp, expVal);
                    }

                    for (int cls = 0; cls < c; cls++)
                    {
                        result[cls, row, col] = numOps.Divide(result[cls, row, col], sumExp);
                    }
                }
            }
        }
        else
        {
            int b = logits.Shape[0], c = logits.Shape[1], h = logits.Shape[2], w = logits.Shape[3];
            for (int batch = 0; batch < b; batch++)
            {
                for (int row = 0; row < h; row++)
                {
                    for (int col = 0; col < w; col++)
                    {
                        T maxVal = logits[batch, 0, row, col];
                        for (int cls = 1; cls < c; cls++)
                        {
                            T val = logits[batch, cls, row, col];
                            if (numOps.GreaterThan(val, maxVal)) maxVal = val;
                        }

                        T sumExp = numOps.Zero;
                        for (int cls = 0; cls < c; cls++)
                        {
                            T expVal = numOps.Exp(numOps.Subtract(logits[batch, cls, row, col], maxVal));
                            result[batch, cls, row, col] = expVal;
                            sumExp = numOps.Add(sumExp, expVal);
                        }

                        for (int cls = 0; cls < c; cls++)
                        {
                            result[batch, cls, row, col] = numOps.Divide(result[batch, cls, row, col], sumExp);
                        }
                    }
                }
            }
        }

        return result;
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
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(values.Shape);
        var one = numOps.FromDouble(1.0);
        var zero = numOps.Zero;
        for (int i = 0; i < values.Length; i++)
            result[i] = numOps.ToDouble(values[i]) > threshold ? one : zero;
        return result;
    }

    /// <summary>
    /// Applies element-wise sigmoid to a tensor: 1 / (1 + exp(-x)).
    /// </summary>
    public static Tensor<T> Sigmoid<T>(Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            double x = numOps.ToDouble(input[i]);
            result[i] = numOps.FromDouble(1.0 / (1.0 + Math.Exp(-x)));
        }
        return result;
    }

    /// <summary>
    /// Generates a 2D Gaussian attention mask centered at (cx, cy) with given sigma.
    /// Output shape is [H, W] with values in (0, 1].
    /// </summary>
    public static Tensor<T> GaussianMask<T>(int height, int width, double cx, double cy, double sigma = 10.0)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>([height, width]);
        double invTwoSigmaSq = 1.0 / (2.0 * sigma * sigma);
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                double dx = col - cx, dy = row - cy;
                result[row, col] = numOps.FromDouble(Math.Exp(-(dx * dx + dy * dy) * invTwoSigmaSq));
            }
        }
        return result;
    }

    /// <summary>
    /// Creates a spatial mask that is 1.0 inside the box [x1, y1, x2, y2) and 0.0 outside.
    /// </summary>
    public static Tensor<T> BoxMask<T>(int height, int width, int x1, int y1, int x2, int y2)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>([height, width]);
        var one = numOps.FromDouble(1.0);
        int clampX1 = Math.Max(0, x1), clampY1 = Math.Max(0, y1);
        int clampX2 = Math.Min(width, x2), clampY2 = Math.Min(height, y2);
        for (int row = clampY1; row < clampY2; row++)
            for (int col = clampX1; col < clampX2; col++)
                result[row, col] = one;
        return result;
    }

    /// <summary>
    /// Computes weighted sum across the channel dimension.
    /// Features shape: [C, H, W], weights length: C. Output: [H, W].
    /// </summary>
    public static Tensor<T> WeightedChannelSum<T>(Tensor<T> features, double[] weights)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
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

    /// <summary>
    /// Generates deterministic channel weights from text using character-level hashing.
    /// The weights are normalized to unit length for use as cosine-similarity queries.
    /// </summary>
    public static double[] TextToWeights(string text, int numChannels)
    {
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
    /// Computes per-pixel cosine similarity between two feature maps [C, H, W].
    /// Output: [H, W] with values in [-1, 1].
    /// </summary>
    public static Tensor<T> PixelAffinity<T>(Tensor<T> features1, Tensor<T> features2)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int c = features1.Shape[0], h = features1.Shape[1], w = features1.Shape[2];
        var result = new Tensor<T>([h, w]);

        for (int row = 0; row < h; row++)
        {
            for (int col = 0; col < w; col++)
            {
                double dot = 0, norm1 = 0, norm2 = 0;
                for (int ch = 0; ch < c; ch++)
                {
                    double v1 = numOps.ToDouble(features1[ch, row, col]);
                    double v2 = numOps.ToDouble(features2[ch, row, col]);
                    dot += v1 * v2;
                    norm1 += v1 * v1;
                    norm2 += v2 * v2;
                }
                double denom = Math.Sqrt(norm1) * Math.Sqrt(norm2);
                result[row, col] = numOps.FromDouble(denom > 1e-8 ? dot / denom : 0.0);
            }
        }

        return result;
    }

    /// <summary>
    /// Warps masks from a reference frame to a target frame using pixel-wise affinity scores.
    /// Reference masks: [N, H, W], affinity: [H, W] with values in [0, 1].
    /// Output: [N, H, W] warped masks weighted by affinity.
    /// </summary>
    public static Tensor<T> WarpMasksByAffinity<T>(Tensor<T> refMasks, Tensor<T> affinity)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = refMasks.Shape[0], h = refMasks.Shape[1], w = refMasks.Shape[2];
        var result = new Tensor<T>([n, h, w]);

        for (int obj = 0; obj < n; obj++)
            for (int row = 0; row < h; row++)
                for (int col = 0; col < w; col++)
                    result[obj, row, col] = numOps.FromDouble(
                        numOps.ToDouble(refMasks[obj, row, col]) *
                        numOps.ToDouble(affinity[row, col]));

        return result;
    }
}
